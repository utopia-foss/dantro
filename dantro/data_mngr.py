"""This module implements the DataManager class, the root of the data tree."""

import copy
import datetime
import functools
import glob
import logging
import multiprocessing as mp
import os
import re
import time
import warnings
from typing import Callable, Dict, List, Tuple, Union

from ._hash import _hash
from ._import_tools import LazyLoader
from .base import PATH_JOIN_CHAR, BaseDataContainer, BaseDataGroup
from .exceptions import *
from .groups import OrderedDataGroup
from .tools import (
    PoolCallbackHandler,
    PoolErrorCallbackHandler,
    clear_line,
    fill_line,
    format_bytesize,
)
from .tools import format_time as _format_time
from .tools import load_yml, print_line, recursive_update, total_bytesize

log = logging.getLogger(__name__)

pkl = LazyLoader("dill")

DATA_TREE_DUMP_EXT = ".d3"
"""File extension for data cache file"""

_fmt_time = lambda seconds: _format_time(seconds, ms_precision=2)
"""Locally used time formatting function"""


# -----------------------------------------------------------------------------


def _load_file_wrapper(
    filepath: str, *, dm: "DataManager", loader: str, **kwargs
) -> Tuple[BaseDataGroup, str]:
    """A wrapper around :py:meth:`~dantro.data_mngr.DataManager._load_file`
    that is used for parallel loading via multiprocessing.Pool.
    It takes care of resolving the loader function and instantiating the file-
    loading method.

    This function needs to be on the module scope such that it is pickleable.
    For that reason, loader resolution also takes place here, because pickling
    the load function may be problematic.

    Args:
        filepath (str): The path of the file to load data from
        dm (DataManager): The DataManager instance to resolve the loader from
        loader (str): The namer of the loader
        **kwargs: Any further loading arguments.

    Returns:
        Tuple[BaseDataContainer, str]: The return value of
            :py:meth:`~dantro.data_mngr.DataManager._load_file`.
    """
    load_func, TargetCls = dm._resolve_loader(loader)
    return dm._load_file(
        filepath,
        loader=loader,
        load_func=load_func,
        TargetCls=TargetCls,
        **kwargs,
    )


def _parse_parallel_opts(
    files: List[str],
    *,
    enabled: bool = True,
    processes: int = None,
    min_files: int = 2,
    min_total_size: int = None,
) -> int:
    """Parser function for the parallel file loading options dict

    Args:
        files (List[str]): List of files that are to be loaded
        enabled (bool, optional): Whether to use parallel loading. If True,
            the threshold arguments will still need to be fulfilled.
        processes (int, optional): The number of processors to use; if this is
            a negative integer, will deduce from available CPU count.
        min_files (int, optional): If there are fewer files to load than this
            number, will *not* use parallel loading.
        min_total_size (int, optional): If the total file size is smaller than
            this file size (in bytes), will *not* use parallel loading.

    Returns:
        int: number of processes to use. Will return 1 if loading should *not*
            happen in parallel. Additionally, this number will never be larger
            than the number of files in order to prevent unnecessary processes.
    """
    if not enabled:
        return 1

    # Minimum files threshold
    if min_files and len(files) < min_files:
        log.remark(
            "Not loading in parallel: there are only %d < %d files to load.",
            len(files),
            min_files,
        )
        return 1

    # Minimum total file size
    if min_total_size:
        min_total_size = int(min_total_size)
        fs = total_bytesize(files)
        if fs < min_total_size:
            log.remark(
                "Not loading in parallel: total file size was %s < %s.",
                format_bytesize(fs),
                format_bytesize(min_total_size),
            )
            return 1

    # Number of processes to use
    cpu_count = os.cpu_count()
    if processes:
        processes = processes if processes >= 0 else (cpu_count + processes)

        # Little warning if choosing too many processes
        if processes > cpu_count:
            log.caution(
                "Loading with more processes (%d) than there are CPUs (%d) "
                "will typically slow down file loading!",
                processes,
                cpu_count,
            )
    else:
        processes = cpu_count

    return min(processes, len(files))


# -----------------------------------------------------------------------------


class DataManager(OrderedDataGroup):
    """The DataManager is the root of a data tree, coupled to a specific data
    directory.

    It handles the loading of data and can be used for interactive work with
    the data.
    """

    # Use a base load configuration to start with
    _BASE_LOAD_CFG = None

    # Define some default groups (same syntax create_groups argument)
    _DEFAULT_GROUPS = None

    # Define as class variable what should be the default group type
    _DATA_GROUP_DEFAULT_CLS = OrderedDataGroup

    # For simple lookups, store class names in a dict; not set by default
    _DATA_GROUP_CLASSES = None

    # The default tree file cache path, parsed by _parse_file_path
    _DEFAULT_TREE_CACHE_PATH = ".tree_cache.d3"

    # .........................................................................
    # Initialization

    def __init__(
        self,
        data_dir: str,
        *,
        name: str = None,
        load_cfg: Union[dict, str] = None,
        out_dir: Union[str, bool] = "_output/{timestamp:}",
        out_dir_kwargs: dict = None,
        create_groups: List[Union[str, dict]] = None,
        condensed_tree_params: dict = None,
        default_tree_cache_path: str = None,
    ):
        """Initializes a DataManager for the specified data directory.

        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            name (str, optional): which name to give to the DataManager. If no
                name is given, the data directories basename will be used
            load_cfg (Union[dict, str], optional): The base configuration used
                for loading data. If a string is given, assumes it to be the
                path to a YAML file and loads it using the
                :py:func:`~dantro._yaml.load_yml` function. If None is given,
                it can still be supplied to the
                :py:meth:`~dantro.data_mngr.DataManager.load` method later on.
            out_dir (Union[str, bool], optional): where output is written to.
                If this is given as a relative path, it is considered relative
                to the ``data_dir``. A formatting operation with the keys
                ``timestamp`` and ``name`` is performed on this, where the
                latter is the name of the data manager. If set to False, no
                output directory is created.
            out_dir_kwargs (dict, optional): Additional arguments that affect
                how the output directory is created.
            create_groups (List[Union[str, dict]], optional): If given, these
                groups will be created after initialization. If the list
                entries are strings, the default group class will be used; if
                they are dicts, the `name` key specifies the name of the group
                and the `Cls` key specifies the type. If a string is given
                instead of a type, the lookup happens from the
                ``_DATA_GROUP_CLASSES`` variable.
            condensed_tree_params (dict, optional): If given, will set the
                parameters used for the condensed tree representation.
                Available options: ``max_level`` and ``condense_thresh``, where
                the latter may be a callable.
                See :py:meth:`dantro.base.BaseDataGroup._tree_repr` for more
                information.
            default_tree_cache_path (str, optional): The path to the default
                tree cache file. If not given, uses the value from the class
                variable ``_DEFAULT_TREE_CACHE_PATH``. Whichever value was
                chosen is then prepared using the
                :py:meth:`~dantro.data_mngr.DataManager._parse_file_path`
                method, which regards relative paths as being relative to the
                associated data directory.
        """
        # Find a name if none was given
        if not name:
            basename = os.path.basename(os.path.abspath(data_dir))
            name = "{}_Manager".format(basename.replace(" ", "_"))

        log.info("Initializing %s '%s'...", self.classname, name)

        # Initialize as a data group via parent class
        super().__init__(name=name)

        # Set condensed tree parameters
        if condensed_tree_params:
            self._set_condensed_tree_params(**condensed_tree_params)

        # Initialize directories
        self.dirs = self._init_dirs(
            data_dir=data_dir,
            out_dir=out_dir,
            **(out_dir_kwargs if out_dir_kwargs else {}),
        )

        # Parse the default tree cache path
        _tcp = (
            default_tree_cache_path
            if default_tree_cache_path
            else self._DEFAULT_TREE_CACHE_PATH
        )
        self._tree_cache_path = self._parse_file_path(_tcp)

        # Start out with the default load configuration or, if not given, with
        # an empty one
        self.load_cfg = {} if not self._BASE_LOAD_CFG else self._BASE_LOAD_CFG

        # Resolve string arguments
        if isinstance(load_cfg, str):
            # Assume this is the path to a configuration file and load it
            log.debug(
                "Loading the default load config from a path:\n  %s", load_cfg
            )
            load_cfg = load_yml(load_cfg)

        # If given, use it to recursively update the base
        if load_cfg:
            self.load_cfg = recursive_update(self.load_cfg, load_cfg)

        # Create default groups, as specified in the _DEFAULT_GROUPS class
        # variable and the create_groups argument
        if self._DEFAULT_GROUPS or create_groups:
            # Parse both into a new list to iterate over
            specs = self._DEFAULT_GROUPS if self._DEFAULT_GROUPS else []
            if create_groups:
                specs += create_groups

            log.debug(
                "Creating %d empty groups from defaults and/or given "
                "initialization arguments ...",
                len(specs),
            )
            for spec in specs:
                if isinstance(spec, dict):
                    # Got a more elaborate group specification
                    self.new_group(**spec)

                else:
                    # Assume this is the group name; use the default class
                    self.new_group(spec)

        # Done
        log.debug("%s initialized.", self.logstr)

    def _set_condensed_tree_params(self, **params):
        """Helper method to set the ``_COND_TREE_*`` class variables"""
        available_keys = ("max_level", "condense_thresh")

        for key, value in params.items():
            if key.lower() not in available_keys:
                raise KeyError(
                    "Invalid condensed tree parameter: '{}'! The "
                    "available keys are: {}."
                    "".format(key, ", ".join(available_keys))
                )
            setattr(self, "_COND_TREE_" + key.upper(), value)

    def _init_dirs(
        self,
        *,
        data_dir: str,
        out_dir: Union[str, bool],
        timestamp: float = None,
        timefstr: str = "%y%m%d-%H%M%S",
        exist_ok: bool = False,
    ) -> Dict[str, str]:
        """Initializes the directories managed by this DataManager and returns
        a dictionary that stores the absolute paths to these directories.

        If they do not exist, they will be created.

        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            out_dir (Union[str, bool]): where output is written to.
                If this is given as a relative path, it is considered relative
                to the **data directory**. A formatting operation with the
                keys ``timestamp`` and ``name`` is performed on this, where
                the latter is the name of the data manager. If set to False,
                no output directory is created.
            timestamp (float, optional): If given, use this time to generate
                the `date` format string key. If not, uses the current time.
            timefstr (str, optional): Format string to use for generating the
                string representation of the current timestamp
            exist_ok (bool, optional): Whether the output directory may exist.
                Note that it only makes sense to set this to True if you can
                be sure that there will be no file conflicts! Otherwise the
                errors will just occur at a later stage.

        Returns:
            Dict[str, str]: The directory paths registered under certain keys,
                e.g. ``data`` and ``out``.
        """

        # Make the data directory absolute
        log.debug("Received `data_dir` argument:\n  %s", data_dir)
        data_dir = os.path.abspath(data_dir)

        # Save dictionary that will hold info on directories
        dirs = dict(data=data_dir)

        # See if an output directory should be created
        if out_dir:
            log.debug("Received `out_dir` argument:\n  %s", out_dir)

            # Make current date and time available for formatting operations
            time = (
                datetime.datetime.fromtimestamp(timestamp)
                if timestamp
                else datetime.datetime.now()
            )
            timestr = time.strftime(timefstr)

            # Perform a format operation on the output directory
            out_dir = out_dir.format(name=self.name.lower(), timestamp=timestr)

            # If it is relative, assume it to be relative to the data directory
            if not os.path.isabs(out_dir):
                # By joining them together, out_dir is now relative
                out_dir = os.path.join(data_dir, out_dir)

            # Make path absolute and store in dict
            dirs["out"] = os.path.abspath(out_dir)

            # Create the directory
            os.makedirs(dirs["out"], exist_ok=exist_ok)

        else:
            dirs["out"] = False

        # Inform about the managed directories, then return
        log.debug(
            "Managed directories:\n%s",
            "\n".join([f"  {k:>8s} : {v}" for k, v in dirs.items()]),
        )

        return dirs

    @property
    def hashstr(self) -> str:
        """The hash of a DataManager is computed from its name and the coupled
        data directory, which are regarded as the relevant parts. While other
        parts of the DataManager are not invariant, it is characterized most by
        the directory it is associated with.

        As this is a string-based hash, it is not implemented as the __hash__
        magic method but as a separate property.

        WARNING Changing how the hash is computed for the DataManager will
                invalidate all TransformationDAG caches.
        """
        return _hash(
            "<DataManager '{}' @ {}>".format(self.name, self.dirs["data"])
        )

    def __hash__(self) -> int:
        """The hash of this DataManager, computed from the hashstr property"""
        return hash(self.hashstr)

    @property
    def tree_cache_path(self) -> str:
        """Absolute path to the default tree cache file"""
        return self._tree_cache_path

    @property
    def tree_cache_exists(self) -> bool:
        """Whether the tree cache file exists"""
        return os.path.isfile(self._tree_cache_path)

    # .........................................................................
    # Loading data

    def load_from_cfg(
        self,
        *,
        load_cfg: dict = None,
        update_load_cfg: dict = None,
        exists_action: str = "raise",
        print_tree: Union[bool, str] = False,
    ) -> None:
        """Load multiple data entries using the specified load configuration.

        Args:
            load_cfg (dict, optional): The load configuration to use. If not
                given, the one specified during initialization is used.
            update_load_cfg (dict, optional): If given, it is used to update
                the load configuration recursively
            exists_action (str, optional): The behaviour upon existing data.
                Can be: ``raise`` (default), ``skip``, ``skip_nowarn``,
                ``overwrite``, ``overwrite_nowarn``. With the ``*_nowarn``
                values, no warning is given if an entry already existed.
            print_tree (Union[bool, str], optional): If True, the full tree
                representation of the DataManager is printed after the data
                was loaded. If ``'condensed'``, the condensed tree will be
                printed.

        Raises:
            TypeError: Raised if a given configuration entry was of invalid
                type, i.e. not a dict
        """
        # Determine which load configuration to use
        if not load_cfg:
            log.debug(
                "No new load configuration given; will use load "
                "configuration given at initialization."
            )
            load_cfg = self.load_cfg

        # Make sure to work on a copy, be it on the defaults or on the passed
        load_cfg = copy.deepcopy(load_cfg)

        if update_load_cfg:
            # Recursively update with the given keywords
            load_cfg = recursive_update(load_cfg, update_load_cfg)
            log.debug("Updated the load configuration.")

        log.hilight("Loading %d data entries ...", len(load_cfg))

        # Loop over the data entries that were configured to be loaded
        for entry_name, params in load_cfg.items():
            # Check if this is of valid type
            if not isinstance(params, dict):
                raise TypeError(
                    "Got invalid load specifications for entry "
                    f"'{entry_name}'! Expected dict, got {type(params)} with "
                    f"value '{params}'. Check the correctness of the given "
                    "load configuration!"
                )

            # Use the public method to load this single entry
            self.load(
                entry_name,
                exists_action=exists_action,
                print_tree=False,  # to not have prints during loading
                **params,
            )

        # All done
        log.success("Successfully loaded %d data entries.", len(load_cfg))

        # Finally, print the tree
        if print_tree:
            if print_tree == "condensed":
                print(self.tree_condensed)
            else:
                print(self.tree)

    def load(
        self,
        entry_name: str,
        *,
        loader: str,
        enabled: bool = True,
        glob_str: Union[str, List[str]],
        base_path: str = None,
        target_group: str = None,
        target_path: str = None,
        print_tree: Union[bool, str] = False,
        load_as_attr: bool = False,
        parallel: Union[bool, dict] = False,
        **load_params,
    ) -> None:
        """Performs a single load operation.

        Args:
            entry_name (str): Name of this entry; will also be the name of the
                created group or container, unless ``target_basename`` is given
            loader (str): The name of the loader to use
            enabled (bool, optional): Whether the load operation is enabled.
                If not, simply returns without loading any data or performing
                any further checks.
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings by which to identify the files within ``data_dir`` that
                are to be loaded using the given loader function
            base_path (str, optional): The base directory to concatenate the
                glob string to; if None, will use the DataManager's data
                directory. With this option, it becomes possible to load data
                from a path outside the associated data directory.
            target_group (str, optional): If given, the files to be loaded will
                be stored in this group. This may only be given if the argument
                target_path is *not* given.
            target_path (str, optional): The path to write the data to. This
                can be a format string. It is evaluated for each file that has
                been matched. If it is not given, the content is loaded to a
                group with the name of this entry at the root level.
                Available keys are: ``basename``, ``match`` (if ``path_regex``
                is used, see ``**load_params``)
            print_tree (Union[bool, str], optional): If True, the full tree
                representation of the DataManager is printed after the data
                was loaded. If ``'condensed'``, the condensed tree will be
                printed.
            load_as_attr (bool, optional): If True, the loaded entry will be
                added not as a new DataContainer or DataGroup, but as an
                attribute to an (already existing) object at ``target_path``.
                The name of the attribute will be the ``entry_name``.
            parallel (Union[bool, dict]): If True, data is loaded in parallel.
                If a dict, can supply more options:

                    - ``enabled``: whether to use parallel loading
                    - ``processes``: how many processes to use; if None, will
                      use as many as are available. For negative integers, will
                      use ``os.cpu_count() + processes`` processes.
                    - ``min_files``: if given, will fall back to non-parallel
                      loading if fewer than the given number of files were
                      matched by ``glob_str``
                    - ``min_size``: if given, specifies the minimum *total*
                      size of all matched files (in bytes) below which to fall
                      back to non-parallel loading

                Note that a single file will *never* be loaded in parallel and
                there will never be more processes used than files that were
                selected to be loaded.
                Parallel loading incurs a constant overhead and is typically
                only speeding up data loading if the task is CPU-bound. Also,
                it requires the data tree to be fully serializable.

            **load_params: Further loading parameters, all optional. These are
                evaluated by :py:meth:`~dantro.data_mngr.DataManager._load`.

                ignore (list):
                    The exact file names in this list will be ignored during
                    loading. Paths are seen as elative to the data directory
                    of the data manager.
                required (bool):
                    If True, will raise an error if no files were found.
                    Default: False.
                path_regex (str):
                    This pattern can be used to match the path of the file
                    that is being loaded. The match result is available to the
                    format string under the ``match`` key.
                exists_action (str):
                    The behaviour upon existing data.
                    Can be: ``raise`` (default), ``skip``, ``skip_nowarn``,
                    ``overwrite``, ``overwrite_nowarn``.
                    With ``*_nowarn`` values, no warning is given if an entry
                    already existed. Note that this is ignored when
                    the ``load_as_attr`` argument is given.
                unpack_data (bool, optional):
                    If True, and ``load_as_attr`` is active, not the
                    DataContainer or DataGroup itself will be stored in the
                    attribute, but the content of its ``.data`` attribute.
                progress_indicator (bool):
                    Whether to print a progress indicator or not. Default: True
                any further kwargs:
                    passed on to the loader function

        Returns:
            None

        Raises:
            ValueError: Upon invalid combination of ``target_group`` and
                ``target_path`` arguments
        """

        def glob_match_single(glob_str: Union[str, List[str]]) -> bool:
            """Returns True if the given glob str matches at most one file."""
            return bool(isinstance(glob_str, str) and glob_str.find("*") < 0)

        def check_target_path(target_path: str):
            """Check that the target path evaluates correctly."""
            log.debug("Checking target path '%s' ...", target_path)
            try:
                _target_path = target_path.format(
                    basename="basename", match="match"
                )

            except (IndexError, KeyError) as err:
                raise ValueError(
                    "Invalid argument `target_path`. Will not be able to "
                    f"properly evaluate '{target_path}' later due to "
                    f"a {type(err)}: {err}"
                ) from err
            else:
                log.debug("Target path will be:  %s", _target_path)

        # Initial checks
        if not enabled:
            log.progress("Skipping loading of data entry '%s' ...", entry_name)
            return
        log.progress("Loading data entry '%s' ...", entry_name)

        t0 = time.time()

        # Parse the arguments that result in the target path
        if load_as_attr:
            if not target_path:
                raise ValueError(
                    "With `load_as_attr`, the `target_path` "
                    "argument needs to be given."
                )

            # The target path should not be adjusted, as it points to the
            # object to store the loaded data as attribute in.
            log.debug(
                "Will load this entry as attribute to the target path "
                "'%s' ...",
                target_path,
            )

            # To communicate the attribute name, store it in the load_as_attr
            # variable; otherwise it would require passing two arguments to
            # _load
            load_as_attr = entry_name

        elif target_group:
            if target_path:
                raise ValueError(
                    "Received both arguments `target_group` and "
                    "`target_path`; make sure to only pass one "
                    "or none of them."
                )

            if glob_match_single(glob_str):
                target_path = target_group + "/" + entry_name
            else:
                target_path = target_group + "/{basename:}"

        elif not target_path:
            if glob_match_single(glob_str):
                target_path = entry_name
            else:
                target_path = entry_name + "/{basename:}"

        # else: target_path was given

        # ...and check that it is working.
        check_target_path(target_path)

        # Try loading the data and handle specific DataManagerErrors
        try:
            num_files, num_success = self._load(
                target_path=target_path,
                loader=loader,
                glob_str=glob_str,
                base_path=base_path,
                load_as_attr=load_as_attr,
                parallel=parallel,
                **load_params,
            )

        except RequiredDataMissingError:
            raise

        except MissingDataError as err:
            warnings.warn(
                "No files were found to import!\n" + str(err),
                MissingDataWarning,
            )
            return  # Does not raise, but does not save anything either

        except LoaderError:
            raise

        else:
            # Everything loaded as desired
            log.progress(
                "Loaded all data for entry '%s' in %s.\n",
                entry_name,
                _fmt_time(time.time() - t0),
            )

        # Done with this entry. Print tree, if desired.
        if print_tree == "condensed":
            print(self.tree_condensed)
        elif print_tree:
            print(self.tree)

    def _load(
        self,
        *,
        target_path: str,
        loader: str,
        glob_str: Union[str, List[str]],
        load_as_attr: Union[str, None],
        base_path: str = None,
        ignore: List[str] = None,
        required: bool = False,
        path_regex: str = None,
        exists_action: str = "raise",
        unpack_data: bool = False,
        progress_indicator: bool = True,
        parallel: Union[bool, dict] = False,
        **loader_kwargs,
    ) -> Tuple[int, int]:
        """Helper function that loads a data entry to the specified path.

        Args:
            target_path (str): The path to load the result of the loader to.
                This can be a format string; it is evaluated for each file.
                Available keys are: basename, match (if ``path_regex`` is
                given)
            loader (str): The loader to use
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings to match files in the data directory
            load_as_attr (Union[str, None]): If a string, the entry will be
                loaded into the object at ``target_path`` under a new attribute
                with this name.
            base_path (str, optional): The base directory to concatenate the
                glob string to; if None, will use the DataManager's data
                directory. With this option, it becomes possible to load data
                from a path outside the associated data directory.
            ignore (List[str], optional): The exact file names in this list
                will be ignored during loading. Paths are seen as relative to
                the data directory.
            required (bool, optional): If True, will raise an error if no files
                were found or if loading of a file failed.
            path_regex (str, optional): The regex applied to the relative path
                of the files that were found. It is used to generate the name
                of the target container. If not given, the basename is used.
            exists_action (str, optional): The behaviour upon existing data.
                Can be: ``raise`` (default), ``skip``, ``skip_nowarn``,
                ``overwrite``, ``overwrite_nowarn``. With ``*_nowarn`` values,
                no warning is given if an entry already existed.
                Note that this is ignored if ``load_as_attr`` is given.
            unpack_data (bool, optional): If True, and ``load_as_attr`` is
                active, not the DataContainer or DataGroup itself will be
                stored in the attribute, but the content of its ``.data``
                attribute.
            progress_indicator (bool, optional): Whether to print a progress
                indicator or not
            parallel (Union[bool, dict]): If True, data is loaded in parallel.
                If a dict, can supply more options:

                    - ``enabled``: whether to use parallel loading
                    - ``processes``: how many processes to use; if None, will
                      use as many as are available. For negative integers, will
                      use ``os.cpu_count() + processes`` processes.
                    - ``min_files``: if given, will fall back to non-parallel
                      loading if fewer than the given number of files were
                      matched by ``glob_str``
                    - ``min_size``: if given, specifies the minimum *total*
                      size of all matched files (in bytes) below which to fall
                      back to non-parallel loading

                Note that a single file will *never* be loaded in parallel and
                there will never be more processes used than files that were
                selected to be loaded.
                Parallel loading incurs a constant overhead and is typically
                only speeding up data loading if the task is CPU-bound. Also,
                it requires the data tree to be fully serializable.

            **loader_kwargs: passed on to the loader function

        Raises:
            ValueError: Bad ``path_regex``

        Returns:
            Tuple[int, int]: Tuple of number of files that matched the glob
                strings, *including* those that may have been skipped, and
                number of successfully loaded and stored entries
        """

        def _print_line(s: str):
            if progress_indicator:
                print_line(s)

        # Create the list of file paths to load
        files = self._create_files_list(
            glob_str=glob_str,
            ignore=ignore,
            required=required,
            base_path=base_path,
            sort=True,
        )

        # If a regex pattern was specified, compile it
        path_sre = re.compile(path_regex) if path_regex else None

        # Check if the `match` key is being used in the target_path
        if path_sre is not None and target_path.find("{match:") < 0:
            raise ValueError(
                "Received the `path_regex` argument to match the "
                "file path, but the `target_path` argument did "
                "not contain the corresponding `{match:}` "
                f"placeholder. `target_path` value: '{target_path}'."
            )

        # Parse the parallel argument, assuming that it's a dict or boolean
        if isinstance(parallel, bool):
            parallel = dict(enabled=parallel)
        elif isinstance(parallel, int):
            parallel = dict(enabled=True, processes=parallel)
        num_procs = _parse_parallel_opts(files, **parallel)

        # Loading . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        num_success = 0
        if num_procs <= 1:
            # Get the loader function; it's the same for all files
            load_func, TargetCls = self._resolve_loader(loader)

            # Go over the files and load them
            for n, filepath in enumerate(files):
                self._progress_info_str = f"  Loading  {n+1}/{len(files)}  ..."
                _print_line(self._progress_info_str)

                # Loading of the file and storing of the resulting object
                _obj, _target_path = self._load_file(
                    filepath,
                    loader=loader,
                    load_func=load_func,
                    target_path=target_path,
                    path_sre=path_sre,
                    load_as_attr=load_as_attr,
                    TargetCls=TargetCls,
                    required=required,
                    **loader_kwargs,
                )
                num_success += self._store_object(
                    _obj,
                    target_path=_target_path,
                    as_attr=load_as_attr,
                    unpack_data=unpack_data,
                    exists_action=exists_action,
                )

        else:
            # Set up callback handlers, load function, and other pool arguments
            cb = PoolCallbackHandler(
                len(files), silent=(not progress_indicator)
            )
            ecb = PoolErrorCallbackHandler()

            _load_file = functools.partial(
                _load_file_wrapper,
                dm=self,
                loader=loader,
                target_path=target_path,
                path_sre=path_sre,
                load_as_attr=load_as_attr,
                required=True,  # errors are handled via ErrorCallbackHandler
                **loader_kwargs,
            )

            # Inform the user about what's going to happen
            # Important to set the progress info string here, which loaders
            # may use to inform about the progress themselves (by concurrently
            # writing to the same line in terminal...)
            log.remark(
                "Loading %d files using %d processes ...",
                len(files),
                num_procs,
            )
            self._progress_info_str = "  Loading in parallel ..."
            _print_line("  Loading ...")

            # Create pool and populate with tasks, storing the AsyncResults
            with mp.Pool(num_procs) as pool:
                results = [
                    pool.apply_async(
                        _load_file, (_f,), callback=cb, error_callback=ecb
                    )
                    for _f in files
                ]
                pool.close()
                pool.join()

            # Merge the results ...
            _print_line("  Merging results ...")
            for result in results:
                try:
                    _obj, _target_path = result.get()
                except Exception as exc:
                    ecb.track_error(exc)
                    continue

                num_success += self._store_object(
                    _obj,
                    target_path=_target_path,
                    as_attr=load_as_attr,
                    unpack_data=unpack_data,
                    exists_action=exists_action,
                )

            if ecb.errors:
                _errors = "\n\n".join(
                    [f" - {e.__class__.__name__}: {e}" for e in ecb.errors]
                )
                err_msg = (
                    f"There were {len(ecb.errors)} errors during parallel "
                    f"loading of data from {len(files)} files!\n"
                    "The errors were:\n\n"
                    f"{_errors}\n\n"
                    "Check if the errors may be caused by loading in "
                    "parallel, e.g. because of unpickleable objects in the "
                    "loaded data tree."
                )
                if required:
                    raise DataLoadingError(
                        err_msg + "\nUnset `required` to ignore this error."
                    )
                log.error(err_msg + "\nSet `required` to raise an error.")

        # Clear the line to get rid of the progress indicator in case there
        # were no errors (otherwise the last message might get chopped off)
        if progress_indicator and num_success == len(files):
            clear_line()

        log.debug("Finished loading data from %d file(s).", len(files))
        return len(files), num_success

    def _load_file(
        self,
        filepath: str,
        *,
        loader: str,
        load_func: Callable,
        target_path: str,
        path_sre: str,
        load_as_attr: str,
        TargetCls: type,
        required: bool,
        **loader_kwargs,
    ) -> Tuple[Union[None, BaseDataContainer], List[str]]:
        """Loads the data of a single file into a dantro object and returns
        the loaded object (or None) and the parsed target path key sequence.
        """
        # Prepare the target path (will be a key sequence, ie. list of keys)
        _target_path = self._prepare_target_path(
            target_path, filepath=filepath, path_sre=path_sre
        )

        # Distinguish regular loading and loading as attribute, and prepare the
        # target class correspondingly to assure that the name is already
        # correct. An object of the target class will then be filled by the
        # load function.
        _name = _target_path[-1] if not load_as_attr else load_as_attr
        _TargetCls = lambda **kws: TargetCls(name=_name, **kws)

        # Let the load function retrieve the data
        try:
            _data = load_func(filepath, TargetCls=_TargetCls, **loader_kwargs)

        except Exception as exc:
            err_msg = (
                f"Failed loading file {filepath} with '{loader}' "
                f"loader!\nGot {exc.__class__.__name__}: {exc}"
            )
            if required:
                raise DataLoadingError(
                    err_msg + "\nUnset `required` to ignore this error."
                ) from exc
            log.error(err_msg + "\nSet `required` to raise an error.")
            return None, _target_path

        log.debug(
            "Successfully loaded file '%s' into %s.", filepath, _data.logstr
        )
        return _data, _target_path

    def _resolve_loader(self, loader: str) -> Tuple[Callable, type]:
        """Resolves the loader function and returns a 2-tuple containing the
        load function and the declared dantro target type to load data to.
        """
        load_func_name = "_load_" + loader.lower()
        try:
            load_func = getattr(self, load_func_name)

        except AttributeError as err:
            raise LoaderError(
                f"Loader '{loader}' was not available to {self.logstr}! "
                "Make sure to use a mixin class that supplies "
                f"the '{load_func_name}' loader method."
            ) from err
        else:
            log.debug("Resolved '%s' loader function.", loader)

        try:
            TargetCls = getattr(load_func, "TargetCls")

        except AttributeError as err:
            raise LoaderError(
                f"Load method '{load_func}' misses required attribute "
                "'TargetCls'. Check your mixin!"
            ) from err

        return load_func, TargetCls

    def _create_files_list(
        self,
        *,
        glob_str: Union[str, List[str]],
        ignore: List[str],
        base_path: str = None,
        required: bool = False,
        sort: bool = False,
    ) -> List[str]:
        """Create the list of file paths to load from.

        Internally, this uses a set, thus ensuring that the paths are
        unique. The set is converted to a list before returning.

        Args:
            glob_str (Union[str, List[str]]): The glob pattern or a list of
                glob patterns
            ignore (List[str]): The list of files to ignore
            base_path (str, optional): The base path for the glob pattern;
                use data directory, if not given.
            required (bool, optional): Will lead to an error being raised
                if no files could be matched
            sort (bool, optional): If true, sorts the list before returning

        Returns:
            list: the file paths to load

        Raises:
            MissingDataError: If no files could be matched
            RequiredDataMissingError: If no files could be matched but were
                required.
        """
        # Create a set to assure that all files are unique
        files = set()

        # Assure it is a list of strings
        if isinstance(glob_str, str):
            # Is a single glob string
            # Put it into a list to handle the same as the given arg
            glob_str = [glob_str]

        # Assuming glob_str to be lists of strings now
        log.debug(
            "Got %d glob string(s) to create set of matching file "
            "paths from.",
            len(glob_str),
        )

        # Handle base path, defaulting to the data directory
        if base_path is None:
            base_path = self.dirs["data"]
            log.debug("Using data directory as base path.")

        else:
            if not os.path.isabs(base_path):
                raise ValueError(
                    "Given base_path argument needs be an "
                    f"absolute path, was not: {base_path}"
                )

        # Go over the given glob strings and add to the files set
        for gs in glob_str:
            # Make the glob string absolute
            gs = os.path.join(base_path, gs)
            log.debug("Adding files that match glob string:\n  %s", gs)

            # Add to the set of files; this assures uniqueness of the paths
            files.update(list(glob.glob(gs, recursive=True)))

        # See if some files should be ignored
        if ignore:
            log.debug("Got list of files to ignore:\n  %s", ignore)

            # Make absolute and generate list of files to exclude
            ignore = [os.path.join(self.dirs["data"], path) for path in ignore]

            log.debug("Removing them one by one now ...")

            # Remove the elements one by one
            while ignore:
                rmf = ignore.pop()
                try:
                    files.remove(rmf)
                except KeyError:
                    log.debug("%s was not found in set of files.", rmf)
                else:
                    log.debug("%s removed from set of files.", rmf)

        # Now the file list is final
        log.note(
            "Found %d file%s with a total size of %s.",
            len(files),
            "s" if len(files) != 1 else "",
            format_bytesize(total_bytesize(files)),
        )
        log.debug("\n  %s", "\n  ".join(files))

        if not files:
            # No files found; exit here, one way or another
            if not required:
                raise MissingDataError(
                    "No files found matching "
                    f"`glob_str` {glob_str} (and ignoring {ignore})."
                )
            raise RequiredDataMissingError(
                f"No files found matching `glob_str` {glob_str} "
                f"(and ignoring {ignore}) were found, but were "
                "marked as required!"
            )

        # Convert to list
        files = list(files)

        # Sort, if asked to do so
        if sort:
            files.sort()

        return files

    def _prepare_target_path(
        self, target_path: str, *, filepath: str, path_sre=None
    ) -> List[str]:
        """Prepare the target path"""
        # The dict to be filled with formatting parameters
        fps = dict()

        # Extract the file basename (without extension)
        fps["basename"] = os.path.splitext(os.path.basename(filepath))[0]
        fps["basename"] = fps["basename"].lower()

        # Use the specified regex pattern to extract a match
        if path_sre:
            try:
                _match = path_sre.findall(filepath)[0]

            except IndexError:
                # nothing could be found
                warnings.warn(
                    "Could not extract a name using the "
                    f"regex pattern '{path_sre}' on the file path:\n"
                    f"{filepath}\nUsing the path's basename instead.",
                    NoMatchWarning,
                )
                _match = fps["basename"]

            else:
                log.debug("Matched '%s' in file path '%s'.", _match, filepath)

            fps["match"] = _match

        # Parse the format string to generate the file path
        log.debug(
            "Parsing format string '%s' to generate target path ...",
            target_path,
        )
        log.debug("  kwargs: %s", fps)
        target_path = target_path.format(**fps)

        log.debug("Generated target path:  %s", target_path)
        return target_path.split(PATH_JOIN_CHAR)

    def _skip_path(self, path: str, *, exists_action: str) -> bool:
        """Check whether a given path exists and — depending on the
        ``exists_action`` – decides whether to skip this path or not.

        Args:
            path (str): The path to check for existence.
            exists_action (str): The behaviour upon existing data. Can be:
                ``raise``, ``skip``, ``skip_nowarn``,
                ``overwrite``, ``overwrite_nowarn``.
                The ``*_nowarn`` arguments suppress the warning.

        Returns:
            bool: Whether to skip this path

        Raises:
            ExistingDataError: Raised when `exists_action == 'raise'`
            ValueError: Raised for invalid `exists_action` value
        """
        if path not in self:
            # Does not exist yet -> no need to skip
            return False
        # else: path exists already
        # NOTE that it is not known whether the path points to a group
        # or to a container

        _msg = f"Path '{PATH_JOIN_CHAR.join(path)}' already exists."

        # Distinguish different actions
        if exists_action == "raise":
            raise ExistingDataError(
                _msg + " Adjust argument `exists_action` to allow skipping "
                "or overwriting of existing entries."
            )

        if exists_action in ["skip", "skip_nowarn"]:
            if exists_action == "skip":
                warnings.warn(
                    _msg + " Loading of this entry will be skipped.",
                    ExistingDataWarning,
                )
            return True  # will lead to the data not being loaded

        elif exists_action in ["overwrite", "overwrite_nowarn"]:
            if exists_action == "overwrite":
                warnings.warn(
                    _msg + " It will be overwritten!", ExistingDataWarning
                )
            return False  # will lead to the data being loaded

        else:
            raise ValueError(
                "Invalid value for `exists_action` argument "
                f"'{exists_action}'! Can be: raise, skip, "
                "skip_nowarn, overwrite, overwrite_nowarn."
            )

    def _store_object(
        self,
        obj: Union[BaseDataGroup, BaseDataContainer],
        *,
        target_path: List[str],
        as_attr: Union[str, None],
        unpack_data: bool,
        exists_action: str,
    ) -> bool:
        """Store the given ``obj`` at the supplied ``target_path``.

        Note that this will automatically overwrite, assuming that all
        checks have been made prior to the call to this function.

        Args:
            obj (Union[BaseDataGroup, BaseDataContainer]): Object to store
            target_path (List[str]): The path to store the object at
            as_attr (Union[str, None]): If a string, store the object in
                the attributes of the container or group at target_path
            unpack_data (bool): Description
            exists_action (str): Description

        Returns:
            bool: Whether storing was successful. May be False in case the
                target path already existed and ``exists_action`` specifies
                that it is to be skipped, or if the object was None.

        Raises:
            ExistingDataError: If non-group-like data already existed at
                that path
            RequiredDataMissingError: If storing as attribute was selected
                but there was no object at the given target_path
        """
        if obj is None:
            log.debug("Object was None, not storing.")
            return False

        # Now handle the (easy) case where the object is to be stored
        # as the attribute at the target_path
        if as_attr:
            # Try to load the object at the target path
            try:
                target = self[target_path]

            except KeyError as err:
                raise RequiredDataMissingError(
                    f"In order to store the object {obj.logstr} at the "
                    f"target path '{target_path}', a group or container "
                    "already needs to exist at that location within "
                    f"{self.logstr}, but there is no such object at that path!"
                ) from err

            # Check whether an attribute with that name already exists
            if as_attr in target.attrs:
                raise ExistingDataError(
                    f"An attribute with the name '{as_attr}' "
                    f"already exists in {target.logstr}!"
                )

            # All checks passed. Can store it now, either directly or with
            # unpacking of its data ...
            if not unpack_data:
                target.attrs[as_attr] = obj
            else:
                target.attrs[as_attr] = obj.data

            log.debug(
                "Stored %s as attribute '%s' of %s.",
                obj.classname,
                as_attr,
                target.logstr,
            )
            return True

        # Check if it is to be skipped
        if self._skip_path(target_path, exists_action=exists_action):
            log.debug("Skipping storing of %s.", obj.logstr)
            return False

        # Extract a target group path and a base name from path list
        group_path = target_path[:-1]
        basename = target_path[-1]

        # Find out the target group, creating it if necessary.
        if not group_path:
            group = self

        else:
            # Need to retrieve or create the group.
            # The difficulty is that the path can also point to a container.
            # Need to assure here, that the group path points to a group.
            if group_path not in self:
                # Needs to be created
                self._create_groups(group_path)

            elif not isinstance(self[group_path], BaseDataGroup):
                # Already exists, but is no group. Cannot continue
                group_path = PATH_JOIN_CHAR.join(group_path)
                target_path = PATH_JOIN_CHAR.join(target_path)
                raise ExistingDataError(
                    f"The object at '{group_path}' in {self.logstr} is "
                    f"not a group but a {type(self[group_path])}. Cannot "
                    f"store {obj.logstr} there because the target path "
                    f"'{target_path}' requires it to be a group."
                )

            # Now the group path will definitely point to a group
            group = self[group_path]

        # Check if any container already exists in that group, overwrite if so.
        # Depending on `exists_action`, this method might have returned above
        # already if overwriting is not intended.
        if basename in group:
            del group[basename]

        # All good, can store the object now.
        group.add(obj)
        log.debug("Successfully stored %s at '%s'.", obj.logstr, obj.path)
        return True

    def _contains_group(
        self, path: Union[str, List[str]], *, base_group: BaseDataGroup = None
    ) -> bool:
        """Recursively checks if the given path is available _and_ a group.

        Args:
            path (Union[str, List[str]]): The path to check.
            base_group (BaseDataGroup): The group to start from. If not
                given, will use self.

        Returns:
            bool: Whether the path points to a group

        """

        def check(path: str, base_group: BaseDataGroup) -> bool:
            """Returns True if the object at path within base_group is
            a group. False otherwise.
            """
            return path in base_group and isinstance(
                base_group[path], BaseDataGroup
            )

        if not isinstance(path, list):
            path = path.split(PATH_JOIN_CHAR)

        if not base_group:
            base_group = self

        if len(path) > 1:
            # Need to continue recursively
            if check(path[0], base_group):
                return self._contains_group(
                    path[1:], base_group=base_group[path[0]]
                )
            return False

        # End of recursion
        return check(path[0], base_group)

    def _create_groups(
        self,
        path: Union[str, List[str]],
        *,
        base_group: BaseDataGroup = None,
        GroupCls: Union[type, str] = None,
        exist_ok: bool = True,
    ):
        """Recursively create groups for the given path. Unlike new_group, this
        also creates the groups at the intermediate paths.

        Args:
            path (Union[str, List[str]]): The path to create groups along
            base_group (BaseDataGroup, optional): The group to start from. If
                not given, uses self.
            GroupCls (Union[type, str], optional): The class to use for
                creating the groups or None if the _DATA_GROUP_DEFAULT_CLS is
                to be used. If a string is given, lookup happens from the
                _DATA_GROUPS_CLASSES variable.
            exist_ok (bool, optional): Whether it is ok that groups along the
                path already exist. These might also be of different type.
                Default: True

        Raises:
            ExistingDataError: If not `exist_ok`
            ExistingGroupError: If not `exist_ok` and a group already exists
        """
        # Parse arguments
        if isinstance(path, str):
            path = path.split(PATH_JOIN_CHAR)

        if base_group is None:
            base_group = self

        GroupCls = self._determine_group_class(GroupCls)

        # Catch the disallowed case as early as possible
        if path[0] in base_group:
            # Check if it is a group that exists there
            if isinstance(base_group[path[0]], BaseDataGroup):
                if not exist_ok:
                    raise ExistingGroupError(path[0])

            else:
                # There is data (that is not a group) existing at the path.
                # Cannot continue
                raise ExistingDataError(
                    f"Tried to create a group '{path[0]}' in "
                    f"{base_group.logstr}, but a container was already stored "
                    "at that path."
                )

        # Create the group, if it does not yet exist
        if path[0] not in base_group:
            log.debug(
                "Creating group '%s' in %s ...", path[0], base_group.logstr
            )
            base_group.new_group(path[0])

        # path[0] is now created
        # Check whether to continue recursion
        if len(path) > 1:
            # Continue recursion
            self._create_groups(
                path[1:], base_group=base_group[path[0]], GroupCls=GroupCls
            )

    def _determine_group_class(self, Cls: Union[type, str]) -> type:
        """Helper function to determine the type of a group from an argument.

        Args:
            Cls (Union[type, str]): If None, uses the _DATA_GROUP_DEFAULT_CLS.
                If a string, tries to extract it from the _DATA_GROUP_CLASSES
                class variable. Otherwise, assumes this is already a type.

        Returns:
            type: The group class to use

        Raises:
            KeyError: If the string class name was not registered
            ValueError: If no _DATA_GROUP_CLASSES variable was populated
        """
        if Cls is None:
            return self._DATA_GROUP_DEFAULT_CLS

        if isinstance(Cls, str):
            cls_name = Cls

            if not self._DATA_GROUP_CLASSES:
                raise ValueError(
                    "The class variable _DATA_GROUP_CLASSES is "
                    "empty; cannot look up class type by the "
                    "given name '{}'.".format(cls_name)
                )

            elif cls_name not in self._DATA_GROUP_CLASSES:
                raise KeyError(
                    "The given class name '{}' was not registered "
                    "with this {}! Available classes: {}"
                    "".format(
                        cls_name, self.classname, self._DATA_GROUP_CLASSES
                    )
                )

            # everything ok, retrieve the class type
            return self._DATA_GROUP_CLASSES[cls_name]

        # else: assume it is already a type and just return the given argument
        return Cls

    # .........................................................................
    # Dumping and restoring the DataManager

    def _parse_file_path(self, path: str, *, default_ext=None) -> str:
        """Parses a file path: if it is a relative path, makes it relative to
        the associated data directory. If a default extension is specified and
        the path does not contain one, that extension is added.

        This helper method is used as part of dumping and storing the data
        tree, i.e. in the :py:meth:`~dantro.data_mngr.DataManager.dump` and
        :py:meth:`~dantro.data_mngr.DataManager.restore` methods.
        """
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self.dirs["data"], path)

        # Handle file extension, adding a default extension if none was given
        if default_ext:
            path, ext = os.path.splitext(path)
            path += ext if ext else default_ext

        return path

    def dump(self, *, path: str = None, **dump_kwargs) -> str:
        """Dumps the data tree to a new file at the given path, creating any
        necessary intermediate data directories.

        For restoring, use :py:meth:`~dantro.data_mngr.DataManager.restore`.

        Args:
            path (str, optional): The path to store this file at. If this is
                not given, use the default tree cache path that was set up
                during initialization.
                If it is given and a relative path, it is assumed relative to
                the data directory.
                If the path does not end with an extension, the ``.d3`` (read:
                "data tree") extension is automatically added.
            **dump_kwargs: Passed on to ``pkl.dump``

        Returns:
            str: The path that was used for dumping the tree file
        """
        if path:
            path = self._parse_file_path(path, default_ext=DATA_TREE_DUMP_EXT)
        else:
            path = self.tree_cache_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        log.progress("Dumping data tree of %s ...", self.logstr)
        with open(path, mode="x+b") as file:
            pkl.dump(self, file, **dump_kwargs)

        log.progress("Successfully stored data tree in cache file.")
        log.note("  Path:       %s", path)
        log.note("  File size:  %s", format_bytesize(os.path.getsize(path)))
        return path

    def restore(
        self, *, from_path: str = None, merge: bool = False, **load_kwargs
    ):
        """Restores the data tree from a dump.

        For dumping, use :py:meth:`~dantro.data_mngr.DataManager.dump`.

        Args:
            from_path (str, optional): The path to restore this DataManager
                from. If it is not given, uses the default tree cache path
                that was set up at initialization.
                If it is a relative path, it is assumed relative to the data
                directory. Take care to add the corresponding file extension.
            merge (bool, optional): If True, uses a recursive update to merge
                the current tree with the restored tree.
                If False, uses :py:meth:`~dantro.data_mngr.DataManager.clear`
                to clear the current tree and then re-populates it with the
                restored tree.
            **load_kwargs: Passed on to ``pkl.load``

        Raises:
            FileNotFoundError: If no file is found at the (expanded) path.
        """
        if from_path:
            path = self._parse_file_path(from_path)
        else:
            path = self.tree_cache_path

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Could not restore {self.logstr} as there was no tree "
                f"cache file at '{path}'!"
            )

        log.progress("Restoring %s from data tree file ...", self.logstr)
        log.note("  Path:       %s", path)
        log.note("  File size:  %s", format_bytesize(os.path.getsize(path)))

        with open(path, mode="rb") as file:
            dm = pkl.load(file, **load_kwargs)

        if not merge:
            log.note("  Mode:       clear and load")
            self.clear()
        else:
            log.note("  Mode:       load and merge")

        self.recursive_update(dm)
        log.progress("Successfully restored the data tree.")

    # .........................................................................
    # Working with the data in the tree

    def new_group(self, path: str, *, Cls: Union[type, str] = None, **kwargs):
        """Creates a new group at the given path.

        This is a slightly advanced version of the new_group method of the
        BaseDataGroup. It not only adjusts the default type, but also allows
        more ways how to specify the type of the group to create.

        Args:
            path (str): Where to create the group. Note that the intermediates
                of this path need to already exist.
            Cls (Union[type, str], optional): If given, use this type to
                create the group. If a string is given, resolves the type from
                the _DATA_GROUP_CLASSES class variable. If None, uses the
                default data group type of the data manager.
            **kwargs: Passed on to Cls.__init__

        Returns:
            The created group of type ``Cls``
        """
        # Use helper function to parse the group class correctly
        Cls = self._determine_group_class(Cls)

        return super().new_group(path, Cls=Cls, **kwargs)
