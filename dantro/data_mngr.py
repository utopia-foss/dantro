"""This module implements the DataManager class, the root of the data tree."""

import os
import copy
import datetime
import re
import glob
import logging
import warnings
from typing import Union, Callable, List, Tuple

from dantro.base import PATH_JOIN_CHAR, BaseDataContainer, BaseDataGroup
from dantro.group import OrderedDataGroup
import dantro.tools as tools

# Local constants
log = logging.getLogger(__name__)


# Exception classes ...........................................................

class DataManagerError(Exception):
    """All DataManager exceptions derive from this one"""
    pass

class RequiredDataMissingError(DataManagerError):
    """Raised if required data was missing."""
    pass

class MissingDataError(DataManagerError):
    """Raised if data was missing, but is not required."""
    pass

class ExistingDataError(DataManagerError):
    """Raised if data already existed."""
    pass

class ExistingGroupError(DataManagerError):
    """Raised if a group already existed."""
    pass

class LoaderError(DataManagerError):
    """Raised if a data loader was not available"""
    pass

class MissingDataWarning(UserWarning):
    """Used as warning instead of MissingDataError"""
    pass

class ExistingDataWarning(UserWarning):
    """If there was data already existing ..."""
    pass

# -----------------------------------------------------------------------------

class DataManager(OrderedDataGroup):
    """The DataManager is the root of a data tree, coupled to a specific data directory.
    
    It handles the loading of data and can be used for interactive work with
    the data.
    """

    # Define as class variables what should be the default groups / containers
    _DATA_GROUP_DEFAULT_CLS = OrderedDataGroup

    # .........................................................................
    # Initialisation

    def __init__(self, data_dir: str, *, name: str=None, load_cfg: Union[dict, str]=None, out_dir: Union[str, bool]="_output/{date:}"):
        """Initialises a DataManager object.
        
        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            name (str, optional): which name to give to the DataManager. If no
                name is given, the data directories basename will be used
            load_cfg (Union[dict, str], optional): The base configuration used
                for loading data. If a string is given, assumes a yaml file and
                loads that. If none is given, it can still be supplied to the
                load_data method.
            out_dir (Union[str, bool], optional): where output is written to.
                If this is given as a relative path, it is considered relative
                to the _data directory_. A formatting operation with the keys 
                `date` and `name` is performed on this, where the latter is
                the name of the data manager. If set to False, the output
                directory is not created.
        """

        # Find a name if none was given
        if not name:
            basename = os.path.basename(os.path.abspath(data_dir))
            name = "{}_Manager".format(basename.replace(" ", "_"))
        
        log.info("Initialising %s '%s'...", self.classname, name)

        # Initialise as a data group via parent class
        super().__init__(name=name)

        # Initialise directories
        self.dirs = self._init_dirs(data_dir=data_dir, out_dir=out_dir)

        # If a specific value for the load configuration was given, store it
        # and use it as the default for the `load_from_cfg` method
        self.load_cfg = {}

        if load_cfg and isinstance(load_cfg, str):
            # Assume this is the path to a configuration file and load it
            log.debug("Loading the default load config from a path:\n  %s",
                      load_cfg)
            self.load_cfg = tools.load_yml(load_cfg)

        elif load_cfg:
            # Assume this is already a mapping and store it as the default
            log.debug("Using the given %s as default load configuration.",
                      type(load_cfg))
            self.load_cfg = load_cfg

        # Done
        log.debug("%s initialised.", self.logstr)

    def _init_dirs(self, *, data_dir: str, out_dir: Union[str, bool]) -> dict:
        """Initialises the directories managed by this DataManager and returns
        a dictionary that stores the absolute paths to these directories.
        
        If they do not exist, they will be created.
        
        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            out_dir (Union[str, bool]): where output is written to.
                If this is given as a relative path, it is considered relative
                to the _data directory_. A formatting operation with the keys 
                `date` and `name` is performed on this, where the latter is
                the name of the data manager. If set to False, the output
                directory is not created.
        
        Returns:
            dict: The directories
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
            datestr = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

            # Perform a format operation on the output directory
            out_dir = out_dir.format(name=self.name.lower(), date=datestr)

            # If it is relative, assume it to be relative to the data directory
            if not os.path.isabs(out_dir):
                # By joining them together, out_dir is now relative
                out_dir = os.path.join(data_dir, out_dir)
    
            # Make path absolute and store in dict
            dirs['out'] = os.path.abspath(out_dir)

            # Create the directory
            os.makedirs(dirs['out'])
        else:
            dirs['out'] = False

        # Inform about the managed directories, then return
        log.debug("Managed directories:\n%s",
                  "\n".join(["  {:>8s} : {}".format(k, v)
                             for k, v in dirs.items()]))

        return dirs

    # .........................................................................
    # Loading data

    def load_from_cfg(self, *, load_cfg: dict=None, update_load_cfg: dict=None, exists_action: str='raise', print_tree: bool=False) -> None:
        """Load multiple data entries using the specified load configuration.
        
        Args:
            load_cfg (dict, optional): The load configuration to use. If not
                given, the one specified during initialisation is used.
            update_load_cfg (dict, optional): If given, it is used to update
                the load configuration recursively
            exists_action (str, optional): The behaviour upon existing data.
                Can be: raise (default), skip, skip_nowarn, overwrite,
                overwrite_nowarn, update, update_nowarn.  With *_nowarn
                values, no warning is given if an entry already existed.
            print_tree (bool, optional): If True, a tree representation of the
                DataManager is printed after the data was loaded
        
        Raises:
            TypeError: Raised if a given configuration entry was of invalid 
                type, i.e. not a dict
        """
        # Determine which load configuration to use
        if not load_cfg:
            log.debug("No load configuration given; will use load "
                      "configuration given at initialisation.")
            load_cfg = self.load_cfg

        # Make sure to work on a copy, be it on the defaults or on the passed
        load_cfg = copy.deepcopy(load_cfg)

        if update_load_cfg:
            # Recursively update with the given keywords
            load_cfg = tools.recursive_update(load_cfg, update_load_cfg)
            log.debug("Updated the load config.")

        log.info("Loading %d data entries ...", len(load_cfg))

        # Loop over the data entries that were configured to be loaded
        for entry_name, params in load_cfg.items():
            # Check if this is of valid type
            if not isinstance(params, dict):
                raise TypeError("Got invalid load specifications for entry "
                                "'{}'! Expected dict, got {} with value '{}'. "
                                "Check the correctness of the given load "
                                "configuration!".format(entry_name,
                                                        type(params), params))

            # Use the public method to load this single entry
            self.load(entry_name, exists_action=exists_action,
                      print_tree=False,  # to not have prints during loading
                      **params)
        
        # All done
        log.info("Successfully loaded %d data entries.", len(load_cfg))
        log.info("Available data entries:\n  %s\n", 
                 ",  ".join(self.data.keys()))

        # Finally, print the tree
        if print_tree:
            print("{:tree}".format(self))


    def load(self, entry_name: str, *, loader: str, glob_str: Union[str, List[str]], target_path: str=None, print_tree: bool=False, **load_params) -> None:
        """Performs a single load operation.
        
        Args:
            entry_name (str): Name of this entry; will also be the name of the
                created group or container, unless `target_basename` is given
            loader (str): The name of the loader to use
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings by which to identify the files within `data_dir` that
                are to be loaded using the given loader function
            target_path (str, optional): The path to write the data to. This
                can be a format string. It is evaluated for each file that has
                been matched. If it is not given, the content is loaded to a
                group with the name of this entry at the root level.
            print_tree (bool, optional): Whether to print the tree at the end
                of the loading operation.
            **load_params: Further loading parameters, all optional!
                ignore (list): The exact file names in this list will be
                    ignored during loading. Paths are seen as elative to the
                    data directory of the data manager.
                required (bool): If True, will raise an error if no files were
                    found. Default: False.
                path_regex (str): This pattern can be used to match the path of
                    the file that is being loaded. The match result is
                    available to the format string under the `match` key.
                exists_action (str): The behaviour upon existing data.
                    Can be: raise (default), skip, skip_nowarn, overwrite,
                    overwrite_nowarn, update, update_nowarn.
                    With *_nowarn values, no warning is given if an entry
                    already existed.
                exist_ok (bool): Whether it is ok that a _group_ along the
                    target_path already exists.
                suppress_group (bool): # TODO write this
                progress_indicator (bool): Whether to print a progress
                    indicator or not. Default: True
                parallel (bool): If True, data is loaded in parallel.
                    This feature is not implemented yet!
                any further kwargs: passed on to the loader function
        
        Returns:
            None
        """

        log.info("Loading data entry '%s' ...", entry_name)
        
        # Create the default target path
        if not target_path:
            target_path = "/" + entry_name + "/{basename:}"
        
        # Check that the target path can be evaluated correctly
        try:
            _target_path = target_path.format(basename="basename",
                                              entry_name=entry_name,
                                              match="match")

        except (IndexError, KeyError) as err:
            raise ValueError("Invalid argument `target_path`. Will not be "
                             "able to properly evalaute '{}' later due to "
                             "a {}: {}".format(target_path,
                                               type(err), err)) from err
        else:
            log.debug("Target path will be:  %s", _target_path)

        # Try loading the data and handle specific DataManagerErrors . . . . .
        try:
            self._load(entry_name=entry_name,
                       loader=loader, glob_str=glob_str, **load_params)

        except RequiredDataMissingError:
            raise

        except MissingDataError as err:
            warnings.warn("No files were found to import!\n"+str(err),
                          MissingDataWarning)
            return  # Does not raise, but does not save anything either

        except LoaderError:
            raise

        else:
            # Everything as desired, _entry is now the imported data
            log.debug("Data successfully loaded.")

        # Done with this config entry
        log.debug("Entry '%s' successfully loaded.", entry_name)

        if print_tree:
            print("{:tree}".format(self))
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Helpers for loading and storing data

    def _create_groups(self, *, path: List[str], base_group: BaseDataGroup=None, GroupCls=None, exist_ok: bool=True):
        """Recursively create groups for the given path.
        
        Args:
            path (List[str]): The path to create groups along
            base_group (BaseDataGroup, optional): The group to start from. If
                not given, uses self.
            GroupCls (None, optional): The class to use for creating the groups
                or None if the _DATA_GROUP_DEFAULT_CLS is to be used
            exist_ok (bool, optional): Whether it is ok that groups along the
                path already exist. These might also be of different type.
                Default: True
        
        Raises:
            ExistingGroupError: If not `exist_ok` and a group already exists
        """
        if base_group is None:
            base_group = self

        if GroupCls is None:
            GroupCls = self._DATA_GROUP_DEFAULT_CLS

        # Catch the disallowed case as early as possible
        if path[0] in base_group and not exist_ok:
            raise ExistingGroupError(path[0])

        # Create the group, if it does not yet exist
        if path[0] not in base_group:
            log.debug("Creating group '%s' in %s ...",
                      path[0], base_group.logstr)
            grp = GroupCls(name=path[0])
            base_group.add(grp)

        # path[0] is now created
        # Check whether to continue recursion
        if len(path) > 1:
            # Continue recursion
            self._create_groups(base_group=base_group[path[0]], path=path[1:],
                                GroupCls=GroupCls)

    def _store(self, obj, *, path: str, exist_action: str):
        """Store the given `obj` at the supplied `path`."""
        # Extract the target group path parameter and resolve it
        target_group = get_target_group(target_group)

        # Extract the desired name of the target container or group
        target_basename = target_basename if target_basename else entry_name

        # Check if the target already exists
        if target_basename in target_group:
            if skip_existing(exists_action,
                             target_group=target_group,
                             target_basename=target_basename):
                log.debug("Skipping entry '%s' as it already exists.",
                          entry_name)
                return

        def skip_existing(exists_action: str, *, target_basename: str, target_group: BaseDataGroup) -> bool:
            """Helper function to generate a meaningful error message if data
            already existed and how the loading will continue ...
            
            Args:
                exists_action (str): The behaviour upon existing data. Can be:
                    raise, skip, skip_nowarn, overwrite, overwrite_nowarn,
                    update, update_nowarn
                target_basename (str): The basename that already existed
                target_group (BaseDataGroup): The group the basename already
                    existed in, i.e. where the conflict occured
            
            Returns:
                bool: Whether to skip loading, i.e. exiting the `load` method
            
            Raises:
                ExistingDataError: Raised when `exists_action == 'raise'`
                ValueError: Raised for invalid `exists_action` value
            """
            _msg = ("The data entry with target basename '{}' already "
                    "exists in target group '{}'!"
                    "".format(target_basename, target_group))

            if exists_action == 'raise':
                raise ExistingDataError(_msg + " Adjust argument "
                                        "`exists_action` to allow skipping "
                                        "or overwriting of existing entries.")

            if exists_action in ['skip', 'skip_nowarn']:
                if exists_action == 'skip':
                    warnings.warn(_msg
                                  + " Loading of this entry will be skipped.",
                                  ExistingDataWarning)
                return True  # will lead to the data not being loaded

            elif exists_action in ['overwrite', 'overwrite_nowarn']:
                if exists_action == 'overwrite':
                    warnings.warn(_msg + " It will be overwritten!",
                                  ExistingDataWarning)
                return False  # will lead to the data being loaded
            
            elif exists_action in ['update', 'update_nowarn']:
                if exists_action == 'update':
                    warnings.warn(_msg + " Will be updated with loaded data.",
                                  ExistingDataWarning)
                return False  # will lead to the data being loaded

            else:
                raise ValueError("Invalid value for `exists_action` "
                                 "argument '{}'! Can be: raise, skip, "
                                 "skip_nowarn, overwrite, overwrite_nowarn"
                                 "".format(exists_action))



    def _load(self, *, target_path: str, loader: str, glob_str: Union[str, List[str]], ignore: List[str]=None, required: bool=False, suppress_group: bool=None, path_regex: str=None, progress_indicator: bool=True, parallel: bool=False, **loader_kwargs) -> None:
        """Helper function that loads a data entry to the specified path.
        
        Args:
            target_path (str): The path to load the result of the loader to.
                This can be a format string; it is evaluated for each file.
            loader (str): The loader to use
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings to match files in the data directory
            ignore (List[str], optional): The exact file names in this list
                will be ignored during loading. Paths are seen as elative to the data directory.
            required (bool, optional): If True, will raise an error if no files
                were found.
            suppress_group (bool, optional): Whether to suppress loading into
                a group. If None, loads into a group only if the `glob_str` is
                a list or contains a wildcard. Will raise an error if more than
                one file is to be loaded.
            path_regex (str, optional): The regex applied to the relative path
                of the files that were found. It is used to generate the name
                of the target container. If not given, the basename is used.
            progress_indicator (bool, optional): Whether to print a progress
                indicator or not
            parallel (bool, optional): If True, data is loaded in parallel -
                not implemented yet!
            **loader_kwargs: passed on to the loader function
        
        Returns:
            None
        
        Raises:
            NotImplementedError: For `parallel == True`
        """

        def resolve_loader(loader: str) -> Tuple[Callable, str]:
            """Resolves the loader function"""
            load_func_name = '_load_' + loader.lower()
            try:
                load_func = getattr(self, load_func_name)
            except AttributeError as err:
                raise LoaderError("Loader '{}' was not available to {}! Make "
                                  "sure to use a mixin class that supplies "
                                  "the '{}' loader method."
                                  "".format(loader, self.logstr,
                                            load_func_name)) from err
            else:
                log.debug("Resolved '%s' loader function.", loader)

            return loader, load_func_name

        def parse_suppress_group_flag(suppress_group: bool, *, glob_str: Union[str, List[str]]) -> bool:
            """Depending on the `glob_str`, re-evaluates `suppress_group`"""
            if suppress_group is not None:
                # Was already set. Return that value
                return suppress_group

            if isinstance(glob_str, str):
                # Check if it includes a wildcard
                if glob_str.find("*") < 0:
                    # Nope. Assume that a specific file is to be loaded
                    log.debug("No wildcard found in `glob_str`; setting "
                              "`suppress_group` flag.")
                    return True

            # Is a list of glob strings or contains a wildcard
            return False

        def create_files_set(*, glob_str: Union[str, List[str]], ignore: List[str]) -> set:
            """Create the set of file paths to load from.
            
            Args:
                glob_str (Union[str, List[str]]): The glob pattern or a list of
                    glob patterns
                ignore (List[str]): The list of files to ignore
                suppress_group (bool): If true, 
            
            Returns:
                set: the file paths to load
            
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
            log.debug("Got %d glob string(s) to create set of matching file "
                      "paths from.", len(glob_str))

            # Go over the given glob strings and add to the files set
            for gs in glob_str:
                # Make the glob string absolute
                gs = os.path.join(self.dirs['data'], gs)
                log.debug("Adding files that match glob string:\n  %s", gs)

                # Add to the set of files; this assures uniqueness of found paths
                files.update(list(glob.glob(gs, recursive=True)))

            # See if some files should be ignored
            if ignore:
                log.debug("Got list of files to ignore:\n  %s", ignore)
                
                # Make absolute and generate list of files to exclude
                ignore = [os.path.join(self.dirs['data'], path)
                          for path in ignore]

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
            log.debug("Found %d files to load:\n  %s",
                      len(files), "\n  ".join(files))

            if not files:
                # No files found; exit here, one way or another
                if not required:
                    raise MissingDataError("No files found matching "
                                           "`glob_str` {} (and ignoring {})."
                                           "".format(glob_str, ignore))
                raise RequiredDataMissingError("No files found matching "
                                               "`glob_str` {} (and ignoring "
                                               "{}) were found, but entry "
                                               "'{}' was marked as required!"
                                               "".format(glob_str, ignore,
                                                         entry_name))

            return files, suppress_group

        def prepare_target_class(load_func: Callable, *, target_name: str) -> Callable:
            """Prepare the target class to load data into."""
            try:
                TargetCls = getattr(load_func, 'TargetCls')

            except AttributeError as err:
                raise LoaderError("Load function {} misses required attribute "
                                  "'TargetCls'. Check your mixin!"
                                  "".format(load_func)) from err

            # Create a new init function where the name is already resolved
            return lambda **kws: TargetCls(name=target_name, **kws)

        def prepare_target(*, load_func, target_group, target_basename: str=None, filepath: str=None, path_sre=None, ignore_existing: bool=False):
            """Fetches the class that the load function specifies and prepares it to be used for initialisation by the load function."""
            tname = None

            # Find a suitable name
            if target_basename:
                # Warn about cases where additional arguments were given that
                # will be ignored
                if path_sre:
                    # Will not use path_regex if loading a single item (not 
                    # into a group)
                    warnings.warn("Argument `path_sre` or `path_regex` was "
                                  "given; will be ignored as the target's "
                                  "basename was already given!", UserWarning)

                # Use the given name as name of the group 
                tname = target_basename

            elif filepath:
                if path_sre:
                    # Use the specified regex pattern to extract a name
                    try:
                        tname = path_sre.findall(filepath)[0]
                    except IndexError:
                        # nothing could be found
                        pass
                    
                    if not tname:
                        # Could not find a basename
                        warnings.warn("Could not extract a name using the "
                                      "regex pattern '{}' on the file path:\n"
                                      "{}\nUsing the path's basename instead."
                                      "".format(path_sre, filepath),
                                      UserWarning)

                if not tname:
                    # use the file's basename, without extension
                    tname = os.path.splitext(os.path.basename(filepath))[0].lower()

            # Ensure that there is nothing under that name in the target group
            if tname in target_group:
                _msg = ("Member '{}' already exists at '{}' of {}!"
                        "".format(tname, target_group.path,
                                  target_group.logstr))
                if not ignore_existing:
                    if not path_sre:
                        raise ExistingDataError(_msg)
                    raise ExistingDataError(_msg + " You might want to check "
                                            "that the given `path_regex` '{}' "
                                            "resolves to unique names."
                                            "".format(path_sre.pattern))
                # else: just log it
                log.debug(_msg)
                    

        # . . . . . . . . . . . End of helper functions . . . . . . . . . . . .
        # Get the loader function
        loader, load_func_name = resolve_loader(loader)

        # Evaluate the suppress_group flag
        suppress_group = parse_suppress_group_flag(suppress_group,
                                                   glob_str=glob_str)

        # Create the set of file paths to load
        files = create_files_set(glob_str=glob_str, ignore=ignore,
                                 required=required)
        
        # If a regex pattern was specified, compile it
        path_sre = re.compile(path_regex) if path_regex else None

        # Ready for loading files now . . . . . . . . . . . . . . . . . . . . .
        # Distinguish between cases where a group should be created and one where the DataContainer-object can be directly returned
        if len(files) == 1 and suppress_group:
            log.debug("Found a single file and will not create a new group.")

            # Prepare the target class, which will be filled by the load func
            # The helper function takes care of the naming of the target cont
            TargetCls = prepare_target_class(load_func=load_func,
                                       target_path=target_path,
                                       path_sre=path_sre,
                                       ignore_existing=True)

            # Load the data using the loader function
            data = load_func(files.pop(), TargetCls=TargetCls, **loader_kwargs)

            log.debug("Finished loading a single file for entry %s.",
                      entry_name)

            # And return it
            return data
    
        # else: more than one file -> need to work with groups

        if parallel:
            # TODO could be implemented by parallelising the below for loop
            raise NotImplementedError("Cannot load in parallel yet.")

        # Create a group wherein all entries are gathered
        # It has the same name as the entry
        log.debug("Creating a %s to group data loaded from %d file(s) in...",
                  self._DATA_GROUP_DEFAULT_CLS.__name__, len(files))
        group = self._DATA_GROUP_DEFAULT_CLS(name=target_basename)

        # Go over the files and load them
        for n, file in enumerate(files):
            if progress_indicator:
                line = "  Loading ... {}/{}".format(n+1, len(files))
                print(tools.fill_tty_line(line), end="\r")

            # Prepare the target class, which will be filled by the load func
            # The helper function takes care of the naming of the target cont
            TargetCls = prepare_target(load_func=load_func,
                                       target_group=group,
                                       filepath=file, path_sre=path_sre)
            # NOTE target_basename should not be given here, as the name is
            # resolved from the filepath or via the given pattern

            # Get the data and add it to the group
            _data = load_func(file, TargetCls=TargetCls, **loader_kwargs)
            group.add(_data)

        # Clear the line
        tools.clear_line()

        # Done
        log.debug("Finished loading %d files for entry %s.", len(files),
                  entry_name)
