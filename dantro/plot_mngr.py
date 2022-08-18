"""Implements the :py:class:`~dantro.plot_mngr.PlotManager`, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the respective :ref:`plot creators <plot_creators>`.
See :ref:`the user manual <plot_manager>` for more information.
"""

import copy
import fnmatch
import logging
import os
import time
import warnings
from collections import OrderedDict
from difflib import get_close_matches as _get_close_matches
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from paramspace import ParamDim, ParamSpace
from pkg_resources import resource_filename as _resource_filename

from .abc import BAD_NAME_CHARS as _BAD_NAME_CHARS
from .data_mngr import DataManager
from .exceptions import *
from .plot import BasePlotCreator, SkipPlot
from .plot._cfg import resolve_based_on as _resolve_based_on
from .plot._cfg import resolve_based_on_single as _resolve_based_on_single
from .plot.creators import ALL_PLOT_CREATORS as ALL_PCRS
from .plot.utils import PlotFuncResolver as _PlotFuncResolver
from .tools import format_time as _format_time
from .tools import load_yml, make_columns, recursive_update, write_yml

# .............................................................................

log = logging.getLogger(__name__)

_fmt_time = lambda seconds: _format_time(seconds, ms_precision=1)

BAD_PLOT_NAME_CHARS = tuple(
    [c for c in _BAD_NAME_CHARS if c not in ("/",)] + ["."]
)
"""Substrings that may not appear in plot names.

Unlike the :py:data:`~dantro.abc.BAD_NAME_CHARS`, these *allow* the ``/`` char
(such that new directories can be created) and disallows the ``.`` character
(in order to not get confused with file extensions).
"""

BASE_PLOTS_CFG_PATH: str = _resource_filename("dantro", "cfg/base_plots.yml")
"""The path to the base plot configurations pool for dantro.

If the ``use_dantro_base_cfg_pool`` flag is set when initializing a
:py:class:`~dantro.plot_mngr.PlotManager`, this file will be used as the first
entry in the sequence of config pools.

Also see :ref:`dantro_base_plots` for more information."""


# -----------------------------------------------------------------------------


class PlotManager:
    """The PlotManager takes care of configuring plots and calling the
    selected :ref:`plot creators <plot_creators>` that then actually carry out
    the plotting operation.

    It is a high-level class that is aware of a larger plot configuration and
    aggregates all general capabilities needed to configure and carry out plots
    using the plotting framework.

    See :ref:`the user manual <plot_manager>` for more information.
    """

    PLOT_FUNC_RESOLVER: type = _PlotFuncResolver
    """The class to use for resolving plot function objects"""

    CREATORS: Dict[str, type] = ALL_PCRS
    """The mapping of creator names to classes.
    By default, all available dantro plot creators are registered.

    When subclassing PlotManager and desiring to *extend* the creator mapping,
    use ``dict(**dantro.plot.creators.ALL, my_new_creator=MyNewCreator)``
    to include the default creator mapping."""

    DEFAULT_OUT_FSTRS: Dict[str, str] = dict(
        timestamp="%y%m%d-%H%M%S",
        #
        # representing the parameter space sweep state and its coordinate
        state_no="{no:0{digits:d}d}",
        state="{name:}_{val:}",
        state_name_replace_chars=[],  # (".", "-")
        state_val_replace_chars=[("/", "-")],
        state_join_char="__",
        state_vector_join_char="-",
        #
        # final fstr for single plot and config path
        path="{name:}{ext:}",
        plot_cfg="{basename:}_cfg.yml",
        #
        # and for sweep
        sweep="{name:}/{state_no:}__{state:}{ext:}",
        plot_cfg_sweep="{name:}/sweep_cfg.yml",
    )
    """The default values for the output format strings, used when composing
    the file name of a plot."""

    SPECIAL_BASE_CFG_POOL_LABELS: Sequence[str] = (
        "plot",
        "plot_from_cfg",
        "plot_from_cfg_unused",
        "plot_pspace",
    )
    """Special keys that may not be used as labels for the base configuration
    pools."""

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def __init__(
        self,
        *,
        dm: DataManager,
        default_plots_cfg: Union[dict, str] = None,
        out_dir: Union[str, None] = "{timestamp:}/",
        base_cfg_pools: Sequence[Tuple[str, Union[dict, str]]] = (),
        use_dantro_base_cfg_pool: bool = True,
        out_fstrs: dict = None,
        plot_func_resolver_init_kwargs: dict = None,
        shared_creator_init_kwargs: dict = None,
        creator_init_kwargs: Dict[str, dict] = None,
        default_creator: str = None,
        save_plot_cfg: bool = True,
        raise_exc: bool = False,
        cfg_exists_action: str = "raise",
    ):
        """Initialize a PlotManager, which provides a uniform configuration
        interface for creating plots and passes tasks on to the respective
        plot creators.

        To avoid copy-paste of plot configurations, the PlotManager comes with
        versatile capabilities to define default plots and re-use other plots.

        - The ``default_plots_cfg`` specifies plot configurations that are
          to be carried out by default when calling the plotting method
          :py:meth:`.plot_from_cfg`.
        - When calling any of the plot methods :py:meth:`.plot_from_cfg` or
          :py:meth:`.plot`, there is the possibility to update the existing
          configuration dict with new entries.
        - At each stage, the ``based_on`` feature allows to make a plot
          configuration inherit entries from an existing configuration.
          These are looked up from the ``base_cfg_pools`` following the
          rules described in :py:func:`~dantro.plot._cfg.resolve_based_on`.

        For more information on how the plot configuration can be defined, see
        :ref:`plot_cfg_inheritance`.

        Args:
            dm (DataManager): The DataManager-derived object to read the plot
                data from.
            default_plots_cfg (Union[dict, str], optional): The default plots
                config or a path to a YAML file to import. Used as defaults
                when calling :py:meth:`.plot_from_cfg`
            out_dir (Union[str, None], optional): If given, will use this
                output directory as basis for the output path for each plot.
                The path can be a format-string; it is evaluated upon call to
                the plot command. Available keys: ``timestamp``, ``name``, ...
                For a relative path, this will be relative to the DataManager's
                output directory. Absolute paths remain absolute.
                If this argument evaluates to False, the DataManager's output
                directory will be the output directory.
            base_cfg_pools (Sequence[Tuple[str, Union[dict, str]]], optional):
                The base configuration pools are used to perform the lookups of
                ``based_on`` entries, see :ref:`plot_cfg_inheritance`.
                The tuples in these sequence consist of ``(label, plots_cfg)``
                pairs and are fed to :py:meth:`.add_base_cfg_pool`; see there
                for more information.
            use_dantro_base_cfg_pool (bool, optional): If set, will use
                dantro's own base plot configuration pool as the *first* entry
                in the pool sequence. Refer to the
                :ref:`corresponding documentation page <dantro_base_plots>`
                for more information on available entries.
            out_fstrs (dict, optional): Format strings that define how the
                output path is generated. The dict given here updates the
                :py:attr:`.DEFAULT_OUT_FSTRS` class variable which holds the
                default values.

                Keys: ``timestamp`` (%-style), ``path``, ``sweep``, ``state``,
                ``plot_cfg``, ``state``, ``state_no``, ``state_join_char``,
                ``state_vector_join_char``.

                Available keys for ``path``: ``name``, ``timestamp``, ``ext``.

                Additionally, for ``sweep``: ``state_no``, ``state_vector``,
                    ``state``.

            plot_func_resolver_init_kwargs (dict, optional): Initialization
                arguments for the plot function resolver, by default
                :py:class:`~dantro.plot.utils.plot_func.PlotFuncResolver`.
            shared_creator_init_kwargs (dict, optional): Initialization
                arguments to the plot creator that are passed to *all* creators
                regardless of type (in contrast to ``creator_init_kwargs``).
            creator_init_kwargs (Dict[str, dict], optional): If given, these
                kwargs are passed to the initialization calls of the respective
                creator classes. These are resolved by the *names* given in the
                :py:attr:`.CREATORS` class variable and are passed to the
                :py:class:`~dantro.plot.creators.base.BasePlotCreator` or the
                respective derived class.
            default_creator (str, optional): If given, a plot without explicit
                ``creator`` declaration will use this creator as default.
            save_plot_cfg (bool, optional): If True, the plot configuration is
                saved to a yaml file alongside the created plot.
            raise_exc (bool, optional): Whether to raise exceptions if there
                are errors raised from the plot creator or errors in the plot
                configuration. If False, the errors will only be logged.
            cfg_exists_action (str, optional): Behaviour when a config file
                already exists. Can be: ``raise`` (default), ``skip``,
                ``append``, ``overwrite``, or ``overwrite_nowarn``.
        """
        # Public
        self.save_plot_cfg = save_plot_cfg
        self.raise_exc = raise_exc

        # Private or property-managed
        self._dm = dm
        self._out_dir = out_dir
        self._plot_info = []
        self._pfr_kwargs = (
            plot_func_resolver_init_kwargs
            if plot_func_resolver_init_kwargs
            else {}
        )
        self._shared_cckwargs = (
            shared_creator_init_kwargs if shared_creator_init_kwargs else {}
        )
        self._cckwargs = creator_init_kwargs if creator_init_kwargs else {}
        self._cfg_exists_action = cfg_exists_action
        self.default_creator = default_creator
        self._default_plots_cfg = None

        # Base configuration pools
        self._base_cfg_pools = OrderedDict()
        if use_dantro_base_cfg_pool:
            self.add_base_cfg_pool(
                label="dantro_base", plots_cfg=BASE_PLOTS_CFG_PATH
            )

        for _label, _plots_cfg in base_cfg_pools:
            self.add_base_cfg_pool(label=_label, plots_cfg=_plots_cfg)

        # Default plot configuration
        if default_plots_cfg:
            self._default_plots_cfg = self._prepare_cfg(default_plots_cfg)

        # Default format strings
        self._out_fstrs = self.DEFAULT_OUT_FSTRS
        if out_fstrs:
            self._out_fstrs = recursive_update(
                copy.deepcopy(self._out_fstrs), out_fstrs
            )

        log.debug("%s initialized.", self.__class__.__name__)
        log.debug(
            "Have %d base configuration pools available:  %s",
            len(self._base_cfg_pools),
            ", ".join(self._base_cfg_pools),
        )

    # .........................................................................
    # Properties

    @property
    def out_fstrs(self) -> dict:
        """The dict of output format strings"""
        return self._out_fstrs

    @property
    def plot_info(self) -> List[dict]:
        """A list of dicts with info on all plots carried out so far"""
        return copy.deepcopy(self._plot_info)

    @property
    def base_cfg_pools(self) -> OrderedDict:
        """The base plot configuration pools, used for lookup the ``based_on``
        entry in plot configurations.

        The order of the entries in the pool is relevant, with later entries
        taking precedence over previous ones. See :ref:`plot_cfg_inheritance`
        for a more detailed description.
        """
        return self._base_cfg_pools

    @property
    def default_creator(self) -> str:
        """The name of the default creator"""
        return self._default_creator

    @default_creator.setter
    def default_creator(self, new_creator: str):
        """Set the name of the default creator, raising if it is invalid"""
        if new_creator and new_creator not in self.CREATORS:
            _avail = ", ".join(self.CREATORS.keys())
            raise InvalidCreator(
                f"No such creator '{new_creator}' available, only: {_avail}"
            )
        self._default_creator = new_creator

    # .........................................................................
    # Configuration

    def add_base_cfg_pool(self, *, label: str, plots_cfg: Union[str, dict]):
        """Adds a base configuration pool entry, allowing for the ``plots_cfg``
        to be a path to a YAML configuration file which is then loaded.

        The new pool is used for ``based_on`` lookups and takes precedence over
        existing entries. For more information on lookup rules, see
        :py:func:`~dantro.plot._cfg.resolve_based_on` and
        :ref:`plot_cfg_inheritance`.

        Args:
            label (str): A label of the pool that is used for identifying it.
            plots_cfg (Union[str, dict]): Description

        Raises:
            ValueError: If ``label`` already exists or is a special label.

        """
        if label in self._base_cfg_pools:
            raise ValueError(
                f"A base configuration labelled '{label}' already exists! "
                "Check if it was already added or choose a different name."
            )

        elif label in self.SPECIAL_BASE_CFG_POOL_LABELS:
            _special = ", ".join(self.SPECIAL_BASE_CFG_POOL_LABELS)
            raise ValueError(
                f"Invalid base configuration pool label '{label}'! Choosing "
                f"one of the special labels ({_special}) is not permitted."
            )

        self._base_cfg_pools[label] = self._prepare_cfg(plots_cfg)
        log.debug("Added base configuration pool '%s'.", label)

    # .........................................................................
    # Helpers

    @staticmethod
    def _prepare_cfg(s: Union[str, dict]) -> dict:
        """Prepares a plots configuration by either loading it from a YAML file
        if the given argument is a string or returning a deep copy of the given
        dict-like object.
        """
        if isinstance(s, str):
            return load_yml(s)
        return copy.deepcopy(s)

    def _handle_exception(
        self,
        exc: Exception,
        *,
        pc: BasePlotCreator,
        debug: bool = None,
        ExcCls: type = PlottingError,
    ):
        """Helper for handling exceptions from the plot creator"""
        should_raise = debug or (debug is None and self.raise_exc)

        e_dbg = (
            "For a full error traceback, specify `debug: True` in the "
            "plot configuration or run the PlotManager in debug mode."
        )
        e_no_dbg = (
            "To ignore the error message and continue plotting with the "
            "other plots, specify `debug: False` in the plot "
            "configuration or disable debug mode for the PlotManager."
        )
        e_msg = (
            f"An error occurred during plotting with {pc.logstr}! "
            f"{e_dbg if not should_raise else e_no_dbg}\n\n"
            f"{exc.__class__.__name__}: {exc}"
        )

        if should_raise:
            raise ExcCls(e_msg) from exc
        log.error(e_msg)

    def _parse_out_dir(self, fstr: str, *, name: str) -> str:
        """Evaluates the format string to create an output directory path.

        Note that the directories are _not_ created; this is outsourced to the
        plot creator such that it happens as late as possible.

        Args:
            fstr (str): The format string to evaluate and create a directory at
            name (str): Name of the plot
            timestamp (float, optional): Description

        Returns:
            str: The path of the created directory
        """
        # Get date format string and current time and create a string
        timefstr = self._out_fstrs.get("timestamp", "%y%m%d-%H%M%S")
        timestr = time.strftime(timefstr)

        out_dir = str(fstr).format(timestamp=timestr, name=name)

        # Make sure it is absolute
        out_dir = os.path.expanduser(out_dir)
        if not os.path.isabs(out_dir):
            # Regard it as relative to the data manager's output directory
            out_dir = os.path.join(self._dm.dirs["out"], out_dir)

        # Return the full path
        return out_dir

    def _parse_out_path(
        self,
        creator: BasePlotCreator,
        *,
        name: str,
        out_dir: str,
        file_ext: str = None,
        state_no: int = None,
        state_no_max: int = None,
        state_vector: Tuple[int] = None,
        dims: dict = None,
    ) -> str:
        """Given a creator and (optionally) parameter sweep information, a full
        and absolute output path is generated, including the file extension.

        Note that the directories are _not_ created; this is outsourced to the
        plot creator such that it happens as late as possible.

        Args:
            creator (BasePlotCreator): The creator instance, used to
                extract information on the file extension.
            name (str): The name of the plot
            out_dir (str): The absolute output directory, prepended to all
                generated paths
            file_ext (str, optional): The file extension to use
            state_no (int, optional): The state number, starting with 0
            state_no_max (int, optional): The maximum state number
            state_vector (Tuple[int], optional): The state vector with info
                on how far each state dimension has progressed in the sweep
            dims (dict, optional): The dict of parameter dimensions of the
                sweep that is carried out.

        Returns:
            str: The fully parsed output path for this plot
        """

        def parse_state_pair(
            name: str, dim: ParamDim, *, fstrs: dict
        ) -> Tuple[str]:
            """Helper method to create a state pair"""
            # Parse the name
            for search, replace in fstrs["state_name_replace_chars"]:
                name = name.replace(search, replace)

            # Parse the value
            val = str(dim.current_value)
            for search, replace in fstrs["state_val_replace_chars"]:
                val = val.replace(search, replace)

            return name, val

        # Get the fstrs
        fstrs = self.out_fstrs

        # Evaluate the keys available for both cases
        keys = dict(timestamp=time.strftime(fstrs["timestamp"]), name=name)

        # Parse file extension and ensure it starts with a dot
        ext = file_ext if file_ext else creator.get_ext()

        if ext and ext[0] != ".":
            ext = "." + ext
        elif ext is None:
            ext = ""

        keys["ext"] = ext

        # Change behaviour depending on whether state information was given
        if state_no is None or (state_no == state_no_max == 0):
            # Assume the other arguments are also None -> Not part of the sweep
            # Evaluate it
            out_path = fstrs["path"].format(**keys)

        else:
            # Is part of a sweep
            # Parse additional keys
            # state number
            digits = len(str(state_no_max))
            keys["state_no"] = fstrs["state_no"].format(
                no=state_no, digits=digits
            )

            # state values -- need to do some parsing here ...
            state_pairs = [
                parse_state_pair(name, dim, fstrs=fstrs)
                for name, dim in dims.items()
            ]

            sjc = fstrs["state_join_char"]
            keys["state"] = sjc.join(
                [fstrs["state"].format(name=k, val=v) for k, v in state_pairs]
            )

            # state vector
            svjc = fstrs["state_vector_join_char"]
            keys["state_vector"] = svjc.join([str(s) for s in state_vector])

            # Evaluate it
            out_path = fstrs["sweep"].format(**keys)

        # Prepend the output directory and return
        out_path = os.path.join(out_dir, out_path)

        return out_path

    def _check_plot_name(self, name: str) -> None:
        """Raises if a plot name contains bad characters"""
        if any([s in name for s in BAD_PLOT_NAME_CHARS]):
            _bad_name_chars = ", ".join(repr(s) for s in BAD_PLOT_NAME_CHARS)
            raise ValueError(
                f"The plot name '{name}' contains unsupported characters! "
                "Remove any of the following offending substrings from the "
                f"plot function name: {_bad_name_chars}"
            )

    def _get_plot_func(self, **resolver_kwargs) -> Callable:
        """Instantiates a plot function resolver,
        :py:class:`~dantro.plot.utils.plot_func.PlotFuncResolver`, and uses
        it to get the desired plot function callable.
        """
        pf_resolver = self._get_plot_func_resolver(**self._pfr_kwargs)
        return pf_resolver.resolve(**resolver_kwargs)

    def _get_plot_func_resolver(self, **init_kwargs) -> _PlotFuncResolver:
        """Instantiates the plot function resolver object with the given
        initialization arguments.

        This method is called from :py:meth:`._get_plot_func` and can be used
        for more conveniently controlling how the resolver is set up.
        By default, the ``init_kwargs`` will be equivalent to the
        ``plot_func_resolver_init_kwargs`` given to :py:meth:`.__init__`.
        """
        return self.PLOT_FUNC_RESOLVER(**init_kwargs)

    def _get_plot_creator(
        self,
        *,
        creator: Union[str, Callable],
        plot_func: Callable,
        name: str,
        init_kwargs: dict,
    ) -> BasePlotCreator:
        """Determines which plot creator to use by looking at the given
        arguments and the plotting function.

        Then, sets up the corresponding creator and returns it.

        This method is called from :py:meth:`._plot`.

        Args:
            creator (Union[str, Callable]): The name of the creator to be
                looked up in :py:attr:`.CREATORS`. Can also be None, in
                which case it is attempted to look it up from the ``plot_func``
                's ``creator`` attribute. If that was not possible either, the
                :py:attr:`.default_creator` is used. If a callable is given,
                will use that as a factory to construct the creator instance.
            name (str): The name that will be used for the plot creator,
                typically the plot name itself.
            init_kwargs (dict): Additional creator initialization parameters

        Returns:
            BasePlotCreator: The selected creator object, fully initialized.
        """
        # If no creator is given, try to retrieve it by some other means
        if creator is None:
            log.debug("No creator specified.")
            if hasattr(plot_func, "creator"):
                creator = plot_func.creator
                log.debug(
                    "  Using creator specified by plot function instead:  %s",
                    creator,
                )

            elif self.default_creator:
                creator = self.default_creator
                log.debug("  Using default creator instead:  %s", creator)

            else:
                raise InvalidCreator(
                    f"Could not determine a plot creator for plot '{name}'!\n"
                    "Either specify it directly via the `creator` argument, "
                    "associate one with the plot function via the decorator, "
                    "or set the `default_creator` argument during the "
                    "initialization of the PlotManager."
                )

        # Determine the actual creator *type* now.
        # More precisely, this can also be a factory that then generates the
        # BasePlotCreator-like object.
        if not callable(creator):
            PlotCreator = self.CREATORS[creator]
        else:
            PlotCreator = creator

        # Parse initialization kwargs, based on the defaults set in __init__
        # FIXME This is not working properly if ``creator`` is not a string!
        pc_kwargs = recursive_update(
            copy.deepcopy(self._shared_cckwargs),
            self._cckwargs.get(creator, {}),
        )
        if init_kwargs:
            log.debug("Recursively updating creator initialization kwargs ...")
            pc_kwargs = recursive_update(copy.deepcopy(pc_kwargs), init_kwargs)

        if "raise_exc" not in pc_kwargs:
            pc_kwargs["raise_exc"] = self.raise_exc

        # Can now instantiate the creator object
        pc = PlotCreator(
            name=name, plot_func=plot_func, dm=self._dm, **pc_kwargs
        )

        log.debug("Initialized %s.", pc.logstr)
        return pc

    def _invoke_plot_creation(
        self,
        plot_creator: BasePlotCreator,
        *,
        out_path: str,
        debug: bool = None,
        **plot_cfg,
    ) -> Union[bool, str]:
        """This method wraps the plot creator's ``__call__`` and is the last
        PlotManager method that is called prior to handing over to the selected
        plot creator. It takes care of invoking the plot creator's ``__call__``
        method and handling potential error messages and return values.

        Args:
            plot_creator (BasePlotCreator): The currently used creator object
            out_path (str): The plot output path
            debug (bool, optional): If given, this overwrites the ``raise_exc``
                option specified during initialization.
            **plot_cfg: The plot configuration

        Returns:
            Union[bool, str]: Whether the plot was carried out successfully.
                Returns the string ``'skipped'`` if the plot was skipped via a
                :py:class:`~dantro.exceptions.SkipPlot` exception.

        Raises:
            PlotCreatorError: On error within the plot creator. This is only
                raised if either ``debug is True`` or
                ``debug is None and self.raise_exc``. Otherwise, the error
                message is merely logged.
        """
        try:
            plot_creator(out_path=out_path, **plot_cfg)

        except SkipPlot as skip_reason:
            log.caution("Skipped. %s\n", skip_reason)
            return "skipped"

        except Exception as exc:
            self._handle_exception(
                exc, pc=plot_creator, debug=debug, ExcCls=PlotCreatorError
            )
            return False

        log.debug("Plot creator call returned successfully.")
        return True

    def _store_plot_info(
        self,
        name: str,
        *,
        plot_cfg: dict,
        plot_cfg_extras: dict,
        creator_name: str,
        save: bool,
        target_dir: str,
        part_of_sweep: bool = False,
        **info,
    ):
        """Stores all plot information in the plot_info list and, if ``save``
        is set, also saves it using :py:meth:`._save_plot_cfg`.
        """
        # Prepare the entry
        entry = dict(
            name=name,
            plot_cfg=plot_cfg,
            plot_cfg_extras=plot_cfg_extras,
            target_dir=target_dir,
            creator_name=creator_name,
            part_of_sweep=part_of_sweep,
            **info,
            plot_cfg_path=None,
        )

        if save:
            # Save the plot configuration
            save_path = self._save_plot_cfg(
                plot_cfg,
                name=name,
                target_dir=target_dir,
                **plot_cfg_extras,
            )

            # Store the path the configuration was saved at
            entry["plot_cfg_path"] = save_path

        # Append to the plot_info list
        self._plot_info.append(entry)

    def _save_plot_cfg(
        self,
        cfg: dict,
        *,
        name: str,
        target_dir: str,
        exists_action: str = None,
        is_sweep: bool = False,
        **plot_cfg_extras,
    ) -> str:
        """Saves the given configuration under the top-level entry ``name`` to
        a yaml file.

        Args:
            cfg (dict): The plot configuration to save
            name (str): The name of the plot
            target_dir (str): The directory path to store the file in
            exists_action (str, optional): What to do if a plot configuration
                already exists. Can be: ``overwrite``, ``overwrite_nowarn``,
                ``skip``, ``append``, ``raise``. If None, uses the value of the
                ``cfg_exists_action`` argument given during initialization.
            is_sweep (bool, optional): Set if the configuration refers to a
                plot in sweep mode, for which a different format string is used
            **plot_cfg_extras: Added to the plot configuration via recursive
                update.

        Returns:
            str: The path the config was saved at (mainly used for testing)

        Raises:
            ValueError: For invalid ``exists_action`` argument
        """
        # Resolve default arguments
        if exists_action is None:
            exists_action = self._cfg_exists_action

        # Build the dict that is to be saved
        d = dict()
        d[name] = copy.deepcopy(cfg)

        # Need to include some extra information like the creator and the
        # plot function that was used
        if not isinstance(cfg, ParamSpace):
            d[name] = recursive_update(d[name], plot_cfg_extras)

        else:
            # FIXME hacky, should not use the internal API!
            d[name]._dict = recursive_update(d[name]._dict, plot_cfg_extras)

        # Generate the filename and save path
        fn_fstr = self.out_fstrs["plot_cfg_sweep" if is_sweep else "plot_cfg"]
        fname = fn_fstr.format(name=name, basename=os.path.basename(name))
        save_path = os.path.join(target_dir, fname)

        # Try to write
        try:
            write_yml(d, path=save_path, mode="x")

        except FileExistsError:
            log.debug("Config file already exists at %s!", save_path)

            if exists_action == "raise":
                raise

            elif exists_action == "skip":
                log.debug("Skipping ...")

            elif exists_action == "append":
                log.debug("Appending ...")
                write_yml(d, path=save_path, mode="a")

            elif exists_action == "overwrite":
                log.warning("Overwriting existing plot configuration ...")
                write_yml(d, path=save_path, mode="w")

            elif exists_action == "overwrite_nowarn":
                log.debug("Overwriting ...")
                write_yml(d, path=save_path, mode="w")

            else:
                raise ValueError(
                    f"Invalid value '{exists_action}' for argument "
                    "`exists_action`! Choose from: raise, skip, append, "
                    "overwrite, overwrite_nowarn"
                )

        else:
            log.debug(
                "Saved plot configuration for '%s' to: %s", name, save_path
            )

        return save_path

    # .........................................................................
    # Plotting

    def plot_from_cfg(
        self,
        *,
        plots_cfg: Union[dict, str] = None,
        plot_only: List[str] = None,
        out_dir: str = None,
        resolve_based_on: bool = True,
        **update_plots_cfg,
    ) -> None:
        """Create multiple plots from a configuration, either a given one or
        the one passed during initialization.

        This is mostly a wrapper around the plot function, allowing additional
        ways of how to configure and create plots.

        Args:
            plots_cfg (Union[dict, str], optional): The plots configuration to
                use. If not given, the ``default_plots_cfg`` specified during
                initialization is used. If a string is given, will assume it
                is a path and load the file.
            plot_only (List[str], optional): If given, create only those plots
                from the resulting configuration that match these names. This
                will lead to the `enabled` key being ignored, regardless of its
                value. The strings given here may also include Unix shell-like
                wildcards like ``*`` and ``? ``, which are matched using the
                Python ``fnmatch`` module.
            out_dir (str, optional): A different output directory; will use the
                one passed at initialization if the given argument evaluates to
                False.
            resolve_based_on (bool, optional): Whether to resolve the
                ``based_on`` entries in ``plots_cfg`` here. If false, will
                postpone this to :py:meth:`.plot`,
                thus *not* including the rest of the ``plots_cfg`` in the base
                configuration pool for name resolution.
                Lookups happen from ``base_cfg_pools`` following the rules
                described in :py:func:`~dantro.plot._cfg.resolve_based_on`.
            **update_plots_cfg: If given, it is used to update the plots_cfg
                recursively. Note that on the top level the _names_ of the
                plots are placed; this cannot be used to make all plots have a
                common property. Furthermore, this update happens *before* the
                ``based_on`` entries are resolved.

        Raises:
            PlotConfigError: Empty or invalid plot configuration
            ValueError: Bad ``plot_only`` argument, e.g. not matching any of
                the available plot names.
        """
        # Determine which plot configuration to use, ensuring a deep copy
        if not plots_cfg:
            if not self._default_plots_cfg and not update_plots_cfg:
                e_msg = (
                    "Got empty `plots_cfg` and `plots_cfg` given at "
                    "initialization was also empty. Nothing to plot."
                )

                if self.raise_exc:
                    raise PlotConfigError(e_msg)

                log.error(e_msg)
                return

            log.debug("Using default plots configuration.")
            plots_cfg = copy.deepcopy(self._default_plots_cfg)

        else:
            plots_cfg = self._prepare_cfg(plots_cfg)

        # Allow update of plots configuration
        if update_plots_cfg:
            plots_cfg = recursive_update(
                plots_cfg, copy.deepcopy(update_plots_cfg)
            )
            log.debug("Updated the plots configuration.")

        # Check the plot configuration for invalid types
        for plot_name, cfg in plots_cfg.items():
            if not isinstance(cfg, (dict, ParamSpace)):
                raise PlotConfigError(
                    "Got invalid plots specifications for the plot named "
                    f"'{plot_name}'! Expected dict or ParamSpace, but got "
                    f"{type(cfg).__name__} with value '{cfg}'."
                )

        # Evaluate `plot_only`, but retain a copy of the full configuration
        full_plots_cfg = copy.deepcopy(plots_cfg)
        if plot_only is None:
            # Resolve all `enabled` entries, creating a new plots_cfg dict
            plots_cfg = {
                k: v for k, v in plots_cfg.items() if v.pop("enabled", True)
            }
        else:
            # Filter the plot selection, resolving glob-like patterns
            to_plot = []
            for name in plot_only:
                if any([s in name for s in ("*", "?", "[", "]")]):
                    # Add all matching plot names
                    to_plot += fnmatch.filter(plots_cfg.keys(), name)
                else:
                    # No globbing; add the name directly
                    to_plot.append(name)

            # Reduce the plot configuration to those entries
            try:
                plots_cfg = {k: plots_cfg[k] for k in to_plot}

            except KeyError as err:
                _plot_only = ", ".join(plot_only)
                raise ValueError(
                    f"Could not find a configuration for a plot named {err} "
                    "while resolving the plot_only argument! Check that it "
                    "was specified correctly:\n"
                    f"  plot_only:  {_plot_only}\n"
                    f"  available:\n{make_columns(plots_cfg.keys())}"
                ) from err

            # Remove all `enabled` keys from the remaining entries, thus also
            # enabling those plots that were set `enabled: False`
            for cfg in plots_cfg.values():
                cfg.pop("enabled", None)

        # Also throw out entries that start with an underscore or dot
        plots_cfg = {
            k: v
            for k, v in plots_cfg.items()
            if not (k.startswith("_") or k.startswith("."))
        }

        # Now resolve the `based_on` entries of all remaining plot entries,
        # but allow lookup from those plots that were not enabled.
        if resolve_based_on:
            # Determine only the _unused_ plot configurations such that they
            # can be added as a separate pool. As these might still contain
            # the already-evaluated `enabled` key, need to pop that out.
            # NOTE The full_plots_cfg no longer contains those entries after
            #      this operation was carried out.
            unused_plots_cfg = {
                k: (v, v.pop("enabled", None))[0]
                for k, v in full_plots_cfg.items()
                if k not in plots_cfg
            }

            plots_cfg = _resolve_based_on(
                plots_cfg,
                label="plot_from_cfg",
                base_pools=(
                    tuple(self.base_cfg_pools.items())
                    + (("plot_from_cfg_unused", unused_plots_cfg),)
                ),
            )

        # Determine the output directory path to use; not creating directories!
        if not out_dir:
            out_dir = self._out_dir
        out_dir = self._parse_out_dir(out_dir, name="{name:}")
        # NOTE creating this here such that all plots from this config are side
        #      by side in one output directory. With the given `name` key, the
        #      evaluation of that part of the out_dir path is postponed
        #      to when the actual plot with that name is created.
        #      Not doing this would lead to multiple output paths for different
        #      time stamps.

        # Provide information on how many plots will be created
        log.hilight(
            "Performing plots from %d plot configuration entr%s:",
            len(plots_cfg),
            "ies" if len(plots_cfg) != 1 else "y",
        )
        t0 = time.time()

        _plot_names = []
        _num_plots = 0
        for plot_name, cfg in plots_cfg.items():
            _n = 1
            _creator = cfg.get("creator", "auto")
            if isinstance(cfg, ParamSpace):
                _n = cfg.volume if cfg.volume else 1

            _plot_names.append(
                "  - {:<50s}  ({:s}, {:d} plot{:s})"
                "".format(plot_name, _creator, _n, "s" if _n != 1 else "")
            )
            _num_plots += _n
        log.note(
            "Have (at least) the following %d plots to perform:\n%s\n",
            _num_plots,
            "\n".join(_plot_names),
        )

        # Loop over the configured plots and invoke the individual plot calls
        for plot_name, cfg in plots_cfg.items():
            if isinstance(cfg, ParamSpace):
                self.plot(plot_name, default_out_dir=out_dir, from_pspace=cfg)

            else:
                self.plot(plot_name, default_out_dir=out_dir, **cfg)

        log.success(
            "Performed plots from %d plot configuration%s in %s.\n",
            len(plots_cfg),
            "s" if len(plots_cfg) != 1 else "",
            _fmt_time(time.time() - t0),
        )

    def plot(
        self,
        name: str,
        *,
        based_on: Union[str, Tuple[str]] = None,
        from_pspace: Union[dict, ParamSpace] = None,
        **plot_cfg,
    ) -> BasePlotCreator:
        """Create plot(s) from a single configuration entry.

        A call to this function resolves the ``based_on`` feature and passes
        the derived plot configuration to :py:meth:`._plot`, which actually
        carries out the plotting. See there for documentation of further
        arguments.

        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.

        For

        Args:
            name (str): The name of this plot. This will be used for generating
                an output file path later on. Some characters are not allowed,
                e.g. ``*`` and ``?``, but a ``/`` can be used to store the plot
                output in a subdirectory.
            based_on (Union[str, Tuple[str]], optional): A key or a sequence
                of keys of entries in the base pool that should be used as
                the basis of this plot. The given plot configuration is then
                used to recursively update (a copy of) those base
                configuration entries.
                Lookups happen from ``base_cfg_pools`` following the rules
                described in :py:func:`~dantro.plot._cfg.resolve_based_on`.
            from_pspace (Union[dict, paramspace.paramspace.ParamSpace], optional):
                If given, execute a parameter sweep over these parameters,
                re-using the same creator instance. If this is a dict, a
                ParamSpace is created from it.
            **plot_cfg: The plot configuration, including some parameters that
                the plot manager will evaluate (and consequently: does not
                pass on to the plot creator).
                If using ``from_pspace``, parameters given here will
                recursively update those given in ``from_pspace``.

        Returns:
            BasePlotCreator: The PlotCreator used for these plots
        """
        self._check_plot_name(name)

        # Distinguish cases that are using a ParamSpace and those that do not.
        # The latter are far simpler to handle ...
        if from_pspace is None:
            plot_cfg = _resolve_based_on_single(
                name=name,
                based_on=based_on,
                plot_cfg=plot_cfg,
                label="plot",
                base_pools=self._base_cfg_pools,
            )
            return self._plot(name, **plot_cfg)

        # Else: It's more complicated now, as the config is in from_pspace, and
        # (partly) in plot_cfg. Urgh.
        # To make it easier, get information out of the `ParamSpace` object,
        # move entries from `plot_cfg` over there, resolve the `based_on`
        # entries and then build a new `ParamSpace` object (done in `_plot`).
        if isinstance(from_pspace, ParamSpace):
            from_pspace = copy.deepcopy(from_pspace._dict)
            # FIXME Should not have to use private API!

        # Combine from_pspace and plot_cfg into one dict
        from_pspace = recursive_update(from_pspace, copy.deepcopy(plot_cfg))

        # Resolve `based_on`
        from_pspace = _resolve_based_on_single(
            name=name,
            based_on=based_on,
            plot_cfg=from_pspace,
            label="plot_pspace",
            base_pools=self._base_cfg_pools,
        )

        # Extract keys which `_plot` expects as separate arguments
        kwargs = {
            k: from_pspace.pop(k, None)
            for k in (
                "plot_func",
                "module",
                "module_file",
                "creator",
                "out_dir",
                "default_out_dir",
                "save_plot_cfg",
                "creator_init_kwargs",
            )
        }

        return self._plot(name, from_pspace=from_pspace, **kwargs)

    def _plot(
        self,
        name: str,
        *,
        plot_func: Union[str, Callable] = None,
        module: str = None,
        module_file: str = None,
        creator: Union[str, Callable] = None,
        out_dir: str = None,
        default_out_dir: str = None,
        file_ext: str = None,
        save_plot_cfg: bool = None,
        creator_init_kwargs: dict = None,
        from_pspace: dict = None,
        **plot_cfg,
    ) -> BasePlotCreator:
        """Create plot(s) from a single configuration entry.

        This first resolves the plot function using the plot function resolver
        class: :py:class:`~dantro.plot.utils.plot_func.PlotFuncResolver` or a
        derived class (depending on the :py:attr:`.PLOT_FUNC_RESOLVER`).

        A call to this function creates a :ref:`plot creator <plot_creators>`,
        which is also returned after all plots are finished.

        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file or when using ``from_pspace``.

        Args:
            name (str): The name of this plot
            plot_func (Union[str, Callable], optional): The name or module
                string of the plot function as it can be imported from
                ``module``. If this is a callable will directly return that
                callable. This argument *needs* be given.
            module (str): If ``plot_func`` was the name of the plot
                function, this needs to be the name of the module to import
                that name from.
            module_file (str): Path to the file to load and look for
                the ``plot_func`` in. If ``base_module_file_dir`` is given
                during initialization, this can also be a path relative to that
                directory.
            creator (Union[str, Callable]): The name of the creator to
                be looked up in :py:attr:`.CREATORS`. Can also be None, in
                which case it is attempted to look it up from the ``plot_func``
                's ``creator`` attribute. If that was not possible either, the
                :py:attr:`.default_creator` is used. If a callable is given,
                will use that as a factory to set up the creator.
            out_dir (str, optional): If given, will use this directory as out
                directory. If not, will use the default value given by
                ``default_out_dir`` or that given at initialization.
            default_out_dir (str, optional): An output directory that was
                determined *in the calling context* and which should be used as
                default if no ``out_dir`` was given explicitly.
            file_ext (str, optional): The file extension to use, including the
                leading dot!
            save_plot_cfg (bool, optional): Whether to save the plot config.
                If not given, uses the default value from initialization.
            creator_init_kwargs (dict, optional): Passed to the plot creator
                during initialization. Note that the arguments given at
                initialization of the PlotManager are updated by this.
            from_pspace (dict, optional): If given, execute a parameter
                sweep over this parameter space, re-using the same creator
                instance. Each point in parameter space will end up calling
                this method with arguments unpacked to the ``plot_cfg``
                argument.
            **plot_cfg: The plot configuration to pass on to the plot creator.
                This may be completely empty if ``from_pspace`` is used!

        Returns:
            BasePlotCreator: The PlotCreator used for these plots. This will
                also be returned in case the plot failed!

        Raises:
            PlotConfigError: If no out directory was specified here or at
                initialization.
            PlotCreatorError: In case the preparation or execution of the plot
                failed for whatever reason. Not raised if not in debug mode.
        """

        log.debug("Preparing plot '%s' ...", name)
        t0 = time.time()

        # Gather arguments that are explicitly handled here; they may still be
        # needed downstream, e.g. when storing the plot configuration to file.
        plot_cfg_extras = dict(
            out_dir=out_dir,
            file_ext=file_ext,
            creator=creator,
            plot_func=str(plot_func),  # ... to not have a callable in here
            module=module,
            module_file=module_file,
            creator_init_kwargs=copy.deepcopy(creator_init_kwargs),
            save_plot_cfg=save_plot_cfg,
        )
        plot_cfg_extras = {
            k: v for k, v in plot_cfg_extras.items() if v is not None
        }

        # Evaluate the output directory and whether to save plot configs
        if not out_dir:
            if default_out_dir:
                out_dir = default_out_dir

            elif self._out_dir:
                out_dir = self._out_dir

            else:
                raise PlotConfigError(
                    "No `out_dir` specified here and at "
                    "initialization; cannot perform plot."
                )

        if save_plot_cfg is None:
            save_plot_cfg = self.save_plot_cfg

        # Retrieve the plot function callable
        if plot_func is None:
            raise PlotConfigError("Missing `plot_func` argument!")
        plot_func = self._get_plot_func(
            plot_func=plot_func, module=module, module_file=module_file
        )

        # Set up the plot creator instance
        plot_creator = self._get_plot_creator(
            creator=creator,
            plot_func=plot_func,
            name=name,
            init_kwargs=creator_init_kwargs,
        )

        # Let the creator process arguments
        try:
            plot_cfg, from_pspace = plot_creator.prepare_cfg(
                plot_cfg=plot_cfg, pspace=from_pspace
            )
        except Exception as exc:
            _debug = plot_cfg.get("debug")
            self._handle_exception(
                exc, pc=plot_creator, debug=_debug, ExcCls=PlotCreatorError
            )
            return plot_creator

        # Distinguish single calls and parameter sweeps
        if not from_pspace:
            log.progress("Plotting '%s' ...", name)

            # Generate the output path
            out_dir = self._parse_out_dir(out_dir, name=name)
            out_path = self._parse_out_path(
                plot_creator, name=name, out_dir=out_dir, file_ext=file_ext
            )

            # Call the plot creator to perform the plot, using the private
            # method to perform exception handling
            rv = self._invoke_plot_creation(
                plot_creator, out_path=out_path, **plot_cfg
            )

            # Store plot information (_save_ only if not skipped)
            self._store_plot_info(
                name=name,
                creator_name=creator,
                out_path=out_path,
                plot_cfg=plot_cfg,
                plot_cfg_extras=plot_cfg_extras,
                save=save_plot_cfg and rv != "skipped",
                target_dir=os.path.dirname(out_path),
                creator_rv=rv,
                part_of_sweep=False,
            )

            if rv is True:
                log.progress(
                    "Performed '%s' plot in %s.\n",
                    name,
                    _fmt_time(time.time() - t0),
                )

            return plot_creator

        # else: Is a parameter sweep over the plot configuration ..............
        # NOTE The parameter space is allowed to have volume 0!

        # Make sure it's a ParamSpace
        if not isinstance(from_pspace, ParamSpace):
            from_pspace = ParamSpace(from_pspace)

        # Extract some info and communicate it
        psp_vol = from_pspace.volume
        psp_dims = from_pspace.dims

        if psp_vol > 0:
            amap_coords = from_pspace.active_state_map.coords
            max_dname_len = max(len(n) for n in amap_coords.keys())
            n_max = psp_vol

            log.progress("Performing %d '%s' plots ...", n_max, name)
            log.note(
                "... iterating over parameter space:\n%s",
                "\n".join(
                    [
                        "  * {0:<{d:}} : {1:}".format(
                            dim_name,
                            ", ".join([str(c) for c in coords.values]),
                            d=max_dname_len,
                        )
                        for dim_name, coords in amap_coords.items()
                    ]
                ),
            )

        else:
            log.progress("Plotting '%s' ...", name)
            log.note("... from default point in zero-volume parameter space.")
            n_max = 1

        # Parse the output directory, such that all plots are together in
        # one directory even if the timestamp varies
        out_dir = self._parse_out_dir(out_dir, name=name)

        # Create the iterator
        it = from_pspace.iterator(
            with_info=("state_no", "state_vector", "coords")
        )

        # Keep track of how many plots are skipped
        num_skipped = 0

        # ...and loop over all points:
        for n, (cfg, state_no, state_vector, coords) in enumerate(it):
            log.progress("Plotting '%s' (%d/%d) ...", name, n + 1, n_max)
            if coords:
                log.note(
                    "Current coordinates of sweep plot configuration:\n%s",
                    "\n".join(
                        "  {:>23s}:   {}".format(*kv) for kv in coords.items()
                    ),
                )

            # Handle the file extension parameter; it might come from the
            # given configuration and then needs to be popped such that it
            # is not propagated to the plot creator.
            _file_ext = cfg.pop("file_ext", file_ext)

            # Generate the output path
            out_path = self._parse_out_path(
                plot_creator,
                name=name,
                out_dir=out_dir,
                file_ext=_file_ext,
                state_no=state_no,
                state_no_max=n_max - 1,
                state_vector=state_vector,
                dims=psp_dims,
            )

            # Call the plot creator to perform the plot, using the private
            # method to perform exception handling
            rv = self._invoke_plot_creation(
                plot_creator, out_path=out_path, **cfg, **plot_cfg
            )
            # NOTE The **plot_cfg is passed here in order to not lose any
            #      arguments that might have been passed to it. While `cfg`
            #      _should_ hold all the arguments from the parameter space
            #      iteration, there might be more arguments in `plot_cfg`;
            #      rather than disallowing this, we pass them on and
            #      forward responsibility downstream ...

            # Always store plot information, regardless of skipping.
            # Saving is enabled only for zero-volume parameter sweeps in order
            # to have the backup file right beside the plot; the file will not
            # be saved again after the for-loop.
            self._store_plot_info(
                name=name,
                creator_name=creator,
                out_path=out_path,
                plot_cfg=dict(**cfg, **plot_cfg),
                plot_cfg_extras=plot_cfg_extras,
                state_no=state_no,
                state_vector=state_vector,
                save=(save_plot_cfg and psp_vol == 0 and rv != "skipped"),
                target_dir=os.path.dirname(out_path),
                creator_rv=rv,
                part_of_sweep=True,
            )

            # Count skipped plots
            if rv == "skipped":
                num_skipped += 1

            elif rv is True:
                log.progress("Finished plot %d/%d.", n + 1, n_max)

                # Estimate for time remaining
                if (n + 1) < n_max:
                    dt = time.time() - t0
                    etl = dt / ((n + 1) / n_max) - dt
                    log.note(
                        "Estimated time needed for %d remaining plots:  %s\n",
                        n_max - (n + 1),
                        _fmt_time(etl),
                    )

        # Finished parameter space iteration.
        # Save the plot configuration alongside, if configured to do so, and
        # if at least one of the plots was *not* skipped and the parameter
        # space was not zero-volume
        if save_plot_cfg and num_skipped < (n + 1) and psp_vol > 0:
            self._save_plot_cfg(
                from_pspace,
                name=name,
                target_dir=out_dir,
                is_sweep=True,
                **plot_cfg_extras,
            )

        dt = time.time() - t0
        if not num_skipped:
            log.progress(
                "Performed all '%s' plots in %s.\n", name, _fmt_time(dt)
            )
        else:
            log.progress(
                "Performed %d/%d '%s' plots in %s, skipped %d.\n",
                (n + 1) - num_skipped,
                n + 1,
                name,
                _fmt_time(dt),
                num_skipped,
            )

        # Done now. Return the plot creator object
        return plot_creator
