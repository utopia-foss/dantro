"""This module implements the PlotManager class, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the PlotCreator.
"""

import os
import time
import copy
import fnmatch
import logging
from typing import Union, List, Dict, Tuple, Any

from paramspace import ParamSpace, ParamDim

from .data_mngr import DataManager
from .plot_creators import ALL as ALL_PCRS
from .plot_creators import BasePlotCreator
from .tools import load_yml, write_yml, recursive_update

# Local constants
log = logging.getLogger(__name__)

# Substrings that may not appear in plot names
BAD_NAME_CHARS = ('*', '?', '[', ']', '!', ':', '.')

# -----------------------------------------------------------------------------
# Custom exception classes

class PlottingError(Exception):
    """Custom exception class for all plotting errors"""


class PlotConfigError(ValueError, PlottingError):
    """Raised when there were errors in the plot configuration"""


class InvalidCreator(ValueError, PlottingError):
    """Raised when an invalid creator was specified"""


class PlotCreatorError(PlottingError):
    """Raised when an error occured in a plot creator"""


# -----------------------------------------------------------------------------


class PlotManager:
    """The PlotManager takes care of configuring plots and calling the
    configured PlotCreator classes that then carry out the plots.

    Attributes:
        CREATORS (dict): The mapping of creator names to classes. When it is
            desired to subclass PlotManager and extend the creator mapping, use
            `dict(**pcr.ALL)` to inherit the default creator mapping.
        DEFAULT_OUT_FSTRS (dict): The default values for the output format
            strings.
    """

    CREATORS = ALL_PCRS
    DEFAULT_OUT_FSTRS = dict(timestamp="%y%m%d-%H%M%S",
                             state_no="{no:0{digits:d}d}",
                             state="{name:}_{val:}",
                             state_name_replace_chars=[],  # (".", "-")
                             state_val_replace_chars=[("/", "-")],
                             state_join_char="__",
                             state_vector_join_char="-",
                             # final fstr for single plot and config path
                             path="{name:}{ext:}",
                             plot_cfg="{basename:}_cfg.yml",
                             # and for sweep
                             sweep="{name:}/{state_no:}__{state:}{ext:}",
                             plot_cfg_sweep="{name:}/sweep_cfg.yml",
                             )

    def __init__(self, *, dm: DataManager, base_cfg: Union[dict, str]=None,
                 update_base_cfg: Union[dict, str]=None,
                 plots_cfg: Union[dict, str]=None,
                 out_dir: Union[str, None]="{timestamp:}/",
                 out_fstrs: dict=None,
                 creator_init_kwargs: Dict[str, dict]=None,
                 default_creator: str=None,
                 auto_detect_creator: bool=False,
                 save_plot_cfg: bool=True,
                 raise_exc: bool=False,
                 cfg_exists_action: str='raise'):
        """Initialize the PlotManager

        The initialization comes with three (optional) hierarchical levels to
        make the configuration of plots versatile, flexible, and avoid copy-
        paste of configurations:
        The first two result in a so-called "base" configuration, a collection
        of available, but disabled plot configs.
        The third specifies the default plots, which can use the `based_on`
        feature to base their configuration on any of the configurations from
        the base plot configuration.

        Specifically:

            1. The ``base_cfg`` contains a set of plot configurations that form
                a repertoire of configurations. These are not performed by
                default, but can be imported.
            2. The ``update_base_cfg`` contains plot configurations that are
                possibly derived from the base repertoire. This happens in the
                following way: First by recursive update of existing entries,
                and second by resolving the ``based_on: a_base_plot`` again by
                recursive update.
            3. The ``plots_cfg`` holds enabled plot configurations, possibly
                derived from the base configuration using the `based_on`
                feature, e.g. ``based_on: a_base_plot``; this happens by
                recursive update.

        Args:
            dm (DataManager): The DataManager-derived object to read the plot
                data from.
            base_cfg (Union[dict, str], optional): The default base config or a
                path to a yaml-file to import. The base config defines a set
                of plot configuration that other plot configurations can
                declare themselves ``based_on``.
            update_base_cfg (Union[dict, str], optional): An update config to
                the base config or a path to a yaml-file to import which
                recursively updates the ``base_cfg``.
            plots_cfg (Union[dict, str], optional): The default plots config or
                a path to a YAML file to import
            out_dir (Union[str, None], optional): If given, will use this
                output directory as basis for the output path for each plot.
                The path can be a format-string; it is evaluated upon call to
                the plot command. Available keys: ``timestamp``, ``name``, ...
                For a relative path, this will be relative to the DataManager's
                output directory. Absolute paths remain absolute.
                If this argument evaluates to False, the DataManager's output
                directory will be the output directory.
            out_fstrs (dict, optional): Format strings that define how the
                output path is generated. The dict given here updates the
                ``DEFAULT_OUT_FSTRS`` class variable which holds the default
                values.

                Keys: ``timestamp`` (%-style), ``path``, ``sweep``, ``state``,
                ``plot_cfg``, ``state``, ``state_no``, ``state_join_char``,
                ``state_vector_join_char``.

                Available keys for ``path``: ``name``, ``timestamp``, ``ext``.

                Additionally, for ``sweep``: ``state_no``, ``state_vector``,
                    ``state``.

            creator_init_kwargs (Dict[str, dict], optional): If given, these
                kwargs are passed to the initialization calls of the respective
                creator classes.
            default_creator (str, optional): If given, a plot without explicit
                ``creator`` declaration will use this creator as default.
            auto_detect_creator (bool, optional): If true, and no default
                creator is given, will try to automatically deduce the creator
                using the given plot arguments. All creators registered with
                this PlotManager instance are candidates.
            save_plot_cfg (bool, optional): If True, the plot configuration is
                saved to a yaml file alongside the created plot.
            raise_exc (bool, optional): Whether to raise exceptions if there
                are errors raised from the plot creator or errors in the plot
                configuration. If False, the errors will only be logged.
            cfg_exists_action (str, optional): Behaviour when a config file
                already exists. Can be: ``raise`` (default), ``skip``,
                ``append``, ``overwrite``, or ``overwrite_nowarn``.

        Raises:
            InvalidCreator: When an invalid default creator was chosen
            KeyError: Upon bad ``based_on`` in ``update_base_cfg``
        """
        # TODO consider making it possible to pass classes for plot creators

        # Initialize attributes and store arguments
        self._plot_info = []

        # Public
        self.save_plot_cfg = save_plot_cfg
        self.raise_exc = raise_exc

        # Private or read-only
        self._dm = dm
        self._out_dir = out_dir
        self._auto_detect_creator = auto_detect_creator

        # Parameters to pass through as defaults to member functions
        self._cfg_exists_action = cfg_exists_action

        # Handle base config
        if isinstance(base_cfg, str):
            # Interpret as path to yaml file
            log.debug("Loading base_cfg from file %s ...", base_cfg)
            self._base_cfg = load_yml(base_cfg)
        else:
            self._base_cfg = copy.deepcopy(base_cfg)

        # Handle the update of base config
        if isinstance(update_base_cfg, str):
            # Interpret as path to yaml file
            log.debug("Loading update_base_cfg from file %s ...",
                      update_base_cfg)
            update_base_cfg = load_yml(update_base_cfg)

        # Perform update of base config: recursive + resolution of `based_on`
        if update_base_cfg:
            # First, make a recursive update of the existing based_on
            self._base_cfg = recursive_update(self._base_cfg, update_base_cfg)
            # Now, potentially existing `based_on` entries from either of
            # these configurations are part of the _base_cfg
            # Resolve these `based_on` keys ...
            for pcfg_name, pcfg in self._base_cfg.items():
                based_on = pcfg.pop('based_on', None)

                if not based_on:
                    continue

                elif isinstance(based_on, str):
                    based_on = (based_on,)

                # Now a sequence of strings; go over all these and build a dict
                # that will be the base that is to be recursively updated with
                # the given update config.
                bcfg = dict()
                for _based_on in based_on:
                    if _based_on not in self._base_cfg:
                        raise KeyError("No base plot configuration named '{}' "
                                       "available to use during resolution of "
                                       "`update_base_cfg`! Available: {}"
                                       "".format(_based_on,
                                                 ", ".join(self._base_cfg)))

                    # Need to work on a deep copy of the original base config
                    # entry in order to not get any mutability issues
                    _bcfg = copy.deepcopy(self._base_cfg[_based_on])
                    bcfg = recursive_update(bcfg, _bcfg)

                # Finally, apply the given update and store as attribute
                self._base_cfg[pcfg_name] = recursive_update(bcfg, pcfg)

        # Handle default plots configuration
        if isinstance(plots_cfg, str):
            # Interpret as path to yaml file
            log.debug("Loading plots_cfg from file %s ...", plots_cfg)
            plots_cfg = load_yml(plots_cfg)
        self._plots_cfg = plots_cfg

        # Update the default format strings, if any were given here
        self._out_fstrs = self.DEFAULT_OUT_FSTRS
        if out_fstrs:
            self._out_fstrs = recursive_update(copy.deepcopy(self._out_fstrs),
                                               out_fstrs)

        # Store creator init kwargs
        self._cckwargs = creator_init_kwargs if creator_init_kwargs else {}

        # Store default creator name
        if default_creator and default_creator not in self.CREATORS:
            raise InvalidCreator("No such creator '{}' available, only: {}"
                                 "".format(default_creator,
                                           [k for k in self.CREATORS.keys()]))
        self._default_creator = default_creator

        log.debug("%s initialized.", self.__class__.__name__)

    # .........................................................................
    # Properties

    @property
    def out_fstrs(self) -> dict:
        """Returns the dict of output format strings"""
        return self._out_fstrs

    @property
    def plot_info(self) -> List[dict]:
        """Returns a list of dicts with info on all plots"""
        return self._plot_info

    @property
    def base_cfg(self) -> dict:
        """Returns a deep copy of the base configuration"""
        return copy.deepcopy(self._base_cfg)

    # .........................................................................
    # Helpers

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
        # TODO allow passing a timestamp?
        timefstr = self._out_fstrs.get('timestamp', "%y%m%d-%H%M%S")
        timestr = time.strftime(timefstr)

        out_dir = fstr.format(timestamp=timestr, name=name)

        # Make sure it is absolute
        out_dir = os.path.expanduser(out_dir)
        if not os.path.isabs(out_dir):
            # Regard it as relative to the data manager's output directory
            out_dir = os.path.join(self._dm.dirs['out'], out_dir)

        # Return the full path
        return out_dir

    def _parse_out_path(self, creator: BasePlotCreator, *, name: str,
                        out_dir: str, file_ext: str=None, state_no: int=None,
                        state_no_max: int=None, state_vector: Tuple[int]=None,
                        dims: dict=None) -> str:
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
        def parse_state_pair(name: str, dim: ParamDim, *,
                             fstrs: dict) -> Tuple[str]:
            """Helper method to create a state pair"""
            # Parse the name
            for search, replace in fstrs['state_name_replace_chars']:
                name = name.replace(search, replace)

            # Parse the value
            val = str(dim.current_value)
            for search, replace in fstrs['state_val_replace_chars']:
                val = val.replace(search, replace)

            return name, val

        # Get the fstrs
        fstrs = self.out_fstrs

        # Evaluate the keys available for both cases
        keys = dict(timestamp=time.strftime(fstrs['timestamp']), name=name)

        # Parse file extension and ensure it starts with a dot
        ext = file_ext if file_ext else creator.get_ext()

        if ext and ext[0] != ".":
            ext = "." + ext
        elif ext is None:
            ext = ""

        keys['ext'] = ext

        # Change behaviour depending on whether state information was given
        if state_no is None:
            # Assume the other arguments are also None -> Not part of the sweep
            # Evaluate it
            out_path = fstrs['path'].format(**keys)

        else:
            # Is part of a sweep
            # Parse additional keys
            # state number
            digits = len(str(state_no_max))
            keys['state_no'] = fstrs['state_no'].format(no=state_no,
                                                        digits=digits)

            # state values -- need to do some parsing here ...
            state_pairs = [parse_state_pair(name, dim, fstrs=fstrs)
                           for name, dim in dims.items()]

            sjc = fstrs['state_join_char']
            keys['state'] = sjc.join([fstrs['state'].format(name=k, val=v)
                                      for k, v in state_pairs])

            # state vector
            svjc = fstrs['state_vector_join_char']
            keys['state_vector'] = svjc.join([str(s) for s in state_vector])

            # Evaluate it
            out_path = fstrs['sweep'].format(**keys)

        # Prepend the output directory and return
        out_path = os.path.join(out_dir, out_path)

        return out_path

    def _resolve_based_on(self, *, cfg: dict,
                          based_on: Union[str, Tuple[str]]=None,
                          work_on_deepcopy: bool=True) -> dict:
        """Resolves the ``based_on`` reference in a plot configuration

        Args:
            cfg (dict): The plot configuration that will be used to recursively
                update the specified base configurations.
            based_on (Union[str, Tuple[str]], optional): The name or names of
                the base configuration entries to use for updating
            work_on_deepcopy (bool, optional): Whether to work on a deepcopy

        Returns:
            dict: The plot configuration derived from the one at ``based_on``

        Raises:
            KeyError: If ``based_on`` value is not a key in ``self._base_cfg``
        """
        if not based_on:
            return cfg

        elif isinstance(based_on, str):
            based_on = (based_on,)

        # Copy, if needed
        if work_on_deepcopy:
            cfg = copy.deepcopy(cfg)

        # Resolve the list of based_on entries
        base_cfg = dict()
        for _based_on in based_on:
            if _based_on not in self.base_cfg.keys():
                raise KeyError("No base plot configuration named '{}' "
                               "available! Choose from: {}"
                               "".format(_based_on,
                                         ", ".join(self._base_cfg.keys())))

            # Do the recursive update. The base_cfg property already returns
            # a deep copy of the base configuration ...
            base_cfg = recursive_update(base_cfg, self.base_cfg[_based_on])

        # As final step, update the base configuration with everything
        return recursive_update(base_cfg, cfg)

    def _get_plot_creator(self, creator: Union[str, None],
                          *, name: str, init_kwargs: dict,
                          from_pspace: ParamSpace=None, plot_cfg: dict,
                          auto_detect: bool=None) -> BasePlotCreator:
        """Determines which plot creator to use by looking at the given
        arguments. If set, tries to auto-detect from the arguments, which
        creator is to be used.

        Then, sets up the corresponding creator and returns it.

        This method is called from the plot() method.

        Args:
            creator (Union[str, None]): The name of the creator to be found.
                Can be None, if no argument was given to the plot method.
            name (str): The name of the plot
            init_kwargs (dict): Additional creator initialization parameters
            from_pspace (ParamSpace, optional): If the plot is to be creatd
                from a parameter space, that parameter space.
            plot_cfg (dict): The plot configuration
            auto_detect (bool, optional): Whether to auto-detect the creator.
                If none, the value given at initialization is used.

        Returns:
            BasePlotCreator: The selected creator object, fully initialized.

        Raises:
            InvalidCreator: If the ``creator`` argument was invalid or auto-
                detection failed.
        """
        # If no creator is given, check if a default one can be used
        if creator is None:
            # Combine auto-detect related settings
            auto_detect = (auto_detect if auto_detect is not None
                           else self._auto_detect_creator)

            # Find other ways to determine the name of the creator
            if self._default_creator:
                # Just use the default
                creator = self._default_creator

            elif auto_detect:
                # Can try to auto-detect from the arguments, which creator
                # could uniquely fit to the arguments
                log.debug("Attempting auto-detection of creator ...")

                # If a ParamSpace plot is to be made, detect feasibility by
                # using its default parameters
                cfg = plot_cfg if not from_pspace else from_pspace.default

                # (name, plot creator) tuples for each candidate
                pc_candidates = []

                # Go over all registered plot creators
                for pc_name in self.CREATORS.keys():
                    try:
                        # Instantiate them by calling this function recursively
                        pc = self._get_plot_creator(pc_name, name=name,
                                                    init_kwargs=init_kwargs,
                                                    from_pspace=from_pspace,
                                                    plot_cfg=plot_cfg)
                        # NOTE Cannot call a class method here, because some
                        #      creators might need the actual arguments in
                        #      order to work properly

                    except:
                        # Failed to initialize for whatever reason, thus not
                        # a candidate
                        continue

                    # Successfully initialized. Check if it's a candidate for
                    # this plot configuration
                    if pc.can_plot(pc_name, **cfg):
                        log.debug("Plot creator '%s' declared itself a "
                                  "candidate.", pc_name)
                        pc_candidates.append((pc_name, pc))

                # If there is more than one candidate, cannot decide
                if len(pc_candidates) > 1:
                    pcc_names = [n for n, _ in pc_candidates]
                    raise InvalidCreator("Tried to auto-detect a plot creator "
                                         "for plot '{}' but could not "
                                         "unambiguously do so! There were {} "
                                         "plot creators declaring themselves "
                                         "as candidates: {}. Consider "
                                         "specifying the creator explicitly."
                                         "".format(name, len(pc_candidates),
                                                   ", ".join(pcc_names)))
                elif len(pc_candidates) < 1:
                    pc_names = ", ".join([k for k in self.CREATORS.keys()])
                    raise InvalidCreator("Tried to auto-detect a plot creator "
                                         "for plot '{}' but none of the "
                                         "available creators ({}) declared "
                                         "itself a candidate! This might also "
                                         "be due to a plot configuration that"
                                         "the candidate plot creators could "
                                         "not interpret. Consider specifying "
                                         "the creator explicitly to find out "
                                         "why auto-detection failed."
                                         "".format(name, pc_names))

                # else: there was only one, use that
                pc_name, pc = pc_candidates[0]
                log.debug("Auto-detected plot creator: %s", pc.logstr)

                # As it is already initialized, can just return it
                return pc

            else:
                raise InvalidCreator("No `creator` argument given and neither "
                                     "`default_creator` specified during "
                                     "initialization nor auto-detection "
                                     "enabled. Cannot plot!")

        # Parse initialization kwargs, based on the defaults set in __init__
        pc_kwargs = self._cckwargs.get(creator, {})
        if init_kwargs:
            log.debug("Recursively updating creator initialization kwargs ...")
            pc_kwargs = recursive_update(copy.deepcopy(pc_kwargs), init_kwargs)

        if 'raise_exc' not in pc_kwargs:
            pc_kwargs['raise_exc'] = self.raise_exc

        # Instantiate the creator class
        pc = self.CREATORS[creator](name=name, dm=self._dm, **pc_kwargs)

        log.debug("Initialized %s.", pc.logstr)
        return pc

    def _invoke_creator(self, pc: BasePlotCreator, *,
                        out_path: str, debug: bool=None,
                        **plot_cfg) -> Union[Any, None]:
        """This method wraps the plot creator's ``__call__`` and is the last
        PlotManager method that is called prior to handing over to the selected
        plot creator. It takes care of invoking the plot creator's ``__call__``
        method and handling potential error messages and return values.

        Args:
            pc (BasePlotCreator): The currently used creator
            out_path (str): The plot output path
            debug (bool, optional): If given, this overwrites the ``raise_exc``
                option specified during initialization.
            **plot_cfg: The plot configuration

        Returns:
            Any: The return value of the plot creator's ``__call__`` method or
                ``None`` if an error occurred but was not raised.

        Raises:
            PlotCreatorError: On error within the plot creator. This is only
                raised if either ``debug is True`` or
                ``debug is None and self.raise_exc``. Otherwise, the error
                message is merely logged.
        """
        try:
            rv = pc(out_path=out_path, **plot_cfg)

        except Exception as err:
            # Conditionally assemble an error message
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
                f"{e_dbg if not should_raise else e_no_dbg}\n"
                f"{err.__class__.__name__}: {err}"
            )

            if should_raise:
                raise PlotCreatorError(e_msg) from err
            log.error(e_msg)
            return None

        log.debug("Plot creator call returned successfully with %s.", type(rv))
        return rv

    def _store_plot_info(self, name: str,
                         *, plot_cfg: dict, creator_name: str, save: bool,
                         target_dir: str, **info):
        """Stores all plot information in the plot_info list and, if `save` is
        set, also saves it using the _save_plot_cfg method.
        """
        # Prepare the entry
        entry = dict(name=name, plot_cfg=plot_cfg, target_dir=target_dir,
                     creator_name=creator_name, **info,
                     plot_cfg_path=None)

        if save:
            # Save the plot configuration
            save_path = self._save_plot_cfg(plot_cfg, name=name,
                                            target_dir=target_dir,
                                            creator_name=creator_name)

            # Store the path the configuration was saved at
            entry['plot_cfg_path'] = save_path

        # Append to the plot_info list
        self._plot_info.append(entry)

    def _save_plot_cfg(self, cfg: dict,
                       *, name: str, creator_name: str, target_dir: str,
                       exists_action: str=None, is_sweep: bool=False) -> str:
        """Saves the given configuration under the top-level entry ``name`` to
        a yaml file.

        Args:
            cfg (dict): The plot configuration to save
            name (str): The name of the plot
            creator_name (str): The name of the creator
            target_dir (str): The directory path to store the file in
            exists_action (str, optional): What to do if a plot configuration
                already exists. Can be: ``overwrite``, ``overwrite_nowarn``,
                ``skip``, ``append``, ``raise``. If None, uses the value of the
                ``cfg_exists_action`` argument given during initialization.
            is_sweep (bool, optional): Set if the configuration refers to a
                plot in sweep mode, for which a different format string is used

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

        if not isinstance(cfg, ParamSpace):
            d[name]['creator'] = creator_name

        else:
            # FIXME hacky, should not use the internal API!
            d[name]._dict['creator'] = creator_name

        # Generate the filename and save path
        fn_fstr = self.out_fstrs['plot_cfg_sweep' if is_sweep else 'plot_cfg']
        fname = fn_fstr.format(name=name,
                               basename=os.path.basename(name))
        save_path = os.path.join(target_dir, fname)

        # Try to write
        try:
            write_yml(d, path=save_path, mode='x')

        except FileExistsError:
            log.debug("Config file already exists at %s!", save_path)

            if exists_action == 'raise':
                raise

            elif exists_action == 'skip':
                log.debug("Skipping ...")

            elif exists_action == 'append':
                log.debug("Appending ...")
                write_yml(d, path=save_path, mode='a')

            elif exists_action == 'overwrite':
                log.warning("Overwriting existing plot configuration ...")
                write_yml(d, path=save_path, mode='w')

            elif exists_action == 'overwrite_nowarn':
                log.debug("Overwriting ...")
                write_yml(d, path=save_path, mode='w')

            else:
                raise ValueError("Invalid value '{}' for argument "
                                 "`exists_action`!".format(exists_action))

        else:
            log.debug("Saved plot configuration for '%s' to: %s",
                      name, save_path)

        return save_path

    # .........................................................................
    # Plotting

    def plot_from_cfg(self, *,
                      plots_cfg: Union[dict, str]=None,
                      plot_only: List[str]=None,
                      out_dir: str=None,
                      **update_plots_cfg) -> None:
        """Create multiple plots from a configuration, either a given one or
        the one passed during initialization.

        This is mostly a wrapper around the plot function, allowing additional
        ways of how to configure and create plots.

        Args:
            plots_cfg (dict, optional): The plots configuration to use. If not
                given, the one specified during initialization is used. If a
                string is given, will assume it is a path and load the file.
            plot_only (List[str], optional): If given, create only those plots
                from the resulting configuration that match these names. This
                will lead to the `enabled` key being ignored, regardless of its
                value. The strings given here may also include Unix shell-like
                wildcards like ``*`` and ``? ``, which are matched using the
                Python ``fnmatch`` module.
            out_dir (str, optional): A different output directory; will use the
                one passed at initialization if the given argument evaluates to
                False.
            **update_plots_cfg: If given, it is used to update the plots_cfg
                recursively. Note that on the top level the _names_ of the
                plots are placed; this cannot be used to make all plots have a
                common property.

        Raises:
            PlotConfigError: Empty or invalid plot configuration
        """
        # Determine which plot configuration to use
        if not plots_cfg:
            if not self._plots_cfg and not update_plots_cfg:
                e_msg = ("Got empty `plots_cfg` and `plots_cfg` given at "
                         "initialization was also empty. Nothing to plot.")

                if self.raise_exc:
                    raise PlotConfigError(e_msg)

                log.error(e_msg)
                return

            log.debug("No new plots configuration given; will use plots "
                      "configuration given at initialization.")
            plots_cfg = self._plots_cfg

        elif isinstance(plots_cfg, str):
            # Interpret as path to yaml file
            log.debug("Loading plots_cfg from file %s ...", plots_cfg)
            plots_cfg = load_yml(plots_cfg)

        # Make sure to work on a copy, be it on the defaults or on the passed
        plots_cfg = copy.deepcopy(plots_cfg)

        if update_plots_cfg:
            # Recursively update with the given keywords
            plots_cfg = recursive_update(plots_cfg, update_plots_cfg)
            log.debug("Updated the plots configuration.")

        # Check the plot configuration for invalid types
        for plot_name, cfg in plots_cfg.items():
            if not isinstance(cfg, (dict, ParamSpace)):
                raise PlotConfigError("Got invalid plots specifications for "
                                      "entry '{}'! Expected dict, got {} with "
                                      "value '{}'. Check the correctness of "
                                      "the given plots configuration!"
                                      "".format(plot_name, type(cfg), cfg))

        # Filter the plot selection
        if plot_only is not None:
            # Resolve glob-like patterns
            to_plot = []
            for name in plot_only:
                if any([s in name for s in ('*', '?', '[', ']')]):
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
                _available = ", ".join(plots_cfg.keys())
                raise ValueError(
                    f"Could not find a configuration for a plot named {err} "
                    "while resolving the plot_only argument! Check that it "
                    "was specified correctly:\n"
                    f"  plot_only:  {_plot_only}\n"
                    f"  available:  {_available}"
                ) from err

            # Remove all `enabled` keys from the remaining entries, thus also
            # enabling those plots that were set `enabled: False`
            for cfg in plots_cfg.values():
                cfg.pop('enabled', None)

        else:
            # Resolve all `enabled` entries, creating a new plots_cfg dict
            plots_cfg = {k: v for k, v in plots_cfg.items()
                         if v.pop('enabled', True)}

        # Throw out entries that start with an underscore or dot
        plots_cfg = {k: v for k, v in plots_cfg.items()
                     if not (k.startswith("_") or k.startswith("."))}

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
        log.hilight("Performing plots from %d plot configuration entr%s:",
                    len(plots_cfg), "ies" if len(plots_cfg) != 1 else "y")

        for plot_name, cfg in plots_cfg.items():
            if isinstance(cfg, ParamSpace):
                log.note("  - %-40s  (%d plot%s)", plot_name,
                         cfg.volume, "s" if cfg.volume != 1 else "")
            else:
                log.note("  - %-40s  (1 plot)", plot_name)

        # Loop over the configured plots
        for plot_name, cfg in plots_cfg.items():
            # Use the public methods to perform the plotting call, depending
            # on the type of the config
            if isinstance(cfg, ParamSpace):
                # Is a parameter space. Use the corresponding call signature
                self.plot(plot_name, default_out_dir=out_dir, from_pspace=cfg)

            else:
                # Just a dict. Use the regular call
                self.plot(plot_name, default_out_dir=out_dir, **cfg)

        # All done
        log.success("Successfully performed plots for %d plot "
                    "configuration%s.\n",
                    len(plots_cfg), "s" if len(plots_cfg) != 1 else "")

    def plot(self, name: str, *,
             based_on: Union[str, Tuple[str]]=None,
             from_pspace: Union[dict, ParamSpace]=None,
             **plot_cfg) -> BasePlotCreator:
        """Create plot(s) from a single configuration entry.

        A call to this function resolves the ``based_on`` feature and passes
        the derived plot configuration to self._plot(), which actually carries
        out the plots.

        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.

        Args:
            name (str): The name of this plot. This will be used for generating
                an output file path later on. Some characters are not allowed,
                e.g. ``*`` and ``?``, but a ``/`` can be used to store the plot
                output in a subdirectory.
            based_on (Union[str, Tuple[str]], optional): A key or a sequence
                of keys of entries in the base config that should be used as
                the basis of this plot.
                The given plot configuration is then used to recursively
                update (a copy of) those base configuration entries.
            from_pspace (Union[dict, ParamSpace], optional): If given, execute
                a parameter sweep over these parameters, re-using the same
                creator instance. If this is a dict, a ParamSpace is created
                from it.
            **plot_cfg: The plot configuration, including some parameters that
                the plot manager will evaluate (and consequently: does not
                pass on to the plot creator)

        Returns:
            BasePlotCreator: The PlotCreator used for these plots
        """
        # Make sure the name does not contain bad characters
        if any([s in name for s in BAD_NAME_CHARS]):
            _bad_name_chars = ", ".join(repr(s) for s in BAD_NAME_CHARS)
            raise ValueError(
                f"The plot name '{name}' contains unsupported characters! "
                "Remove any of the following offending substrings from the "
                f"plot function name: {_bad_name_chars}"
            )

        # Derive the plot_cfg using based_on. Do this first only for the case
        # of the non-ParamSpace plot configuration, because it's easier.
        plot_cfg = self._resolve_based_on(cfg=plot_cfg, based_on=based_on)

        if from_pspace is None:
            # Can just invoke the helper and be done with it
            return self._plot(name, from_pspace=from_pspace, **plot_cfg)

        # Else: It's more complicated now, as the config is in from_pspace, and
        # (partly) in plot_cfg. Urgh.
        # Distinguish between dict and actual ParamSpace objects
        if isinstance(from_pspace, ParamSpace):
            # Already is a Paramspace. If so, will need to extract the
            # underlying dict to be able to do a recursive update
            pspace_plot_cfg = copy.deepcopy(from_pspace._dict)
            # FIXME Should not have to use private API!

            # Resolve `based_on`
            based_on = pspace_plot_cfg.pop("based_on", None)
            pspace_plot_cfg = self._resolve_based_on(cfg=pspace_plot_cfg,
                                                     based_on=based_on,
                                                     work_on_deepcopy=False)

            # Extract info, then re-create the ParamSpace with the updated cfg
            creator = pspace_plot_cfg.pop("creator", None)
            from_pspace = ParamSpace(pspace_plot_cfg)

        else:
            # Assume it's something dict-like
            # ... but need a copy in order to safely pop elements.
            from_pspace = copy.deepcopy(from_pspace)
            based_on = from_pspace.pop("based_on", None)

            # Do the update; no deepcopy needed in there because done here
            from_pspace = self._resolve_based_on(cfg=from_pspace,
                                                 based_on=based_on,
                                                 work_on_deepcopy=False)

            # Now also extract the creator; might have come in only by the
            # based_on resolution
            creator = from_pspace.pop("creator", None)

        # NOTE creator needs to be singled out above because _plot is not able
        #      to extract it from whatever `from_pspace` is.

        # Now have all the information extracted / removed from from_pspace to
        # be ready to call _plot. Finally.
        return self._plot(name, creator=creator, from_pspace=from_pspace,
                          **plot_cfg) # **plot_cfg is anything remaining ...

    def _plot(self, name: str, *,
              creator: str=None,
              out_dir: str=None,
              default_out_dir: str=None,
              from_pspace: ParamSpace=None,
              file_ext: str=None,
              save_plot_cfg: bool=None,
              auto_detect_creator: bool=None,
              creator_init_kwargs: dict=None,
              **plot_cfg
              ) -> BasePlotCreator:
        """Create plot(s) from a single configuration entry.

        A call to this function creates a single PlotCreator, which is also
        returned after all plots are finished.

        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.

        Args:
            name (str): The name of this plot
            creator (str, optional): The name of the creator to use. Has to be
                part of the CREATORS class variable. If not given, the argument
                `default_creator` given at initialization will be used.
            out_dir (str, optional): If given, will use this directory as out
                directory. If not, will use the default value given by
                ``default_out_dir`` or that given at initialization.
            default_out_dir (str, optional): An output directory that was
                determined in the calling context and which should be used as
                default if no ``out_dir`` was given explicitly.
            file_ext (str, optional): The file extension to use, including the
                leading dot!
            from_pspace (ParamSpace, optional): If given, execute a parameter
                sweep over these parameters, re-using the same creator instance
            save_plot_cfg (bool, optional): Whether to save the plot config.
                If not given, uses the default value from initialization.
            auto_detect_creator (bool, optional): Whether to attempt auto-
                detection of the ``creator`` argument. If given, this argument
                overwrites the value given at PlotManager initialization.
            creator_init_kwargs (dict, optional): Passed to the plot creator
                during initialization. Note that the arguments given at
                initialization of the PlotManager are updated by this.
            **plot_cfg: The plot configuration to pass on to the plot creator.

        Returns:
            BasePlotCreator: The PlotCreator used for these plots

        Raises:
            PlotConfigError: If no out directory was specified here or at
                initialization.
        """

        log.debug("Preparing plot '%s' ...", name)

        # Decide on the output directory to use ...
        if not out_dir:
            if default_out_dir:
                out_dir = default_out_dir

            elif self._out_dir:
                out_dir = self._out_dir

            else:
                raise PlotConfigError("No `out_dir` specified here and at "
                                      "initialization; cannot perform plot.")

        # Whether to save the plot config
        if save_plot_cfg is None:
            save_plot_cfg = self.save_plot_cfg

        # Get the plot creator, either by name or using auto-detect feature
        plot_creator = self._get_plot_creator(creator, name=name,
                                              init_kwargs=creator_init_kwargs,
                                              from_pspace=from_pspace,
                                              plot_cfg=plot_cfg,
                                              auto_detect=auto_detect_creator)

        # Let the creator process arguments
        plot_cfg, from_pspace = plot_creator.prepare_cfg(plot_cfg=plot_cfg,
                                                         pspace=from_pspace)

        # Distinguish single calls and parameter sweeps
        if not from_pspace:
            log.progress("Performing '%s' plot ...", name)

            # Generate the output path
            out_dir = self._parse_out_dir(out_dir, name=name)
            out_path = self._parse_out_path(plot_creator, name=name,
                                            out_dir=out_dir, file_ext=file_ext)

            # Call the plot creator to perform the plot, using the private
            # method to perform exception handling
            self._invoke_creator(plot_creator, out_path=out_path, **plot_cfg)

            # Store plot information
            self._store_plot_info(name=name, creator_name=creator,
                                  out_path=out_path, plot_cfg=plot_cfg,
                                  save=save_plot_cfg,
                                  target_dir=os.path.dirname(out_path))

            log.progress("Finished '%s' plot.\n", name)

        else:
            # Is a parameter sweep over the plot configuration.
            # NOTE The parameter space is allowed to have volume 0!

            # If it is not already a ParamSpace, create one; useful if not
            # calling from plot_from_cfg, but directly ...
            if not isinstance(from_pspace, ParamSpace):
                from_pspace = ParamSpace(from_pspace)

            # Extract some info
            psp_vol = from_pspace.volume
            psp_dims = from_pspace.dims

            # ... and provide it to the logger
            if psp_vol > 0:
                amap_coords = from_pspace.active_state_map.coords
                max_dname_len = max(len(n) for n in amap_coords.keys())

                log.progress("Performing %d '%s' plots ...", psp_vol, name)
                log.note("... iterating over parameter space:\n%s",
                         "\n".join(["  * {0:<{d:}} : {1:}"
                                    "".format(dim_name,
                                             ", ".join([str(c) for c in
                                                        coords.values]),
                                             d=max_dname_len)
                                    for dim_name, coords
                                    in amap_coords.items()]))

            else:
                log.progress("Performing '%s' plot ...", name)
                log.note("... from default point in zero-volume parameter "
                         "space.")

            # Parse the output directory, such that all plots are together in
            # one directory even if the timestamp varies
            out_dir = self._parse_out_dir(out_dir, name=name)

            # Create the iterator
            it = from_pspace.iterator(with_info=('state_no', 'state_vector',
                                                 'coords'))

            # ...and loop over all points:
            for n, (cfg, state_no, state_vector, coords) in enumerate(it):
                log.progress("Performing plot {n:{d:}d} / {v:} ..."
                             "".format(n=n+1, d=len(str(psp_vol)), v=psp_vol))
                log.note("Current coordinates:  %s",
                         ",  ".join("{}: {}".format(*kv)
                                    for kv in coords.items()))

                # Handle the file extension parameter; it might come from the
                # given configuration and then needs to be popped such that it
                # is not propagated to the plot creator.
                _file_ext = cfg.pop('file_ext', file_ext)

                # Generate the output path
                out_path = self._parse_out_path(plot_creator,
                                                name=name,
                                                out_dir=out_dir,
                                                file_ext=_file_ext,
                                                state_no=state_no,
                                                state_no_max=psp_vol-1,
                                                state_vector=state_vector,
                                                dims=psp_dims)

                # Call the plot creator to perform the plot, using the private
                # method to perform exception handling
                self._invoke_creator(plot_creator, out_path=out_path,
                                     **cfg, **plot_cfg)
                # NOTE The **plot_cfg is passed here in order to not loose any
                # arguments that might have been passed to it. While `cfg`
                # _should_ hold all the arguments from the parameter space
                # iteration, there might be more arguments in `plot_cfg`;
                # rather than disallowing this, we pass them on and forward
                # responsibility downstream ...

                # Store plot information
                self._store_plot_info(name=name, creator_name=creator,
                                      out_path=out_path, plot_cfg=plot_cfg,
                                      state_no=state_no,
                                      state_vector=state_vector,
                                      save=False, # TODO check if reasonable
                                      target_dir=os.path.dirname(out_path))

                # Done with these coordinates

            # Save the plot configuration alongside, if configured to do so
            if save_plot_cfg:
                self._save_plot_cfg(from_pspace, name=name,
                                    creator_name=creator,
                                    target_dir=out_dir, is_sweep=True)

            log.progress("Finished all '%s' plots.\n", name)

        # Done now. Return the plot creator object
        return plot_creator
