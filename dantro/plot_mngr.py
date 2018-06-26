"""This module implements the PlotManager class, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the PlotCreator.
"""

import os
import time
import copy
import logging
from typing import Union, List, Dict, Tuple

from paramspace import ParamSpace

import dantro.tools as tools
from dantro.data_mngr import DataManager
import dantro.plot_creators as pcr

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class PlotManager:
    """The PlotManager takes care of configuring plots and calling the
    configured PlotCreator classes that then carry out the plots.
    
    Attributes:
        CREATORS (dict): The mapping of creator names to classes. When it is
            desired to subclass PlotManager and extend the creator mapping, use
            `dict(**pcr.ALL)` to inherit the default creator mapping.
        DEFAULT_FSTRS (dict): The default values for the output format strings
    """

    CREATORS = pcr.ALL
    DEFAULT_FSTRS = dict(date="%y%m%d-%H%M%S",
                         state_no="{no:0{digits:d}d}",
                         state="{key:}_{val:}", state_join_char="-",
                         state_vector_join_char="-",
                         path="{name:}.{ext:}",
                         sweep="{name:}/{state_no:}-{state:}.{ext:}")

    def __init__(self, *, dm: DataManager, plots_cfg: Union[dict, str]=None, out_dir: Union[str, None]="{date:}/", out_fstrs: dict=None, common_creator_kwargs: Dict[str, dict]=None, default_creator: str=None):
        """Initialize the PlotManager
        
        Args:
            dm (DataManager): The DataManager-derived object to read the plot
                data from.
            plots_cfg (Union[dict, str], optional): The default plots config.
            out_dir (Union[str, None], optional): If given, will use this
                output directory as basis for the output path for each plot.
                The path can be a format-string; it is evaluated upon call to
                the plot command. Available keys: `date`, `name`, ...
                For a relative path, this will be relative to the DataManager's
                output directory. Absolute paths remain absolute.
                If this argument evaluates to False, the DataManager's output
                directory will be the output directory.
            out_fstrs (dict, optional): Format strings that define how the
                output path is generated.
                Keys to be set: `date` (%-style), `path`, `sweep`, `state`
                Available keys for `path`:
            common_creator_kwargs (Dict[str, dict], optional): If given, these
                kwargs are passed to the initialisation calls of the respective
                creator classes.
            default_creator (str, optional): If given, a plot without explicit
                `creator` declaration will use this creator as default.
        
        Raises:
            ValueError: Invalid default creator
        """
        # TODO consider making it possible to pass classes for plot creators

        # Store arguments
        self._dm = dm
        self._plots_cfg = plots_cfg

        self._out_dir = out_dir
        if out_fstrs:
            # Update defaults
            d = copy.deepcopy(self.DEFAULT_FSTRS)
            d.update(out_fstrs)
            self._out_fstrs = d
        else:
            # Use defaults
            self._out_fstrs = self.DEFAULT_FSTRS

        self._cckwargs = common_creator_kwargs if common_creator_kwargs else {}

        if default_creator and default_creator not in self.CREATORS:
            raise ValueError("No such creator '{}' available, only: {}"
                             "".format(default_creator,
                                       [k for k in self.CREATORS.keys()]))
        self._default_creator = default_creator

        log.debug("%s initialised.", self.__class__.__name__)

    # .........................................................................
    # Properties

    @property
    def out_fstrs(self) -> dict:
        """Returns the dict of output format strings"""
        return self._out_fstrs

    # .........................................................................
    # Helpers

    def _parse_out_dir(self, fstr: str, *, name: str) -> str:
        """Evaluates the format string to create an output directory.

        Note that the directories are _not_ created; this is outsourced to the
        plot creator such that it happens as late as possible.
        
        Args:
            fstr (str): The format string to evaluate and create a directory at
        
        Returns:
            str: The path of the created directory
        """
        # Get date format string and evaluate
        date_fstr = self._out_fstrs.get('date', "%y%m%d-%H%M%S")
        date = time.strftime(date_fstr)

        out_dir = fstr.format(date=date, name=name)

        # Make sure it is absolute
        if not os.path.isabs(out_dir):
            # Regard it as relative to the data manager's output directory
            out_dir = os.path.join(self._dm.dirs['out'], out_dir)

        # Return the full path
        return out_dir

    def _parse_out_path(self, creator: pcr.BasePlotCreator, *, name: str, out_dir: str, state_no: int=None, state_no_max: int=None, state_vector: Tuple[int]=None, dims: dict=None) -> str:
        """Given a creator and (optionally) parameter sweep information, a full
        and absolute output path is generated, including the file extension.
        
        Note that the directories are _not_ created; this is outsourced to the
        plot creator such that it happens as late as possible.
        
        Args:
            creator (pcr.BasePlotCreator): The creator instance, used to
                extract information on the file extension.
            name (str): The name of the plot
            out_dir (str): The absolute output directory, prepended to all
                generated paths
            state_no (int, optional): The state number, starting with 0
            state_no_max (int, optional): The maximum state number
            state_vector (Tuple[int], optional): The state vector with info
                on how far each state dimension has progressed in the sweep
            dims (dict, optional): The dict of parameter dimensions of the
                sweep that is carried out.
        
        Returns:
            str: Description
        """
        # Get the fstrs
        fstrs = self.out_fstrs
        
        # Evaluate the keys available for both cases
        keys = dict(date=time.strftime(fstrs['date']),
                    name=name, ext=creator.get_ext())

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

            # state values
            state_pairs = [(name, dim.current_value)
                           for name, dim in dims.items()]
            sjc = fstrs['state_join_char']
            keys['state'] = sjc.join([fstrs['state'].format(key=k, val=v)
                                      for k, v in state_pairs])

            # state vector
            svjc = fstrs['state_vector_join_char']
            keys['state_vector'] = svjc.join([str(s) for s in state_vector])

            # Evaluate it
            out_path = fstrs['sweep'].format(**keys)

        # Prepend the output directory and return
        out_path = os.path.join(out_dir, out_path)

        return out_path


    # .........................................................................
    # Plotting

    def plot_from_cfg(self, *, plots_cfg: dict=None, plot_only: List[str]=None, out_dir: str=None, **update_plots_cfg) -> None:
        """Create multiple plots from a configuration, either a given one or
        the one passed during initialisation.
        
        This is mostly a wrapper around the plot function, allowing additional
        ways of how to configure and create plots.
        
        Args:
            plots_cfg (dict, optional): The plots configuration to use. If not
                given, the one specified during initialisation is used.
            plot_only (List[str], optional): If given, create only those plots
                from the resulting configuration that match these names. This
                will lead to the `enabled` key being ignored, regardless of its
                value.
            out_dir (str, optional): A different output directory; will use the
                one passed at initialisation if the given argument evaluates to
                False.
            **update_plots_cfg: If given, it is used to update the plots_cfg
                recursively. Note that on the top level the _names_ of the
                plots are placed; this cannot be used to make all plots have a
                common property.
        
        Raises:
            TypeError: Invalid plot configuration type
        """
        # Determine which plot configuration to use
        if not plots_cfg:
            log.debug("No new plots configuration given; will use plots "
                      "configuration given at initialisation.")
            plots_cfg = self._plots_cfg

        # Make sure to work on a copy, be it on the defaults or on the passed
        plots_cfg = copy.deepcopy(plots_cfg)

        if update_plots_cfg:
            # Recursively update with the given keywords
            plots_cfg = tools.recursive_update(plots_cfg, update_plots_cfg)
            log.debug("Updated the plots configuration.")

        # Check the plot configuration for invalid types
        for plot_name, cfg in plots_cfg.items():
            if not isinstance(cfg, (dict, ParamSpace)):
                raise TypeError("Got invalid plots specifications for entry "
                                "'{}'! Expected dict, got {} with value '{}'. "
                                "Check the correctness of the given plots "
                                "configuration!".format(plot_name, type(cfg),
                                                        cfg))

        # Filter the plot selection
        if plot_only:
            # Only plot these entries
            plots_cfg = {k:plots_cfg[k] for k in plot_only}
            # NOTE that this deliberately raises an error for an invalid entry
            #      in the `plot_only` argument

            # Remove all `enabled` keys from the remaining entries
            for cfg in plots_cfg.values():
                cfg.pop('enabled', None)

        else:
            # Resolve all `enabled` entries, creating a new plots_cfg dict
            plots_cfg = {k:v for k, v in plots_cfg.items()
                         if v.pop('enabled', True)}

        # Determine and create the plot directory to use
        if not out_dir:
            out_dir = self._out_dir
        out_dir = self._parse_out_dir(out_dir, name="{name:}")
        # NOTE creating this here such that all plots from this config are side
        #      by side in one output directory. With the given name value, the
        #      evaluation of that part of the output directory is postponed
        #      to when the actual plot with that name is created.

        log.info("Performing plots from %d entries ...", len(plots_cfg))

        # Loop over the configured plots
        for plot_name, cfg in plots_cfg.items():
            # Use the public methods to perform the plotting call, depending
            # on the type of the config
            if isinstance(cfg, ParamSpace):
                # Is a parameter space. Use the corresponding call signature
                self.plot(plot_name, out_dir=out_dir, from_pspace=cfg)
            
            else:
                # Just a dict. Use the regular call
                self.plot(plot_name, out_dir=out_dir, **cfg)

        # All done
        log.info("Successfully performed plots for %d configuration(s).",
                 len(plots_cfg))


    def plot(self, name: str, *, creator: str=None, out_dir: str=None, from_pspace: ParamSpace=None, **plot_cfg) -> pcr.BasePlotCreator:
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
                `default_creator` given at initialisation will be used.
            out_dir (str, optional): If given, will use this directory as out
                directory. If not, will use the default value given at
                initialisation.
            from_pspace (ParamSpace, optional): If given, execute a parameter
                sweep over these parameters, re-using the same creator instance
            **plot_cfg: The plot configuration to pass on to the plot creator.
        
        Returns:
            pcr.BasePlotCreator: The PlotCreator used for these plots
        
        Raises:
            ValueError: If no creator was given here and at initialisation or
                when out_dir was neither given here nor during initialisation
        """
        # Check that the output directory is given
        if not out_dir:
            if not self._out_dir:
                raise ValueError("No `out_dir` specified here and at "
                                 "initialisation; cannot perform plot.")

            out_dir = self._out_dir

        # If no creator is given, use the default one
        if creator is None:
            if not self._default_creator:
                raise ValueError("No `creator` argument given and no "
                                 "`default_creator` specified during "
                                 "initialisation; cannot perform plot!")

            creator = self._default_creator

        # Instantiate the creator class, also passing initialisation kwargs
        init_kwargs = self._cckwargs.get(creator, {})
        creator = self.CREATORS[creator](name=name, dm=self._dm, **init_kwargs)

        log.debug("Received creator: %s", creator.classname)

        # Distinguish single calls and parameter sweeps
        if not from_pspace:
            log.info("Performing plot '%s' ...", name)

            # Generate the output path
            out_dir = self._parse_out_dir(out_dir, name=name)
            out_path = self._parse_out_path(creator, name=name,
                                             out_dir=out_dir)

            # Call the plot creator to perform the plot
            creator(out_path=out_path, **plot_cfg)

        else:
            # If it is not already a ParamSpace, create one
            # This is useful if not calling from plot_from_cfg
            if not isinstance(from_pspace, ParamSpace):
                from_pspace = ParamSpace(from_pspace)

            # Extract some info
            psp_vol = from_pspace.volume
            psp_dims = from_pspace.dims

            log.info("Performing plot '%s' from parameter space ...", name)
            log.info("  Volume:  %d", psp_vol)

            # Parse the output directory, such that all plots are together in
            # one directory even if the timestamp varies
            out_dir = self._parse_out_dir(out_dir, name=name)

            # Create the iterator
            it = from_pspace.all_points(with_info=('state_no', 'state_vector'))
            
            # ...and loop over all points:
            for cfg, state_no, state_vector in it:
                # Generate the output path
                out_path = self._parse_out_path(creator,
                                                name=name,
                                                out_dir=out_dir,
                                                state_no=state_no,
                                                state_no_max=psp_vol-1,
                                                state_vector=state_vector,
                                                dims=psp_dims)

                # Call the plot creator to perform the plot
                creator(out_path=out_path, **cfg)

        # Done now. Return the plot creator.
        return creator
