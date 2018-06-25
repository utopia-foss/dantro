"""This module implements the PlotManager class, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the PlotCreator.
"""

import copy
import logging
from typing import Union, List, Dict

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
    """

    CREATORS = pcr.ALL

    def __init__(self, *, dm: DataManager, plots_cfg: Union[dict, str]=None, out_dir: Union[str, None]="{name:}", common_creator_kwargs: Dict[str, dict]=None, default_creator: str=None):
        """Initialize the PlotManager
        
        Args:
            dm (DataManager): The DataManager-derived object to read the plot
                data from.
            plots_cfg (Union[dict, str], optional): The default plots config.
            out_dir (Union[str, None], optional): If given, will use this
                output directory, creating it if it does not yet exist.
                For a relative path, this will be relative to the DataManager's
                output directory. Absolute paths remain absolute.
                The path can be a format-string; it is evaluated upon call to
                the plot command. Available keys: date, plot_name, ...
            common_creator_kwargs (Dict[str, dict], optional): If given, these
                kwargs are passed to the initialisation calls of the respective
                creator classes.
            default_creator (str, optional): If given, a plot without explicit
                `creator` declaration will use this creator as default.
        """
        # Store arguments
        self._dm = dm
        self._plots_cfg = plots_cfg
        self._out_dir = out_dir
        self._cckwargs = common_creator_kwargs if common_creator_kwargs else {}

        if default_creator and default_creator not in self.CREATORS:
            raise ValueError("No such creator '{}' available, only: {}"
                             "".format(default_creator,
                                       [k for k in self.CREATORS.keys()]))
        self._default_creator = default_creator

        log.debug("%s initialised.", self.__class__.__name__)

    # .........................................................................
    # Properties


    # .........................................................................
    # Plotting

    def plot_from_cfg(self, *, plots_cfg: dict=None, update_plots_cfg: dict=None, plot_only: List[str]=None) -> None:
        """Create multiple plots from a configuration, either a given one or
        the one passed during initialisation.
        
        This is mostly a wrapper around the plot function, allowing additional
        ways of how to configure and create plots.
        
        Args:
            plots_cfg (dict, optional): The plots configuration to use. If not
                given, the one specified during initialisation is used.
            update_plots_cfg (dict, optional): If given, it is used to update
                the plots_cfg recursively
            plot_only (List[str], optional): If given, create only those plots
                from the resulting configuration that match these names.
        
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
            load_cfg = tools.recursive_update(load_cfg, update_plots_cfg)
            log.debug("Updated the plots configuration.")

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
            # Resolve all `enabled` entries
            plots_cfg = {k:v for k, v in plots_cfg.items()
                         if v.pop('enabled', True)}

        log.info("Performing plots from %d entries ...", len(plots_cfg))

        # Loop over the configured plots
        for plot_name, cfg in plots_cfg.items():
            # Use the public methods to perform the plotting call, depending
            # on the type of the config
            if isinstance(cfg, dict):
                # Just a dict. Use the regular call
                self.plot(plot_name, **cfg)

            elif isinstance(cfg, ParamSpace):
                # Is a parameter space. Use the alternative signature
                self.plot(plot_name, from_pspace=cfg)

            else:
                raise TypeError("Got invalid plots specifications for entry "
                                "'{}'! Expected dict, got {} with value '{}'. "
                                "Check the correctness of the given plots "
                                "configuration!".format(plot_name, type(cfg),
                                                        cfg))
        
        # All done
        log.info("Successfully performed plots for %d configuration(s).",
                 len(plots_cfg))



    def plot(self, name: str, *, creator: str=None, from_pspace: ParamSpace=None, **plot_cfg) -> pcr.BasePlotCreator:
        """Create plot(s) from a single configuration entry.
        
        A call to this function creates a single PlotCreator, which is also
        returned after all plots are finished.
        
        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.
        
        Args:
            name (str): The name of this plot
            creator (str, optional): The name of the creator to use. Has to be
                part of the CREATORS class variable.
            from_pspace (ParamSpace, optional): If given, execute a parameter
                sweep over these parameters, re-using the same creator instance
            **plot_cfg: The plot configuration to pass on to the plot creator.
        """
        log.info("Performing plot '%s' ...", name)

        # If no creator is given, use the default one
        if not creator:
            if not self._default_creator:
                raise ValueError("No `creator` argument given and no "
                                 "`default_creator` specified during "
                                 "initialisation; cannot perform plot!")

            creator = self._default_creator

        # Get the creator class and directly instantiate it
        plot_creator = self.CREATORS[creator](name=name, dm=self._dm)

        # Distinguish single calls and parameter sweeps
        if not from_pspace:
            # Generate the output path

            # Call the plot creator
            plot_creator(out_path=out_path, **plot_cfg)

        else:
            # Generate the base output path

            # Create the iterator
            it = from_pspace.all_points(with_info=('state_no', 'state_vector'))
            # ...and loop over all points:
            for cfg, state_no in it:
                # Generate the output path
                # TODO

                # Call the plot creator
                plot_creator(out_path=out_path, **cfg)
