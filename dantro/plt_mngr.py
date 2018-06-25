"""This module implements the PlotManager class, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the PlotCreator.
"""

import logging
from typing import Union, List, Dict

from paramspace import ParamSpace

from dantro.data_mngr import DataManager
import dantro.plt_creator as pcr

# Local constants
log = logging.getLogger(__name__)

# The default mapping for creator names to classes
DANTRO_CREATORS = dict(external=pcr.ExternalPlotCreator,
                       declarative=pcr.DeclarativePlotCreator,
                       vega=pcr.VegaPlotCreator,
                       )


# -----------------------------------------------------------------------------

class PlotManager:
    """The PlotManager takes care of configuring plots and calling the
    configured PlotCreator classes that then carry out the plots.

    Attributes:
        CREATORS (dict): The mapping of creator names to classes. When it is
            desired to subclass PlotManager and extend the creator mapping, use
            `dict(**DANTRO_CREATORS)` to inherit the default creator mapping.
    """

    CREATORS = DANTRO_CREATORS

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
                # TODO implement this functionality
            common_creator_kwargs (Dict[str, dict], optional): If given, these
                kwargs are passed to the initialisation calls of the respective
                creator classes.
            default_creator (str, optional): If given, a plot without explicit
                `creator` declaration will use this creator as default.
        """
        pass

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
        """
        pass

    def plot(self, name: str, *, from_pspace: ParamSpace=None, **plot_cfg) -> pcr.BasePlotCreator:
        """Create plot(s) from a single configuration entry.

        A call to this function creates a single PlotCreator, which is also
        returned after all plots are finished.
    
        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.
        """
        pass
