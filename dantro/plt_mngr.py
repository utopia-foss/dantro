"""This module implements the PlotManager class, which handles the
configuration of multiple plots and prepares the data and configuration to pass
to the PlotCreator.
"""

import logging
from typing import Union, List

from dantro.data_mngr import DataManager

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class PlotManager:
    
    def __init__(self, *, dm: DataManager, plots_cfg: Union[dict, str]=None, out_dir: Union[str, None]="{name:}", custom_modules_base_dir: str=None):
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

    def plot(self, name: str, ) -> None:
        """Create plot(s) from a single configuration entry.

        Note that more than one plot can result from a single configuration
        entry, e.g. when plots were configured that have more dimensions than
        representable in a single file.
        """
        pass
