"""This module implements the base PlotCreator class.

Classes derived from this class create plots for single files.

The interface is defined as an abstract base class and partly implemented by
the BasePlotCreator (which still remains abstract).
"""

import os
import copy
import logging

import dantro.abc
import dantro.tools as tools
from dantro.data_mngr import DataManager

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class BasePlotCreator(dantro.abc.AbstractPlotCreator):
    """The base class for PlotCreators
    
    Note that the `_plot` method remains abstract and needs to be subclassed!
    
    Attributes:
        DEFAULT_EXT (str): The class variable to use for default extension
        default_ext (str): The property-managed actual value for the default
            extension to use
        EXTENSIONS (tuple): The supported extensions
        POSTPONE_PATH_PREPARATION (bool): Whether to create paths in the base
            class's __call__ method or not. If the derived class wants to
            take care of this on their own, this should be set to True.
    """
    EXTENSIONS = ()
    DEFAULT_EXT = None
    POSTPONE_PATH_PREPARATION = False

    def __init__(self, name: str, *, dm: DataManager, default_ext: str=None, **plot_cfg):
        """Create a PlotCreator instance for a plot with the given `name`.
        
        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            default_ext (str, optional): The default extension to use; needs
                to be in EXTENSIONS.
            **plot_cfg: The default plot configuration
        """
        # Store arguments as private attributes
        self._dm = dm
        self._plot_cfg = plot_cfg
        self._default_ext = None

        # And others via their property setters
        self.default_ext = default_ext if default_ext else DEFAULT_EXT

        log.debug("%s initialised.", self.classname)

    # .........................................................................
    # Properties

    @property
    def dm(self) -> DataManager:
        """Return the DataManager"""
        return self._dm

    @property
    def plot_cfg(self) -> dict:
        """Returns a deepcopy of the plot configuration, assuring that plot
        configurations are completely independent of each other.
        """
        return copy.deepcopy(self._plot_cfg)

    @property
    def default_ext(self) -> str:
        """Returns the default extension to use for the plots"""
        return self._default_ext

    @default_ext.setter
    def default_ext(self, val: str) -> None:
        """Sets the default extension. Needs to be in EXTENSIONS"""
        if val.lower() not in self.EXTENSIONS:
            raise ValueError("Extension '{}' not supported. Supported "
                             "extensions are: {}"
                             "".format(val, self.EXTENSIONS))

    # .........................................................................
    # Main API functions

    def __call__(self, *, out_path: str, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling _plot.
        """
        # TODO add logging messages
        # Get (a deep copy of) the initial plot config
        cfg = self.plot_cfg

        # Check if a recursive update needs to take place
        if update_plot_cfg:
            cfg = tools.recursive_update(cfg, update_plot_cfg)

        # Prepare the output path
        if not self.POSTPONE_PATH_PREPARATION:
            out_path = self._prepare_path(out_path)

        # Now call the plottig function with these arguments
        return self._plot(out_path=out_path, **cfg)

    # .........................................................................
    # Helpers

    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot by checking
        the supported extensions and """
        return self.default_ext

    def _prepare_path(self, out_path: str) -> str:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.
        
        This is called from __call__ and is meant to postpone directory
        creation as far as possible.
        
        Args:
            out_path (str): The absolute output path to start with
        
        Returns:
            str: The (possibly adjusted) output path
        """
        # Create necessary directories
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Nothing more to do here (at least not in the base class)
        return out_path
