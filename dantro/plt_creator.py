"""This module implements the PlotCreator classes, which create plots on the
level of a single file.

The interface is defined as an abstract base class and partly implemented by
the BasePlotCreator (which still remains abstract).
In this module, the following non-abstract plot creators are implemented:
  - ExternalPlotCreator: imports and calls an external plot script
  - DeclarativePlotCreator: creates plots using a declarative syntax
  - VegaPlotCreator: interfaces with Altair to provide a Vega-Lite interface
"""

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

    NOTE that the `_plot` method remains abstract and needs to be subclassed!
    """

    def __init__(self, name: str, *, dm: DataManager, **plot_cfg):
        """Create a PlotCreator instance for a plot with the given `name`.
        
        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            **plot_cfg: Description
        """
        # Store arguments as private attributes
        self._dm = dm
        self._plot_cfg = plot_cfg

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

    # .........................................................................

    def __call__(self, *, out_path: str=None, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling _plot.
        """
        # TODO add logging messages
        # Get (a deep copy of) the initial plot config
        cfg = self.plot_cfg

        # Check if a recursive update needs to take place
        if update_plot_cfg:
            cfg = tools.recursive_update(cfg, update_plot_cfg)

        # Check if a different output path was given, and if yes, use that one
        if out_path:
            out_path = self._resolve_out_path(out_path)
        else:
            out_path = self.out_path

        # Now call the plottig function with these arguments
        return self._plot(out_path=out_path, **cfg)


# -----------------------------------------------------------------------------

class ExternalPlotCreator(BasePlotCreator):
    """This PlotCreator uses external scripts to create plots."""
    pass



class DeclarativePlotCreator(BasePlotCreator):
    """This PlotCreator can create plots from a dantro-specific declarative
    plot configuration. The language is inspired by Vega-Lite but is adapted to
    working with the different data structures stored in a DataManager.
    """
    pass


class VegaPlotCreator(BasePlotCreator):
    """This PlotCreator interfaces with Altair to provide a Vega-Lite interface
    for plot creation.
    """
    pass
