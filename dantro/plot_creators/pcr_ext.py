"""This module implements the ExternalPlotCreator class"""

import logging

from .pcr_base import BasePlotCreator


# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ExternalPlotCreator(BasePlotCreator):
    """This PlotCreator uses external scripts to create plots."""

    EXTENSIONS = 'all'  # no checks performed
    DEFAULT_EXT = None
    DEFAULT_EXT_REQUIRED = False

    # TODO base_module_file_dir in init

    def _plot(self, *, out_path: str, plot_func: str, module_file: str=None,
              **func_kwargs):
        """Performs the plot operation by calling a specified plot function.

        The plot function is specified by its name, which is interpreted as a
        full module string, or by directly passing a callable.

        Alternatively, the base module can be loaded from a file path.
        """


    def _get_module(self, modstr: str):
        """Returns the module corresponding to the given `modstr`"""

    def _get_module_from_file(self, path: str):
        """Returns the module corresponding to the file at the given `path`"""
