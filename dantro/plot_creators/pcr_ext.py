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

    def _plot(self, *args, **kwargs):
        pass
