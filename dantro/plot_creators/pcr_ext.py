"""This module implements the ExternalPlotCreator class"""

import logging

from .pcr_base import BasePlotCreator


# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ExternalPlotCreator(BasePlotCreator):
    """This PlotCreator uses external scripts to create plots."""

    def _plot(self):
        pass
