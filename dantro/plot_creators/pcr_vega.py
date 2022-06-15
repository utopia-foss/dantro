"""This module implements the VegaPlotCreator class"""

import logging

from .pcr_base import BasePlotCreator

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class VegaPlotCreator(BasePlotCreator):
    """This PlotCreator interfaces with Altair to provide a Vega-Lite interface
    for plot creation.
    """

    pass
