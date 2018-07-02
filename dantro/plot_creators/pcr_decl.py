"""This module implements the DeclarativePlotCreator class"""

import logging

from .pcr_base import BasePlotCreator


# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class DeclarativePlotCreator(BasePlotCreator):
    """This PlotCreator can create plots from a dantro-specific declarative
    plot configuration. The language is inspired by Vega-Lite but is adapted to
    working with the different data structures stored in a DataManager.
    """
    pass

