"""This sub-package implements non-abstract dantro plot creator classes,
based on :py:class:`~dantro.plot.creators.base.BasePlotCreator`"""

from .base import BasePlotCreator
from .ext import ExternalPlotCreator
from .psp import MultiversePlotCreator, UniversePlotCreator

ALL_PLOT_CREATORS = dict(
    external=ExternalPlotCreator,
    universe=UniversePlotCreator,
    multiverse=MultiversePlotCreator,
)
"""A mapping of plot creator names to the corresponding types"""
