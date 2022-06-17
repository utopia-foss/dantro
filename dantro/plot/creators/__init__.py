"""This sub-package implements non-abstract dantro plot creator classes,
based on :py:class:`~dantro.plot.creators.base.BasePlotCreator`"""

from .base import BasePlotCreator
from .psp import MultiversePlotCreator, UniversePlotCreator
from .pyplot import PyPlotCreator

ALL_PLOT_CREATORS = dict(
    base=BasePlotCreator,
    external=PyPlotCreator,  # NOTE Old name, kept for compatibility
    pyplot=PyPlotCreator,
    universe=UniversePlotCreator,
    multiverse=MultiversePlotCreator,
)
"""A mapping of plot creator names to the corresponding types"""
