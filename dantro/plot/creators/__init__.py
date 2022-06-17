"""This sub-package implements non-abstract dantro plot creator classes,
based on :py:class:`~dantro.plot.creators.base.BasePlotCreator`"""

from .base import BasePlotCreator
from .psp import MultiversePlotCreator, UniversePlotCreator
from .pyplot import PyPlotCreator

ALL_PLOT_CREATORS = dict(
    pyplot=PyPlotCreator,
    external=PyPlotCreator,
    universe=UniversePlotCreator,
    multiverse=MultiversePlotCreator,
)
"""A mapping of plot creator names to the corresponding types"""
