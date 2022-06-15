"""This sub-package implements non-abstract dantro plot creator classes,
based on :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator`"""

from .pcr_base import BasePlotCreator, SkipPlot
from .pcr_decl import DeclarativePlotCreator
from .pcr_ext import ExternalPlotCreator, PlotHelper, is_plot_func
from .pcr_psp import MultiversePlotCreator, UniversePlotCreator
from .pcr_vega import VegaPlotCreator

ALL = dict(
    external=ExternalPlotCreator,
    declarative=DeclarativePlotCreator,
    universe=UniversePlotCreator,
    multiverse=MultiversePlotCreator,
    vega=VegaPlotCreator,
)
"""A mapping of plot creator names to the corresponding types"""
