"""This sub-package implements non-abstract dantro plot creator classes"""

from .pcr_base import BasePlotCreator, SkipPlot
from .pcr_decl import DeclarativePlotCreator
from .pcr_ext import ExternalPlotCreator, PlotHelper, is_plot_func
from .pcr_psp import MultiversePlotCreator, UniversePlotCreator
from .pcr_vega import VegaPlotCreator

# And gather them into a dictionary that gives names to each of them
ALL = dict(
    external=ExternalPlotCreator,
    declarative=DeclarativePlotCreator,
    universe=UniversePlotCreator,
    multiverse=MultiversePlotCreator,
    vega=VegaPlotCreator,
)
