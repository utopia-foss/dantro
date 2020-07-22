"""This sub-package implements non-abstract dantro plot creator classes"""

# Supply the base class
from .pcr_base import BasePlotCreator

# And all derived PlotCreator classes
from .pcr_ext import ExternalPlotCreator
from .pcr_decl import DeclarativePlotCreator
from .pcr_psp import UniversePlotCreator
from .pcr_psp import MultiversePlotCreator
from .pcr_vega import VegaPlotCreator

# Make some associated objects easier to import
from .pcr_base import SkipPlot
from .pcr_ext import is_plot_func, PlotHelper

# And gather them into a dictionary that gives names to each of them
ALL = dict(external=ExternalPlotCreator,
           declarative=DeclarativePlotCreator,
           universe=UniversePlotCreator,
           multiverse=MultiversePlotCreator,
           vega=VegaPlotCreator,
           )
