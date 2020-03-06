"""This sub-package implements non-abstract dantro plot creator classes"""

# Supply the base class
from dantro.plot_creators.pcr_base import BasePlotCreator

# And all derived PlotCreator classes
from dantro.plot_creators.pcr_ext import ExternalPlotCreator
from dantro.plot_creators.pcr_decl import DeclarativePlotCreator
from dantro.plot_creators.pcr_psp import UniversePlotCreator
from dantro.plot_creators.pcr_psp import MultiversePlotCreator
from dantro.plot_creators.pcr_vega import VegaPlotCreator

# Make some associated objects easier to import
from dantro.plot_creators.pcr_ext import is_plot_func, PlotHelper

# And gather them into a dictionary that gives names to each of them
ALL = dict(external=ExternalPlotCreator,
           declarative=DeclarativePlotCreator,
           universe=UniversePlotCreator,
           multiverse=MultiversePlotCreator,
           vega=VegaPlotCreator,
           )
