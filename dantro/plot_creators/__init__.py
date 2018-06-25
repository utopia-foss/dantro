"""This sub-package implements the dantro PlotCreators"""

# Supply the base class
from dantro.plot_creators.pcr_base import BasePlotCreator

# And all derived PlotCreator classes
from dantro.plot_creators.pcr_ext import ExternalPlotCreator
from dantro.plot_creators.pcr_decl import DeclarativePlotCreator
from dantro.plot_creators.pcr_vega import VegaPlotCreator

# And gather them into a dictionary that gives names to each of them
ALL = dict(external=ExternalPlotCreator,
           declarative=DeclarativePlotCreator,
           vega=VegaPlotCreator,
           )
