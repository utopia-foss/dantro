"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import logging
from typing import Union

import seaborn as sns

from ..pcr_ext import PlotHelper, is_plot_func

# Local constants
log = logging.getLogger(__name__)

# fmt: off

# The available plot kinds for the multiplot interface that require an axes 
# to plot data onto. 
# Details of the seaborn-related plots can be found here in the seaborn API:
# https://seaborn.pydata.org/api.html
_MULTIPLOT_PLOT_KINDS = {
    # Relational plots
    "sns.scatterplot": sns.scatterplot,
    "sns.lineplot": sns.lineplot,
    # Distribution plots
    "sns.histplot": sns.histplot,
    "sns.kdeplot": sns.kdeplot,
    "sns.ecdfplot": sns.ecdfplot,
    "sns.rugplot": sns.rugplot,
    # Categorical plots
    "sns.stripplot": sns.stripplot,
    "sns.swarmplot": sns.swarmplot,
    "sns.boxplot": sns.boxplot,
    "sns.violinplot": sns.violinplot,
    "sns.boxenplot": sns.boxenplot,
    "sns.pointplot": sns.pointplot,
    "sns.barplot": sns.barplot,
    "sns.countplot": sns.countplot,
    # Regression plots
    "sns.regplot": sns.regplot,
    "sns.residplot": sns.residplot,
    # Matrix plots
    "sns.heatmap": sns.heatmap,
    # Utility functions
    "sns.despine": sns.despine,
}

# fmt: on


# -- Helper functions ---------------------------------------------------------

def apply_plot_func(ax, plot_spec: list, **kwargs):
    pass

def get_multiplot_function(name: str):
    pass

# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------

@is_plot_func(
    use_dag=True, required_dag_tags=(("data",))
)
def multiplot (
    *,
    data: dict,
    to_plot: Union[list, dict],
    hlpr: PlotHelper,
    **multiplot_kwargs,
) -> None:
    # dict-like to_plot is not yet implemented
    if isinstance(to_plot, dict):
        raise NotImplementedError("'to_plot' needs to be list-like but was "
            f"of type {type(to_plot)}. Specifying multi-axis plots through "
            "a dict-like 'to_plot' argument is not yet implemented.")
    
    # to_plot needs to be a list
    elif not isinstance(to_plot, list):
        raise TypeError("'to_plot' needs to be list-like but was "
            f"of type {type(to_plot)}. Please assure to pass a list.")
    
    pass
