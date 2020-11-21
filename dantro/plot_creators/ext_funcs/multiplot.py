"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import logging
from typing import Callable, Union

import seaborn as sns

from ..pcr_ext import PlotHelper, is_plot_func

# Local constants
log = logging.getLogger(__name__)

# fmt: off

# The available plot kinds for the multiplot interface that require an axes
# to plot data onto.
# Details of the seaborn-related plots can be found here in the seaborn API:
# https://seaborn.pydata.org/api.html
_MULTIPLOT_PLOT_KINDS = { # --- start literalinclude
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
}   # --- end literalinclude

# fmt: on


# -- Helper functions ---------------------------------------------------------


def apply_plot_func(*, ax, func: Callable, **kwargs) -> None:
    """Apply a plot function to a given axis.

    Args:
        ax:                 The matplotlib Axes object to plot the data on.
        func (Callable):    The callable plot function.
    """
    func(ax=ax, **kwargs)


def get_multiplot_func(name: str) -> Callable:
    """Get the multiplot function from the _MULTIPLOT_PLOT_KINDS dict
    containing the following entries:

    .. literalinclude:: ../../dantro/plot_creators/ext_funcs/multiplot.py
        :language: python
        :start-after: _MULTIPLOT_PLOT_KINDS = { # --- start literalinclude
        :end-before:  }   # --- end literalinclude

    Args:
        name (str):             The name of the multiplot function to plot.

    Returns:
        A callable function object
    """
    try:
        plot_func = _MULTIPLOT_PLOT_KINDS[name]
    except KeyError as err:
        raise KeyError(
            f"The function '{name}' is not a valid multiplot function. "
            f"Valid options are: {_MULTIPLOT_PLOT_KINDS.values()}."
        ) from err
    return plot_func


# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True)
def multiplot(
    *, data: dict, hlpr: PlotHelper, to_plot: Union[list, dict]
) -> None:
    """Consecutively plot multiple functions on one or multiple axes.

    Args:
        data (dict): Data from TransformationDAG selection.
        hlpr (PlotHelper): The PlotHelper instance for this plot
        to_plot (Union[list, dict]): The data to plot. If to_plots is list-like
            the plot functions are plotted on the axes provided through hlpr.ax.
            If to_plot is dict-like, the keys specify the coordinate pair
            selecting an ax to plot on, e.g. (0,0), while the values specify
            a list of plot function configurations to apply consecutively.
            Each single plot function is configured via the following kwargs:

                ``function (str)``
                    The name of the function to plot. The corresponding
                    plot function is retreaved from the
                    :py:func:`~dantro.plot_creators.ext_funcs.multiplot.get_multiplot_func`
                    function.

                ``**func_kwargs (dict, optional)``
                    The function kwargs passed on to
                    the selected function to plot.

            Examples:
                A simple ``to_plot`` configuration on the hlpr.ax is:

                .. code-block:: yaml

                    to_plot:
                    - function: sns.lineplot
                      data: !dag_result data
                    - function: sns.despine

                A simple ``to_plot`` configuration specifying two axis is:

                .. code-block:: yaml

                    to_plot:
                    [0,0]: - function: sns.lineplot
                             data: !dag_result data
                    [1,0]: - function: sns.scatterplot
                             data: !dag_result data

    .. note::

        On a failing plot function call the logger will emit a warning.
        This allows to still show the plots of other functions applied on the
        same axis.

    Raises:
        NotImplementedError: On a dict-like ``to_plot`` argument that would
            define the ax to plot on in case of multiple axes to select from.
        TypeError: On a non-list-like or non-dict-like ``to_plot`` argument.
    """
    # dict-like to_plot is not yet implemented
    if isinstance(to_plot, dict):
        raise NotImplementedError(
            "'to_plot' needs to be list-like but was "
            f"of type {type(to_plot)}. Specifying multi-axis plots through "
            "a dict-like 'to_plot' argument is not yet implemented."
        )

    # to_plot needs to be a list
    elif not isinstance(to_plot, list):
        raise TypeError(
            "'to_plot' needs to be list-like but was "
            f"of type {type(to_plot)}. Please assure to pass a list."
        )

    for func_kwargs in to_plot:
        # Extract the multiplot function
        func_name = func_kwargs.pop("function")
        func = get_multiplot_func(func_name)

        # Warn if no data-key was given in the case of functions requiring
        # a `data` specification. Otherwise, most of the plots functions would
        # just not plot anything without any hint to the user.
        if func_name != "sns.despine" and ("data" not in func_kwargs):
            if hlpr.raise_on_error:
                raise KeyError(
                    "The required 'data' key is missing in the plot "
                    f"configuration of {func_name}"
                )

            log.warning(
                "The required 'data' key is missing in the plot "
                "configuration of '%s'",
                func_name,
            )

        # Apply the function to the PlotHelper axis hlpr.ax.
        # Allow for a single plot to fail.
        try:
            apply_plot_func(ax=hlpr.ax, func=func, **func_kwargs)

        # If plotting fails, just pass and try to plot the next plot :)
        except Exception as exc:
            log.warning(
                "Plotting '%s' did not succeed with the following error "
                "message: '%s'",
                func_name,
                exc,
            )
