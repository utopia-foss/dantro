"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import logging
from typing import Callable, Union

import seaborn as sns

from ...exceptions import PlottingError
from ..pcr_ext import PlotHelper, is_plot_func

# Local constants
log = logging.getLogger(__name__)

# fmt: off

# The available plot kinds for the multiplot interface that require an axes
# to plot data onto.
# Details of the seaborn-related plots can be found here in the seaborn API:
# https://seaborn.pydata.org/api.html
_MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
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

# The multiplot functions that emit a warning if they do not get any arguments
# when called.
# This is helpful for functions that e.g. require a 'data' argument but do
# not fail or warn if no 'data' is passed on to them.
_MULTIPLOT_CAUTION_FUNC_NAMES = {
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
}

# fmt: on


# -- Helper functions ---------------------------------------------------------


def apply_plot_func(*, ax, func: Callable, **kwargs) -> None:
    """Apply a plot function to a given axis.

    Args:
        ax:                 The matplotlib Axes object to plot the data on.
        func (Callable):    The callable plot function.
    """
    func(ax=ax, **kwargs)


def parse_func_kwargs(function: Union[str, Callable], **func_kwargs):
    """Parse the multiplot function kwargs.

    Args:
        function (Union[str, Callable]):  The callable function object or the
            name of the plot function to look up in the _MULTIPLOT_FUNC_KINDS
            dict containing the following entries:

            .. literalinclude:: ../../dantro/plot_creators/ext_funcs/multiplot.py
                :language: python
                :start-after: _MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
                :end-before:  }   # --- end literalinclude


        **func_kwargs (dict): The function kwargs to be passed on to the
            function object.

    Returns:
        func_name:      The plot function name
        func:           A callable function object
        func_kwargs:    The kwargs for the multiplot function
    """
    if callable(function):
        func_name = function.__name__
        func = function
    else:
        func_name = function

        # Look up the function in the _MULTIPLOT_FUNC_KINDS dict
        try:
            func = _MULTIPLOT_FUNC_KINDS[func_name]
        except KeyError as err:
            raise KeyError(
                f"The function `{func_name}` is not a valid multiplot function. "
                f"Available functions: {', '.join(_MULTIPLOT_FUNC_KINDS.keys())}."
            ) from err

    return func_name, func, func_kwargs


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
            Each list entry specifies one function plot and is parsed via the
            :py:func:`~dantro.plot_creators.ext_funcs.multiplot.parse_func_kwargs`
            function.

            Examples:
                A simple ``to_plot`` configuration on the hlpr.ax is:

                .. code-block:: yaml

                    to_plot:
                    - function: sns.lineplot
                      data: !dag_result data
                      # Note that seaborn plot functions require a `data`
                      # input argument that can conveniently be provided via
                      # the !dag_result YAML-tag. If not provided, nothing
                      # is plotted without showing a warning.
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

    for func_num, func_kwargs in enumerate(to_plot):
        # Get the function name, the function object and all function kwargs
        # from the configuration entry.
        func_name, func, func_kwargs = parse_func_kwargs(**func_kwargs)

        # Notify user if plot functions do not get any kwargs passed on.
        # This is e.g. helpful and relevant for seaborn functions that require
        # a 'data' kwarg but do not fail or warn if no 'data' is passed on to
        # them.
        if not func_kwargs and func_name in _MULTIPLOT_CAUTION_FUNC_NAMES:
            log.caution(
                "Oops, you seem to have called '%s' without any function "
                "arguments. If the plot produces unexpected output, check that "
                "all required arguments (e.g. `data`, `x`, ...) were given."
            )

        # Apply the plot function and allow it to fail to make sure that
        # potential other plots are still plotted and shown.
        try:
            apply_plot_func(ax=hlpr.ax, func=func, **func_kwargs)

        except Exception as exc:
            msg = (
                f"Plotting with '{func_name}', plot number '{func_num}', "
                f"did not succeed! Got a {type(exc).__name__}: {exc}"
            )
            if hlpr.raise_on_error:
                raise PlottingError(msg) from exc
            log.warning(
                f"{msg}\nEnable debug mode to get a full traceback."
                "Proceeding with next plot ..."
            )
