"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import logging
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt

from ..._import_tools import LazyLoader, import_module_or_object
from .._plot_helper import PlotHelper, parse_and_invoke_function
from ..pcr_ext import is_plot_func

# Local constants
log = logging.getLogger(__name__)

# Lazy module loading for packages that take a long time to import
_sns = LazyLoader("seaborn", _depth=1)
# NOTE This import is specifically for the definition of the seaborn-based
#      multiplot functions and should not be used for anything else!
#      Depth 1 means: `_sns.foobar` is still a LazyLoader object and will only
#      be resolved upon an *attribute* call! Subsequently, these need to be
#      resolved manually, as done via `_resolve_lazy_imports`.

# fmt: off

# The available plot kinds for the multiplot interface.
# Details of the seaborn-related plots can be found here in the seaborn API:
# https://seaborn.pydata.org/api.html
#
# NOTE Seaborn plot functions are defined here in a lazy fashion, thus being
#      actually LazyLoader instances. They are resolved upon a first call to
#      the multiplot function
_MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
    # Seaborn - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # https://seaborn.pydata.org/api.html

    # Relational plots
    "sns.scatterplot":      _sns.scatterplot,
    "sns.lineplot":         _sns.lineplot,

    # Distribution plots
    "sns.histplot":         _sns.histplot,
    "sns.kdeplot":          _sns.kdeplot,
    "sns.ecdfplot":         _sns.ecdfplot,
    "sns.rugplot":          _sns.rugplot,

    # Categorical plots
    "sns.stripplot":        _sns.stripplot,
    "sns.swarmplot":        _sns.swarmplot,
    "sns.boxplot":          _sns.boxplot,
    "sns.violinplot":       _sns.violinplot,
    "sns.boxenplot":        _sns.boxenplot,
    "sns.pointplot":        _sns.pointplot,
    "sns.barplot":          _sns.barplot,
    "sns.countplot":        _sns.countplot,

    # Regression plots
    "sns.regplot":          _sns.regplot,
    "sns.residplot":        _sns.residplot,

    # Matrix plots
    "sns.heatmap":          _sns.heatmap,

    # Utility functions
    "sns.despine":          _sns.despine,

    # Matplotlib - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # https://matplotlib.org/tutorials/introductory/sample_plots.html

    # Relational plots
    "plt.fill":             plt.fill,
    "plt.scatter":          plt.scatter,
    "plt.plot":             plt.plot,
    "plt.polar":            plt.polar,
    "plt.loglog":           plt.loglog,
    "plt.semilogx":         plt.fill,
    "plt.semilogy":         plt.semilogy,

    # Distribution plots
    "plt.hist":             plt.hist,
    "plt.hist2d":           plt.hist2d,

    # Categorical plots
    "plt.bar":              plt.bar,
    "plt.barh":             plt.barh,
    "plt.pie":              plt.pie,
    "plt.table":            plt.table,

    # Matrix plots
    "plt.imshow":           plt.imshow,
    "plt.pcolormesh":       plt.pcolormesh,

    # Vector plots
    "plt.contour":          plt.contour,
    "plt.quiver":           plt.quiver,
    "plt.streamplot":       plt.streamplot,
}   # --- end literalinclude

# The multiplot functions that emit a warning if they do not get any arguments
# when called.
# This is helpful for functions that e.g. require a 'data' argument but do
# not fail or warn if no 'data' is passed on to them.
_MULTIPLOT_CAUTION_FUNC_NAMES = tuple([
    func_name for func_name in _MULTIPLOT_FUNC_KINDS
    if func_name not in ("sns.despine",)
])


# fmt: on


# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True)
def multiplot(
    *,
    hlpr: PlotHelper,
    to_plot: Union[List[dict], Dict[Tuple[int, int], List[dict]]],
    data: dict,
    show_hints: bool = True,
    **shared_kwargs,
) -> None:
    """Consecutively call multiple plot functions on one or multiple axes.

    ``to_plot`` contains all relevant information for the functions to plot.
    If ``to_plot`` is list-like the plot functions are plotted on the current
    axes created through the hlpr.
    If ``to_plot`` is dict-like, the keys specify the coordinate pair selecting
    an ax to plot on, e.g. (0,0), while the values specify a list of plot
    function configurations to apply consecutively.
    Each list entry specifies one function plot and is parsed via the
    :py:func:`~dantro.plot_creators._plot_helper.parse_function_specs`
    function.

    The multiplot works with any plot function that either operates on the
    current axis and does *not* create a new figure or does not require an
    axis at all.

    .. note::

        While most functions will automatically operate on the current axis,
        some function calls may require an axis object.
        If so, use the ``pass_axis_object_as`` argument to specify the name of
        the keyword argument as which the current axis is to be passed to the
        function call.

    Look at the :ref:`multiplot documentation <dag_multiplot>` for further
    information.

    Example:

        A simple ``to_plot`` specification for a single axis may look like
        this:

        .. code-block:: yaml

            to_plot:
              - function: sns.lineplot
                data: !dag_result data
                # Note that especially seaborn plot functions require a
                # `data` input argument that can conveniently be
                # provided via the `!dag_result` YAML-tag.
                # If not provided, nothing is plotted without emitting
                # a warning.
              - function: sns.despine

        A ``to_plot`` specification for a two-column subplot could look like
        this:

        .. code-block:: yaml

            to_plot:
              [0,0]:
                - function: sns.lineplot
                  data: !dag_result data
                - # ... more here ...
              [1,0]:
                - function: sns.scatterplot
                  data: !dag_result data

    If ``function`` is a string it is looked up from the following dictionary:

    .. literalinclude:: ../../dantro/plot_creators/ext_funcs/multiplot.py
        :language: python
        :start-after: _MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    It is also possible to *import* callables on the fly. To do so, pass a
    2-tuple of ``(module, name)`` to ``function``, which will then be loaded
    using :py:func:`~dantro._import_tools.import_module_or_object`.

    Args:
        hlpr (PlotHelper): The PlotHelper instance for this plot
        to_plot (Union[list, dict]): The plot specifications.
            If list-like, assumes that there is only a single axis and applies
            all functions to that axis.
            If dict-like, expects 2-tuples for keys and selects the axis before
            commencing to plot. Beforehand, the figure needs to have been set
            up accordingly via the ``setup_figure`` helper.
        data (dict): Data from TransformationDAG selection. These results are
            ignored; data needs to be passed via the result placeholders!
            See above.
        show_hints (bool): Whether to show hints in the case of not passing
            any arguments to a plot function.
        **shared_kwargs (dict): Shared kwargs for all plot functions.
            They are recursively updated, if to_plot specifies the same
            kwargs.

    .. warning::

        Note that especially seaborn plot functions require a ``data``
        argument that needs to be passed via a ``!dag_result`` key,
        see :ref:`dag_result_placeholder`.
        The multiplot function neither expects nor automatically passes a
        ``data`` DAG-node to the individual functions.

    .. note::

        If a plot fails and the helper is configured to not raise on a failing
        invocation, the logger will inform about the error. This allows to
        still apply other functions on the same axis.

    Raises:
        TypeError: On a non-list-like or non-dict-like ``to_plot`` argument.
    """
    if not isinstance(to_plot, (list, tuple, dict)):
        raise TypeError(
            "The `to_plot` argument needs to be list-like or a dict but was "
            f"of type {type(to_plot)} with value {to_plot}."
        )

    if show_hints and data:
        log.caution(
            "Got the following transformation results via the "
            f"`data` argument: {', '.join(data)}. "
            "Note that the multiplot function ignores these; pass "
            "them via result placeholders instead. Remove these tags "
            "from the `compute_only` argument to avoid passing them "
            "or set `show_hints` to False to suppress this hint."
        )

    if isinstance(to_plot, dict):
        for ax_coords, specs in to_plot.items():
            hlpr.select_axis(*ax_coords)
            for call_num, func_kwargs in enumerate(specs):
                parse_and_invoke_function(
                    hlpr=hlpr,
                    funcs=_MULTIPLOT_FUNC_KINDS,
                    shared_kwargs=shared_kwargs,
                    func_kwargs=func_kwargs,
                    show_hints=show_hints,
                    call_num=call_num,
                    caution_func_names=_MULTIPLOT_CAUTION_FUNC_NAMES,
                )

    else:
        for call_num, func_kwargs in enumerate(to_plot):
            parse_and_invoke_function(
                hlpr=hlpr,
                funcs=_MULTIPLOT_FUNC_KINDS,
                shared_kwargs=shared_kwargs,
                func_kwargs=func_kwargs,
                show_hints=show_hints,
                call_num=call_num,
                caution_func_names=_MULTIPLOT_CAUTION_FUNC_NAMES,
            )
