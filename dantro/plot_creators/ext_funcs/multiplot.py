"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import copy
import logging
from typing import Callable, Union

import matplotlib.pyplot as plt

from ..._import_tools import LazyLoader
from ...exceptions import PlottingError
from ...tools import make_columns, recursive_update
from ..pcr_ext import PlotHelper, is_plot_func

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


# -- Helper functions ---------------------------------------------------------


def _resolve_lazy_imports(d: dict):
    """In-place resolves lazy imports in the given dict"""
    for k, v in d.items():
        if isinstance(v, LazyLoader):
            d[k] = v.resolve()


def _parse_func_kwargs(
    function: Union[str, Callable],
    args: list = None,
    shared_kwargs: dict = None,
    **func_kwargs,
):
    """Parse a multiplot callable and its positional and keyword arguments.
    If ``function`` is a string it is looked up and mapped from the following
    dictionary:

    .. literalinclude:: ../../dantro/plot_creators/ext_funcs/multiplot.py
        :language: python
        :start-after: _MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    Args:
        function (Union[str, Callable]):  The callable function object or the
            name of the plot function to look up.

        args (list, optional): The positional arguments for the plot function

        shared_kwargs (dict, optional): Shared kwargs that passed on to
            all multiplot functions. They are recursively updated with
            the individual plot functions' func_kwargs.

        **func_kwargs (dict): The function kwargs to be passed on to the
            function object.

    .. note::

        The function kwargs cannot pass on a ``function`` or ``args`` key
        because both are parsed and translated into the plot function to use
        and the optional positional function arguments, respectively.

    Returns:
        (str, Callable, list, dict): (function name, function object, function
            arguments, function kwargs)
    """
    # First need to resolve all lazy imports in _MULTIPLOT_FUNC_KINDS
    _resolve_lazy_imports(_MULTIPLOT_FUNC_KINDS)

    if shared_kwargs is None:
        shared_kwargs = {}

    func_kwargs = recursive_update(copy.deepcopy(shared_kwargs), func_kwargs)

    if callable(function):
        func_name = function.__name__
        func = function
    else:
        func_name = function

        # Look up the function in the _MULTIPLOT_FUNC_KINDS dict
        try:
            func = _MULTIPLOT_FUNC_KINDS[func_name]
        except KeyError as err:
            _mp_funcs = _MULTIPLOT_FUNC_KINDS
            if _mp_funcs:
                _mp_funcs = "\n" + make_columns(_MULTIPLOT_FUNC_KINDS)
            else:
                _mp_funcs = " (none)\n"

            raise ValueError(
                f"The function `{func_name}` is not a valid multiplot "
                f"function. Available functions: \n {_mp_funcs} \n"
                "Alternatively, pass a callable instead of the name of a plot "
                "function."
            ) from err

    if args is None:
        args = []

    return func_name, func, args, func_kwargs


# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True)
def multiplot(
    *,
    hlpr: PlotHelper,
    to_plot: Union[list, dict],
    data: dict,
    show_hints: bool = True,
    **shared_kwargs,
) -> None:
    """Consecutively plot multiple functions on one or multiple axes.

    ``to_plot`` contains all relevant information for the functions to plot.
    If ``to_plot`` is list-like the plot functions are plotted on the current
    axes created through the hlpr.
    If ``to_plot`` is dict-like, the keys specify the coordinate pair selecting
    an ax to plot on, e.g. (0,0), while the values specify a list of plot
    function configurations to apply consecutively.
    Each list entry specifies one function plot and is parsed via the
    :py:func:`~dantro.plot_creators.ext_funcs.multiplot._parse_func_kwargs`
    function.
    The multiplot works with any plot function that either operates on the
    current axis and does _not_ create a new figure or does not require an
    axis at all.

    Look at the :ref:`multiplot documentation <dag_multiplot>` for further
    information.

    Examples:
        A simple ``to_plot`` configuration looks like this:

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

        A simple ``to_plot`` configuration specifying two axis is:

        .. code-block:: yaml

            to_plot:
              [0,0]:
                - function: sns.lineplot
                  data: !dag_result data
              [1,0]:
                - function: sns.scatterplot
                  data: !dag_result data

    Args:
        hlpr (PlotHelper): The PlotHelper instance for this plot
        to_plot (Union[list, dict]): The data to plot.
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
    elif not isinstance(to_plot, (list, tuple)):
        raise TypeError(
            "'to_plot' needs to be list-like but was "
            f"of type {type(to_plot)}. Please assure to pass a list."
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

    for func_num, func_kwargs in enumerate(to_plot):
        # Get the function name, the function object and all function kwargs
        # from the configuration entry.
        func_name, func, func_args, func_kwargs = _parse_func_kwargs(
            shared_kwargs=shared_kwargs, **func_kwargs
        )

        # Notify user if plot functions do not get any kwargs passed on.
        # This is e.g. helpful and relevant for seaborn functions that require
        # a 'data' kwarg but do not fail or warn if no 'data' is passed on to
        # them.
        if (
            show_hints
            and not func_kwargs
            and func_name in _MULTIPLOT_CAUTION_FUNC_NAMES
        ):
            log.caution(
                "Oops, you seem to have called '%s' without any function "
                "arguments. If the plot produces unexpected output, check "
                "that all required arguments (e.g. `data`, `x`, ...) were "
                "given.\n"
                "To silence this warning, set `show_hints` to `False`."
            )

        # Apply the plot function and allow it to fail to make sure that
        # potential other plots are still plotted and shown.
        try:
            func(*func_args, **func_kwargs)

        except Exception as exc:
            msg = (
                f"Plotting with '{func_name}', plot number {func_num}, "
                f"did not succeed! Got a {type(exc).__name__}: {exc}"
            )
            if hlpr.raise_on_error:
                raise PlottingError(msg) from exc
            log.warning(
                f"{msg}\nEnable debug mode to get a full traceback. "
                "Proceeding with next plot ..."
            )
