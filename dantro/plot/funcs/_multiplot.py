"""Implements config-configurable function invocation that can be used for
applying function calls to a plot.
This is used in :py:func:`~dantro.plot.funcs.multiplot.multiplot` plot and
the :py:meth:`~dantro.plot.plot_helper.PlotHelper._hlpr_call` helper function.
"""

import copy
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

from ..._import_tools import (
    LazyLoader,
    import_module_or_object,
    resolve_lazy_imports,
)
from ...exceptions import *
from ...tools import make_columns, recursive_update

log = logging.getLogger(__name__)

# Lazy module loading for packages that take a long time to import
_plt = LazyLoader("matplotlib.pyplot", _depth=1)
_sns = LazyLoader("seaborn", _depth=1)
# NOTE This import is specifically for the definition of the seaborn-based
#      multiplot functions and should not be used for anything else!
#      Depth 1 means: `_sns.foobar` is still a LazyLoader object and will only
#      be resolved upon an *attribute* call! Subsequently, these need to be
#      resolved manually, as done via `resolve_lazy_imports`.

# fmt: off

# -----------------------------------------------------------------------------

# Define some default plot functions
# NOTE These are defined here in a lazy fashion, thus being actually
#      LazyLoader instances. They are (and need to be) resolved upon a first
#      call to the multiplot function
MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
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
    "plt.fill":             _plt.fill,
    "plt.scatter":          _plt.scatter,
    "plt.plot":             _plt.plot,
    "plt.polar":            _plt.polar,
    "plt.loglog":           _plt.loglog,
    "plt.semilogx":         _plt.fill,
    "plt.semilogy":         _plt.semilogy,

    # Distribution plots
    "plt.hist":             _plt.hist,
    "plt.hist2d":           _plt.hist2d,

    # Categorical plots
    "plt.bar":              _plt.bar,
    "plt.barh":             _plt.barh,
    "plt.pie":              _plt.pie,
    "plt.table":            _plt.table,

    # Matrix plots
    "plt.imshow":           _plt.imshow,
    "plt.pcolormesh":       _plt.pcolormesh,

    # Vector plots
    "plt.contour":          _plt.contour,
    "plt.quiver":           _plt.quiver,
    "plt.streamplot":       _plt.streamplot,
}   # --- end literalinclude
"""The default-available plot kinds for the
:py:func:`~dantro.plot.funcs.multiplot.multiplot` function.

Details of the seaborn-related plots can be found here in the
`seaborn docs <https://seaborn.pydata.org/api.html>`_.
"""

MULTIPLOT_CAUTION_FUNC_NAMES = tuple(
    func_name for func_name in MULTIPLOT_FUNC_KINDS
    if func_name not in (
        "sns.despine",
    )
)
"""The multiplot functions that emit a warning if they do not get any arguments
when called. This is helpful for functions that e.g. require a ``data``
argument but do *not* fail or warn if no such argument is passed on to them.
"""

# fmt: on

# -----------------------------------------------------------------------------


def parse_function_specs(
    *,
    _hlpr: "PlotHelper",
    _funcs: Dict[str, Callable] = None,
    _shared_kwargs: dict = {},
    function: Union[str, Callable, Tuple[str, str]],
    args: list = None,
    pass_axis_object_as: str = None,
    pass_helper: bool = False,
    **func_kwargs,
) -> Tuple[str, Callable, list, dict]:
    """Parses a function specification used in the ``invoke_function`` helper.
    If ``function`` is a string it is looked up from the ``_funcs`` dict.

    See :py:func:`~.parse_and_invoke_function` and
    :py:func:`~dantro.plot.funcs.multiplot.multiplot`.

    Args:
        _hlpr (dantro.plot.plot_helper.PlotHelper): The currently
            used PlotHelper instance
        _funcs (Dict[str, Callable]): The lookup dictionary for callables
        _shared_kwargs (dict, optional): Shared kwargs that passed on to
            all multiplot functions. They are recursively updated with
            the individual plot functions' ``func_kwargs``.
        function (Union[str, Callable, Tuple[str, str]]): The callable
            function object or the name of the plot function to look up.
            If given as 2-tuple ``(module, name)``, will attempt an import of
            that module.
        args (list, optional): The positional arguments for the plot function
        pass_axis_object_as (str, optional): If given, will add a keyword
            argument with this name to pass the current axis object to the
            to-be-invoked function.
        pass_helper (bool, optional): If true, passes the helper instance to
            the function call as keyword argument ``hlpr``.
        **func_kwargs (dict): The function kwargs to be passed on to the
            function object.

    Returns:
        Tuple[str, Callable, list, dict]: A tuple of function name, callable,
            positional arguments, and keyword arguments.
    """
    # Parse positional and keyword arguments
    if args is None:
        args = []

    if _funcs is None:
        _funcs = MULTIPLOT_FUNC_KINDS

    func_kwargs = recursive_update(copy.deepcopy(_shared_kwargs), func_kwargs)

    if pass_axis_object_as:
        func_kwargs[pass_axis_object_as] = _hlpr.ax

    if pass_helper:
        func_kwargs["hlpr"] = _hlpr

    # Get the function object and a readable name
    if callable(function):
        func_name = function.__name__
        func = function

    elif isinstance(function, (list, tuple)):
        # Import
        mod, name = function
        func = import_module_or_object(mod, name)
        func_name = ".".join(function)

    else:
        # Look up the function in the `_funcs` dict.
        # Still need to resolve all lazy imports in the lookup dictionary.
        resolve_lazy_imports(_funcs)

        func_name = function
        try:
            func = _funcs[func_name]

        except KeyError as err:
            _avail = " (none)\n"
            if _funcs:
                _avail = make_columns(_funcs)

            raise ValueError(
                f"A function called '{func_name}' could not be found "
                f"by name!\nAvailable functions:\n{_avail}\n"
                "Alternatively, pass a callable instead of the function name "
                "or pass a 2-tuple of (module, name) to import a callable."
            ) from err

    return func_name, func, args, func_kwargs


def parse_and_invoke_function(
    *,
    hlpr: "PlotHelper",
    shared_kwargs: dict,
    func_kwargs: dict,
    show_hints: bool,
    call_num: int,
    funcs: Dict[str, Callable] = None,
    caution_func_names: List[str] = None,
) -> Any:
    """Parses function arguments and then calls
    :py:func:`~dantro.plot.funcs.multiplot.multiplot`.

    Args:
        hlpr (PlotHelper): The currently used PlotHelper instance
        funcs (Dict[str, Callable], optional): The lookup dictionary for the
            plot functions. If not given, will use a default lookup dictionary
            with a set of seaborn and matplotlib functions.
        shared_kwargs (dict): Arguments shared between function calls
        func_kwargs (dict): Arguments for *this* function in particular
        show_hints (bool): Whether to show hints
        call_num (int): The number of this plot, for easier identification
        caution_func_names (List[str], optional): a list of function names that
            will trigger a log message if no function kwargs were given.
            If not explicitly given, will use some defaults.

    Returns:
        Any: return value of plot function call
    """
    if caution_func_names is None:
        caution_func_names = MULTIPLOT_CAUTION_FUNC_NAMES

    # Get the function name, the function object and all function kwargs
    # from the configuration entry.
    func_name, func, func_args, func_kwargs = parse_function_specs(
        _hlpr=hlpr, _funcs=funcs, _shared_kwargs=shared_kwargs, **func_kwargs
    )

    # Notify user if plot functions do not get any kwargs passed on.
    # This is e.g. helpful and relevant for seaborn functions that require
    # a 'data' kwarg but do not fail or warn if no 'data' is passed on to them.
    if show_hints and not func_kwargs and func_name in caution_func_names:
        log.caution(
            "You seem to have called '%s' without any function arguments. "
            "If this produces unexpected output, check that all required "
            "arguments (e.g. `data`, `x`, ...) were given.\n"
            "To silence this warning, set `show_hints` to `False`.",
            func_name,
        )

    # Apply the plot function and allow it to fail to make sure that potential
    # other plots are still plotted and shown.
    rv = None
    try:
        rv = func(*func_args, **func_kwargs)

    except Exception as exc:
        msg = (
            f"The call to '{func_name}' (call no. {call_num} on axis "
            f"{hlpr.ax_coords}) did not succeed!\n"
            f"Got a {type(exc).__name__}: {exc}"
        )
        if hlpr.raise_on_error:
            raise PlottingError(msg) from exc
        log.warning(
            f"{msg}\nEnable debug mode to get a full traceback. "
            "Proceeding with next plot ..."
        )

    return rv
