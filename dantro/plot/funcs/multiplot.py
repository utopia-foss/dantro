"""Generic, DAG-based multiplot function for the
:py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` and derived plot
creators.
"""

import logging
from typing import Callable, Dict, List, Tuple, Union

from ..utils import is_plot_func
from ._multiplot import parse_and_invoke_function

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True)
def multiplot(
    *,
    hlpr: "dantro.plot.plot_helper.PlotHelper",
    to_plot: Union[List[dict], Dict[Tuple[int, int], List[dict]]],
    data: dict,
    funcs: Dict[str, Callable] = None,
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
    :py:func:`~dantro.plot.funcs._multiplot.parse_function_specs`
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

    .. literalinclude:: ../../dantro/plot/funcs/_multiplot.py
        :language: python
        :start-after: MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    It is also possible to *import* callables on the fly. To do so, pass a
    2-tuple of ``(module, name)`` to ``function``, which will then be loaded
    using :py:func:`~dantro._import_tools.import_module_or_object`.

    Args:
        hlpr (dantro.plot.plot_helper.PlotHelper): The PlotHelper instance for
            this plot, carrying the to-be-plotted-on figure object.
        to_plot (Union[list, dict]): The plot specifications.
            If list-like, assumes that there is only a single axis and applies
            all functions to that axis.
            If dict-like, expects 2-tuples for keys and selects the axis before
            commencing to plot. Beforehand, the figure needs to have been set
            up accordingly via the ``setup_figure`` helper.
        data (dict): Data from TransformationDAG selection. These results are
            ignored; data needs to be passed via the result placeholders!
            See above.
        funcs (Dict[str, Callable], optional): If given, use this dictionary
            to look up functions by name. If not given, will use a default
            dict with a set of matplotlib and seaborn functions.
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
                    funcs=funcs,
                    shared_kwargs=shared_kwargs,
                    func_kwargs=func_kwargs,
                    show_hints=show_hints,
                    call_num=call_num,
                )

    else:
        for call_num, func_kwargs in enumerate(to_plot):
            parse_and_invoke_function(
                hlpr=hlpr,
                funcs=funcs,
                shared_kwargs=shared_kwargs,
                func_kwargs=func_kwargs,
                show_hints=show_hints,
                call_num=call_num,
            )
