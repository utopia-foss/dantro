"""Plot functions to draw :py:class:`networkx.Graph` objects.

.. todo::

    Should really integrate utopya ``GraphPlot`` here!
"""

import logging
import os
from typing import Callable, Union

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def _get_positions(
    g: "networkx.Graph",
    *,
    model: Union[str, Callable] = "spring",
    fallback_model: str = None,
    fallback_kwargs: dict = None,
    silent_fallback: bool = False,
    **kwargs,
) -> dict:
    """Returns the positions dict for the given graph, created from a networkx
    layouting algorithm.

    Args:
        g (networkx.Graph): The graph object for which to create the layout
        model (Union[str, Callable], optional): Name of the layouting model.
            If starting with ``graphviz_<prog>``, will invoke
            :py:func:`networkx.drawing.nx_agraph.graphviz_layout` with the
            given value for ``prog``. Note that these only take a single
            keyword argument, ``args``.
            If it is a string, it's looked up from the networkx namespace.
            If it is a callable, it is invoked with ``g`` as only positional
            argument and ``**kwargs`` as keyword arguments.
        fallback_model (str, optional): If given, and ``model`` fails for *any*
            reason, will use this fallback model instead.
        fallback_kwargs (dict, optional): Keyword arguments for the fallback
            model invocation.
        silent_fallback (bool, optional): If True, will not log warnings in
            case there was need to switch to the fallback.
        **kwargs: Passed on to layouting algorithm.
    """

    import networkx as nx

    _suffix = "_layout"
    POSITIONING_MODELS_NETWORKX = {
        l[: -len(_suffix)]: getattr(nx, l)
        for l in dir(nx)
        if l.endswith(_suffix)
    }

    def invoke_model(model, **kwargs) -> dict:
        if callable(model):
            log.debug("Invoking callable for node layouting ...")
            return model(g, **kwargs)

        elif model.startswith("graphviz_"):
            log.debug("Invoking %s model for node layouting ...", model)
            try:
                model = model[len("graphviz_") :]
                return nx.drawing.nx_agraph.graphviz_layout(
                    g, prog=model, **kwargs
                )

            except ImportError as err:
                raise ImportError(
                    "Could not apply graphviz layout, probably because "
                    "pygraphviz is not installed!"
                ) from err

        else:
            try:
                log.debug("Invoking %s model for node layouting ...", model)
                return POSITIONING_MODELS_NETWORKX[model](g, **kwargs)

            except KeyError as err:
                _avail = ", ".join(POSITIONING_MODELS_NETWORKX)
                raise ValueError(
                    f"No layouting model '{model}' available in networkx! "
                    f"Available models: {_avail}"
                ) from err

    # .........................................................................

    if not fallback_model:
        return invoke_model(model, **kwargs)

    # Fallback available
    try:
        return invoke_model(model, **kwargs)

    except Exception as exc:
        if not silent_fallback:
            log.caution(
                "Node layouting with '%s' model failed with a %s: %s",
                model,
                type(exc).__name__,
                exc,
            )
            log.remark("Invoking fallback model '%s' ...", fallback_model)
        else:
            log.remark(
                "Node layouting with '%s' model failed; "
                "invoking fallback model '%s' ...",
                model,
                fallback_model,
            )

    return invoke_model(
        fallback_model, **(fallback_kwargs if fallback_kwargs else {})
    )


def _draw_graph(
    g: "networkx.Graph",
    *,
    ax: "matplotlib.axes.Axes" = None,
    drawing: dict = {},
    layout: dict = {},
):
    """Draws a graph using
    :py:func:`networkx.drawing.nx_pylab.draw_networkx_nodes`,
    :py:func:`networkx.drawing.nx_pylab.draw_networkx_edges`, and
    :py:func:`networkx.drawing.nx_pylab.draw_networkx_labels`.

    .. warning::

        This function is not yet completed and may change anytime.

    Args:
        g (networkx.Graph): The graph to draw
        out_path (str): Where to store it to
        drawing (dict, optional): Drawing arguments, containing the
            ``nodes``, ``edges`` and ``labels`` keys. The ``labels`` key
            can contain the ``from_attr`` key which will read the attribute
            specified there and use it for the label.
        layout (dict, optional): Used to generate node positions via the
            :py:func:`~dantro.plot.funcs.graph._get_positions` function.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    def draw(
        g, *, ax, pos, nodes: dict = {}, edges: dict = {}, labels: dict = {}
    ):
        # Parse some attributes
        if "from_attr" in labels:
            labels["labels"] = nx.get_node_attributes(
                g, labels.pop("from_attr")
            )

        # Now draw
        nx.draw_networkx_nodes(g, pos=pos, ax=ax, **nodes)
        nx.draw_networkx_edges(g, pos=pos, ax=ax, **edges)
        nx.draw_networkx_labels(g, pos=pos, ax=ax, **labels)

    ax = ax if ax is not None else plt.gca()
    pos = _get_positions(g, **layout)

    # Draw
    draw(g, ax=ax, pos=pos, **drawing)

    # Post-process
    ax.axis("off")
