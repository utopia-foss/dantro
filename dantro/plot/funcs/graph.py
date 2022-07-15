"""Plot functions to draw :py:class:`networkx.Graph` objects.

.. todo::

    Should really integrate utopya ``GraphPlot`` here!
"""

import copy
import logging
import os
from typing import Callable, Union

from ...tools import recursive_update as _recursive_update

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def _wiggle_pos(
    pos: dict, *, x: float = None, y: float = None, seed: int = None
) -> dict:
    """Wiggles positions by absolute random amplitudes in x and y direction

    Args:
        pos (dict): Positions dict with values being x and y positions
        x (float, optional): Absolute wiggle amplitude
        y (float, optional): Absolute wiggle amplitude
        seed (int, optional): Seed for the :py:class:`numpy.random.RandomState`
            that is used for drawing random numbers. Set to a fixed value to
            always get the same positions.
    """
    import numpy as np
    import numpy.random

    rng = np.random.RandomState(seed)
    wiggle = lambda old_pos, amp: old_pos + amp * rng.uniform(-1, +1)

    if x is not None:
        pos = {n: [wiggle(p[0], x), p[1]] for n, p in pos.items()}

    if y is not None:
        pos = {n: [p[0], wiggle(p[1], y)] for n, p in pos.items()}

    return pos


def _get_positions(
    g: "networkx.Graph",
    *,
    model: Union[str, Callable],
    wiggle: dict = None,
    **kwargs,
) -> dict:
    """Returns the positions dict for the given graph, created from a networkx
    layouting algorithm of a certain name or an arbitrary callable.

    Args:
        g (networkx.Graph): The graph object for which to create the layout
        model (Union[str, Callable]): Name of the layouting model or the
            layouting function itself.
            If starting with ``graphviz_<prog>``, will invoke
            :py:func:`networkx.drawing.nx_agraph.graphviz_layout` with the
            given value for ``prog``. Note that these only take a single
            keyword argument, ``args``.
            If it is a string, it's looked up from the networkx namespace.
            If it is a callable, it is invoked with ``g`` as only positional
            argument and ``**kwargs`` as keyword arguments.
        wiggle (dict, optional): If given, will postprocess the positions dict
            by randomly wiggling x and y coordinates according to the absolute
            amplitudes given as values.
        **kwargs: Passed on to the layouting algorithm.
    """

    import networkx as nx

    if callable(model):
        log.debug("Invoking callable for node layouting ...")
        pos = model(g, **kwargs)

    elif model.startswith("graphviz_"):
        log.debug("Invoking %s model for node layouting ...", model)
        try:
            model = model[len("graphviz_") :]
            pos = nx.drawing.nx_agraph.graphviz_layout(g, prog=model, **kwargs)

        except ImportError as err:
            raise ImportError(
                "Could not apply graphviz layout, probably because "
                "pygraphviz is not installed!"
            ) from err

    else:
        _suffix = "_layout"
        POSITIONING_MODELS_NETWORKX = {
            l[: -len(_suffix)]: getattr(nx, l)
            for l in dir(nx)
            if l.endswith(_suffix)
        }

        try:
            log.debug("Invoking %s model for node layouting ...", model)
            layout_func = POSITIONING_MODELS_NETWORKX[model]
            pos = layout_func(g, **kwargs)

        except KeyError as err:
            _avail = ", ".join(POSITIONING_MODELS_NETWORKX)
            raise ValueError(
                f"No layouting model '{model}' available in networkx! "
                f"Available models: {_avail}"
            ) from err

    if wiggle:
        pos = _wiggle_pos(pos, **wiggle)
    return pos


def get_positions(
    g: "networkx.Graph",
    *,
    model: Union[str, Callable] = "spring",
    model_kwargs: dict = {},
    fallback: Union[str, dict] = None,
    silent_fallback: bool = False,
    **kwargs,
) -> dict:
    """Returns the positions dict for the given graph, created from a networkx
    layouting algorithm of a certain name or an arbitrary callable.

    This is a wrapper around :py:func:`._get_positions` which allows to specify
    a fallback layouting model to use in case the first one fails for whatever
    reason.

    Args:
        g (networkx.Graph): The graph object for which to create the layout
        model (Union[str, Callable], optional): Name of the layouting model or
            the layouting function itself.
            If starting with ``graphviz_<prog>``, will invoke
            :py:func:`networkx.drawing.nx_agraph.graphviz_layout` with the
            given value for ``prog``. Note that these only take a single
            keyword argument, ``args``.
            If it is a string, it's looked up from the networkx namespace.
            If it is a callable, it is invoked with ``g`` as only positional
            argument and ``**kwargs`` as keyword arguments.
        model_kwargs (dict, optional): A dict where keys correspond to names
            of layouting models and values are parameters that are to be passed
            to the layouting function. This dict may contain more arguments
            than required, only the ``model`` key is looked up here. This can
            be useful for providing a wider set of defaults. These defaults
            are not considered when ``model`` is a callable.
        fallback (Union[str, dict], optional): The fallback model name (if
            a string) or a dict containing the key ``model`` and further
            kwargs.
        silent_fallback (bool, optional): Whether to log a visible message
            about the fallback or a more discrete one.
        **kwargs: Passed on to the layouting algorithm in addition to the
            selected entry from ``model_kwargs``. Keys given here update those
            from ``model_kwargs``.
            Also, these are *not* passed on to the fallback invocation.
    """

    def parse_kwargs(model: Union[str, Callable], **kwargs) -> dict:
        """Performs lookup and update of the layouting model arguments"""
        if callable(model):
            return kwargs

        return _recursive_update(
            copy.deepcopy(model_kwargs.get(model, {})), kwargs
        )

    # Prepare arguments
    if isinstance(fallback, str):
        fallback = dict(model=fallback)

    # Get the positions, potentially using a fallback
    try:
        return _get_positions(g, model=model, **parse_kwargs(model, **kwargs))

    except Exception as exc:
        if not fallback:
            raise

        if not silent_fallback:
            log.caution(
                "Node layouting with '%s' model failed with a %s: %s",
                model,
                type(exc).__name__,
                exc,
            )
            log.remark("Invoking fallback layouting:  %s", fallback)
        else:
            log.remark(
                "Node layouting with '%s' model failed; using fallback:  %s",
                model,
                fallback,
            )

    return _get_positions(
        g, model=fallback["model"], **parse_kwargs(**fallback)
    )


# .............................................................................


def _draw_graph(
    g: "networkx.Graph",
    *,
    ax: "matplotlib.axes.Axes" = None,
    drawing: dict = {},
    layout: dict = {},
) -> list:
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
            :py:func:`~dantro.plot.funcs.graph.get_positions` function.
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
        _nodes = nx.draw_networkx_nodes(g, pos=pos, ax=ax, **nodes)
        _edges = nx.draw_networkx_edges(g, pos=pos, ax=ax, **edges)
        _labels = nx.draw_networkx_labels(g, pos=pos, ax=ax, **labels)

        # Gather artists, adapting to the specific way they are passed back
        artists = []
        artists.append(_nodes)
        artists += _edges
        artists += list(_labels.values())

        return artists

    ax = ax if ax is not None else plt.gca()
    pos = get_positions(g, **layout)

    # Draw
    artists = draw(g, ax=ax, pos=pos, **drawing)

    # Post-process
    ax.axis("off")

    return artists
