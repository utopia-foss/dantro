"""Plot functions to draw :py:class:`networkx.Graph` objects.

.. todo::

    Should really integrate utopya ``GraphPlot`` here!
"""

import os

# -----------------------------------------------------------------------------


def _draw_graph(
    g: "networkx.Graph",
    *,
    out_path: str,
    drawing: dict = {},
    layout: dict = {},
    figure_kwargs: dict = {},
    save_kwargs: dict = {},
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
        layout (dict, optional): Passed to (currently hard-coded) layouting
            functions.
        figure_kwargs (dict, optional): Passed to
            :py:func:`matplotlib.pyplot.figure` for setting up the figure
        save_kwargs (dict, optional): Passed to
            :py:func:`matplotlib.pyplot.savefig` for saving the figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    def get_positions(g, model=None, **layout) -> dict:
        """Performs layouting on the given graph"""
        if model:
            raise NotImplementedError("Cannot change the layouting model yet!")

        try:
            return nx.nx_agraph.graphviz_layout(
                g, prog="dot", args="-y", **layout
            )

        except ImportError:
            pass

        return nx.multipartite_layout(
            g, align="horizontal", subset_key="layer", scale=-1, **layout
        )

    def draw_graph(
        g: "networkx.Graph",
        *,
        ax,
        pos,
        nodes: dict = {},
        edges: dict = {},
        labels: dict = {},
    ):
        """Draws the graph onto the given matplotlib axes"""
        # Parse some attributes
        if "from_attr" in labels:
            labels["labels"] = nx.get_node_attributes(
                g, labels.pop("from_attr")
            )

        # Draw
        nx.draw_networkx_nodes(g, pos=pos, ax=ax, **nodes)
        nx.draw_networkx_edges(g, pos=pos, ax=ax, **edges)
        nx.draw_networkx_labels(g, pos=pos, ax=ax, **labels)

        # Post-process
        ax.axis("off")

    def save_plot(*, out_path: str, bbox_inches="tight", **save_kwargs):
        """Saves the matplotlib plot to the given output path"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(
            out_path,
            bbox_inches=bbox_inches,
            **(save_kwargs if save_kwargs else {}),
        )

    # .....................................................................

    # Create figure
    fig = plt.figure(constrained_layout=True, **figure_kwargs)

    # Now layout, draw, and save the DAG visualization
    try:
        pos = get_positions(g, **layout)
        draw_graph(g, ax=plt.gca(), pos=pos, **drawing)
        save_plot(out_path=out_path, **save_kwargs)

    finally:
        plt.close(fig)
