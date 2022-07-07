"""networkx-related utility functions"""

import copy
import logging
import os
from typing import Sequence, Union

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def keep_node_attributes(g: "networkx.Graph", *to_keep):
    """Iterates over the given graph object and removes all node attributes
    *but* those in ``to_keep``.

    .. note:: This function works in-place on the given graph object

    Args:
        g (networkx.Graph): The graph object with the nodes
        *to_keep: Sequence of attribute names to *keep*
    """
    for _, node_attrs in g.nodes(data=True):
        to_pop = [a for a in node_attrs if a not in to_keep]
        for a in to_pop:
            node_attrs.pop(a, None)


def keep_edge_attributes(g: "networkx.Graph", *to_keep):
    """Iterates over the given graph object and removes all edge attributes
    *but* those in ``to_keep``.

    .. note:: This function works in-place on the given graph object

    Args:
        g (networkx.Graph): The graph object with the edges
        *to_keep: Sequence of attribute names to *keep*
    """
    for _, _, edge_attrs in g.edges(data=True):
        to_pop = [a for a in edge_attrs if a not in to_keep]
        for a in to_pop:
            edge_attrs.pop(a, None)


def export_graph(
    g: "networkx.Graph",
    *,
    out_path: str,
    keep_node_attrs: Union[bool, Sequence[str]] = True,
    keep_edge_attrs: Union[bool, Sequence[str]] = True,
    **export_specs,
):
    """Takes care of exporting a networkx graph object using one or many of the
    ``nx.write_`` methods.

    Args:
        g (networkx.Graph): The graph to export
        out_path (str): Path to export it to; extensions will be dropped and
            replaced by the corresponding export format. Add the ``file_ext``
            key to a export format specification to set it to some other value.
        keep_node_attrs (Union[bool, Sequence[str]], optional): Which node
            attributes to *keep*, all others are dropped. Set to True to keep
            all existing node attributes; for all other values the
            :py:func:`~.keep_node_attributes` function is invoked.
        keep_edge_attrs (Union[bool, Sequence[str]], optional): Which edge
            attributes to *keep*, all others are dropped. Set to True to keep
            all existing edge attributes; for all other values the
            :py:func:`~.keep_edge_attributes` function is invoked.
        **export_specs: Keys need to correspond to valid ``nx.write_*``
            function names, values are passed on to the write function. There
            are two special keys ``enabled`` and ``file_ext`` that can control
            the behaviour of the respective export operation.
    """
    import networkx as nx

    NX_WRITERS = {
        f[6:]: getattr(nx, f) for f in dir(nx) if f.startswith("write_")
    }
    try:
        NX_WRITERS["dot"] = nx.drawing.nx_agraph.write_dot

    except ImportError:
        pass

    # Need to work on a copy because certain attributes will be dropped
    g = copy.deepcopy(g)

    # Keep only certain node and/or edge attributes
    if keep_node_attrs is not True:
        keep_node_attributes(g, *(keep_node_attrs if keep_node_attrs else ()))

    if keep_edge_attrs is not True:
        keep_edge_attributes(g, *(keep_edge_attrs if keep_edge_attrs else ()))

    # No go over the export specifications
    export_specs = copy.deepcopy(export_specs)
    for export_format, specs in export_specs.items():
        if isinstance(specs, bool):
            specs = dict(enabled=specs)

        if not specs.pop("enabled", True):
            continue

        try:
            writer = NX_WRITERS[export_format]

        except KeyError as err:
            _avail = ", ".join(NX_WRITERS)
            raise ValueError(
                f"Invalid export format '{export_format}'! "
                "No such writer available in networkx.\n"
                f"Available formats: {_avail}"
            ) from err

        log.remark("Exporting %s as %s ...", g, export_format)

        file_ext = specs.pop("file_ext", export_format)
        out_path = f"{os.path.splitext(out_path)[0]}.{file_ext}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer(g, out_path, **specs)
