"""networkx-related utility functions"""

import copy
import logging
import os
from typing import Callable, Dict, Sequence, Union

from .._dag_utils import parse_dag_minimal_syntax as _parse_dag_minimal_syntax
from .._dag_utils import parse_dag_syntax as _parse_dag_syntax
from ..data_ops import apply_operation as _apply_operation
from ..data_ops import is_operation as _is_operation
from ..exceptions import *

log = logging.getLogger(__name__)

ATTR_MAPPER_OP_PREFIX = "attr_mapper"
"""A prefix used for registring attribute mapping data operations"""

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


def map_attributes(
    g: "networkx.Graph", kind: str, mappers: Dict[str, Union[str, dict]]
):
    """Maps attributes of nodes or edges (specified by ``kind``).

    Args:
        g (networkx.Graph): Graph object to map the node or edge attributes of
        kind (str): Name of a valid graph iterator, e.g. ``nodes``, ``edges``
        mappings (Dict[str, Union[str, Callable]]): The mappings dict.
            Will set node attributes that have as their value the result of a
            single data operation. The dict values can either be the name of a
            registered data operation or a dict that defines an operation and
            the corresponding arguments, supporting the
            :ref:`typical DAG syntax <dag_minimal_syntax>`.

            .. note::

                Note that the operation needs to be part of an extended
                dantro operations database. It may *not* be a
                meta-operation and can also not be a *sequence* of
                operations. The operation will always get node's or edge's
                existing ``attrs`` dict as a keyword argument. The return value
                of the operation is used as the new attribute value.
    """
    # Register some custom data operations
    _prefix = ATTR_MAPPER_OP_PREFIX

    @_is_operation(f"{_prefix}.copy_from_attr", skip_existing=True)
    def copy_from_attr(attr_to_copy_from: str, *, attrs: dict):
        return copy.copy(attrs[attr_to_copy_from])

    # .........................................................................

    def parse_op_params(p: Union[str, dict]) -> dict:
        """Parses operation parameters using the usual DAG syntax"""
        p = _parse_dag_minimal_syntax(p, with_previous_result=False)
        return _parse_dag_syntax(**p)

    # Prepare the mappers
    mappers = mappers if mappers else {}
    mappers = {
        attr: parse_op_params(op_params)
        for attr, op_params in mappers.items()
        if op_params is not None
    }

    # Prepare the node or edge iterator and then start iterating ...
    obj_it = getattr(g, kind)(data=True)

    for obj_and_attrs in obj_it:
        attrs = obj_and_attrs[-1]

        for target_attr, op_params in mappers.items():
            try:
                attrs[target_attr] = _apply_operation(
                    op_params["operation"],
                    *op_params["args"],
                    attrs=attrs,
                    **op_params["kwargs"],
                )

            except BadOperationName as err:
                raise BadOperationName(
                    f"Failed mapping {kind}' attributes due to an invalid "
                    f"operation name. Use operations prefixed with {_prefix} "
                    f"for common attribute mapping tasks.\n\n{err}"
                ) from err

            except Exception as exc:
                _op_name = op_params["operation"]
                raise type(exc)(
                    f"Failed mapping {kind}' attributes:\n"
                    f"Make sure the data operation ({_op_name}) and arguments "
                    "are valid and inspect the chained traceback for more "
                    f"information.\n\nGot a {type(exc).__name__}: {exc}"
                ) from exc


def export_graph(
    g: "networkx.Graph",
    *,
    out_path: str,
    map_node_attrs: Dict[str, Union[str, Callable]] = None,
    map_edge_attrs: Dict[str, Union[str, Callable]] = None,
    keep_node_attrs: Union[bool, Sequence[str]] = True,
    keep_edge_attrs: Union[bool, Sequence[str]] = True,
    **export_specs,
):
    """Takes care of exporting a networkx graph object using one or many of the
    ``nx.write_`` methods.

    Allows some pre-processing or node and edge attributes.

    Args:
        g (networkx.Graph): The graph to export
        out_path (str): Path to export it to; extensions will be dropped and
            replaced by the corresponding export format. Add the ``file_ext``
            key to a export format specification to set it to some other value.
        map_node_attrs (Dict[str, Union[str, Callable]], optional): Sets
            the node attributes given by the keys of this dict with those at
            the value. If a callable is given, is invoked with the unpacked
            dict of node attributes as arguments and writes the return value
            to the attribute given by the key.
        map_edge_attrs (Dict[str, Union[str, Callable]], optional): Sets
            the edge attributes given by the keys of this dict with those at
            the value. If a callable is given, is invoked with the unpacked
            dict of edge attributes as arguments and writes the return value
            to the attribute given by the key.
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

    # Map some attributes over
    if map_node_attrs:
        map_attributes(g, "nodes", map_node_attrs)

    if map_edge_attrs:
        map_attributes(g, "edges", map_edge_attrs)

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

        log.remark("Exporting %s using %s writer ...", g, export_format)

        file_ext = specs.pop("file_ext", export_format)
        out_path = f"{os.path.splitext(out_path)[0]}.{file_ext}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer(g, out_path, **specs)
