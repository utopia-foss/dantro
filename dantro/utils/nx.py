"""networkx-related utility functions"""

import copy
import logging
import os
from typing import Any, Callable, Dict, Sequence, Union

from .._dag_utils import parse_dag_minimal_syntax as _parse_dag_minimal_syntax
from .._dag_utils import parse_dag_syntax as _parse_dag_syntax
from ..data_ops import apply_operation as _apply_operation
from ..data_ops import is_operation as _is_operation
from ..exceptions import *

log = logging.getLogger(__name__)

ATTR_MAPPER_OP_PREFIX = "attr_mapper"
"""A prefix used for registring attribute mapping data operations"""

ATTR_MAPPER_OP_PREFIX_DAG = f"{ATTR_MAPPER_OP_PREFIX}.dag"
"""A prefix used for registring attribute mapping data operations that are
specialized for use in the DAG, e.g. in
:py:meth:`dantro.dag.TransformationDAG.generate_nx_graph`.
"""

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
                _prefix = ATTR_MAPPER_OP_PREFIX
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


def manipulate_attributes(
    g: "networkx.Graph",
    *,
    map_node_attrs: Dict[str, Union[str, Callable]] = None,
    map_edge_attrs: Dict[str, Union[str, Callable]] = None,
    keep_node_attrs: Union[bool, Sequence[str]] = True,
    keep_edge_attrs: Union[bool, Sequence[str]] = True,
):
    """Manipulates the given graph's edge and/or node attributes

    Args:
        g (networkx.Graph): The graph the node and edge attributes of which are
            to be manipulated
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
    """
    if map_node_attrs:
        map_attributes(g, "nodes", map_node_attrs)

    if map_edge_attrs:
        map_attributes(g, "edges", map_edge_attrs)

    # Keep only certain node and/or edge attributes
    if keep_node_attrs is not True:
        keep_node_attrs = keep_node_attrs if keep_node_attrs else ()
        keep_node_attributes(g, *(keep_node_attrs if keep_node_attrs else ()))

    if keep_edge_attrs is not True:
        keep_edge_attrs = keep_edge_attrs if keep_edge_attrs else ()
        keep_edge_attributes(g, *(keep_edge_attrs if keep_edge_attrs else ()))


def export_graph(
    g: "networkx.Graph",
    *,
    out_path: str,
    manipulate_attrs: dict = None,
    **export_specs,
):
    """Takes care of exporting a networkx graph object using one or many of the
    ``nx.write_`` methods. See the
    `networkx documentation <https://networkx.org/documentation/stable/reference/readwrite/>`_
    for available output formats.

    This also allows some pre-processing or node and edge attributes using the
    :py:func:`.manipulate_attributes` function.

    Example:

    .. code-block:: yaml

        manipulate_attrs: {}

        # Export formats
        dot: true  # needs graphviz
        graphml:
          file_ext: gml
          # ... further kwargs passed to writer
        gml: false

    Args:
        g (networkx.Graph): The graph to export
        out_path (str): Path to export it to; extensions will be dropped and
            replaced by the corresponding export format. Add the ``file_ext``
            key to a export format specification to set it to some other value.
        manipulate_attrs (dict, optional): If given, is passed to
            :py:func:`~dantro.utils.nx.manipulate_attributes` to manipulate the
            node and/or edge attributes of a (copy of) the given graph ``g``.
        **export_specs: Keys need to correspond to valid ``nx.write_*``
            function names, values are passed on to the write function. There
            are two special keys ``enabled`` and ``file_ext`` that can control
            the behaviour of the respective export operation.
            Alternatively, values can be a boolean that enables or disables
            the writer.
    """
    import networkx as nx

    NX_WRITERS = {
        f[6:]: getattr(nx, f) for f in dir(nx) if f.startswith("write_")
    }
    try:
        import nx.drawing.nx_agraph

        NX_WRITERS["dot"] = nx.drawing.nx_agraph.write_dot

    except ImportError:
        pass

    if manipulate_attrs:
        # Need to work on a copy because certain attributes may be dropped
        g = copy.deepcopy(g)
        manipulate_attributes(g, **manipulate_attrs)

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


# -----------------------------------------------------------------------------
# -- Custom attribute mappers -------------------------------------------------
# -----------------------------------------------------------------------------
# These rely on ``attrs["obj"]`` containing the associated node object.


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX}.copy_from_attr")
def copy_from_attr(attr_to_copy_from: str, *, attrs: dict):
    """:py:func:`Attribute mapper operation <map_attributes>` that copies an
    attribute by name.
    """
    return copy.copy(attrs[attr_to_copy_from])


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX}.set_value")
def set_value(value: Any, *, attrs: dict):
    """:py:func:`Attribute mapper operation <map_attributes>` that simply sets
    a value, regardless of other attributes.
    """
    return value


# .. Specializations for use within the DAG ...................................


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_operation")
def get_operation(*, attrs: dict) -> str:
    """:py:func:`Attribute mapper operation <map_attributes>` that returns the
    transformation's operation name.
    See :py:attr:`dantro.dag.Transformation.operation`.

    Used in :ref:`dag_graph_vis`.
    """
    obj = attrs["obj"]
    return getattr(obj, "operation", "")


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_meta_operation")
def get_meta_operation(*, attrs: dict) -> str:
    """:py:func:`Attribute mapper operation <map_attributes>` that returns the
    transformation's meta-operation name, *if* it was added as part of a meta-
    operation. Otherwise returns an empty string. This information stems from
    the :py:attr:`dantro.dag.Transformation.context` attribute.

    Used in :ref:`dag_graph_vis`.
    """
    obj = attrs["obj"]
    return getattr(obj, "context", {}).get("meta_operation", "")


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.format_arguments")
def format_arguments(*, attrs: dict, join_str: str = "\n") -> str:
    """:py:func:`Attribute mapper operation <map_attributes>` that formats
    node arguments in a nice and readable way.

    Used in :ref:`dag_graph_vis`.
    """
    obj = attrs["obj"]
    if not hasattr(obj, "_args") or not hasattr(obj, "_kwargs"):
        return ""

    _args = join_str.join(f"{v}" for _, v in enumerate(obj._args))
    _kwargs = join_str.join(f"{k}: {v} " for k, v in obj._kwargs.items())
    return f"args: {_args}\nkwargs: {_kwargs}"
    # TODO Improve, e.g. by hiding reference hashes


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_layer")
def get_layer(*, attrs: dict) -> int:
    """:py:func:`Attribute mapper operation <map_attributes>` that returns the
    transformation's layer value.
    See :py:attr:`dantro.dag.Transformation.layer`.

    Used in :ref:`dag_graph_vis`.
    """
    obj = attrs["obj"]
    return getattr(obj, "layer", 0)


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_status")
def get_status(*, attrs: dict) -> str:
    """:py:func:`Attribute mapper operation <map_attributes>` that returns the
    transformation's status value.
    See :py:attr:`dantro.dag.Transformation.status`.

    Used in :ref:`dag_graph_vis`.
    """
    obj = attrs["obj"]
    return getattr(obj, "status", "none")


@_is_operation(f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_description")
def get_description(
    *,
    attrs: dict,
    join_str: str = "\n",
    to_include: list = ("operation", "meta_operation", "tag", "result"),
    abbreviate_result: int = 12,
) -> str:
    """:py:func:`Attribute mapper operation <map_attributes>` that creates a
    description string from the transformation.

    Used in :ref:`dag_graph_vis`.

    Args:
        attrs (dict): Node attributes dict
        join_str (str, optional): How to join the individual segments together
        to_include (list, optional): Which segments to include.
            Can be ``'all'`` or a sequence of keys referring to individual
            segments. Available segments:

                - ``operation``
                - ``meta_operation``
                - ``tag``
                - ``result`` (if available)
                - ``status`` (if available)

            Note that the order is also given by the order in this list.
    """
    obj = attrs["obj"]
    tag = attrs.get("tag")

    # Operation
    op = getattr(obj, "operation", "")

    # Result
    result_str = ""
    if attrs.get("has_result"):
        result = attrs.get("result")
        result_str = str(result)

        if abbreviate_result:
            # Use only the first line
            result_str = result_str.split("\n")[0]

            # If there are more characters, abbreviate to type
            if len(result_str) > abbreviate_result:
                result_str = str(type(result).__name__)

    # Status
    status = getattr(obj, "status", "none").replace("_", " ")

    # Meta-operation this node may have been part of
    meta_op = getattr(obj, "context", {}).get("meta_operation")

    # Assemble, evaluate and join descriptions
    desc_specs = dict(
        operation=dict(fstr="{}", content=op),
        meta_operation=dict(fstr="({})", content=meta_op),
        result=dict(fstr="= {}", content=result_str),
        tag=dict(fstr="— {} —", content=tag),
        status=dict(fstr="status: {}", content=status),
    )
    if to_include == "all":
        to_include = list(desc_specs.keys())

    return join_str.join(
        desc_specs[name]["fstr"].format(desc_specs[name]["content"])
        for name in to_include
        if desc_specs[name]["content"]
    )
