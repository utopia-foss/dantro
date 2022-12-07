"""In this module, the :py:class:`~dantro.groups.graph.GraphGroup` is
implemented, which provides an interface between hierarchically stored data
and the creation of graph objects using the networkx package and the therein
implemented :py:class:`networkx.Graph` classes.

See :ref:`data_structures_graph_group` for more information.
"""

import logging
import warnings
from typing import List, Union

import numpy as np

from .._import_tools import LazyLoader
from ..containers import XrDataContainer
from ..tools import recursive_update
from . import BaseDataGroup, is_group
from .labelled import LabelledDataGroup

# Local constants
log = logging.getLogger(__name__)

# Lazy imports for packages that take a long time to import
nx = LazyLoader("networkx")
xr = LazyLoader("xarray")


# -----------------------------------------------------------------------------


@is_group
class GraphGroup(BaseDataGroup):
    """The GraphGroup class manages groups of graph data containers and
    provides the possibility to create networkx graph objects using the data
    inside this group.

    See :ref:`data_structures_graph_group` for more information.
    """

    # Define allowed member container types
    _ALLOWED_CONT_TYPES = (XrDataContainer, LabelledDataGroup)

    # Define, as class variables, in which containers or attributes to find the
    # info on the nodes and edges.
    _GG_node_container = "nodes"
    _GG_edge_container = "edges"
    _GG_attr_directed = "directed"
    _GG_attr_parallel = "parallel"
    _GG_attr_edge_container_is_transposed = "edge_container_is_transposed"
    _GG_attr_keep_dim = "keep_dim"

    # Define whether warning is raised upon bad alignment of property data
    _GG_WARN_UPON_BAD_ALIGN = True

    # .........................................................................

    def __init__(self, *args, **kwargs):
        """Initialize a GraphGroup.

        Args:
            *args: passed to :py:meth:`dantro.base.BaseDataGroup.__init__`
            **kwargs: passed to :py:meth:`dantro.base.BaseDataGroup.__init__`
        """
        super().__init__(*args, **kwargs)
        self._property_maps = dict()

    @property
    def property_maps(self) -> dict:
        """The property maps associated with this group, keyed by name."""
        return self._property_maps

    @property
    def node_container(self):
        """Returns the associated node container of this graph group"""
        try:
            return self[self._GG_node_container]

        except KeyError as err:
            raise KeyError(
                f"No container with name '{self._GG_node_container}' "
                f"available in {self.logstr}! Check if the class variable "
                "_GG_node_container is set to the correct value."
            ) from err

    @property
    def edge_container(self):
        """Returns the associated edge container of this graph group"""
        try:
            return self[self._GG_edge_container]

        except KeyError as err:
            raise KeyError(
                f"No container with name '{self._GG_edge_container}' "
                f"available in {self.logstr}! Check if the class variable "
                "_GG_edge_container is set to the correct value."
            ) from err

    @property
    def default_keep_dim(self):
        """The default dimensions not to be squeezed during data selection as
        specified in the respective group attribute.
        """
        if self._GG_attr_keep_dim in self.attrs:
            return tuple(self.attrs[self._GG_attr_keep_dim])
        return None

    # .........................................................................

    def _get_item_or_pmap(self, key: Union[str, List[str]]):
        """Returns the object accessible via ``key``. Apart from allowing to
        retrieve objects in this group, the method additionally allows to
        access data stored in property maps.

        Args:
            key (Union[str, List[str]]): The object to retrieve.
                If this is a path, will recurse down until at the end.

        Returns:
            The object at ``key``

        Raises:
            KeyError: If no such key can be found
        """
        # First, look for the key in the property maps
        if key in self._property_maps:
            return self._property_maps[key]

        # Else, invoke the parent method
        try:
            return super().__getitem__(key=key)

        except KeyError as err:
            _available_keys = [str(k) for k in self.keys()]
            _available_keys += [str(k) for k in self._property_maps.keys()]
            _available_keys = ", ".join(_available_keys)

            raise KeyError(
                f"No key, key sequence, or property '{key}' in {self.logstr}! "
                f"Available keys at top level: {_available_keys}"
            ) from err

    def _get_data_at(
        self,
        *,
        data: Union[XrDataContainer, LabelledDataGroup],
        sel: dict = None,
        isel: dict = None,
        at_time: int = None,
        at_time_idx: int = None,
        keep_dim=None,
    ) -> Union["xarray.DataArray", XrDataContainer]:
        """Returns a :py:class:`xarray.DataArray` containing the data
        specified via the selectors ``sel`` and ``isel``. Any dimension of
        size 1 is removed from the selected data.

        .. warning::

            Any invalid key in ``sel`` and ``isel`` is ignored silently.

        Args:
            data (Union[XrDataContainer, LabelledDataGroup]): Data to select
                from.
            sel (dict, optional): Dict of coordinate values keyed by
                dimensions, passed to ``data.sel``. Used to select data via
                index label. May be given together with ``isel`` if no key
                exists in both.
            isel (dict, optional): Dict of indexes keyed by dimensions,
                passed to ``data.isel``. Used to select data via index.
                May be given together with ``sel`` if no key exists in both.
            at_time (int, optional): Select along ``time`` dimension via index
                label. Translated to ``sel = dict(time=at_time)``, potentially
                overwriting an existing ``time`` entry.
            at_time_idx (int, optional): Select along ``time`` dimension via
                index. Translated to ``isel = dict(time=at_time_idx)``,
                potentially overwriting an existing ``time`` entry.
            keep_dim (optional): Iterable containing names of the dimensions
                that can not be squeezed.

        Returns:
            xarray.DataArray: The selected data

        Raises:
            ValueError: On keys that exist in both ``sel`` and ``isel``
        """
        # Update `sel` and `isel` with the explicit time specifications
        if at_time is not None:
            sel = recursive_update(sel if sel else {}, dict(time=at_time))

        if at_time_idx is not None:
            isel = recursive_update(
                isel if isel else {}, dict(time=at_time_idx)
            )

        # Check that no key is in both sel and isel
        if sel and isel:
            # Check that the intersection of the key-sets is empty
            if sel.keys() & isel.keys():
                _duplicate_keys = ", ".join(sel.keys() & isel.keys())
                raise ValueError(
                    "Received keys that appear in both `sel` and `isel` for "
                    f"the selection of {data}: {_duplicate_keys}"
                )

        log.debug(
            "Received the following selectors:\n  sel: %s\n  isel: %s",
            sel,
            isel,
        )
        log.debug("Now applying these to the following data ...\n  %s", data)

        # Get the dimensions available for selection
        dims = {d for d in data.dims}
        # If `data` is a group, extract the member level dimensions. It is
        # sufficient to look at a single group member, since dimensions along
        # which is selected have to be shared among all group members.
        if isinstance(data, BaseDataGroup):
            dims |= {d for d in data[next(iter(data))].dims}

        # Remove any entries for which no matching dimension exists and apply
        # sel _and_ isel (if given).
        if sel is not None:
            sel = {k: v for k, v in sel.items() if k in dims}
            data = data.sel(**sel)

        if isel is not None:
            isel = {k: v for k, v in isel.items() if k in dims}
            data = data.isel(**isel)

        log.debug(
            "Applied the following selectors:\n  sel: %s\nisel: %s", sel, isel
        )

        # If `data` is still a group, select everything to get a `xr.DataArray`
        # with the coordinates also from the member level.
        if isinstance(data, BaseDataGroup):
            data = data.sel()

        # `data` is now a `xr.DataArray`. Squeeze out any dimension of size 1
        # that is not in `keep_dim`.
        if keep_dim is None:
            keep_dim = (
                self.default_keep_dim
                if self.default_keep_dim is not None
                else []
            )

        for dim in data.dims:
            if data.sizes[dim] == 1 and dim not in keep_dim:
                data = data.squeeze(dim=dim)

        if not data.dims:
            # Oops, no dimension left. Add one default dimension.
            data = data.expand_dims("dim_0")

        return data

    def _prepare_edge_data(self, *, edges, max_tuple_size: int):
        """Prepares the edge data. Depending on the
        ``_GG_attr_edge_container_is_transposed`` class attribute, the edge
        data is transposed or not. If the attribute does not exist, the data is
        transposed only if the correct shape could unambiguously be deduced.

        Args:
            edges: The edge data stored in a 2-dimensional container
            max_tuple_size (int): The maximum allowed edge tuple size (4 for
                :py:class:`networkx.MultiGraph`, else 3). Used if the correct
                shape is tried to be deduced automatically.

        Returns:
            The edge data, possibly transposed

        Raises:
            TypeError: Edge data is not 2-dimensional
        """
        # Check that edga data is 2-dimensional (if not empty)
        if edges.values.size != 0 and edges.values.ndim != 2:
            _msg_squeezed_dims = ""

            if edges.values.ndim == 1:
                _msg_squeezed_dims = (
                    "\nThe edge dimension might have been squeezed during "
                    f"data selection. Use the `{self._GG_attr_keep_dim}` "
                    "group attribute or the `keep_dim` argument to specify "
                    "dimensions that are not to be squeezed."
                )

            raise TypeError(
                "Edge data must be 2-dimensional. Received edge data with "
                f"{edges.values.ndim} dimension(s). Also note that the "
                f"edge-tuples must be of equal size.{_msg_squeezed_dims}"
            )

        # If information on edge container shape is given as class attribute,
        # rely on that.
        if self._GG_attr_edge_container_is_transposed in self.attrs:
            if self.attrs[self._GG_attr_edge_container_is_transposed]:
                return edges.T
            return edges

        # Transpose if needed and if the correct shape is unambiguous,
        # i.e., if the size of one dimension lies not in [2, max_tuple_size].
        elif (
            not 2 <= edges.values.shape[-1] <= max_tuple_size
            and 2 <= edges.values.shape[0] <= max_tuple_size
        ):
            return edges.T

        return edges

    def _prepare_property_data(self, name: str, data):
        """Prepares external property data.

        Args:
            name (str): The properties' name
            data: The property data

        Returns:
            The data, potentially converted to a
                :py:class: `~dantro.containers.xr.XrDataContainer`

        Raises:
            TypeError: On invalid type of ``data``
        """
        if not any([isinstance(data, t) for t in self._ALLOWED_CONT_TYPES]):
            try:
                data = XrDataContainer(name=name, data=data)
            except:
                _allowed_types = ", ".join(
                    [str(t) for t in self._ALLOWED_CONT_TYPES]
                )
                raise TypeError(
                    f"Received invalid type for 'data' argument: {type(data)}."
                    f" Expected one of: {_allowed_types} ... or a type that "
                    "could be converted to an XrDataContainer."
                )

        return data

    def _check_alignment(self, *, ent, prop):
        """Checks the alignment of property data and entity (node or edge)
        data. If ``self._GG_WARN_UPON_BAD_ALIGN`` is True, warn on possible
        pitfalls.

        Args:
            ent: The entity (node or edge) data
            prop: The property data
        """
        if not self._GG_WARN_UPON_BAD_ALIGN:
            return

        # Check if any matching dimension (of size >1) exists
        if not any([d in prop.squeeze().dims for d in ent.squeeze().dims]):
            warnings.warn(
                "No matching dimensions found in property data "
                f"('{prop.name}') and entity data ('{ent.name}') for "
                "dimensions of size >1. No re-ordering was done during "
                "alinment.",
                UserWarning,
            )

        # Check for missing values in the property data after alignment
        if prop.isnull().any():
            warnings.warn(
                f"Found missing values in property data ('{prop.name}') after "
                f"alignment with entity data ('{ent.name}'). Make sure that "
                "the coordinate values are available in the property data.",
                UserWarning,
            )

    # .........................................................................

    def register_property_map(self, key: str, data):
        """Registers a new property map. It allows for the given data to be
        accessed internally by the specified key.

        Args:
            key (str): The key via which the registered data will be available
            data: The data to be mapped. If the given data is not an allowed
                container type, an attempt is made to construct an
                :py:class:`~dantro.containers.xr.XrDataContainer` with
                the data. Only if this operation fails, will property map
                registration fail.

        Raises:
            ValueError: On invalid key
        """
        data = self._prepare_property_data(key, data)

        # Ensure that the key does not exist already, which could lead to
        # unexpected shadowing of an existing group member.
        if key in self or key in self._property_maps:
            raise ValueError(
                f"The given key '{key}' conflicts with a valid member path in "
                f"{self.logstr}. Please choose a unique name."
            )

        # Everything ok, register the new map
        self._property_maps[key] = data

        log.remark(
            "Registered tag '%s' as property map of %s.", key, self.logstr
        )

    def create_graph(
        self,
        *,
        directed: bool = None,
        parallel_edges: bool = None,
        node_props: list = None,
        edge_props: list = None,
        sel: dict = None,
        isel: dict = None,
        at_time: int = None,
        at_time_idx: int = None,
        align: bool = False,
        keep_dim=None,
        **graph_kwargs,
    ) -> "networkx.Graph":
        """Create a networkx :py:class:`networkx.Graph` (or a more specialized
        graph type) object from the node and edge data associated with this
        graph group. Optionally, node and edge properties can be added from
        data stored or registered in the graph group.
        The coordinates for the selected or squeezed dimensions of the node,
        edge, and property data are stored as graph attributes in ``g.graph``.

        .. note::

            Any pre-selection specified by ``sel``, ``isel``, ``at_time``, or
            ``at_time_idx`` will be applied to the node data, edge data, as
            well as any given property data.

        .. warning::

            Any invalid key in ``sel`` and ``isel`` is ignored silently (see
            :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`).

        Args:
            directed (bool, optional): If true, the graph will be directed.
                If not given, the value given by the group attribute with name
                ``_GG_attr_directed`` is used instead.
            parallel_edges (bool, optional): If true, the graph will allow
                parallel edges. If not given, the value is tried to be read
                from the group attribute with name ``_GG_attr_parallel``.
            node_props (list, optional): List of names specifying the
                containers that contain the node property data.
            edge_props (list, optional): List of names specifying the
                containers that contain the edge property data.
            sel (dict, optional): Dict of coordinate values keyed by
                dimensions, passed to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`.
                Used to select data via index label.
            isel (dict, optional): Dict of indexes keyed by dimensions,
                passed to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`.
                Used to select data via index.
            at_time (int, optional): Select along ``time`` dimension via index
                label. Translated to ``sel = dict(time=at_time)``.
            at_time_idx (int, optional): Select along ``time`` dimension via
                index. Translated to ``isel = dict(time=at_time_idx)``.
            align (bool, optional): If True, the property data is aligned
                with the node/edge data using ``xarray.align``
                (default: False). The indexes of the ``<node/edge>_container``
                are used for each dimension. If the class variable
                ``_GG_WARN_UPON_BAD_ALIGN`` is True, warn upon missing values
                or if no re-ordering was done. Any dimension of size 1 is
                squeezed and thus alignment (via ``align=True``) will have no
                effect on such dimensions.
            keep_dim (optional): Iterable containing names of the dimensions
                that can not be squeezed. Passed on to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`.
            **graph_kwargs: Passed to the constructor of the respective
                networkx graph object.

        Returns:
            The networkx graph object. Depending on the provided information,
            one of the following graph objects is created:
            :py:class:`networkx.Graph`, :py:class:`networkx.DiGraph`,
            :py:class:`networkx.MultiGraph`, :py:class:`networkx.MultiDiGraph`.
        """
        # Get the node and edge data stored in the graph group
        log.debug("Checking whether node and edge data is available...")

        node_cont = self._get_data_at(
            data=self.node_container,
            sel=sel,
            isel=isel,
            at_time=at_time,
            at_time_idx=at_time_idx,
            keep_dim=keep_dim,
        )
        edge_cont = self._get_data_at(
            data=self.edge_container,
            sel=sel,
            isel=isel,
            at_time=at_time,
            at_time_idx=at_time_idx,
            keep_dim=keep_dim,
        )

        # Get info on directed and parallel edges from attributes, if not
        # explicitly given
        if directed is None:
            directed = self.attrs[self._GG_attr_directed]

        if parallel_edges is None:
            parallel_edges = self.attrs[self._GG_attr_parallel]

        # Create a networkx graph corresponding to the graph properties
        log.debug("Creating a networkx graph object...")

        if not directed and not parallel_edges:
            g = nx.Graph(**graph_kwargs)

        elif directed and not parallel_edges:
            g = nx.DiGraph(**graph_kwargs)

        elif not directed and parallel_edges:
            g = nx.MultiGraph(**graph_kwargs)

        else:
            g = nx.MultiDiGraph(**graph_kwargs)

        # Prepare the edge data. If needed, the data is transposed
        edge_cont = self._prepare_edge_data(
            edges=edge_cont,
            max_tuple_size=(4 if isinstance(g, nx.MultiGraph) else 3),
        )

        # Drop missing values from the node-data and edge-data. After calling
        # `_prepare_edge_data`, the first edge-data dim should be the edge-dim.
        # This is also why the dropna kwarg can't be evaluated in _get_data_at.
        if np.isnan(node_cont).any():
            node_cont = node_cont.dropna(dim=node_cont.dims[0])
            log.debug(
                "Removed entries containg missing (NaN) values along dim '%s' "
                "from %s.",
                node_cont.dims[0],
                self.node_container.logstr,
            )

        if np.isnan(edge_cont).any():
            edge_cont = edge_cont.dropna(dim=edge_cont.dims[0])
            log.debug(
                "Removed entries containg missing (NaN) values along dim '%s' "
                "from %s.",
                edge_cont.dims[0],
                self.node_container.logstr,
            )

        # Add nodes and edges to the graph
        log.debug("Adding nodes to the graph...")
        g.add_nodes_from(node_cont.values)

        log.debug("Adding edges to the graph...")
        g.add_edges_from(edge_cont.values)

        # Store node and edge coordinates as graph attribute for dimensions
        # of size 1; they are discarded otherwise. Since the networkx graph
        # does not ensure the same edge-ordering, the edge-idx coordinates
        # should not be stored. For the sake of consistency, this is not done
        # for any of the dimensions of size > 1 (node-idx, edge-idx, edge).
        # If needed, the node-idx coordinates can be added as property.
        node_coords = {
            d: node_cont.coords[d].item()
            for d in node_cont.coords
            if node_cont.coords[d].size == 1
        }
        edge_coords = {
            d: edge_cont.coords[d].item()
            for d in edge_cont.coords
            if edge_cont.coords[d].size == 1
        }
        # The order of the dictionary update should not matter
        g.graph.update(node_coords, **edge_coords)

        # Add properties to nodes and edges
        if node_props:
            for prop_name in node_props:
                self.set_node_property(
                    g=g,
                    name=prop_name,
                    sel=sel,
                    isel=isel,
                    at_time=at_time,
                    at_time_idx=at_time_idx,
                    align=align,
                    keep_dim=keep_dim,
                )
        if edge_props:
            for prop_name in edge_props:
                self.set_edge_property(
                    g=g,
                    name=prop_name,
                    sel=sel,
                    isel=isel,
                    at_time=at_time,
                    at_time_idx=at_time_idx,
                    align=align,
                    keep_dim=keep_dim,
                )

        # Return the graph
        log.info(
            "Created graph (%d node%s, %d edge%s) from %s.",
            g.number_of_nodes(),
            "s" if g.number_of_nodes() != 1 else "",
            g.number_of_edges(),
            "s" if g.number_of_edges() != 1 else "",
            self.logstr,
        )

        return g

    def set_node_property(
        self,
        *,
        g: "networkx.Graph",
        name: str,
        data=None,
        align: bool = False,
        keep_dim=None,
        **selector,
    ):
        """Sets a property to every node in Graph ``g`` that is also in the
        ``node_container`` of the graph group. The coordinates for the selected
        or squeezed dimensions of the property data are stored as Graph
        attributes (in ``g.graph``).

        Args:
            g (networkx.Graph): The networkx graph object
            name (str): If ``data`` is ``None``, ``name`` must specify the
                container within the graph group that contains the property
                values, or be valid key in ``property_maps``. ``name`` is used
                as the name for the property in the graph object, potentially
                overwriting an existing property.
            data (None, optional): If given, load node properties directly
                from ``data``. If the given data is not an allowed container
                type, an attempt is made to construct an
                :py:class:`~dantro.containers.xr.XrDataContainer` with
                the data. Only if this operation fails, the node property
                setting will fail.
            align (bool, optional): If True, the property data is aligned
                with the node data using ``xarray.align``. The indexes of the
                ``node_container`` are used for each dimension. If the class
                variable ``_GG_WARN_UPON_BAD_ALIGN`` is True, warn upon
                missing values or if no re-ordering was done. Any dimension of
                size 1 is squeezed and thus alignment (via ``align=True``) will
                have no effect on such dimensions.
            keep_dim (optional): Iterable containing names of the dimensions
                that can not be squeezed. Passed on to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`.
            **selector: Specifies the selection applied to both node data and
                property data. Passed on to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`. Use
                the ``sel`` (``isel``) dict to select data via coordinate value
                (index).

        Raises:
            ValueError: Lenght mismatch of the selected property and node data
        """
        # Get the property data
        if data is None:
            prop_data = self._get_item_or_pmap(name)
        else:
            prop_data = self._prepare_property_data(name, data)

        prop_data = self._get_data_at(
            data=prop_data, keep_dim=keep_dim, **selector
        )

        # Get the node data
        node_cont = self._get_data_at(
            data=self.node_container, keep_dim=keep_dim, **selector
        )

        if np.isnan(node_cont).any():
            node_cont = node_cont.dropna(dim=node_cont.dims[0])
            log.debug(
                "Removed entries containg missing (NaN) values along dim '%s' "
                "from %s",
                node_cont.dims[0],
                self.node_container.logstr,
            )

        # Optionally, align the property data with the node data.
        if align:
            node_cont, prop_data = xr.align(node_cont, prop_data, join="left")
            self._check_alignment(ent=node_cont, prop=prop_data)

        # Check if the data can be added as node property
        if len(prop_data) != len(node_cont):
            # Prepare error message
            _msg_squeezed_dims = ""
            _msg_match_with_node_number = ""

            if len(g.nodes) == 1:
                _msg_squeezed_dims = (
                    "\n"
                    "The node dimension in the property data might have been "
                    "squeezed. Use the `keep_dim` argument to specify "
                    "dimensions not to be squeezed."
                )

            if len(g.nodes) == len(prop_data):
                _msg_match_with_node_number = (
                    "\n"
                    f"The size of '{name}' matches the current number of "
                    "nodes, which changed since the graph creation. However, "
                    "properties can only be added to those nodes that are in "
                    f"{self.node_container.logstr}. Alternatively, add them "
                    "manually using nx.set_node_attributes."
                )

            raise ValueError(
                f"Length mismatch: Failed to add '{name}' data as a node "
                f"property. Received {len(prop_data)} property values for "
                f"{len(node_cont)} nodes in {self.node_container.logstr}! "
                f"{_msg_match_with_node_number}{_msg_squeezed_dims}"
            )

        n_data = len(node_cont)
        n_graph = len(g.nodes)

        if n_data != n_graph:
            warnings.warn(
                "The number of nodes "
                f"{'decreased' if n_graph < n_data else 'increased'} since "
                f"graph creation (length of {self.node_container.logstr}: "
                f"{n_data}, nodes in graph: {n_graph}). Some property values "
                "will not be added to the graph! Reasons might be previous "
                "modifications of the graph structure, duplicate entries in "
                f"'{self.node_container.logstr}', or nodes in "
                f"'{self.edge_container.logstr}' that are not in "
                f"'{self.node_container.logstr}'.",
                UserWarning,
            )

        # Create dict of properties keyed by node
        props = {n: val for n, val in zip(node_cont.values, prop_data.values)}

        # Add property to graph
        nx.set_node_attributes(g, values=props, name=name)

        # Store the coords of the selected property data for dimensions of size
        # 1 as graph attribute (potentially overwriting existing attributes).
        prop_coords = {
            d: prop_data.coords[d].item()
            for d in prop_data.coords
            if prop_data.coords[d].size == 1
        }
        g.graph.update(prop_coords)

        log.remark("Node property '%s' added from %s.", name, self.logstr)

    def set_edge_property(
        self,
        *,
        g: "networkx.Graph",
        name: str,
        data=None,
        align: bool = False,
        keep_dim=None,
        **selector,
    ):
        """Sets a property to every edge in Graph ``g`` that is also in the
        ``edge_container`` of the graph group. The coordinates for the selected
        or squeezed dimensions of the property data are stored as Graph
        attributes (in ``g.graph``).

        Args:
            g (networkx.Graph): The networkx graph object
            name (str): If ``data`` is ``None``, ``name`` must specify the
                container within the graph group that contains the property
                values, or be valid key in ``property_maps``. ``name`` is used
                as the name for the property in the graph object, potentially
                overwriting an existing property.
            data (None, optional): If given, load edge properties directly
                from ``data``. If the given data is not an allowed container
                type, an attempt is made to construct an
                :py:class:`~dantro.containers.xr.XrDataContainer` with
                the data. Only if this operation fails, the edge property
                setting will fail.
            align (bool, optional): If True, the property data is aligned
                with the edge data using ``xarray.align``. The indexes of the
                ``edge_container`` are used for each dimension. If the class
                variable ``_GG_WARN_UPON_BAD_ALIGN`` is True, warn upon
                missing values or if no re-ordering was done. Any dimension of
                size 1 is squeezed and thus alignment (via ``align=True``) will
                have no effect on such dimensions.
            keep_dim (optional): Iterable containing names of the dimensions
                that can not be squeezed. Passed on to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`.
            **selector: Specifies the selection applied to both edge data and
                property data. Passed on to
                :py:meth:`~dantro.groups.graph.GraphGroup._get_data_at`. Use
                the ``sel`` (``isel``) dict to select data via coordinate value
                (index).

        Raises:
            ValueError: Lenght mismatch of the selected property and edge data
        """

        def new_edge_key(e, edges, directed: bool):
            """Returns an unused edge key.

            Finds the last edge in `edges` that represents the same edge as
            ``e`` and return its key incremented by 1.
            If `e` is not found, it is the first being added. Thus, return 0.
            """
            e = tuple(e)

            if directed:
                match = lambda edge: e == edge[:2]
            else:
                match = lambda edge: e == edge[:2] or e == edge[:2][::-1]

            for edge in edges[::-1]:
                if match(edge):
                    return edge[2] + 1
            # First key to add, may return 0
            return 0

        # Get the property data
        if data is None:
            prop_data = self._get_item_or_pmap(name)
        else:
            prop_data = self._prepare_property_data(name, data)

        prop_data = self._get_data_at(
            data=prop_data, keep_dim=keep_dim, **selector
        )

        # Get and prepare the edge data
        edge_cont = self._get_data_at(
            data=self.edge_container, keep_dim=keep_dim, **selector
        )
        edge_cont = self._prepare_edge_data(
            edges=edge_cont,
            max_tuple_size=(4 if isinstance(g, nx.MultiGraph) else 3),
        )

        if np.isnan(edge_cont).any():
            edge_cont = edge_cont.dropna(dim=edge_cont.dims[0])
            log.debug(
                "Removed entries containg missing (NaN) values along dim '%s' "
                "from %s",
                edge_cont.dims[0],
                self.node_container.logstr,
            )

        # Optionally, align the property data with the node data.
        if align:
            edge_cont, prop_data = xr.align(edge_cont, prop_data, join="left")
            self._check_alignment(ent=edge_cont, prop=prop_data)

        # Check if the data can be added as edge property
        if len(prop_data) != len(edge_cont):
            # Prepare error message
            _msg_duplicate_edges = ""
            _msg_match_with_edge_number = ""

            if type(g) is nx.DiGraph or type(g) is nx.Graph:
                _msg_duplicate_edges = (
                    "\n"
                    f"Multiple entries in {self.edge_container.logstr} might "
                    "be interpreted as a single one, due to the chosen graph "
                    f"type: '{type(g)}'"
                )

            if len(g.edges) == len(prop_data):
                _msg_match_with_edge_number = (
                    "\n"
                    f"The size of '{name}' matches the current number of "
                    "edges, which changed since the graph creation. However, "
                    "properties can only be added to those edges that are in "
                    f"{self.edge_container.logstr}. Alternatively, add them "
                    "manually using nx.set_edge_attributes."
                )

            raise ValueError(
                f"Length mismatch! Failed to add '{name}' data as an edge "
                f"property. Received {len(prop_data)} property values for "
                f"{len(edge_cont)} edges in {self.edge_container.logstr}! "
                f"{_msg_duplicate_edges}{_msg_match_with_edge_number}"
            )

        n_data = len(edge_cont)
        n_graph = len(g.edges)

        if n_data != n_graph:
            warnings.warn(
                "The number of edges "
                f"{'decreased' if n_graph < n_data else 'increased'} since "
                f"graph creation (length of {self.edge_container.logstr}: "
                f"{n_data}, edges in graph: {n_graph}). Some property values "
                "will not be added to the graph! Reasons might be previous "
                "modifications of the graph structure or duplicate entries in "
                f"{self.edge_container.logstr}.",
                UserWarning,
            )

        # Now, prepare creation of dict of properties keyed by edge. If the
        # graph type does not allow for parallel edges, the keys (edges) must
        # be 2-tuples. Else, they must be 3-tuples, additionally containing an
        # edge key.
        # Thus, in the case of parallel edges: If edge data contains 2-tuples,
        # create the same edge keys as it was done during graph creation.
        if isinstance(g, nx.MultiGraph):
            edge_length = edge_cont.values.shape[-1]

            if edge_length == 2:
                edges = []
                for e in edge_cont:
                    edges.append(
                        (
                            int(e[0]),
                            int(e[1]),
                            new_edge_key(e, edges, g.is_directed()),
                        )
                    )

            else:
                edges = [tuple(e) for e in edge_cont.values]

        else:
            edges = [tuple(e) for e in edge_cont.values]

        # Create dict of property values keyed by edge
        props = {n: val for n, val in zip(edges, prop_data.values)}

        # Add property to graph
        nx.set_edge_attributes(g, values=props, name=name)

        # Store the coords of the selected property data for dimensions of size
        # 1 as graph attribute (potentially overwriting existing attributes).
        prop_coords = {
            d: prop_data.coords[d].item()
            for d in prop_data.coords
            if prop_data.coords[d].size == 1
        }
        g.graph.update(prop_coords)

        log.remark("Edge property '%s' added from %s.", name, self.logstr)
