"""In this module, the NetworkGroup is implemented"""

import logging

import networkx as nx

from ..base import BaseDataGroup
from ..containers import NumpyDataContainer

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class NetworkGroup(BaseDataGroup):
    """The NetworkGroup class manages groups of network data containers."""

    # Define, as class variables, in which containers or attributes to find the
    # info on the nodes and edges.
    _NWG_node_container = "nodes"
    _NWG_edge_container = "edges"
    _NWG_attr_directed = "directed"
    _NWG_attr_parallel = "parallel"
    _NWG_attr_is_node_property = "is_node_property"
    _NWG_attr_is_edge_property = "is_edge_property"

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (NumpyDataContainer,)

    def __init__(self, *, name: str,  containers: list=None, **kwargs):
        """Initialize a NetworkGroup from the list of given containers.
        
        Args:
            name (str): The name of this group
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("NetworkGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # Done.
        log.debug("NetworkGroup.__init__ finished.")


    def create_graph(self, *, directed: bool=None, parallel_edges: bool=None,
                     with_node_properties: bool=False,
                     with_edge_properties: bool=False,
                     **graph_kwargs):
        """Create a networkx graph object.
        
        Args:
            directed (bool, optional): The graph is directed.
            parallel_edges (bool, optional): The graph allows for parallel edges.
            with_node_properties (bool, optional): Create the graph with node
                properties.
            with_edge_properties (bool, optional): Create the graph with edge
                properties.
            **graph_kwargs: Further initialisation kwargs for the graph.
        """

        log.debug("Create a networkx graph.")

        # Check whether the node and edge containers are available
        log.debug("Check whether the node and edge container are available.")
        node_cont = self[self._NWG_node_container]
        edge_cont = self[self._NWG_edge_container]

        # Get info on directed and parallel edges from attributes, if not
        # explicitly given
        if directed is None:
            directed = self.attrs[self._NWG_attr_directed]
        
        if parallel_edges is None:
            parallel_edges = self.attrs[self._NWG_attr_parallel]

        # Create a networkx graph corresponding to the graph properties.
        log.debug("Create a networkx graph object.")
        if not directed and not parallel_edges:
            g = nx.Graph(**graph_kwargs)
        elif directed and not parallel_edges:
            g = nx.DiGraph(**graph_kwargs)
        elif not directed and parallel_edges:
            g = nx.MultiGraph(**graph_kwargs)
        else:
            g = nx.MultiDiGraph(**graph_kwargs)

        # Add nodes and edges to the graph
        log.debug("Add nodes and edges to the graph.")
        g.add_nodes_from(node_cont)
        g.add_edges_from(edge_cont)

        # Set node properties
        if with_node_properties:
            self.set_node_properties(g)          

        # Set edge properties
        if with_edge_properties:
            self.set_edge_properties(g)

        # Return the graph
        return g


    def set_node_properties(self, g):
        # Get the node container
        node_cont = self[self._NWG_node_container]

        # Gather additional containers that could be used as node attributes
        log.debug("Gather node attribute container.")

        node_props = {name: cont for name, cont in self.items()
                     if cont.attrs.get(self._NWG_attr_is_node_property)}

        log.debug("Set node properties.")

        for name, cont in node_props.items():
            # Create a dictionary with the node as key 
            # and the property as value
            props = {n: p for n, p in zip(node_cont, cont.data)}
            nx.set_node_attributes(g, props, name=name)


    def set_edge_properties(self, g):
        # In the case of multigraphs edges can be parallel, thus, it is not
        # sufficient any more to characterize them via their source and target.
        # An additional edge key is necessary to correctly set edge attributes.
        # Further, in the case of unparallel edges networkx does not keep the
        # order of the edges internally in the construction of the graph object.
        # That is why adding edge properties to the correct graph cannot be
        # trivially done after the graph initialization.
        # NOTE: Adding edges is not yet implemented due to the complications 
        #       mentioned directly above.
        raise NotImplementedError("Adding edge properties to multigraph "
                                    "objects is not yet implemented!")