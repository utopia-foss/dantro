"""In this module, the NetworkGroup is implemented"""

import logging
from typing import List

import networkx as nx
import networkx.exception

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

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (NumpyDataContainer,)


    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a NetworkGroup from the list of given containers.
        
        Args:
            name (str): The name of this group
            containers (list, optional): A list of containers to add
            **kwargs: Further initialization kwargs, e.g. `attrs` ...
        """

        log.debug("NetworkGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # Done.
        log.debug("NetworkGroup.__init__ finished.")


    def create_graph(self, *, directed: bool=None, parallel_edges: bool=None,
                     **graph_kwargs) -> nx.Graph:
        """Create a networkx graph object.
        
        Args:
            directed (bool, optional): The graph is directed. If not given, the
                value given by the group attribute with name _NWG_attr_directed
                is used instead.
            parallel_edges (bool, optional): If true, the graph will allow
                parallel edges. If not given, the value is tried to be read
                from the group attribute with name _NWG_attr_parallel.
            **graph_kwargs: Further initialization kwargs for the graph.
        
        Returns:
            nx.Graph
        
        Raises:
            KeyError: For non-existant node or edge container under the names
                specified by the respective class variables.
        """

        log.debug("Checking whether node and edge container are available...")
        
        try:
            node_cont = self[self._NWG_node_container]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in {}! "
                           "Check if the class variable _NWG_node_container "
                           "is set to the correct value."
                           "".format(self._NWG_node_container, self.logstr)
                           ) from err

        try:
            edge_cont = self[self._NWG_edge_container]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in {}! "
                           "Check if the class variable _NWG_edge_container "
                           "is set to the correct value."
                           "".format(self._NWG_edge_container, self.logstr)
                           ) from err

        # Get info on directed and parallel edges from attributes, if not
        # explicitly given
        if directed is None:
            directed = self.attrs[self._NWG_attr_directed]
        
        if parallel_edges is None:
            parallel_edges = self.attrs[self._NWG_attr_parallel]

        # Create a networkx graph corresponding to the graph properties.
        log.debug("Creating a networkx graph object...")
        
        if not directed and not parallel_edges:
            g = nx.Graph(**graph_kwargs)
            
        elif directed and not parallel_edges:
            g = nx.DiGraph(**graph_kwargs)
            
        elif not directed and parallel_edges:
            g = nx.MultiGraph(**graph_kwargs)
            
        else:
            g = nx.MultiDiGraph(**graph_kwargs)

        # Add nodes to the graph
        log.debug("Adding nodes to the graph...")
        g.add_nodes_from(node_cont)
        
        # Add edges to the graph
        log.debug("Adding edges to the graph...")
        try:
            g.add_edges_from(edge_cont)
        
        except networkx.exception.NetworkXError as err:
            # Probably the data had an unexpected shape. 
            # Just try transposing :)
            try:
                g.add_edges_from(edge_cont.T)
            
            except:
                # It doesn't work. Give up on it... :(
                raise err

        # Return the graph
        log.info("Successfully created graph (%d node%s, %d edge%s) from %s.",
                 g.number_of_nodes(), "s" if g.number_of_nodes() != 1 else "",
                 g.number_of_edges(), "s" if g.number_of_edges() != 1 else "",
                 self.logstr)

        return g
