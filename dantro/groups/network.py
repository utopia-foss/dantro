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
    _NWG_attr_is_node_attribute = "is_node_attribute"
    _NWG_attr_is_edge_attribute = "is_edge_attribute"

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (NumpyDataContainer,)

    def __init__(self, *, name: str,  containers: list=None, **kwargs):
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
                     with_node_attributes: bool=False,
                     with_edge_attributes: bool=False,
                     **graph_kwargs) -> nx.Graph:
        """Create a networkx graph object.
        
        Args:
            directed (bool, optional): The graph is directed. If not given, the
                value given by the group attribute with name _NWG_attr_directed
                is used instead.
            parallel_edges (bool, optional): If true, the graph will allow
                parallel edges. If not given, the 
            with_node_attributes (bool, optional): Additionally load the node
                attributes into the graph by looking at those containers in
                this group that have the attribute with name given by class
                variable _NWG_attr_is_node_attribute set to True.
            with_edge_attributes (bool, optional): Additionally load the edge
                attributes into the graph by looking at those containers in
                this group that have the attribute with name given by class
                variable _NWG_attr_is_edge_attribute set to True.
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

        # Set node properties
        if with_node_attributes:
            self.set_node_attributes(g)          

        # Set edge properties
        if with_edge_attributes:
            self.set_edge_attributes(g)

        # Return the graph
        return g


    def set_node_attributes(self, g: nx.Graph, *,
                            from_containers: List[NumpyDataContainer]=None):
        """Sets the node attributes of the given graph using the containers in
        this group.
        
        Which containers are used for node properties is determined by their
        container attribute; they have to have a boolean attribute set that has
        the name that is specified in the class variable
        _NWG_attr_is_node_attribute.
        
        CAREFUL: This assumes that the groups node container is ordered in the
                 same way as the container that the attributes are read from is
                 ordered. The attribute values are not selected by node ID, but
                 just by their order within the container!
                 As a consistency check, the shape of the node container and
                 the attribute data is compared; it needs to be the same to
                 not raise an error.
                 Additionally, the node container shape is checked against the
                 number of nodes in the graph.
        
        Args:
            g (nx.Graph): The graph object to set the node attributes of
            from_containers (List[NumpyDataContainer], optional): If given,
                these containers are used instead of those in this group. This
                argument can also be used to select only a subset of the
                containers in this group.
        
        Raises:
            ValueError: If node container size did not match number of nodes or
                node container shape did not match attribute data shape.
        """
        # Get the node container to associate node IDs
        node_cont = self[self._NWG_node_container]

        # Make sure the network did not change the number of vertices
        if g.number_of_nodes() != node_cont.shape[0]:
            raise ValueError("Number of nodes ({}) is not the same as it was "
                             "when constructing the graph ({}). Refusing to "
                             "add node attributes, because the association "
                             "of node IDs and attributes might be wrong!"
                             "".format(g.number_of_nodes(),
                                       node_cont.shape[0]))

        # Will need to select by attribute if all containers in this group are
        # to be used
        select_by_attr = bool(from_containers is None)
        
        # Determine in which containers to look for attributes
        conts = from_containers if from_containers else self.values()

        # Now gather those containers that may be used for node attributes by
        # going through the list of containers. If from_containers was _not_
        # given, only containers that have the required attribute set are
        # selected for extraction of attributes.
        log.debug("Gathering containers for use as node attributes ...")

        attrs = {cont.name: cont for cont in conts
                 if ((select_by_attr
                      and cont.attrs.get(self._NWG_attr_is_node_attribute))
                     or not select_by_attr)}

        log.debug("Setting node attributes from %d containers ...", len(attrs))

        for name, cont in attrs.items():
            # Make a consistency check: node container and attribute container
            # data match in shape
            if node_cont.shape != cont.shape:
                raise ValueError("Node container and {} from which attribute "
                                 "data is meant to be read do not match in "
                                 "shape: {} != {} . Refusing to add these "
                                 "attributes to the nodes as the association "
                                 "to nodes might be wrong."
                                 "".format(cont.logstr,
                                           node_cont.shape, cont.shape))

            # Create a dictionary with node as key and attribute data as value
            attr_dict = {node_id: data
                         for node_id, data in zip(node_cont, cont.data)}

            # Set the attributes using this dictionary
            nx.set_node_attributes(g, attr_dict, name=name)


    def set_edge_attributes(self, g: nx.Graph):
        """Set the edge attributes of the given graph using the containers in
        this group.
        
        Which containers are used for edge attributes is determined by their
        container attribute; they have to have a boolean attribute set that has
        the name that is specified in class variable
        _NWG_attr_is_edge_attribute.
        
        NOTE: In the case of multigraphs edges can be parallel. Thus, it is not
        sufficient any more to characterize them via their source and target.
        An additional edge ID is necessary to correctly set edge attributes.
        Further, in the case of unparallel edges, networkx does not keep the
        order of the edges internally in the construction of the graph object.
        That is why adding edge attributes to the correct graph cannot be
        trivially done _after_ the graph has been initialized.
        
        Due to these complications, this method is currently not implemented!
        """
        raise NotImplementedError("Adding edge attributes to multigraph "
                                  "objects is not yet implemented!")
