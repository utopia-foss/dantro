"""In this module, the GraphGroup is implemented"""

import logging
from typing import List, Union

import numpy as np
import xarray as xr
import networkx as nx
import networkx.exception

from . import TimeSeriesGroup
from ..base import BaseDataGroup
from ..containers import XrDataContainer

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class GraphGroup(BaseDataGroup):
    """The GraphGroup class manages groups of graph data containers."""

    # Define allowed member container types
    _ALLOWED_CONT_TYPES = (XrDataContainer, TimeSeriesGroup)

    # Define, as class variables, in which containers or attributes to find the
    # info on the nodes and edges.
    _GG_node_container = "nodes"
    _GG_edge_container = "edges"
    _GG_attr_directed = "directed"
    _GG_attr_parallel = "parallel"
    _GG_attr_node_prop = "node_prop"
    _GG_attr_edge_prop = "edge_prop"

    # .........................................................................

    def _get_data_at(self, *, data: Union[XrDataContainer, TimeSeriesGroup],
                     at_time: int=None, at_time_idx: int=None
                     ) -> Union[xr.DataArray, XrDataContainer]:
        """Extract the data (at a certain time).
        
        Args:
            data (Union[XrDataContainer, TimeSeriesGroup]): data to select from
            at_time (int, optional): access data via time coordinate
            at_time_idx (int, optional): access data via index
        
        Returns:
            Union[xr.DataArray, XrDataContainer]: selected property data
        
        Raises:
            IndexError: selected time index not available
            KeyError: selected time coordinate not available
            TypeError: wrong container for single graph data
            ValueError: Both ``at_time`` and ``at_time_idx`` are given
        
        """
        if at_time_idx is not None and at_time is not None:
            raise ValueError("Arguments 'at_time' and 'at_time_idx' can not "
                             "be given at the same time!")

        # If `time` is not available, assume 1d data and return as it is.
        if 'time' not in data.dims:
            return data

        if at_time is not None:

            try:
                data = data.sel(time=at_time)

            except KeyError as err:
                raise KeyError("Time key '{}' not available!"
                               "".format(at_time)) from err

        elif at_time_idx is not None:

            try:
                data = data.isel(time=at_time_idx)

            except IndexError as err:
                raise IndexError("Time index '{}' not available!"
                                 "".format(at_time_idx)) from err

        else:
            if not isinstance(data, XrDataContainer):
                raise TypeError("'data' has to be a XrDataContainer"
                                ", got '{}'".format(type(data)))

        return data


    def create_graph(self, *, directed: bool=None, parallel_edges: bool=None,
                     at_time: int=None, at_time_idx: int=None,
                     node_props: list=None, edge_props: list=None,
                     **graph_kwargs) -> nx.Graph:
        """Create a networkx graph object.
        
        Args:
            directed (bool, optional): The graph is directed. If not given, the
                value given by the group attribute with name
                ``_GG_attr_directed`` is used instead.
            parallel_edges (bool, optional): If true, the graph will allow
                parallel edges. If not given, the value is tried to be read
                from the group attribute with name _GG_attr_parallel.
            at_time (int, optional): access data via time coordinate
            at_time_idx (int, optional): access data via index
            node_props (list, optional): list of names specifying the
                containers that contain the node property data
            edge_props (list, optional): list of names specifying the
                containers that contain the edge property data
            **graph_kwargs: Further initialization kwargs for the graph.
        
        Returns:
            nx.Graph
        
        Raises:
            KeyError: For non-existant node or edge container under the names
                specified by the respective class variables.
        """

        # store nodes and edges here
        node_cont = None
        edge_cont = None

        # Check types and extract data as it is
        log.debug("Checking whether node and edge container are available...")
        
        try:
            node_cont = self[self._GG_node_container]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in {}! "
                           "Check if the class variable _GG_node_container "
                           "is set to the correct value."
                           "".format(self._GG_node_container, self.logstr)
                           ) from err

        try:
            edge_cont = self[self._GG_edge_container]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in {}! "
                           "Check if the class variable _GG_edge_container "
                           "is set to the correct value."
                           "".format(self._GG_edge_container, self.logstr)
                           ) from err

        # Get node and edge data. If only data for a single time step was
        # written, 'at_time' and 'at_time_idx' are ignored.
        if ('time' in node_cont.dims and
            len(node_cont.coords['time']) == 1):
            node_cont = self._get_data_at(data=node_cont, at_time_idx=0)
        else:
            node_cont = self._get_data_at(data=node_cont, at_time=at_time,
                                            at_time_idx=at_time_idx)

        if ('time' in edge_cont.dims and
            len(edge_cont.coords['time']) == 1):
            edge_cont = self._get_data_at(data=edge_cont, at_time_idx=0)
        else:
            edge_cont = self._get_data_at(data=edge_cont, at_time=at_time,
                                            at_time_idx=at_time_idx)

        # Get info on directed and parallel edges from attributes, if not
        # explicitly given
        if directed is None:
            directed = self.attrs[self._GG_attr_directed]
        
        if parallel_edges is None:
            parallel_edges = self.attrs[self._GG_attr_parallel]

        # sort nodes
        node_cont = node_cont[np.argsort(node_cont)]

        # NOTE: Only the nodes and the node data has to be sorted as the edges
        #       are defined via the node id

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
        g.add_nodes_from(node_cont.values)
        
        # Add edges to the graph
        log.debug("Adding edges to the graph...")
        try:
            g.add_edges_from(edge_cont.values)
        
        except networkx.exception.NetworkXError as err:
            # Probably the data had an unexpected shape. 
            # Just try transposing :)
            try:
                g.add_edges_from(edge_cont.values.T)
            
            except:
                # It doesn't work. Give up on it... :(
                raise err


        # Add properties to nodes and edges
        # The properties will be sorted along the node id's
        if node_props:
            for prop_name in node_props:
                self.set_node_property(g=g, name=prop_name,
                            at_time=at_time, at_time_idx=at_time_idx)
        if edge_props:
            for prop_name in edge_props:
                self.set_edge_property(g=g, name=prop_name,
                            at_time=at_time, at_time_idx=at_time_idx)

        # Return the graph
        log.info("Successfully created graph (%d node%s, %d edge%s) from %s.",
                 g.number_of_nodes(), "s" if g.number_of_nodes() != 1 else "",
                 g.number_of_edges(), "s" if g.number_of_edges() != 1 else "",
                 self.logstr)

        return g


    def set_node_property(self, *, g: nx.Graph, name: str, at_time: int=None,
                          at_time_idx: int=None):
        """Sets a property to every node in Graph g.
        
        Args:
            g (nx.Graph): The networkx Graph to work on
            name (str): Name of the container which contains the property
                values
            at_time (int, optional): access data via time coordinate
            at_time_idx (int, optional): access data via index
        
        Raises:
            AttributeError: specified container not marked as property via
                the according class attribute
            KeyError: The name is not available in ``graph_grp`` or ``at_time``
                is not available in ``graph_grp[name]``.
            ValueError: lenght of datasets does not match
        """

        try:
            prop_data = self[name]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in"
                           " {}!".format(name, self.logstr)) from err

        if not prop_data.attrs.get(self._GG_attr_node_prop, False):
            raise AttributeError("The data in '{}' is not marked as node"
                " property! Check that it has the attribute '{}' set to"
                " True".format(name, self._GG_attr_node_prop))

        prop_data = self._get_data_at(data=prop_data, at_time=at_time,
                                        at_time_idx=at_time_idx)

        # sort data along the node id's
        node_cont = self[self._GG_node_container]

        if ('time' in node_cont.dims
            and len(node_cont.coords['time']) == 1):
            node_cont = self._get_data_at(data=node_cont, at_time_idx=0)
        else:
            node_cont = self._get_data_at(data=node_cont, at_time=at_time,
                                            at_time_idx=at_time_idx)
            
        prop_data = prop_data[np.argsort(node_cont)]

        if len(g.nodes) != len(prop_data.values):
            raise ValueError("Mismatch! '{}' property values for '{}' nodes!"
                             "".format(len(prop_data.values), len(g.nodes)))

        props = dict()

        for node, val in zip(g.nodes, prop_data.values):
            props[node] = val

        # add property to graph
        nx.set_node_attributes(g, props, name=name)
        

    def set_edge_property(self, *, g: nx.Graph, name: str, at_time: int=None,
                          at_time_idx: int=None):
        """Sets a property to every edge in Graph g.
        
        Args:
            g (nx.Graph): The networkx Graph to work on
            name (str): Name of the container which contains the property
                values
            at_time (int, optional): access data via time coordinate
            at_time_idx (int, optional): access data via index
        
        Raises:
            AttributeError: specified container not marked as property via
                the according class attribute
            KeyError: The name is not available in ``graph_grp`` or ``at_time``
                is not available in ``graph_grp[name]``.
            ValueError: dataset lengths mismatch
        """
        try:
            prop_data = self[name]

        except KeyError as err:
            raise KeyError("No container with name '{}' available in {}!"
                           "".format(name, self.logstr)) from err

        if not prop_data.attrs.get(self._GG_attr_edge_prop, False):
            raise AttributeError("The data in '{}' is not marked as edge "
                                 "property! Check that it has the attribute "
                                 "'{}' set to True"
                                 "".format(name, self._GG_attr_edge_prop))

        prop_data = self._get_data_at(data=prop_data, at_time=at_time,
                                      at_time_idx=at_time_idx)

        if len(g.edges) != len(prop_data.values):
            raise ValueError("Mismatch! '{}' property values for '{}' edges"
                             "".format(len(prop_data.values), len(g.edges)))

        props = dict()
        for edge, val in zip(g.edges, prop_data.values):
            props[edge] = val

        # add property to graph
        nx.set_edge_attributes(g, props, name=name)
