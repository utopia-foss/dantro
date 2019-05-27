"""Test the NetworkGroup"""

from pkg_resources import resource_filename
from typing import Union

import pytest

import networkx as nx
import networkx.exception
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

# Import the dantro objects to test here
from dantro.base import BaseDataGroup
from dantro.containers import NumpyDataContainer, XrDataContainer
from dantro.groups import NetworkGroup, TimeSeriesGroup
from dantro.tools import load_yml


# Local paths
NW_GRP_PATH = resource_filename('tests', 'cfg/nw_grps.yml')

# Helper functions ------------------------------------------------------------


# Fixtures --------------------------------------------------------------------

@pytest.fixture()
def nw_grp_cfgs() -> dict:
    """Returns the dict of NetworkGroup configurations"""
    return load_yml(NW_GRP_PATH)

@pytest.fixture()
def nw_grps(nw_grp_cfgs) -> Union[dict, dict]:
    """Creates a NetworkGroup to be tested below"""
    grps = dict()

    for name, cfg in nw_grp_cfgs.items():
        grps[name] = NetworkGroup(name=name, attrs=cfg["attrs"])
        
        # Add nodes, edges and properties from config
        # ... if this is not one of the keys where no nodes should be added:
        # The wrong_* config entries have the nodes or edges missing.
        # The wrong_type config entries have invalid container types
        if name.startswith("grp"):

            sub_grp = grps[name].new_container('nodes', Cls=TimeSeriesGroup)
            for time in cfg['times']:
                sub_grp.new_container(str(time), Cls=XrDataContainer,
                                    data=cfg['nodes'][time])

            if name == "grp_less_edge_times":
                sub_grp = grps[name].new_container('edges', Cls=TimeSeriesGroup)
                sub_grp.new_container(str(time), Cls=XrDataContainer,
                                        data=cfg['edges'][0])
            else:
                sub_grp = grps[name].new_container('edges', Cls=TimeSeriesGroup)
                for time in cfg['times']:
                    sub_grp.new_container(str(time), Cls=XrDataContainer,
                                        data=cfg['edges'][time])

            sub_grp = grps[name].new_container('np1', Cls=TimeSeriesGroup,
                                    attrs=cfg.get('attrs_node_props', None))
            for time in cfg['times']:
                sub_grp.new_container(str(time), Cls=XrDataContainer,
                                        data=cfg['np1'][time])

            sub_grp = grps[name].new_container('ep1', Cls=TimeSeriesGroup,
                                    attrs=cfg.get('attrs_edge_props', None))
            for time in cfg['times']:
                sub_grp.new_container(str(time), Cls=XrDataContainer,
                                        data=cfg['ep1'][time])

        else:

            if name != "wrong_nodes":
                if name == "xr_wrong_type_nodes":
                    sub_grp = grps[name].new_container('nodes', Cls=TimeSeriesGroup)
                    for i in [0,1]:
                        sub_grp.new_container(str(i), Cls=XrDataContainer,
                                                data=cfg['nodes'][i])
                elif name == "xr_time_series_static_nodes_unlabelled":
                    grps[name].new_container('nodes', Cls=XrDataContainer,
                                             data=cfg['nodes'])
                else:
                    grps[name].new_container('nodes', Cls=XrDataContainer,
                                             data=cfg['nodes'], dims=['time'])

            if name != "wrong_edges":
                if name == "xr_time_series_static_edges_unlabelled":
                    grps[name].new_container('edges', Cls=XrDataContainer,
                                             data=cfg['edges'])
                else:
                    grps[name].new_container('edges', Cls=XrDataContainer,
                                         data=cfg['edges'], dims=['time'])

            grps[name].new_container('np1', Cls=XrDataContainer,
                                     data=cfg.get('np1',[]), dims=['time'],
                                     attrs=cfg.get('attrs_node_props', None))

            grps[name].new_container('ep1', Cls=XrDataContainer,
                                     data=cfg.get('ep1',[]), dims=['time'],
                                     attrs=cfg.get('attrs_edge_props', None))

    return (grps, nw_grp_cfgs)

# Tests -----------------------------------------------------------------------

def test_network_group_basics(nw_grps):
    """Test the NetworkGroup.create_graph function"""

    # Helper functions --------------------------------------------------------
    def basic_network_creation_test(nw, cfg, *, name: str):
        # Get the attributes
        attrs = cfg["attrs"]
        directed = attrs["directed"]
        parallel = attrs["parallel"]

        # Check that the network is not empty, (not) directed ...
        assert nx.is_empty(nw) == False
        assert nx.is_directed(nw) == directed

        # Check the data type of the network
        if not directed and not parallel:
            assert isinstance(nw, nx.Graph)

        elif directed and not parallel:
            assert isinstance(nw, nx.DiGraph)

        elif not directed and parallel:
            assert isinstance(nw, nx.MultiGraph)

        else:
            assert isinstance(nw, nx.MultiDiGraph)

        # Check that the nodes and edges given in the config coincide with
        # the ones stored inside of the network
        nodes = cfg["nodes"]
        edges = cfg["edges"]

        for v, n in zip(range(5), nx.nodes(nw)):
            assert v == n
        
        # Need to preprocess the case with transposed edges
        if name == "transposed_edges":
            edges = [[edges[0][i], edges[1][i]] for i,_ in enumerate(edges[0])]

        if name.startswith("grp") or name.startswith("xr_time_series"):
            if (not name.startswith("xr_time_series_static_edges")
                and (name != "grp_less_edge_times")):
                for e in edges[cfg['at_time']]:
                    assert tuple(e) in nx.edges(nw)

            elif name == "xr_time_series_static_edges_unlabelled":
                for e in edges:
                    assert tuple(e) in nx.edges(nw)

            else:
                for e in edges[0]:
                    assert tuple(e) in nx.edges(nw)

        else:    
            for e in edges:
                assert tuple(e) in nx.edges(nw)

        # Check that properties are set correctly
        if name.endswith("prop") and name.startswith("grp"):
            for prop_name in cfg['node_props']:
                for v, val in zip(range(5), cfg[prop_name+'_sorted'][cfg['at_time']]):
                    assert nw.nodes[v][prop_name] == val

            for prop_name in cfg['edge_props']:
                for e, val in zip(nw.edges, cfg[prop_name][cfg['at_time']]):
                    assert nw.edges[e][prop_name] == val

        elif name.endswith("prop"):
            for prop_name in cfg['node_props']:
                for v, val in zip(range(5), cfg[prop_name+'_sorted']):
                    assert nw.nodes[v][prop_name] == val

            for prop_name in cfg['edge_props']:
                for e, val in zip(nw.edges, cfg[prop_name]):
                    assert nw.edges[e][prop_name] == val

    # Actual test -------------------------------------------------------------
    # Get the groups and their corresponding configurations
    (grps, cfgs) = nw_grps

    for name, grp in grps.items():
        print("Testing configuration {} ...".format(name))

        # Get the config
        cfg = cfgs[name]

        # Get the attributes
        attrs = cfg['attrs']

        # test the failing cases
        if name == 'wrong_nodes':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_node_container"):
                grp.create_graph(at_time_idx=(cfg.get('at_time', None)),
                                 node_props=(cfg.get('node_props', None)),
                                 edge_props=(cfg.get('edge_props', None)))

            # Nothing else to check
            continue

        elif name == 'wrong_edges':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_edge_container"):
                grp.create_graph(at_time_idx=(cfg.get('at_time', None)),
                                 node_props=(cfg.get('node_props', None)),
                                 edge_props=(cfg.get('edge_props', None)))

            # Nothing else to check
            continue

        elif name == 'bad_edges':
            with pytest.raises(nx.exception.NetworkXError,
                               match="must be a 2-tuple, 3-tuple or 4-tuple."):
                grp.create_graph(at_time_idx=(cfg.get('at_time', None)),
                                 node_props=(cfg.get('node_props', None)),
                                 edge_props=(cfg.get('edge_props', None)))

            # Nothing else to check
            continue

        elif name.startswith("xr") and "wrong_type_" in name:
            with pytest.raises(TypeError, match="'data' has to be a "
                                          "XrDataContainer"):
                grp.create_graph(at_time_idx=(cfg.get('at_time', None)),
                                 node_props=(cfg.get('node_props', None)),
                                 edge_props=(cfg.get('edge_props', None)))

            # Nothing else to check
            continue

        elif name == "grp_less_edge_times":
            with pytest.raises(IndexError, match="Time index '10' not "
                                               "available!"):
                grp.create_graph(at_time_idx=10,
                                 node_props=(cfg.get('node_props', None)),
                                 edge_props=(cfg.get('edge_props', None)))

        # Check the regular cases
        else:
            # This should work
            nw = grp.create_graph(at_time_idx=(cfg.get('at_time', None)),
                                  node_props=(cfg.get('node_props', None)),
                                  edge_props=(cfg.get('edge_props', None)))

        # Check that the basic graph creation works
        basic_network_creation_test(nw, cfg, name=name)

def test_set_property_functions(nw_grps):
    """Test the NetworkGroup.set_node_property
            and NetworkGroup.set_edge_property function
    """
    (grps, cfgs) = nw_grps
    grp = grps['grp_prop']
    cfg = cfgs['grp_prop']

    nw = grp.create_graph(at_time_idx=1)

    # test the failing cases
    # container name does not exist
    with pytest.raises(KeyError, match="No container with name 'foo' "
                                         "available"):
        grp.set_node_property(g=nw, name='foo')

    with pytest.raises(KeyError, match="No container with name 'foo' "
                                         "available"):
        grp.set_edge_property(g=nw, name='foo')

    # trying to add from DataSet that is not marked as property
    with pytest.raises(AttributeError, match="The data in 'ep1' is not marked"
                                             " as node property!"):
        grp.set_node_property(g=nw, name='ep1', at_time_idx=0)

    with pytest.raises(AttributeError, match="The data in 'np1' is not marked"
                                             " as edge property!"):
        grp.set_edge_property(g=nw, name='np1', at_time_idx=0)

    # invalid time key
    with pytest.raises(KeyError, match="Time key '42' not available!"):
        grp.set_node_property(g=nw, name='np1', at_time=42)

    with pytest.raises(IndexError, match="Time index '42' not available!"):
        grp.set_edge_property(g=nw, name='ep1', at_time_idx=42)

    # at_time and at_time_idx both given
    with pytest.raises(ValueError, match="'at_time' and 'at_time_idx' can not"
                                            " be given at the same time!"):
        grp.set_node_property(g=nw, name='np1', at_time=0, at_time_idx=0)

    # test sorting
    grp.set_node_property(g=nw, name='np1', at_time_idx=1)
    for v, val in zip(range(5), cfg['np1_sorted'][1]):
        assert nw.nodes[v]['np1'] == val

    # this should work
    grp.set_node_property(g=nw, name='np1', at_time=0)

    # another failing case
    # lengths do not match
    nw.remove_node(4)
    with pytest.raises(ValueError, match="Mismatch! '5' property values for"
                                         " '4' nodes"):
        grp.set_node_property(g=nw, name='np1', at_time_idx=1)
    
    nw.remove_edge(0,1)
    with pytest.raises(ValueError, match="Mismatch! '5' property values for"
                                         " '3' edges"):
        grp.set_edge_property(g=nw, name='ep1', at_time_idx=1)