"""Test the NetworkGroup"""

from pkg_resources import resource_filename
from typing import Union

import pytest

import networkx as nx
import networkx.exception
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

# Import the dantro objects to test here
from dantro.containers import NumpyDataContainer
from dantro.groups import NetworkGroup
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
        
        # Add nodes and edges from config
        # ... if this is not one of the keys where no nodes should be added:
        # The wrong_* config entries have the nodes or edges missing.
        if name != "wrong_nodes":
            grps[name].new_container('nodes', Cls=NumpyDataContainer,
                                     data=cfg['nodes'])

        if name != "wrong_edges":
            grps[name].new_container('edges', Cls=NumpyDataContainer,
                                     data=cfg['edges'])

    return (grps, nw_grp_cfgs)

# Tests -----------------------------------------------------------------------

def test_NetworkGroup(nw_grps):
    """Test the NetworkGroup"""

    # Helper functions ........................................................
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

        for v in nodes:
            assert v in nx.nodes(nw)
        
        # Need to preprocess the case with transposed edges
        if name == "transposed_edges":
            edges = [[edges[0][i], edges[1][i]] for i,_ in enumerate(edges[0])]

        for e in edges:
            assert tuple(e) in nx.edges(nw)

    # Actual test .............................................................
    # Get the groups and their corresponding configurations
    (grps, cfgs) = nw_grps

    for name, grp in grps.items():
        print("Testing configuration {} ...".format(name))

        # Get the config
        cfg = cfgs[name]

        # Get the attributes
        attrs = cfg['attrs']

        ### Case: Graph without any node or edge attributes
        # Create the graph without any node or edge attributes
        # Check the regular cases
        if name not in ['wrong_nodes', 'wrong_edges', 'bad_edges']:
            # This should work
            nw = grp.create_graph()

        # Also test the failing cases
        elif name == 'wrong_nodes':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_node_container"):
                grp.create_graph()

            # Nothing else to check
            continue

        elif name == 'wrong_edges':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_edge_container"):
                grp.create_graph()

            # Nothing else to check
            continue

        elif name == 'bad_edges':
            with pytest.raises(nx.exception.NetworkXError,
                               match="must be a 2-tuple, 3-tuple or 4-tuple."):
                grp.create_graph()

            # Nothing else to check
            continue

        # Check that the basic graph creation works
        basic_network_creation_test(nw, cfg, name=name)
