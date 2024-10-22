"""Test the GraphGroup"""

from math import isnan
from typing import Union

import networkx as nx
import networkx.exception
import numpy as np
import pytest
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

from dantro._import_tools import get_resource_path

# Import the dantro objects to test here
from dantro.base import BaseDataGroup
from dantro.containers import NumpyDataContainer, XrDataContainer
from dantro.groups import GraphGroup, LabelledDataGroup, TimeSeriesGroup
from dantro.tools import load_yml

# Local paths
GRAPH_GRP_PATH = get_resource_path("tests", "cfg/graph_grps.yml")

# Helper functions ------------------------------------------------------------


# Fixtures --------------------------------------------------------------------


@pytest.fixture()
def graph_data() -> dict:
    """Returns the graph data to compare with"""
    return dict(
        nodes=[0, 1, 2, 3, 4],
        edges=[(1, 0), (2, 0), (3, 4), (3, 4), (4, 3)],
        node_props={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        edge_props={(1, 0): 10, (2, 0): 20, (3, 4): 34, (4, 3): 43},
    )


@pytest.fixture()
def graph_grp_cfgs() -> dict:
    """Returns the dict of GraphGroup configurations"""
    return load_yml(GRAPH_GRP_PATH)


@pytest.fixture()
def graph_grps(graph_grp_cfgs) -> Union[dict, dict]:
    """Creates a GraphGroup to be tested below"""
    grps = dict()

    for name, cfg in graph_grp_cfgs.items():
        grps[name] = GraphGroup(name=name, attrs=cfg["attrs"])

        # Add nodes, edges and properties from config
        # ... if this is not one of the keys where no nodes should be added:
        # The wrong_* config entries have the nodes or edges missing.
        # The wrong_type config entries have invalid container types
        if name.startswith("grp"):

            sub_grp = grps[name].new_container("nodes", Cls=TimeSeriesGroup)
            for time in cfg["times"]:
                sub_grp.new_container(
                    str(time), Cls=XrDataContainer, data=cfg["nodes"][time]
                )

            sub_grp = grps[name].new_container("edges", Cls=TimeSeriesGroup)
            for time in cfg["times"]:
                sub_grp.new_container(
                    str(time), Cls=XrDataContainer, data=cfg["edges"][time]
                )

            sub_grp = grps[name].new_container("np", Cls=TimeSeriesGroup)
            for time in cfg["times"]:
                sub_grp.new_container(
                    str(time), Cls=XrDataContainer, data=cfg["np"][time]
                )

            sub_grp = grps[name].new_container("ep", Cls=TimeSeriesGroup)
            for time in cfg["times"]:
                sub_grp.new_container(
                    str(time), Cls=XrDataContainer, data=cfg["ep"][time]
                )

        else:

            if name != "wrong_nodes":
                grps[name].new_container(
                    "nodes",
                    Cls=XrDataContainer,
                    data=cfg["nodes"],
                    dims=["time", "node_idx"],
                )

            if name != "wrong_edges":
                grps[name].new_container(
                    "edges",
                    Cls=XrDataContainer,
                    data=cfg["edges"],
                    dims=["time", "edge_idx"],
                )

            grps[name].new_container(
                "np",
                Cls=XrDataContainer,
                data=cfg.get("np", []),
                dims=["time", "node_idx"],
            )

            grps[name].new_container(
                "ep",
                Cls=XrDataContainer,
                data=cfg.get("ep", []),
                dims=["time", "edge_idx"],
            )

    return (grps, graph_grp_cfgs)


# Tests -----------------------------------------------------------------------


def test_create_graph_function(graph_grps, graph_data):
    """Test the GraphGroup.create_graph function"""

    # Helper functions --------------------------------------------------------
    def basic_graph_creation_test(g, cfg):
        """Check that the nodes, edges, and properties were set correctly"""
        # Get the attributes
        attrs = cfg["attrs"]
        directed = attrs["directed"]
        parallel = attrs["parallel"]

        # Check that the graph is not empty, (not) directed ...
        assert nx.is_empty(g) == False
        assert nx.is_directed(g) == directed

        # Check the data type of the graph
        if not directed and not parallel:
            assert type(g) is nx.Graph

        elif directed and not parallel:
            assert type(g) is nx.DiGraph

        elif not directed and parallel:
            assert type(g) is nx.MultiGraph

        else:
            assert type(g) is nx.MultiDiGraph

        # Get the graph data to compare with
        _nodes = graph_data["nodes"]
        _edges = graph_data["edges"]
        _node_props = graph_data["node_props"]
        _edge_props = graph_data["edge_props"]

        # Check that the nodes and edges match with the given data
        assert g.number_of_nodes() == len(set(_nodes))

        for n in _nodes:
            assert n in g.nodes

        # Make sure to count the edges correctly depending on whether the graph
        # is directed and whether it allows for parallel edges.
        if not directed and not parallel:
            edge_count = []
            for i, j in _edges:
                if (j, i) not in edge_count:
                    edge_count.append((i, j))
            assert g.number_of_edges() == len(set(edge_count))

        elif directed and not parallel:
            assert g.number_of_edges() == len(set(_edges))

        else:
            assert g.number_of_edges() == len(_edges)

        for e in _edges:
            if directed:
                assert e in g.edges()
            else:
                assert e in g.edges() or e in [
                    edge[::-1] for edge in g.edges()
                ]

        # Check that properties are set correctly
        if cfg.get("node_props", False):
            for n in g.nodes:
                assert g.nodes[n]["np"] == _node_props[n]

        if cfg.get("edge_props", False):
            for e in g.edges:
                assert g.edges[e]["ep"] == _edge_props[e[:2]]

    # Actual test -------------------------------------------------------------
    # Get the groups and their corresponding configurations
    (grps, cfgs) = graph_grps

    for name, grp in grps.items():
        print(f"Testing configuration {name} ...")

        # Get the config
        cfg = cfgs[name]

        # Check the failing cases
        if name == "wrong_nodes":
            with pytest.raises(
                KeyError, match=r"Check if .* _GG_node_container"
            ):
                grp.create_graph(
                    at_time_idx=cfg.get("at_time_idx", None),
                    node_props=(cfg.get("node_props", None)),
                    edge_props=(cfg.get("edge_props", None)),
                )

            continue

        elif name == "wrong_edges":
            with pytest.raises(
                KeyError, match=r"Check if .* _GG_edge_container"
            ):
                grp.create_graph(
                    at_time_idx=cfg.get("at_time_idx", None),
                    node_props=(cfg.get("node_props", None)),
                    edge_props=(cfg.get("edge_props", None)),
                )

            continue

        elif name == "bad_edges":
            with pytest.raises(
                nx.exception.NetworkXError, match="must be a 2-tuple"
            ):
                # "...or 3-tuple, or 4-tuple..."
                grp.create_graph(
                    at_time_idx=cfg.get("at_time_idx", None),
                    node_props=(cfg.get("node_props", None)),
                    edge_props=(cfg.get("edge_props", None)),
                )

            continue

        # Check the regular cases
        else:
            # This should work
            g = grp.create_graph(
                at_time_idx=cfg.get("at_time_idx", None),
                node_props=(cfg.get("node_props", None)),
                edge_props=(cfg.get("edge_props", None)),
            )

        # Check the node and edge container type
        if name.startswith("grp"):
            assert isinstance(grp.node_container, TimeSeriesGroup)
            assert isinstance(grp.edge_container, TimeSeriesGroup)

        else:
            assert isinstance(grp.node_container, XrDataContainer)
            assert isinstance(grp.edge_container, XrDataContainer)

        # Check that the basic graph creation works
        basic_graph_creation_test(g, cfg)

    # -------------------------------------------------------------------------
    # Check graph creation for single node/edge using `keep_dim`
    # Single node
    gg = GraphGroup(
        name="foo",
        attrs=dict(directed=False, parallel=True, keep_dim=("node_idx",)),
    )
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[0],
    )
    gg.new_container("edges", Cls=XrDataContainer, data=[])
    gg.new_container(
        "np", Cls=XrDataContainer, data=[[0, 1]], dims=("node_idx", "foo")
    )
    g = gg.create_graph(node_props=["np"])

    with pytest.raises(
        ValueError, match="Received 2 property values for 1 nodes"
    ):
        gg.create_graph(node_props=["np"], keep_dim=("bar",))

    # Single edge
    gg = GraphGroup(
        name="foo",
        attrs=dict(directed=False, parallel=True, keep_dim=("edge_idx",)),
    )
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[0, 1],
        dims=("node_idx"),
    )
    gg.new_container(
        "edges",
        Cls=XrDataContainer,
        data=[[1, 0]],
        dims=("edge_idx", "type"),
    )
    g = gg.create_graph()

    with pytest.raises(
        TypeError, match="The edge dimension might have been squeezed"
    ):
        gg.create_graph(keep_dim=("bar",))

    # -------------------------------------------------------------------------
    # Check graph attributes being set correctly from node and edge coordinates
    gg = GraphGroup(name="foo", attrs=dict(directed=False, parallel=True))
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[[0, 1, 2]],
        dims=("time", "node_idx"),
        coords={"time": [42], "node_idx": [0, 1, 2]},
    )
    gg.new_container(
        "edges",
        Cls=XrDataContainer,
        data=[[[[0, 1], [1, 2], [2, 0]]]],
        dims=("time", "foo", "edge_idx", "type"),
        coords={
            "time": [42],
            "foo": [42],
            "edge_idx": [0, 1, 2],
            "type": ["source", "target"],
        },
    )
    gg.new_container(
        "np",
        Cls=XrDataContainer,
        data=[[99, 99, 99]],
        dims=("bar", "node_idx"),
        coords={"bar": [42], "node_idx": [0, 1, 2]},
    )
    g = gg.create_graph(node_props=["np"], isel=dict(time=0))

    assert g.graph == {"time": 42, "foo": 42, "bar": 42}


def test_set_property_functions(graph_grps, graph_data):
    """Test the GraphGroup.set_node_property
    and GraphGroup.set_edge_property function
    """
    (grps, cfgs) = graph_grps
    grp = grps["grp_prop"]
    cfg = cfgs["grp_prop"]

    # Get the graph data to compare with
    _node_props = graph_data["node_props"]
    _edge_props = graph_data["edge_props"]

    g = grp.create_graph(at_time_idx=cfg.get("at_time_idx", None))

    # this should work
    grp.set_node_property(
        g=g, name="np", at_time_idx=cfg.get("at_time_idx", None)
    )
    grp.set_edge_property(
        g=g, name="ep", at_time_idx=cfg.get("at_time_idx", None)
    )

    # Check that properties are set correctly
    assert all([g.nodes[n]["np"] == _node_props[n] for n in g.nodes])
    assert all([g.edges[e]["ep"] == _edge_props[e[:2]] for e in g.edges])

    # -------------------------------------------------------------------------
    # Test the failing cases
    # container name does not exist
    with pytest.raises(
        KeyError,
        match=(
            "No key, key sequence, or property 'foo' in GraphGroup 'grp_prop'"
        ),
    ):
        grp.set_node_property(g=g, name="foo")

    with pytest.raises(
        KeyError,
        match=(
            "No key, key sequence, or property 'foo' in GraphGroup 'grp_prop'"
        ),
    ):
        grp.set_edge_property(g=g, name="foo")

    # invalid time key
    with pytest.raises(KeyError):
        grp.set_node_property(g=g, name="np", at_time=42)

    with pytest.raises(IndexError, match="42"):
        grp.set_edge_property(g=g, name="ep", at_time_idx=42)

    # at_time and at_time_idx both given (raises in _get_data_at)
    with pytest.raises(
        ValueError,
        match="Received keys that appear in both `sel` and "
        r"`isel` for the selection of .*: time",
    ):
        grp.set_node_property(g=g, name="np", at_time=0, at_time_idx=0)

    # length mismatch
    # first, add data to the group that does not fit to the node/edge number
    sub_grp = grp.new_container("np2", Cls=TimeSeriesGroup)
    for time in cfg["times"]:
        sub_grp.new_container(
            str(time), Cls=XrDataContainer, data=cfg["np"][time] + [42]
        )

    sub_grp = grp.new_container("ep2", Cls=TimeSeriesGroup)
    for time in cfg["times"]:
        sub_grp.new_container(
            str(time), Cls=XrDataContainer, data=cfg["ep"][time] + [42]
        )

    with pytest.raises(
        ValueError, match="Received 6 property values for 5 nodes"
    ):
        grp.set_node_property(
            g=g, name="np2", at_time_idx=cfg.get("at_time_idx", None)
        )

    with pytest.raises(
        ValueError, match="Received 6 property values for 5 edges"
    ):
        grp.set_edge_property(
            g=g, name="ep2", at_time_idx=cfg.get("at_time_idx", None)
        )

    # also try adding a single property value
    grp.new_container("np_single", Cls=XrDataContainer, data=[42])

    with pytest.raises(
        ValueError, match="Received 1 property values for 5 nodes"
    ):
        grp.set_node_property(
            g=g, name="np_single", at_time_idx=cfg.get("at_time_idx", None)
        )

    # modify the graph
    g.add_node(42)
    g.add_edge(0, 42)

    # warnings should be raised now because the graph was changed
    with pytest.warns(UserWarning, match="The number of nodes increased"):
        grp.set_node_property(
            g=g, name="np", at_time_idx=cfg.get("at_time_idx", None)
        )

    with pytest.warns(UserWarning, match="The number of edges increased"):
        grp.set_edge_property(
            g=g, name="ep", at_time_idx=cfg.get("at_time_idx", None)
        )

    # error when trying to add invalid data
    with pytest.raises(
        ValueError, match="'np2' matches the current number of nodes"
    ):
        grp.set_node_property(
            g=g, name="np2", at_time_idx=cfg.get("at_time_idx", None)
        )

    with pytest.raises(
        ValueError, match="'ep2' matches the current number of edges"
    ):
        grp.set_edge_property(
            g=g, name="ep2", at_time_idx=cfg.get("at_time_idx", None)
        )

    g = grp.create_graph(
        at_time_idx=cfg.get("at_time_idx", None), parallel_edges=False
    )
    g.add_edge(0, 42)

    with pytest.raises(
        ValueError, match="might be interpreted as a single one"
    ):
        grp.set_edge_property(
            g=g, name="ep2", at_time_idx=cfg.get("at_time_idx", None)
        )

    # -------------------------------------------------------------------------
    # Test set_edge_property for nx.MultiGraph, nx.Graph, and nx.DiGraph
    gg = GraphGroup(name="foo")
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[[0, 1, 2]],
        dims=("time", "node_idx"),
    )
    gg.new_container(
        "edges",
        Cls=XrDataContainer,
        data=[[[1, 0], [0, 2], [2, 1]]],
        dims=("time", "edge_idx"),
    )
    gg.new_container(
        "ep", Cls=XrDataContainer, data=[[1, 0, 2]], dims=("time", "edge_idx")
    )

    # nx.MultiGraph
    g = gg.create_graph(directed=False, parallel_edges=True)
    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(1, 0, 0)]["ep"] == 1
    assert g.edges[(0, 2, 0)]["ep"] == 0
    assert g.edges[(2, 1, 0)]["ep"] == 2

    # nx.Graph
    g = gg.create_graph(directed=False, parallel_edges=False)
    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(1, 0)]["ep"] == 1
    assert g.edges[(0, 2)]["ep"] == 0
    assert g.edges[(2, 1)]["ep"] == 2

    # nx.DiGraph
    g = gg.create_graph(directed=True, parallel_edges=False)
    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(1, 0)]["ep"] == 1
    assert g.edges[(0, 2)]["ep"] == 0
    assert g.edges[(2, 1)]["ep"] == 2

    # -------------------------------------------------------------------------
    # Test that the setting of edge properties works for parallel, undirected
    # edges with different edge properties.
    gg = GraphGroup(name="foo", attrs=dict(directed=False, parallel=True))

    gg.new_container("nodes", Cls=XrDataContainer, data=[0, 1, 2, 3, 4])
    gg.new_container(
        "edges", Cls=XrDataContainer, data=[[3, 4], [3, 4], [4, 3], [3, 4]]
    )
    gg.new_container("ep", Cls=XrDataContainer, data=[340, 341, 342, 343])
    g = gg.create_graph()

    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(3, 4, 0)]["ep"] == 340
    assert g.edges[(3, 4, 1)]["ep"] == 341
    assert g.edges[(3, 4, 2)]["ep"] == 342
    assert g.edges[(3, 4, 3)]["ep"] == 343

    # -------------------------------------------------------------------------
    # Test handling of transposed edge containers
    gg = GraphGroup(name="gg", attrs=dict(directed=True, parallel=True))
    gg.new_container("nodes", Cls=XrDataContainer, data=[2, 1, 0])
    # Edge data that could be added both as it is or transposed
    gg.new_container("edges", Cls=XrDataContainer, data=[[1, 0, 2], [0, 2, 1]])
    # Property data for the non-transposed and the transposed case
    gg.new_container("ep", Cls=XrDataContainer, data=[1, 0])
    gg.new_container("ep_transposed", Cls=XrDataContainer, data=[1, 0, 2])

    # Class attribute not given, edge container should _not_ be transposed
    # because the correct shape can't be deduced unambiguously.
    g = gg.create_graph()
    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(1, 0, 2)]["ep"] == 1
    assert g.edges[(0, 2, 1)]["ep"] == 0

    # Enforce transposing of the edge data
    gg.attrs["edge_container_is_transposed"] = True
    g = gg.create_graph()
    gg.set_edge_property(g=g, name="ep_transposed")

    assert g.edges[(1, 0, 0)]["ep_transposed"] == 1
    assert g.edges[(0, 2, 0)]["ep_transposed"] == 0
    assert g.edges[(2, 1, 0)]["ep_transposed"] == 2

    # Enforce leaving the edge data untransposed
    gg.attrs["edge_container_is_transposed"] = False
    g = gg.create_graph()
    gg.set_edge_property(g=g, name="ep")

    assert g.edges[(1, 0, 2)]["ep"] == 1
    assert g.edges[(0, 2, 1)]["ep"] == 0

    # Check that the desired error is thrown
    with pytest.raises(
        ValueError, match="Received 3 property values for 2 edges"
    ):
        gg.set_edge_property(g=g, name="ep_transposed")


def test_loading_external_data():
    """Tests the `register_property_map` function and the setting of properties
    from external data. Also tests extra kwargs such as `align`, `dropna`.
    """
    # Create a simple graph from a GraphGroup
    gg = GraphGroup(name="gg", attrs=dict(directed=False, parallel=False))
    gg._GG_WARN_UPON_BAD_ALIGN = False
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[[2, 1, 0]],
        dims=("time", "node_idx"),
        coords=dict(time=[10], node_idx=[2, 1, 0]),
    )
    gg.new_container(
        "edges",
        Cls=XrDataContainer,
        data=[[[1, 0], [0, 2], [2, 1]]],
        dims=("time", "edge_idx", "type"),
        coords=dict(time=[10], edge_idx=[1, 0, 2], type=["source", "target"]),
    )
    g = gg.create_graph()

    # Define some XrDataContainers to be added as property
    np1 = XrDataContainer(
        name="np1",
        data=[[0, 0, 0], [2, 1, 0]],
        dims=("time", "node_idx"),
        coords=dict(time=[0, 10], node_idx=[2, 1, 0]),
    )
    ep1 = XrDataContainer(
        name="ep1",
        data=[[0, 0, 0], [1, 0, 2]],
        dims=("time", "edge_idx"),
        coords=dict(time=[0, 10], edge_idx=[1, 0, 2]),
    )

    # Register property maps for the data
    gg.register_property_map("np1", np1)
    gg.register_property_map("ep1", ep1)

    assert "np1" in gg.property_maps
    assert "ep1" in gg.property_maps
    assert (gg.property_maps["np1"] == np1).all()
    assert (gg.property_maps["ep1"] == ep1).all()

    # Add the external properties to the graph
    gg.set_node_property(g=g, name="np1", at_time_idx=-1)
    gg.set_edge_property(g=g, name="ep1", at_time_idx=-1)

    assert g.nodes[0]["np1"] == 0
    assert g.nodes[1]["np1"] == 1
    assert g.nodes[2]["np1"] == 2
    assert g.edges[(1, 0)]["ep1"] == 1
    assert g.edges[(0, 2)]["ep1"] == 0
    assert g.edges[(2, 1)]["ep1"] == 2

    # Now using align=True
    gg.set_node_property(g=g, name="np1", at_time_idx=-1, align=True)
    gg.set_edge_property(g=g, name="ep1", at_time_idx=-1, align=True)

    assert g.nodes[0]["np1"] == 0
    assert g.nodes[1]["np1"] == 1
    assert g.nodes[2]["np1"] == 2
    assert g.edges[(1, 0)]["ep1"] == 1
    assert g.edges[(0, 2)]["ep1"] == 0
    assert g.edges[(2, 1)]["ep1"] == 2

    # -------------------------------------------------------------------------
    # Test for TimeSeriesGroups
    np2 = TimeSeriesGroup(name="np2", dims=["time"])
    # Also add _invalid_ (wrong-sized) data for time '0' to test the case of
    # different sized data stored in a TimeSeriesGroup.
    np2.new_container(
        "0",
        Cls=XrDataContainer,
        data=np.arange(10),
        dims=["node_idx"],
        coords=dict(node_idx=range(10)),
    )
    np2.new_container(
        "10",
        Cls=XrDataContainer,
        data=["foo", "bar", "baz"],
        dims=["node_idx"],
        coords=dict(node_idx=[0, 1, 2]),
    )
    ep2 = TimeSeriesGroup(name="ep2", dims=["time"])
    ep2.new_container(
        "0",
        Cls=XrDataContainer,
        data=np.arange(10),
        dims=["edge_idx"],
        coords=dict(edge_idx=range(10)),
    )
    ep2.new_container(
        "10",
        Cls=XrDataContainer,
        data=["foo", "bar", "baz"],
        dims=["edge_idx"],
        coords=dict(edge_idx=[0, 1, 2]),
    )

    gg.register_property_map("np2", np2)
    gg.register_property_map("ep2", ep2)

    # Set properties without aligning. The data is assumed to be in the same
    # order as in `nodes`/`edges`.
    gg.set_node_property(g=g, name="np2", at_time_idx=-1)
    gg.set_edge_property(g=g, name="ep2", at_time_idx=-1)

    assert g.nodes[0]["np2"] == "baz"
    assert g.nodes[1]["np2"] == "bar"
    assert g.nodes[2]["np2"] == "foo"
    assert g.edges[(1, 0)]["ep2"] == "foo"
    assert g.edges[(0, 2)]["ep2"] == "bar"
    assert g.edges[(2, 1)]["ep2"] == "baz"

    # Now using align=True
    gg.set_node_property(g=g, name="np2", at_time_idx=-1, align=True)
    gg.set_edge_property(g=g, name="ep2", at_time_idx=-1, align=True)

    assert g.nodes[0]["np2"] == "foo"
    assert g.nodes[1]["np2"] == "bar"
    assert g.nodes[2]["np2"] == "baz"
    assert g.edges[(1, 0)]["ep2"] == "bar"
    assert g.edges[(0, 2)]["ep2"] == "foo"
    assert g.edges[(2, 1)]["ep2"] == "baz"

    # Check whether we get the same result when doing it via create_graph
    g = gg.create_graph(
        node_props=["np2"], edge_props=["ep2"], at_time_idx=-1, align=True
    )

    assert g.nodes[0]["np2"] == "foo"
    assert g.nodes[1]["np2"] == "bar"
    assert g.nodes[2]["np2"] == "baz"
    assert g.edges[(1, 0)]["ep2"] == "bar"
    assert g.edges[(0, 2)]["ep2"] == "foo"
    assert g.edges[(2, 1)]["ep2"] == "baz"

    # -------------------------------------------------------------------------
    # Test for LabelledDataGroup
    np3 = LabelledDataGroup(name="np3", dims=["dim"])
    np3.new_container(
        "foo",
        Cls=XrDataContainer,
        data=[[["11", "12"], ["21", "22"], ["31", "32"]]],
        dims=["dim", "subdim1", "subdim2"],
        coords=dict(subdim2=["coord1", "coord2"]),
    )
    ep3 = LabelledDataGroup(name="ep3", dims=["dim"])
    ep3.new_container(
        "foo",
        Cls=XrDataContainer,
        data=[[["11", "12"], ["21", "22"], ["31", "32"]]],
        dims=["dim", "subdim1", "subdim2"],
        coords=dict(subdim2=["coord1", "coord2"]),
    )

    gg.register_property_map("np3", np3)
    gg.register_property_map("ep3", ep3)

    # Deep-select along 'subdim2'
    gg.set_node_property(
        g=g, name="np3", sel=dict(subdim2="coord2"), isel=dict(dim=0)
    )
    gg.set_edge_property(
        g=g, name="ep3", sel=dict(subdim2="coord2"), isel=dict(dim=0)
    )

    assert g.nodes[0]["np3"] == "32"
    assert g.nodes[1]["np3"] == "22"
    assert g.nodes[2]["np3"] == "12"
    assert g.edges[(1, 0)]["ep3"] == "12"
    assert g.edges[(0, 2)]["ep3"] == "22"
    assert g.edges[(2, 1)]["ep3"] == "32"

    # Don't select at all. Should squeeze 'dim' and add arrays as properties.
    gg.set_node_property(g=g, name="np3")
    gg.set_edge_property(g=g, name="ep3")

    assert all(g.nodes[0]["np3"] == ["31", "32"])
    assert all(g.nodes[1]["np3"] == ["21", "22"])
    assert all(g.nodes[2]["np3"] == ["11", "12"])
    assert all(g.edges[(1, 0)]["ep3"] == ["11", "12"])
    assert all(g.edges[(0, 2)]["ep3"] == ["21", "22"])
    assert all(g.edges[(2, 1)]["ep3"] == ["31", "32"])

    # -------------------------------------------------------------------------
    # More tests for the alignment of property data
    # Also test loading the external data directly
    gg._GG_WARN_UPON_BAD_ALIGN = True

    # Aligning external data without coordinates
    np5 = XrDataContainer(
        name="np5", data=[[0, 0, 0], [2, 1, 0]], dims=("time",)
    )
    ep5 = XrDataContainer(
        name="ep5", data=[[0, 0, 0], [1, 0, 2]], dims=("time",)
    )

    gg.set_node_property(g=g, name="np5", data=np5, at_time_idx=-1, align=True)
    gg.set_edge_property(g=g, name="ep5", data=ep5, at_time_idx=-1, align=True)

    assert g.nodes[0]["np5"] == 0
    assert g.nodes[1]["np5"] == 1
    assert g.nodes[2]["np5"] == 2
    assert g.edges[(1, 0)]["ep5"] == 1
    assert g.edges[(0, 2)]["ep5"] == 0
    assert g.edges[(2, 1)]["ep5"] == 2

    # Do it again, now checking for the warning
    with pytest.warns(UserWarning, match="No matching dimensions found"):
        gg.set_node_property(
            g=g, name="np5", data=np5, at_time_idx=-1, align=True
        )

    # Aligning external data with coordinate mismatch
    np6 = XrDataContainer(
        name="np6",
        data=[[42], [24]],
        dims=("node_idx", "foo"),
        coords=dict(node_idx=[555, 999]),
    )
    ep6 = XrDataContainer(
        name="ep6",
        data=[[42], [24]],
        dims=("edge_idx", "foo"),
        coords=dict(edge_idx=[555, 999]),
    )

    gg.set_node_property(g=g, name="np6", data=np6, at_time_idx=-1, align=True)
    gg.set_edge_property(g=g, name="ep6", data=ep6, at_time_idx=-1, align=True)

    assert all([isnan(g.nodes[n]["np6"]) for n in g.nodes])
    assert all([isnan(g.edges[e]["ep6"]) for e in g.edges])

    # Do it again, now checking for the warning
    with pytest.warns(
        UserWarning, match="Found missing values in property data"
    ):
        gg.set_node_property(
            g=g, name="np6", data=np6, at_time_idx=-1, align=True
        )

    # -------------------------------------------------------------------------
    # Test failing cases
    # registering data of the wrong type
    with pytest.raises(TypeError, match="Received invalid type for 'data'"):
        gg.register_property_map("foo", dict(bar="baz"))

    # key exists already
    with pytest.raises(ValueError, match="Please choose a unique name"):
        gg.register_property_map("np1", np1)

    # -------------------------------------------------------------------------
    # Test the NaN-removal together with `align`
    # Create node and edge data that contains Nan's. The missing values should
    # be dropped. Using `align=True` should drop also the respective property
    # values.
    gg = GraphGroup(name="gg", attrs=dict(directed=True, parallel=False))
    gg.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=[0, np.nan, 2],
        dims=("node_idx",),
        coords=dict(node_idx=[0, 1, 2]),
    )
    gg.new_container(
        "edges",
        Cls=XrDataContainer,
        data=[[2, 0], [0, np.nan], [0, 2]],
        dims=("edge_idx", "type"),
        coords=dict(edge_idx=[0, 1, 2], type=["source", "target"]),
    )
    gg.new_container(
        "np1",
        Cls=XrDataContainer,
        data=[2, 1, 0],
        dims=("node_idx",),
        coords=dict(node_idx=[2, 1, 0]),
    )
    gg.new_container(
        "ep1",
        Cls=XrDataContainer,
        data=[2, 1, 0],
        dims=("edge_idx",),
        coords=dict(edge_idx=[2, 1, 0]),
    )

    g = gg.create_graph(
        node_props=["np1"], edge_props=["ep1"], align=True, dropna=True
    )

    assert list(g.nodes) == [0, 2]
    assert set(g.edges) == {(2, 0), (0, 2)}
    assert g.nodes[0]["np1"] == 0
    assert g.nodes[2]["np1"] == 2
    assert g.edges[(2, 0)]["ep1"] == 0
    assert g.edges[(0, 2)]["ep1"] == 2
