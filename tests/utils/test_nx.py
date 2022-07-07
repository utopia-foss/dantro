"""Tests the utils.nx module"""

import copy
import os

import networkx as nx
import pytest

import dantro.utils.nx as dunx


@pytest.fixture
def g():
    g = nx.fast_gnp_random_graph(100, 0.3)

    for node, attrs in g.nodes(data=True):
        attrs["foo"] = "bar"
        attrs["id_squared"] = node**2

    for s, t, attrs in g.edges(data=True):
        attrs["some_edge_attr"] = "foobar"
        attrs["as_string"] = f"{s} -> {t}"

    return g


@pytest.fixture
def g_with_more_attrs(g):
    def SomeCustomClass():
        foo = "bar"

    g = copy.deepcopy(g)

    for node, attrs in g.nodes(data=True):
        # Cannot be written by most networkx writers
        attrs["some_dict"] = dict(foo="bar")
        attrs["some_bad_object"] = SomeCustomClass()

        # ... can add more here

    return g


# -----------------------------------------------------------------------------


def test_keep_node_attributes(g):
    for node, attrs in g.nodes(data=True):
        assert "foo" in attrs
        assert "id_squared" in attrs

    dunx.keep_node_attributes(g, "foo", "bar")

    for node, attrs in g.nodes(data=True):
        assert "foo" in attrs
        assert "id_squared" not in attrs


def test_keep_edge_attributes(g):
    for _, _, attrs in g.edges(data=True):
        assert "some_edge_attr" in attrs
        assert "as_string" in attrs

    dunx.keep_edge_attributes(g, "some_edge_attr", "as_string")

    for _, _, attrs in g.edges(data=True):
        assert "some_edge_attr" in attrs
        assert "as_string" in attrs

    dunx.keep_edge_attributes(g, "as_string")

    for _, _, attrs in g.edges(data=True):
        assert "some_edge_attr" not in attrs
        assert "as_string" in attrs


def test_export_graph(tmpdir, g, g_with_more_attrs):
    export = dunx.export_graph

    out1 = str(tmpdir.join("one/exported"))
    export(g, out_path=out1, graphml=True, gml=True, dot=False)

    files1 = os.listdir(os.path.dirname(out1))
    assert "exported.graphml" in files1
    assert "exported.gml" in files1
    assert "exported.dot" not in files1

    # Drop some attributes
    out2 = str(tmpdir.join("two/exported.some_ext"))
    export(
        g_with_more_attrs,
        out_path=out2,
        keep_node_attrs=("foo", "id_squared"),
        graphml=True,
        gml=True,
        adjlist=True,
    )

    files2 = os.listdir(os.path.dirname(out2))
    assert "exported.graphml" in files2
    assert "exported.gml" in files2
    assert "exported.adjlist" in files2

    # ... and need to do that, otherwise cannot write some formats
    with pytest.raises(Exception, match="does not support"):
        export(
            g_with_more_attrs,
            out_path=out2,
            keep_node_attrs=True,
            graphml=True,
            gml=True,
        )

    # Invalid writer
    with pytest.raises(ValueError, match="Invalid export format"):
        export(g, out_path=out1, bad_writer=True)
