"""Tests graph-related plotting"""

import networkx as nx
import pytest

import dantro.plot.funcs.graph as dg

# -----------------------------------------------------------------------------


@pytest.fixture
def g():
    return nx.erdos_renyi_graph(100, 0.3)


# -----------------------------------------------------------------------------


def test_get_positions(g):
    """Tests the node layouting helper function"""
    get_positions = dg._get_positions

    pos = get_positions(g, model="spring", iterations=23)
    assert isinstance(pos, dict)

    # Model can also be a callable
    def my_layouting_algo(g, **kwargs) -> dict:
        assert isinstance(g, nx.Graph)
        assert "some_kwarg" in kwargs

        return nx.spring_layout(g)

    pos = get_positions(g, model=my_layouting_algo, some_kwarg="foo")
    assert isinstance(pos, dict)

    # Can also have a fallback if the first one fails for whatever reason
    get_positions(
        g,
        model="spring",
        bad_arg="some_bad_arg",
        fallback_model="spring",
        fallback_kwargs=dict(iterations=23),
    )
    get_positions(
        g,
        model="spring",
        bad_arg="some_bad_arg",
        fallback_model="spring",
        fallback_kwargs=dict(iterations=23),
        silent_fallback=True,
    )

    # Bad layouting name
    with pytest.raises(ValueError, match="No layouting model"):
        get_positions(g, model="bad_layouting_model")
