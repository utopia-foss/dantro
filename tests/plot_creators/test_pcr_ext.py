"""Tests the ExternalPlotCreator class"""

import pytest

from dantro.plot_creators import ExternalPlotCreator


# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..test_plot_mngr import dm

@pytest.fixture
def init_kwargs(dm) -> dict:
    """Default initialisation kwargs"""
    return dict(dm=dm, default_ext="pdf")

# Tests -----------------------------------------------------------------------

def test_init(init_kwargs):
    """Tests initialisation"""
    pc = ExternalPlotCreator("init", **init_kwargs)
