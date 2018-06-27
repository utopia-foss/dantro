"""Tests the ExternalPlotCreator class."""

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

def test_init(init_kwargs, tmpdir):
    """Tests initialisation"""
    ExternalPlotCreator("init", **init_kwargs)

    # Test passing a base_module_file_dir
    ExternalPlotCreator("init", **init_kwargs,
                        base_module_file_dir=tmpdir)
    
    # Check with invalid directories
    with pytest.raises(ValueError, match="needs to be an absolute path"):
        ExternalPlotCreator("init", **init_kwargs,
                            base_module_file_dir="foo/bar/baz")

    with pytest.raises(ValueError, match="does not exists or does not point"):
        ExternalPlotCreator("init", **init_kwargs,
                            base_module_file_dir=tmpdir.join("foo.bar"))
