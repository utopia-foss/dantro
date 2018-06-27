"""Tests features of the base class."""

import pytest

from dantro.data_mngr import DataManager
from dantro.plot_creators import BasePlotCreator

# Test classes ----------------------------------------------------------------

class MockPlotCreator(BasePlotCreator):
    """Test class to test the base class implementation"""

    EXTENSIONS = ("one", "two", "ext")

    def _plot(self, out_path: str=None, **cfg):
        """Does nothing but writing the content of cfg to a file at out_path"""
        with open(out_path, "w") as f:
            f.write(str(cfg))


class MockPlotCreator2(MockPlotCreator):
    """Test class to test the base class implementation"""

    DEFAULT_EXT_REQUIRED = False

class MockPlotCreator3(MockPlotCreator):
    """Test class to test the base class implementation"""

    DEFAULT_EXT = "zero"
    
# Fixtures --------------------------------------------------------------------

@pytest.fixture
def init_kwargs(tmpdir) -> dict:
    """Default initialisation kwargs"""
    return dict(dm=DataManager(data_dir=tmpdir), default_ext="ext")

# Tests -----------------------------------------------------------------------

def test_init(init_kwargs):
    """Tests initialisation"""
    # Basic case
    MockPlotCreator("init", **init_kwargs)

    # Test check for existence of default extension
    init_kwargs.pop("default_ext")
    with pytest.raises(ValueError, match="requires a default extension"):
        MockPlotCreator("init", **init_kwargs)

    # Without a required default extension, this should work:
    MockPlotCreator2("init", **init_kwargs)

    # This one should create a ValueError, as the default extension does not
    # match the supported extensions
    with pytest.raises(ValueError, match="Extension 'zero' not supported in"):
        MockPlotCreator3("init", **init_kwargs)


def test_properties(init_kwargs):
    """Tests the properties"""
    mpc = MockPlotCreator("init", **init_kwargs)

    assert mpc.name
    assert mpc.classname
    assert mpc.logstr
    assert isinstance(mpc.dm, DataManager)
    assert mpc.plot_cfg == mpc._plot_cfg
    assert not mpc.plot_cfg is mpc._plot_cfg
    assert mpc.default_ext == "ext"

    # Assert setting of default extension checks work
    mpc.default_ext = "one"
    assert mpc.default_ext == "one"

    with pytest.raises(ValueError, match="Extension 'three' not supported"):
        mpc.default_ext = "three"
    assert mpc.default_ext == "one"


def test_call(init_kwargs, tmpdir):
    """Test the call to the plot creator"""
    mpc = MockPlotCreator("test", **init_kwargs)

    mpc(out_path=tmpdir.join("call0"), foo="bar")
    mpc(out_path=tmpdir.join("call1"), foo="bar")

    # Same output path should fail:
    with pytest.raises(FileExistsError, match="There already exists a"):
        mpc(out_path=tmpdir.join("call1"), foo="bar")

