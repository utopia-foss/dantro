"""Tests the PlotManager class"""

from pkg_resources import resource_filename

import pytest

import paramspace as psp

from dantro.tools import load_yml
from dantro.data_mngr import DataManager
from dantro.container import NumpyDataContainer as NumpyDC
from dantro.plot_mngr import PlotManager


# Files -----------------------------------------------------------------------
PLOTS_EXT = load_yml(resource_filename("tests", "cfg/plots_ext.yml"))
# PLOTS_DECL = load_yml(resource_filename("tests", "cfg/plots_decl.yml"))
# PLOTS_VEGA = load_yml(resource_filename("tests", "cfg/plots_vega.yml"))


# Test classes ----------------------------------------------------------------


# Fixtures --------------------------------------------------------------------

@pytest.fixture
def dm(tmpdir) -> DataManager:
    """Returns a DataManager with some test data for plotting."""
    # Initialize it to a temporary direcotry and without load config
    dm = DataManager(tmpdir)

    # Now add data to it
    # TODO

    return dm

@pytest.fixture
def pm_kwargs() -> dict:
    """Common plot manager kwargs to use"""
    return dict(default_creator="external",
                common_creator_kwargs=dict())

@pytest.fixture
def pspace_plots() -> dict:
    """Returns a plot configuration (external creator) with parameter sweeps"""
    pc = dict()
    pc["sweep"] = psp.ParamSpace(dict(plot_func="my_module.my_func",
                                      foo=psp.ParamDim(default=0, range=[5])))

    return pc


# Tests -----------------------------------------------------------------------

def test_init(dm, tmpdir):
    """Tests initialisation"""
    # Test different ways to initialise
    # Only with DataManager; will then later have to pass configuration
    PlotManager(dm=dm)

    # With a configuration dict
    PlotManager(dm=dm, plots_cfg={})

    # With a configuration file path
    PlotManager(dm=dm, plots_cfg=tmpdir.join("foo.yml"))
    
    # With a separate output directory
    PlotManager(dm=dm, out_dir=tmpdir.mkdir("out"))

    # With updating out_fstrs
    pm = PlotManager(dm=dm, out_fstrs=dict(state="foo"))
    assert pm._out_fstrs.get('state') == "foo"

    # Giving an invalid default creator
    with pytest.raises(ValueError, match="No such creator 'invalid'"):
        PlotManager(dm=dm, default_creator="invalid")


def test_plotting(dm, pm_kwargs):
    """Test the plotting functionality of the PlotManager"""
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_EXT, **pm_kwargs)

    # Plot that config
    pm.plot_from_cfg()

    # Plot only specific entries
    pm.plot_from_cfg(plot_only=["from_func", "from_file"])

    # An invalid key should be propagated
    with pytest.raises(KeyError, match="invalid_key"):
        pm.plot_from_cfg(plot_only=["invalid_key"])

    # Invalid plot specification
    with pytest.raises(TypeError, match="Got invalid plots specifications"):
        pm.plot_from_cfg(invalid_entry=(1,2,3))

    # Now directly to the plot function
    # If default values were given during init, this should work
    pm.plot("foo")

    # Otherwise, without out_dir or creator arguments, not:
    with pytest.raises(ValueError, match="No `out_dir` specified"):
        PlotManager(dm=dm, out_dir=None).plot("foo")
    
    with pytest.raises(ValueError, match="No `creator` argument"):
        PlotManager(dm=dm).plot("foo")


def test_sweep(dm, pm_kwargs, pspace_plots):
    """Test that sweeps work"""
    pm = PlotManager(dm=dm, **pm_kwargs)

    pm.plot_from_cfg(**pspace_plots)

    # By passing a config to `from_pspace` that is no ParamSpace, a config
    # should be created
    pm.plot("foo", from_pspace=dict(plot_func="my_module.my_func",
                                    foo=psp.ParamDim(default="foo",
                                                     values=["bar", "baz"])))

@pytest.mark.skip("TODO")
def test_save_plot_cfg():
    pass
