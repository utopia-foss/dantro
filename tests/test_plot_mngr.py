"""Tests the PlotManager class"""

import os
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
    return dict(default_creator="external")

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
    assert len(pm.plot_info) == len(PLOTS_EXT)

    # Plot only specific entries
    pm.plot_from_cfg(plot_only=["from_func", "from_file"])
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT)

    # An invalid key should be propagated
    with pytest.raises(KeyError, match="invalid_key"):
        pm.plot_from_cfg(plot_only=["invalid_key"])
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT)

    # Invalid plot specification
    with pytest.raises(TypeError, match="Got invalid plots specifications"):
        pm.plot_from_cfg(invalid_entry=(1,2,3))
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT)

    # Empty plot config
    with pytest.raises(ValueError, match="Got empty `plots_cfg`"):
        PlotManager(dm=dm).plot_from_cfg()

    # Now directly to the plot function
    # If default values were given during init, this should work
    pm.plot("foo")
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT) + 1

    # Otherwise, without out_dir or creator arguments, not:
    with pytest.raises(ValueError, match="No `out_dir` specified"):
        PlotManager(dm=dm, out_dir=None).plot("foo")
    
    with pytest.raises(ValueError, match="No `creator` argument"):
        PlotManager(dm=dm).plot("foo")

    # Test storage of config files
    pm.plot("bar")
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT) + 2
    assert pm.plot_info[-1]['plot_cfg_path']
    assert os.path.exists(pm.plot_info[-1]['plot_cfg_path'])
    
    pm.plot("baz", save_plot_cfg=False)
    assert len(pm.plot_info) == 2 * len(PLOTS_EXT) + 3
    assert pm.plot_info[-1]['plot_cfg_path'] is None

    # Assert that all plot files were created
    for pi in pm.plot_info:
        assert pi['out_path']
        # assert os.path.exists(pi['out_path'])
        # FIXME activate once implemented


def test_sweep(dm, pm_kwargs, pspace_plots):
    """Test that sweeps work"""
    pm = PlotManager(dm=dm, **pm_kwargs)

    pm.plot_from_cfg(**pspace_plots)

    # By passing a config to `from_pspace` that is no ParamSpace, a config
    # should be created
    pm.plot("foo", from_pspace=dict(plot_func="my_module.my_func",
                                    foo=psp.ParamDim(default="foo",
                                                     values=["bar", "baz"])))


def test_file_ext(dm, pm_kwargs):
    """Check file extension handling"""
    # Without given default extension
    PlotManager(dm=dm, **pm_kwargs, plots_cfg=PLOTS_EXT).plot_from_cfg()

    # With extension (with dot)
    cc_kwargs = dict(external=dict(default_ext=".pdf"))
    PlotManager(dm=dm, **pm_kwargs, plots_cfg=PLOTS_EXT,
                common_creator_kwargs=cc_kwargs).plot_from_cfg()

    # ...and without dot
    cc_kwargs = dict(external=dict(default_ext="pdf"))
    PlotManager(dm=dm, **pm_kwargs, plots_cfg=PLOTS_EXT,
                common_creator_kwargs=cc_kwargs).plot_from_cfg()
