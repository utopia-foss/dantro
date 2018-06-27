"""Tests the PlotManager class"""

import os
from pkg_resources import resource_filename

import numpy as np

import pytest

import paramspace as psp

from dantro.tools import load_yml, recursive_update
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
    # Groups
    vectors = dm.new_group("vectors")
    ndarray = dm.new_group("ndarrays")

    # Vectorial datasets
    vals = 100
    vectors.add(NumpyDC(name="times", data=np.linspace(0, 1, vals)))
    vectors.add(NumpyDC(name="values", data=np.random.rand(vals)))
    vectors.add(NumpyDC(name="more_values", data=np.random.rand(vals)))

    # Multidimensional datasets
    # TODO

    return dm

@pytest.fixture
def pm_kwargs(tmpdir) -> dict:
    """Common plot manager kwargs to use; uses the ExternalPlotCreator for all
    the tests."""
    # Create a test module that just writes a file to the given path
    write_something_funcdef = (
        "def write_something(dm, *, out_path, **kwargs):\n"
        "    '''Writes the kwargs to the given path'''\n"
        "    with open(out_path, 'w') as f:\n"
        "        f.write(str(kwargs))\n"
        )

    tmpdir.join("test_module.py").write(write_something_funcdef)

    # Pass the tmpdir to the ExternalPlotCreator __init__
    cik = dict(external=dict(default_ext="pdf",
                             base_module_file_dir=str(tmpdir)))

    return dict(default_creator="external", creator_init_kwargs=cik)

@pytest.fixture
def pcr_ext_kwargs() -> dict:
    """Returns valid kwargs to make a ExternalPlotCreator plot"""
    return dict(module=".basic", plot_func="lineplot", y="vectors/values")

@pytest.fixture
def pspace_plots() -> dict:
    """Returns a plot configuration (external creator) with parameter sweeps"""

    # Create a sweep over the y-keys for the lineplot
    y_pdim = psp.ParamDim(default="vectors/values",
                          values=["vectors/values", "vectors/more_values"])

    # Assemble the dict
    return dict(sweep=psp.ParamSpace(dict(module=".basic",
                                          plot_func="lineplot",
                                          # kwargs to the plot function
                                          y=y_pdim)))


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


def test_plotting(dm, pm_kwargs, pcr_ext_kwargs):
    """Test the plotting functionality of the PlotManager"""
    def assert_num_plots(pm: PlotManager, num: int):
        """Helper function to check if the plot info is ok"""
        assert len(pm.plot_info) == num

    # The plot manager to test everything with
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_EXT, **pm_kwargs)


    # Plot all from the given default config file
    pm.plot_from_cfg()
    assert_num_plots(pm, len(PLOTS_EXT))


    # Plot only specific entries
    pm.plot_from_cfg(plot_only=[]) # Plotting none to not create overlaps
    assert_num_plots(pm, len(PLOTS_EXT))


    # Assert that plot files were created
    for pi in pm.plot_info:
        print("Checking plot info: ", pi)
        assert pi['out_path']
        assert os.path.exists(pi['out_path'])


    # Check invalid specifications and that they create no plots
    # An invalid key should be propagated
    with pytest.raises(KeyError, match="invalid_key"):
        pm.plot_from_cfg(plot_only=["invalid_key"])
    assert_num_plots(pm, len(PLOTS_EXT))

    # Invalid plot specification
    with pytest.raises(TypeError, match="Got invalid plots specifications"):
        pm.plot_from_cfg(invalid_entry=(1,2,3))
    assert_num_plots(pm, len(PLOTS_EXT))

    # Empty plot config
    with pytest.raises(ValueError, match="Got empty `plots_cfg`"):
        PlotManager(dm=dm).plot_from_cfg()


    # Now, directly using the plot function
    # If default values were given during init, this should work
    pm.plot("foo", **pcr_ext_kwargs)
    assert_num_plots(pm, len(PLOTS_EXT) + 1)

    # Otherwise, without out_dir or creator arguments, not:
    with pytest.raises(ValueError, match="No `out_dir` specified"):
        PlotManager(dm=dm, out_dir=None).plot("foo")
    
    with pytest.raises(ValueError, match="No `creator` argument"):
        PlotManager(dm=dm).plot("foo")


    # Assert that config files were created
    pm.plot("bar", **pcr_ext_kwargs)
    assert_num_plots(pm, len(PLOTS_EXT) + 2)
    assert pm.plot_info[-1]['plot_cfg_path']
    assert os.path.exists(pm.plot_info[-1]['plot_cfg_path'])
    
    pm.plot("baz", **pcr_ext_kwargs, save_plot_cfg=False)
    assert_num_plots(pm, len(PLOTS_EXT) + 3)
    assert pm.plot_info[-1]['plot_cfg_path'] is None


def test_sweep(dm, pm_kwargs, pspace_plots):
    """Test that sweeps work"""
    pm = PlotManager(dm=dm, **pm_kwargs)

    # Plot the sweep
    pm.plot_from_cfg(**pspace_plots)

    # This should have created 2 plots
    assert len(pm.plot_info) == 2

    # By passing a config to `from_pspace` that is no ParamSpace (in this case 
    # the internally stored dict) a ParamSpace should be created from that dict
    pm.plot("foo", from_pspace=pspace_plots["sweep"]._dict)

    # This should have created two more plots
    assert len(pm.plot_info) == 2 + 2

    # Assert that all plots were created
    for pi in pm.plot_info:
        assert os.path.exists(pi['out_path'])

        # None of them should have a plot config saved
        assert pi['plot_cfg_path'] is None


def test_file_ext(dm, pm_kwargs):
    """Check file extension handling"""
    # Without given default extension
    PlotManager(dm=dm, plots_cfg=PLOTS_EXT, out_dir="no1/",
                **pm_kwargs).plot_from_cfg()

    # With extension (with dot)
    pm_kwargs['creator_init_kwargs']['external']['default_ext'] = "pdf"
    PlotManager(dm=dm, plots_cfg=PLOTS_EXT, out_dir="no2/",
                **pm_kwargs).plot_from_cfg()

    # ...and without dot
    pm_kwargs['creator_init_kwargs']['external']['default_ext'] = ".pdf"
    PlotManager(dm=dm, plots_cfg=PLOTS_EXT, out_dir="no3/",
                **pm_kwargs).plot_from_cfg()
