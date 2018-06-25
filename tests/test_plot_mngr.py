"""Tests the PlotManager class"""

from pkg_resources import resource_filename

import pytest

from dantro.data_mngr import DataManager
from dantro.container import NumpyDataContainer as NumpyDC
from dantro.plot_mngr import PlotManager


# Files -----------------------------------------------------------------------
PLOTS_CUSTOM = resource_filename("tests", "cfg/plots_custom.yml")
PLOTS_DECL = resource_filename("tests", "cfg/plots_decl.yml")
PLOTS_VEGA = resource_filename("tests", "cfg/plots_vega.yml")


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


def test_plot(dm):
    """Test the plotting functionality of the PlotManager"""
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_CUSTOM)

    # Plot that config
    pm.plot_from_cfg()

    # TODO continue test-driven development here
    # TODO try to make interface similar to that of DataManager
