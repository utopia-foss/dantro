"""Tests the multiplot external plot function."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from pkg_resources import resource_filename

from dantro.containers import PassthroughContainer, XrDataContainer
from dantro.plot_creators import ExternalPlotCreator, PlotHelper
from dantro.plot_creators.ext_funcs.multiplot import multiplot
from dantro.tools import load_yml

from .test_generic import create_nd_data, out_dir

# Local variables and configuration ...........................................

# Whether to write test output to a temporary directory
# NOTE When manually debugging, it's useful to set this to False, such that the
#      output can be inspected in TEST_OUTPUT_PATH
USE_TMPDIR = False

# If not using a temporary directory, the desired output directory
TEST_OUTPUT_PATH = os.path.abspath(os.path.expanduser("~/dantro_test_output"))

# Test configurations
PLOTS_CFG_MP = load_yml(resource_filename("tests", "cfg/plots_multiplot.yml"))

# Disable matplotlib logger (much too verbose)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# -- Helpers ------------------------------------------------------------------


# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests
from ...test_plot_mngr import dm as _dm


@pytest.fixture
def dm(_dm):
    """Returns a data manager populated with test data as pd.DataFrame"""

    df = pd.DataFrame(
        np.random.randn(6, 2).cumsum(axis=0), columns=["dim_0", "dim_1"]
    )

    grp_df = _dm.new_group("test_data")

    grp_df.add(PassthroughContainer(name="tips", data=df))

    return _dm


# .. Multiplot Tests .........................................................


def test_multiplot(dm, out_dir):
    """Tests the basic features and special cases of the multiplot plot"""
    epc = ExternalPlotCreator("test_multiplot", dm=dm)
    epc._exist_ok = True

    # Shortcuts
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Make sure there are no figures currently open, in order to be able to
    # track whether any figures leak from the plot function ...
    plt.close("all")
    assert len(plt.get_fignums()) == 0

    # Invoke the plotting function with data of different dimensionality.
    for plots in PLOTS_CFG_MP.values():
        for key, plot_cfg in plots.items():
            epc(
                **out_path("multiplot_" + str(key)),
                **plots[key],
                select=dict(data="test_data/tips"),
                module=".multiplot",
            )

    # The last figure should survive from this.
    assert len(plt.get_fignums()) == 1

    # Error message upon invalid kind. There should be no figure surviving from
    # such an invocation ...
    plt.close("all")

    assert len(plt.get_fignums()) == 0
