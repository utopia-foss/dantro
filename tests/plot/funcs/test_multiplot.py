"""Tests the multiplot external plot function."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from dantro._import_tools import get_resource_path
from dantro.containers import PassthroughContainer, XrDataContainer
from dantro.exceptions import PlottingError
from dantro.plot import PlotHelper, PyPlotCreator
from dantro.plot.funcs._multiplot import parse_function_specs
from dantro.plot.funcs.multiplot import multiplot
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
PLOTS_CFG_MP = load_yml(get_resource_path("tests", "cfg/plots_multiplot.yml"))

# Disable matplotlib logger (much too verbose)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# -- Helpers ------------------------------------------------------------------


from ..._fixtures import *

# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests
from ...test_plot_mngr import dm as _dm


@pytest.fixture
def dm(_dm):
    """Returns a data manager populated with test data as pd.DataFrame"""

    # Test data for seaborn plots
    df = pd.DataFrame(
        np.random.randn(6, 2).cumsum(axis=0), columns=["dim_0", "dim_1"]
    )

    # Test data for the plt.plot function
    plot_x_data = np.arange(5)
    plot_y_data = np.arange(5)

    grp_df = _dm.new_group("test_data")

    grp_df.add(PassthroughContainer(name="2D_random", data=df))
    grp_df.add(PassthroughContainer(name="1D_x", data=plot_x_data))
    grp_df.add(PassthroughContainer(name="1D_y", data=plot_y_data))

    return _dm


# .. Multiplot Tests .........................................................


def test_multiplot(dm, out_dir):
    """Tests the basic features and special cases of the multiplot plot"""
    ppc = PyPlotCreator("test_multiplot", dm=dm, plot_func=multiplot)
    ppc._exist_ok = True

    # Shortcuts
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Make sure there are no figures currently open, in order to be able to
    # track whether any figures leak from the plot function ...
    plt.close("all")
    assert len(plt.get_fignums()) == 0

    # Invoke the plotting function with data of different dimensionality.
    for case, plot_cfg in PLOTS_CFG_MP["multiplots"].items():
        if plot_cfg.pop("_raises", False):
            # Expecting an error to be raised
            match = plot_cfg.pop("_match", None)
            with pytest.raises(Exception, match=match):
                ppc(
                    **out_path("multiplot_" + str(case)),
                    **plot_cfg,
                    select=dict(
                        data="test_data/2D_random",
                        plot_x_data="test_data/1D_x",
                        plot_y_data="test_data/1D_y",
                    ),
                )

        else:
            # No error expected
            ppc(
                **out_path("multiplot_" + str(case)),
                **plot_cfg,
                select=dict(
                    data="test_data/2D_random",
                    plot_x_data="test_data/1D_x",
                    plot_y_data="test_data/1D_y",
                ),
            )

    # The last figure should survive from this.
    assert len(plt.get_fignums()) == 1

    # Error message upon invalid kind. There should be no figure surviving from
    # such an invocation ...
    plt.close("all")

    assert len(plt.get_fignums()) == 0

    # Test for correct error handling
    with pytest.raises(
        TypeError,
        match="`to_plot` argument needs to be list-like or a dict but",
    ):
        ppc(
            **out_path("multiplot_string_func"),
            to_plot="some bad type (string)",
            select={},
        )

    # Enable raising errors to check whether errors are risen
    ppc.raise_exc = True

    with pytest.raises(PlottingError, match="sns.regplot.*did not succeed!"):
        ppc(
            **out_path("multiplot_fail"),
            to_plot=[dict(function="sns.regplot")],
            select=dict(data="test_data/2D_random"),
        )

    # Reset raising errors
    ppc.raise_exc = False
