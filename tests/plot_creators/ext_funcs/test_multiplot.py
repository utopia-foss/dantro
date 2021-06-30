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
from dantro.exceptions import PlottingError
from dantro.plot_creators import ExternalPlotCreator, PlotHelper
from dantro.plot_creators.ext_funcs.multiplot import (
    _parse_func_kwargs,
    multiplot,
)
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
    epc = ExternalPlotCreator("test_multiplot", dm=dm)
    epc._exist_ok = True

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
                epc(
                    **out_path("multiplot_" + str(case)),
                    **plot_cfg,
                    select=dict(
                        data="test_data/2D_random",
                        plot_x_data="test_data/1D_x",
                        plot_y_data="test_data/1D_y",
                    ),
                    module=".multiplot",
                )

        else:
            # No error expected
            epc(
                **out_path("multiplot_" + str(case)),
                **plot_cfg,
                select=dict(
                    data="test_data/2D_random",
                    plot_x_data="test_data/1D_x",
                    plot_y_data="test_data/1D_y",
                ),
                module=".multiplot",
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
        epc(
            **out_path("multiplot_string_func"),
            **dict(plot_func=multiplot, to_plot="some bad type (string)"),
            select={},
            module=".multiplot",
        )

    # Enable raising errors to check whether errors are risen
    epc.raise_exc = True

    with pytest.raises(PlottingError, match="Plotting with "):
        epc(
            **out_path("multiplot_fail"),
            **dict(
                plot_func=multiplot, to_plot=[dict(function="sns.regplot")]
            ),
            select=dict(data="test_data/2D_random"),
            module=".multiplot",
        )

    # Reset raising errors
    epc.raise_exc = False


def test_parse_func_kwargs():
    """Tests the basic features of the _parse_func_kwargs"""
    # Check that it is possible to get a plot function from the
    # _MULTIPLOT_FUNC_KINDS as well as from directly passing a function
    _parse_func_kwargs("sns.lineplot")
    _parse_func_kwargs(plt.scatter)

    # Check that an error is emitted for a wrong key
    with pytest.raises(ValueError, match="is not a valid multiplot function."):
        _parse_func_kwargs("wrong_func_name")
