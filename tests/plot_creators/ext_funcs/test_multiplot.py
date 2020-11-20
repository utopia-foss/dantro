"""Tests the multiplot external plot function."""

import logging
import os

import pytest
import matplotlib.pyplot as plt
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


def create_pd_data(
    n: int, *, name: str, shape=None, with_coords: bool = False, **data_array_kwargs
):
    """Creates n-dimensional random data of a certain shape. If no shape is
    given, will use ``(3, 4, 5, ..)``.
    Can also add coords.
    """
    xr_data = create_nd_data(n, shape=shape, with_coords=with_coords, **data_array_kwargs)
    return xr_data.to_dataframe(name=name)


# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests
from ...test_plot_mngr import dm as _dm


@pytest.fixture
def dm(_dm):
    """Returns a data manager populated with some high-dimensional test data"""
    # Add ndim random data for DataArrays, going from 0 to 7 dimensions
    grp_ndim_df = _dm.new_group("ndim_df")
    grp_ndim_df.add(
        *[
            PassthroughContainer(name="{:d}D".format(n), data=create_pd_data(n, name="{:d}D".format(n)))
            for n in [2, 3]
        ]
    )

    grp_labelled = _dm.new_group("labelled")
    grp_labelled.add(
        *[
            PassthroughContainer(
                name="{:d}D".format(n),
                data=create_pd_data(n, with_coords=True, name="{:d}D".format(n)),
            )
            for n in [2, 3]
        ]
    )

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

        for cont_name, key in zip(dm["ndim_df"], plots.keys()):
            epc(
                **out_path("multiplot_" + str(key) + cont_name),
                **plots[key],
                select=dict(data="ndim_df/" + cont_name),
                module=".multiplot",
            )

    # The last figure should survive from this.
    assert len(plt.get_fignums()) == 1

    # Error message upon invalid kind. There should be no figure surviving from
    # such an invocation ...
    plt.close("all")

    assert len(plt.get_fignums()) == 0
