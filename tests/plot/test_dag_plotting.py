"""Tests the DAG plotting features.

This module contains tests for multiple plot creators. The tests have in common
that they all use the data transformation framework for plotting.
They are invoked on the level of the PlotManager, thus allowing to test the
full integration...
"""

import logging
import os
from builtins import *  # to have Exception types available in globals

import numpy as np
import pytest
import xarray as xr
from pkg_resources import resource_filename

from dantro import DataManager, PlotManager
from dantro._yaml import load_yml
from dantro.containers import PassthroughContainer
from dantro.exceptions import *
from dantro.plot_mngr import (
    InvalidCreator,
    PlotConfigError,
    PlotCreatorError,
    PlottingError,
)

# The associated configuration file
DAG_PLOTS_CONFIG = resource_filename("tests", "cfg/dag_plots.yml")

# Set up logging, and disable some loggers (much too verbose)
log = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("dantro.utils.ordereddict").setLevel(logging.INFO)


from .. import TEST_VERBOSITY

# Fixtures --------------------------------------------------------------------
from .._fixtures import *


@pytest.fixture
def dm(psp_grp, psp_grp_default, psp_grp_missing_data, tmpdir) -> DataManager:
    """Returns a DataManager, containing some test data for plotting."""
    # Initialize it to a temporary direcotry and without load config
    dm = DataManager(tmpdir)

    # For ParamSpace plot creators, need some paramspace data. Set new names
    # and add them under a separate group
    psp_grps = dm.new_group("psp")
    psp_grp.name = "regular"
    psp_grp_default.name = "only_default"
    psp_grp_missing_data.name = "missing_data"
    psp_grps.add(psp_grp, psp_grp_default, psp_grp_missing_data)

    # Add some datasets
    dsets = dm.new_group("datasets")
    dsets.new_container(
        path="mean_and_std_4D",
        data=xr.Dataset(
            dict(
                mean=xr.DataArray(
                    np.random.rand(5, 4, 3, 2),
                    dims=("foo", "bar", "baz", "spam"),
                    coords=dict(
                        foo=range(5), bar=range(4), baz=range(3), spam=range(2)
                    ),
                ),
                std=xr.DataArray(
                    np.random.rand(5, 4, 3, 2),
                    dims=("foo", "bar", "baz", "spam"),
                    coords=dict(
                        foo=range(5), bar=range(4), baz=range(3), spam=range(2)
                    ),
                ),
            )
        ),
        Cls=PassthroughContainer,
    )

    # Add xarrax tutorial examples as xr.Dataset and XrDataContainer
    xr_tut_dsets = dm.new_group("xr_tutorial")
    xr_tut_darrs = dm.new_group("xr_tutorial/arrays")
    XR_TUT_DATASETS = (
        "air_temperature",
        "rasm",
        "ROMS_example",
    )
    try:
        for ds_name in XR_TUT_DATASETS:
            ds = xr.tutorial.open_dataset(ds_name, decode_times=True)
            xr_tut_dsets.new_container(
                path=ds_name,
                data=ds,
                Cls=PassthroughContainer,
            )
            xr_tut_darrs.new_container(
                path=ds_name,
                data=ds.to_array(),
                Cls=XrDataContainer,
            )
    except Exception as exc:
        log.error(
            "Failed loading xr.tutorial datasets; probably because there is "
            "no locally cached copy available and you have no internet "
            f"connection.\nError was a {type(exc).__name__}: {exc}"
        )

    # NOTE Can add more test data here, if desired

    if TEST_VERBOSITY >= 2:
        print(dm.tree_condensed)
    return dm


@pytest.fixture
def dag_plots_cfg() -> dict:
    """The test configuration file, freshly loaded from the YAML file"""
    return load_yml(path=DAG_PLOTS_CONFIG)


@pytest.fixture
def pm(dm, out_dir, dag_plots_cfg) -> PlotManager:
    """Creates a PlotManager instance with the specified output directory"""
    return PlotManager(
        dm=dm,
        out_dir=str(out_dir),
        raise_exc=True,
        **dag_plots_cfg["_pm_init_kwargs"],
    )


# Tests -----------------------------------------------------------------------


def test_config_based(pm, dag_plots_cfg):
    """Carries out fully config-based tests using a PlotManager"""

    def invoke_plot(pm: PlotManager, *, name: str, plot_cfg: dict):
        return pm.plot(name=name, **plot_cfg)

    # .. Automate creation of individual plots with their respective config ...
    for case_name, case_cfg in dag_plots_cfg["config_based"].items():
        print(f"\n\n\n--- Testing plot case '{case_name}' ... ---\n")

        # Find out whether this is expected to succeed or not
        _raises = case_cfg.get("_raises", False)
        _exp_exc = (
            Exception if not isinstance(_raises, str) else globals()[_raises]
        )
        _warns = case_cfg.get("_warns", False)
        _exp_warning = (
            UserWarning if not isinstance(_warns, str) else globals()[_warns]
        )
        _match = case_cfg.get("_match")

        if not _raises and not _warns:
            invoke_plot(pm, name=case_name, plot_cfg=case_cfg["plot_cfg"])

        elif _warns and not _raises:
            print(f"Expecting {_exp_warning.__name__} (match: {_match}) ...")
            with pytest.warns(_exp_warning, match=_match):
                invoke_plot(pm, name=case_name, plot_cfg=case_cfg["plot_cfg"])

        elif _raises and not _warns:
            print(f"Expecting {_exp_exc.__name__} (match: {_match}) ...")

            with pytest.raises(_exp_exc, match=_match):
                invoke_plot(pm, name=case_name, plot_cfg=case_cfg["plot_cfg"])

        else:
            raise ValueError(
                "Cannot have `_raises` AND `_warns`! "
                f"Check config of case {case_name}"
            )

        print(f"\n\n\n--- Test case '{case_name}' succeeded ---\n")
