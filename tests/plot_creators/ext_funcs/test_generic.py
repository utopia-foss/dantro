"""Tests the generic external plot functions."""

from ...test_plot_mngr import dm as _dm
import pytest
import xarray as xr
import numpy as np
import copy
import os

from dantro.plot_creators.ext_funcs.generic import facet_grid
from dantro.plot_creators import PlotHelper, ExternalPlotCreator
from dantro.containers import XrDataContainer, PassthroughContainer

CREATE_OUTPUT_DIR = False
OUTPUT_PATH = os.path.abspath("test_output")

# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests


@pytest.fixture
def data_dataarray() -> dict:
    """Create a test xarray.DaraArray"""

    return xr.DataArray(np.random.rand(5, 4, 3))


@pytest.fixture
def data_dataset() -> dict:
    """Create a test xarray.Daraset"""
    d0 = xr.DataArray(np.random.rand(5, 4, 3))
    d1 = xr.DataArray(np.random.rand(5, 4, 3))

    ds = xr.Dataset()
    ds['d0'] = d0
    ds['d1'] = d1

    return ds


@pytest.fixture
def dm(_dm, data_dataarray, data_dataset):
    """Returns a data manager populated with some high-dimensional test data"""
    # Add a test xr.DataArray
    grp_dataarray = _dm.new_group("dataarray")
    grp_dataarray.add(XrDataContainer(name="data", data=data_dataarray))

    # Add a test xr.Dataset
    grp_dataset = _dm.new_group("dataset")
    grp_dataset.add(PassthroughContainer(name="data", data=data_dataset))

    return _dm


@pytest.fixture
def anim_disabled() -> dict:
    """Returns a dict with default (disabled) animation kwargs"""
    return dict(enabled=False, writer='frames',
                writer_kwargs=dict(frames=dict(saving=(dict(dpi=96)))))


@pytest.fixture
def anim_enabled(anim_disabled) -> dict:
    """Returns a dict with default (enabled) animation kwargs"""
    d = copy.deepcopy(anim_disabled)
    d['enabled'] = True
    return d


@pytest.fixture
def out_dir(tmpdir) -> str:
    if CREATE_OUTPUT_DIR:
        # Create an output path if it does not yet exist
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        return OUTPUT_PATH
    else:
        return str(tmpdir)

# -- Tests --------------------------------------------------------------------


def test_facet_grid_animation(dm, anim_disabled, anim_enabled, out_dir):
    """Test the FacetGrid animation; this requires invocation via a plot creator"""
    epc = ExternalPlotCreator("test", dm=dm, base_module_file_dir=out_dir)

    # The data paths to the test xr.DataArray and xr.Dataset
    DATAARRAY_PATH = dict(select=dict(data='dataarray/data'))
    DATASET_PATH = dict(select=dict(data='dataset/data'))

    # The kinds of plots that can be created with the generic plot function
    # NOTE There are special cases that are require different parameters
    #      and, therefore, are tested separately, below, e.g. 'hist'
    KINDS_1D = ('line', 'step')
    KINDS_2D = ('contourf', 'contour', 'imshow', 'pcolormesh')

    # Dictionary to collect passing test cases
    test_cases = {}

    # Dictionary to collect from dimension cases
    test_dim_error_cases = {}

    # Add general cases without kind specification to the test cases
    test_cases['default_1'] = dict(col='dim_1')
    test_cases['default_2'] = dict(row='dim_1', hue='dim_2', )
    test_cases['default_anim'] = dict(frames='dim_1', col='dim_2', )

    # Add 1D plot kinds to the test cases
    for k in KINDS_1D:
        test_dim_error_cases['_'.join([k, '0'])] = dict(kind=k)
        test_dim_error_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')
        test_cases['_'.join([k, '2'])] = dict(kind=k,
                                              row='dim_1',
                                              hue='dim_2',
                                              col='dim_0', )

    # Add 2D plot cases to the test cases
    for k in KINDS_2D:
        test_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')
        test_cases['_'.join([k, 'anim_1'])] = dict(
            kind=k, frames='dim_1')

    # .. Tests ................................................................
    for name, plot_kwargs in test_dim_error_cases.items():
        with pytest.raises(Exception):
            # Invoke plotting function via plot creator
            epc.plot(out_path=None,
                     plot_func=facet_grid,
                     animation=anim_disabled, **plot_kwargs, **DATAARRAY_PATH)

    for name, plot_kwargs in test_cases.items():
        # Invoke plotting function via plot creator
        epc.plot(out_path=os.path.join(out_dir, "test_{}".format(name)),
                 plot_func=facet_grid,
                 animation=anim_disabled, **plot_kwargs, **DATAARRAY_PATH)

    # # .. Special Cases ........................................................
    # hist
    # Invoke plotting function via plot creator
    epc.plot(out_path="/".join([out_dir, "test_{}".format(name)]),
             plot_func=facet_grid,
             animation=anim_disabled, kind='hist', **DATAARRAY_PATH)

    # scatter: Is only valid for dataset data
    # Invoke plotting function via plot creator
    epc.plot(out_path="/".join([out_dir, "test_{}".format(name)]),
             plot_func=facet_grid,
             animation=anim_disabled, kind='scatter', x='d0', y='d1',
             **DATASET_PATH)

    # .. Errors ...............................................................
    with pytest.raises(ValueError, match="Got an unknown plot kind"):
        # Invoke plotting function via plot creator
        epc.plot(out_path="/".join([out_dir, "test_{}".format(name)]),
                 plot_func=facet_grid,
                 animation=anim_disabled, kind='wrong', **DATAARRAY_PATH)
