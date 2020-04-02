"""Tests the generic external plot functions."""

import pytest
import xarray as xr
import numpy as np

from dantro.plot_creators.ext_funcs.generic import facet_grid
from dantro.plot_creators import PlotHelper

# -- Fixtures -----------------------------------------------------------------
@pytest.fixture
def data() -> dict:
    """Create a test xarray.DaraArray"""

    return dict(data=xr.DataArray(np.random.rand(5, 4, 3)))


@pytest.fixture
def data_dataset() -> dict:
    """Create a test xarray.Daraset"""
    d0 = xr.DataArray(np.random.rand(5, 4, 3))
    d1 = xr.DataArray(np.random.rand(5, 4, 3))

    ds = xr.Dataset()
    ds['d0'] = d0
    ds['d1'] = d1

    return dict(data=ds)


@pytest.fixture
def hlpr(tmpdir) -> PlotHelper:
    return PlotHelper(out_path=tmpdir)

# -- Tests --------------------------------------------------------------------


def test_facet_grid(tmpdir, hlpr, data, data_dataset):
    """Test the facet_grid function"""
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
    test_dim_error_cases['default_0'] = dict()
    test_cases['default_1'] = dict(col='dim_1')
    test_cases['default_2'] = dict(row='dim_1', hue='dim_2')

    # Add 1D plot kinds to the test cases
    for k in KINDS_1D:
        test_dim_error_cases['_'.join([k, '0'])] = dict(kind=k)
        test_dim_error_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')
        test_cases['_'.join([k, '2'])] = dict(kind=k,
                                              row='dim_1',
                                              hue='dim_2',
                                              col='dim_0')

    # Add 2D plot cases to the test cases
    for k in KINDS_2D:
        test_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')

    # .. Tests ................................................................
    for _, plot_kwargs in test_dim_error_cases.items():
        with pytest.raises(Exception):
            facet_grid(data=data, hlpr=hlpr, **plot_kwargs)

    for _, plot_kwargs in test_cases.items():
        facet_grid(data=data, hlpr=hlpr, **plot_kwargs)

    # .. Special Cases ........................................................
    # hist
    facet_grid(data=data, hlpr=hlpr, kind='hist')

    # scatter: Is only valid for dataset data
    facet_grid(data=data_dataset, hlpr=hlpr, kind='scatter', x='d0', y='d1')
    with pytest.raises(AttributeError, match="The plot kind"):
        facet_grid(data=data, hlpr=hlpr, kind='scatter', x='d0', y='d1')

    # .. Errors ...............................................................
    with pytest.raises(ValueError, match="Got an unknown plot kind"):
        facet_grid(data=data, hlpr=hlpr, kind='wrong')
