"""Tests the utils.data_ops module"""

import pytest

import numpy as np
import xarray as xr

import dantro
from dantro.groups import OrderedDataGroup
from dantro.containers import ObjectContainer
from dantro.utils import register_operation, apply_operation
from dantro.utils.data_ops import _OPERATIONS as OPERATIONS
import dantro.utils.data_ops as dops

# -----------------------------------------------------------------------------

# Interface tests -------------------------------------------------------------

def test_OPERATIONS():
    """Test the operations database"""
    assert isinstance(OPERATIONS, dict)

    # Make sure some basics are in there
    assert 'print' in OPERATIONS
    assert 'getitem' in OPERATIONS
    assert 'getattr' in OPERATIONS
    assert 'increment' in OPERATIONS

def test_register_operation():
    """Test operation registration"""
    # Can add something
    assert 'op_foobar' not in OPERATIONS
    func_foobar = lambda: 'foobar'
    register_operation(name='op_foobar', func=func_foobar)
    assert 'op_foobar' in OPERATIONS
    assert OPERATIONS['op_foobar'] is func_foobar

    # Cannot overwrite it ...
    with pytest.raises(ValueError, match="already exists!"):
        register_operation(name='op_foobar', func=lambda: 'some_other_func')

    # No error if it's to be skipped
    register_operation(name='op_foobar', func=lambda: 'some_other_func',
                       skip_existing=True)
    assert OPERATIONS['op_foobar'] is func_foobar

    # ... unless explicitly allowed
    func_foobar2 = lambda: 'foobar2'
    register_operation(name='op_foobar', func=func_foobar2,
                       overwrite_existing=True)
    assert OPERATIONS['op_foobar'] is not func_foobar
    assert OPERATIONS['op_foobar'] is func_foobar2

    # Needs be a callable
    with pytest.raises(TypeError, match="is not callable!"):
        register_operation(name='some_invalid_func', func=123)

    # Name needs be a string
    with pytest.raises(TypeError, match="need be a string, was"):
        register_operation(name=123, func=func_foobar)

    # Remove the test entry again
    del OPERATIONS['op_foobar']
    assert 'op_foobar' not in OPERATIONS
    

def test_apply_operation():
    """Test operation application"""
    assert apply_operation('add', 1, 2) == 3

    # Test the "did you mean" feature
    with pytest.raises(KeyError, match="Did you mean: add ?"):
        apply_operation("addd")
    
    # Test application failure error message
    with pytest.raises(TypeError,
                       match="Failed applying operation 'add':.*missing 1"):
        apply_operation("add", 1)


# Tests of specific operations ------------------------------------------------

def test_op_print_data():
    """Tests the print_data operation

    Does not test the print output; but that should be ok.
    """
    # Test passthrough
    d = dict(foo="bar")
    assert dops.print_data(d) is d

    # Coverage test for dantro objects
    dops.print_data(OrderedDataGroup(name="foo"))
    dops.print_data(ObjectContainer(name="objs", data=d))


def test_op_create_mask():
    """Tests the create_mask operation"""
    da = xr.DataArray(name="foo", data=np.random.random((2,3,4)),
                      dims=('x', 'y', 'z'),
                      coords=dict(x=[1,2], y=[1,2,3], z=[1,2,3,4]))

    da_masked = dops.create_mask(da, "<", 0.5)
    assert isinstance(da_masked, xr.DataArray)
    assert all(da_masked.coords['x'] == da.coords['x'])
    assert all(da_masked.coords['y'] == da.coords['y'])
    assert all(da_masked.coords['z'] == da.coords['z'])
    assert da_masked.dims == da.dims

    da_mask_neg = dops.create_mask(da, ">", 0.)
    assert da_mask_neg.all()
    assert "(masked by '> 0.0')" in da_mask_neg.name

    da_mask_larger1 = dops.create_mask(da, ">", 1.)
    assert not da_mask_larger1.any()
    assert "(masked by '> 1.0')" in da_mask_larger1.name

    # Error messages
    with pytest.raises(KeyError, match="No boolean operator '123' available!"):
        dops.create_mask(da, "123", 0.5)


    # Also test where
    da_all_nan = dops.where(da, "<", 0.)
    assert np.isnan(da_all_nan).all()
    
    da_no_nan = dops.where(da, "<=", 1.0)
    assert not np.isnan(da_no_nan).any()


def test_op_count_unique():
    """Test the count_unique operation"""
    da = xr.DataArray(name="foo", data=np.random.randint(0, 5, size=(20, 20)))

    cu = dops.count_unique(da)
    assert isinstance(cu, xr.DataArray)
    assert cu.dims == ('unique',)
    assert (cu.coords['unique'] == [0, 1, 2, 3, 4]).all()
    assert "(unique counts)" in cu.name
