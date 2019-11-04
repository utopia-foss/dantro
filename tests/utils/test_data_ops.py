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

@pytest.fixture
def darrs() -> np.ndarray:
    da000 = xr.DataArray(name="foo",
                         data=np.random.randint(0,10, size=(1,2,3)),
                         dims=('a', 'x', 'y'),
                         coords=dict(a=[0], x=[0, 1], y=[0, 1, 2]))
    da100 = xr.DataArray(name="foo",
                         data=np.random.randint(0,10, size=(1,2,3)),
                         dims=('a', 'x', 'y'),
                         coords=dict(a=[1], x=[0, 1], y=[0, 1, 2]))
    
    a = np.empty((2,1,1), dtype=object)
    a[0,0,0] = da000
    a[1,0,0] = da100

    return a

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
    with pytest.raises(ValueError, match="Did you mean: add ?"):
        apply_operation("addd")

    # ... and check that a list of available operations is posted
    with pytest.raises(ValueError, match="  - getitem"):
        apply_operation("addd")
    
    # Test application failure error message
    with pytest.raises(RuntimeError,
                       match="Failed applying operation 'add'! Got a "
                             "TypeError: .*unexpected keyword argument"):
        apply_operation("add", 1, foo="bar")

    # Check again if kwargs are part of the error message
    with pytest.raises(RuntimeError,
                       match="kwargs: {'foo': 'bar'}"):
        apply_operation("add", 1, foo="bar")


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


def test_op_where():
    """Tests the where operation"""
    da = xr.DataArray(name="foo", data=np.random.random((2,3,4)),
                      dims=('x', 'y', 'z'),
                      coords=dict(x=[1,2], y=[1,2,3], z=[1,2,3,4]))

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


def test_op_populate_ndarray():
    """Test np.ndarray population from a sequence of objects"""
    a0 = dops.populate_ndarray(1,2,3,4,5,6, shape=(2,3), dtype=object)

    # Shape and dtype as requested
    assert (a0 == np.arange(1,7).reshape((2,3))).all()
    assert a0.dtype == object

    # Specifying a different order should have an effect
    a0f = dops.populate_ndarray(1,2,3,4,5,6, shape=(2,3), dtype=float,
                                order='F')
    assert (a0f == np.arange(1,7).reshape((2,3), order='F')).all()

    # Argument mismatch should raise
    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray(1,2,3,4,5, shape=(2,3))
    
    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray(1,2,3,4,5,6,7, shape=(2,3))

def test_op_multi_concat(darrs):
    """Test dantro specialization of xr.concat"""
    c1 = dops.multi_concat(darrs, dims=('a', 'x', 'y'))
    print(c1)

    assert isinstance(c1, xr.DataArray)
    assert c1.dims == ('a', 'x', 'y')
    assert c1.shape == (2,   2,   3)

    # dtype is maintained
    assert c1.dtype == int

    # The case is different when aggregating over a new dimension, which is
    # interpreted as missing data, thus leading to a dtype change to allow NaNs
    c2 = dops.multi_concat(darrs.squeeze(), dims=('b',))
    print(c2)
    assert c2.dtype == float

    # When the number of to-be-concatenated dimensions does not match the
    # number of array dimensions, an error should be raised
    with pytest.raises(ValueError, match="did not match the number of dimens"):
        dops.multi_concat(darrs.squeeze(), dims=('b', 'c', 'd'))
    
    with pytest.raises(ValueError, match="did not match the number of dimens"):
        dops.multi_concat(darrs, dims=('b',))


def test_op_merge(darrs):
    """Test dantro specialization of xr.merge"""
    m1 = dops.merge(list(darrs.flat))
    assert isinstance(m1, xr.Dataset)
    assert m1['foo'].shape == (2,2,3)
    assert m1['foo'].dtype == float
    assert m1['foo'].dims == ('a', 'x', 'y')

    # Can also pass it as an object array
    m2 = dops.merge(darrs)
    assert isinstance(m2, xr.Dataset)
    assert (m1 == m2).all()

    # Can also reduce it to a DataArray
    m3 = dops.merge(darrs, reduce_to_array=True)
    assert isinstance(m3, xr.DataArray)

    # Cannot reduce it if there are multiple data variables
    darrs[0,0,0].name = 'bar'
    with pytest.raises(ValueError, match="one and only one data variable"):
        dops.merge(darrs, reduce_to_array=True)

def test_op_expand_dims():
    """Tests dantro specialization of xarray's expand_dims method"""
    data = np.random.randint(0, 5, size=(20, 20))
    da = xr.DataArray(data=data, dims=('x', 'y'),
                      coords=dict(x=list(range(20)), y=list(range(20))))

    da_e1 = dops.expand_dims(da, dim=dict(a=[0]))
    print(da_e1)
    assert da_e1.dims == ('a', 'x', 'y')
    assert (da_e1.coords['a'] == [0]).all()

    # Also works directly on the data, just without coordinates
    da_e2 = dops.expand_dims(data, dim=dict(a=[0]))
    print(da_e2)
    assert da_e2.dims == ('a', 'dim_0', 'dim_1')
    assert (da_e2.coords['a'] == [0]).all()
