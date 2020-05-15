"""Tests the utils.data_ops module"""

import builtins

import pytest

import numpy as np
import xarray as xr
import sympy as sym

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


def test_op_import_module_or_object():
    """Test the import_module_or_object operation"""
    _import = dops.import_module_or_object

    # Module import
    assert _import() is builtins
    assert _import(".utils.data_ops") is dantro.utils.data_ops
    assert _import("numpy.random") is np.random
    assert _import("numpy.random") is _import(module="numpy.random")

    # Name import, including traversal
    assert _import(name='abs') is builtins.abs is abs
    assert _import(name='abs.__name__') == 'abs'
    assert _import(".utils", "register_operation") is dantro.utils.register_operation
    assert _import("numpy", "pi") is np.pi
    assert _import("numpy.random", "randint.__name__") == 'randint'

    # Errors
    with pytest.raises(ModuleNotFoundError, match="foobar"):
        _import("foobar")

    with pytest.raises(AttributeError, match="abs> has no attribute 'has'!"):
        _import(name="abs.has.no.attributes")

    with pytest.raises(AttributeError, match="has no attribute 'not_pi_but_"):
        _import("numpy", "not_pi_but_something_else")


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
    a0 = dops.populate_ndarray([1,2,3,4,5,6], shape=(2,3), dtype=object)

    # Shape and dtype as requested
    assert (a0 == np.arange(1,7).reshape((2,3))).all()
    assert a0.dtype == object

    # Specifying a different order should have an effect
    a0f = dops.populate_ndarray([1,2,3,4,5,6], shape=(2,3), dtype=float,
                                order='F')
    assert (a0f == np.arange(1,7).reshape((2,3), order='F')).all()

    # Argument mismatch should raise
    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray([1,2,3,4,5], shape=(2,3))

    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray([1,2,3,4,5,6,7], shape=(2,3))


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


def test_op_expand_object_array():
    """Tests dantro specialization of xarray's expand_dims method"""
    expand = dops.expand_object_array

    arr = dops.populate_ndarray([i * np.ones((4, 5), dtype=int)
                                 for i in range(6)],
                                shape=(2,3), dtype=object)
    da = xr.DataArray(arr, dims=('a', 'b'),
                      coords=dict(a=range(10, 12), b=range(20, 23)))
    assert da.shape == (2,3)
    assert da.dtype == np.dtype('O')

    eda = expand(da, inner_shape=(4,5), inner_dims=('c', 'd'))
    print(eda)
    assert eda.shape == (2,3,4,5)
    assert eda.dtype == int
    assert eda.dims == ('a', 'b', 'c', 'd')
    assert (eda.coords["a"] == [10, 11]).all()
    assert (eda.coords["b"] == [20, 21, 22]).all()
    assert (eda.coords["c"] == range(4)).all()
    assert (eda.coords["d"] == range(5)).all()


def test_op_expression():
    """Tests the ``expression`` data operation"""
    expr = dops.expression

    # Basics
    assert expr("1 + 2*3 / (4 - 5)") == -5.
    assert expr("1 + 2*3 / (4 - five)", symbols=dict(five=5)) == -5.
    assert expr("1 + 2*3 / (4 - 5) + .1", astype=int) == -4
    assert expr("1 + 2*3 / (4 - 5) + .1", astype='uint8') == 256 - 4

    # ... also works with expr evaluating to a literal type
    assert expr("foo - foo") == 0
    assert expr("(foo - bar)*0 + 1") == 1
    assert expr("(foo - bar)*0 + 1", evaluate=False) == 1

    # XOR operator ^ is _not_ converted to exponentiation
    assert expr("2**3") == 8
    assert expr("true^false", astype=bool) == True

    # Expressions that retain free symbols cannot be fully evaluated
    with pytest.raises(TypeError, match="Failed casting.*free symbols.*"):
        expr("foo + bar", symbols=dict(foo=1))

    # ... unless they do _not_ specify a dtype
    assert (   expr("foo + bar", symbols=dict(foo=1), astype=None).free_symbols
            == {sym.Symbol("bar")})

    # --- Array support ---
    arrs = dict()
    a1 = arrs['a1'] = np.array([1, 2, 3])
    a2 = arrs['a2'] = np.array([.1, .2, .3])
    a3 = arrs['a3'] = np.array([[1, 2], [3, 4]])

    assert (expr("a1 + a2 / 2", symbols=arrs) == a1 + a2 / 2).all()
    assert np.allclose(expr("a1 ** exp(3)", symbols=arrs), a1 ** np.exp(3))

    # --- Advanced features ---
    # List definition ...
    assert (expr("[1,2,3]") == [1,2,3]).all()
    assert (expr("[1,2,3] * 2") == [1,2,3,1,2,3]).all()
    assert (expr("[1,2,3] * foo", symbols=dict(foo=2)) == [1,2,3,1,2,3]).all()
    assert (expr("[1,2,3]", evaluate=False) == [1,2,3]).all()

    # ... does not work with free or unevaluated symbols, though
    with pytest.raises(ValueError, match="can't multiply sequence by non-int"):
        expr("[1,2,3] * foo")
    with pytest.raises(ValueError, match="SympifyError"):
        expr("[1,2,3] * foo", symbols=dict(foo=2), evaluate=False)
    with pytest.raises(ValueError, match="SympifyError"):
        expr("[1,2,3] * foo", evaluate=False)

    # Item access ...
    assert expr("[foo, bar][i]", symbols=dict(i=1, bar=42)) == 42

    # Attribute access ...
    assert expr("a1.ndim + a2.size", symbols=arrs) == 4
    assert expr("a1.ndim + a2.size", symbols=arrs, evaluate=False) == 4

    # ... will fail without symbols being specified
    with pytest.raises(ValueError, match="Failed parsing.*none specified.*"):
        expr("a1.ndim + a2.size")
    with pytest.raises(ValueError, match="Failed parsing.*a1.*"):
        expr("a1.ndim + a2.size", symbols=dict(a1=a1))


def test_op_generate_lambda():
    """Tests the ``generate_lambda`` data operation"""
    gen_and_call = lambda e, *a, **k: dops.generate_lambda(e)(*a, **k)

    # Basic
    assert gen_and_call("lambda *_: 3") == 3
    assert gen_and_call("lambda a: a+2", 1) == 3
    assert gen_and_call("lambda a, b: a + b", 1, 2) == 3
    assert gen_and_call("lambda a, *, b: a + b", 1, b=2) == 3
    assert gen_and_call("lambda *, a_b, c: a_b + c", a_b=1, c=2) == 3
    assert gen_and_call("lambda **ks: ks", foo=1, bar=2) == dict(foo=1, bar=2)

    # Some more tests; omits the leading lambda to save space
    valid = {
        "a, b: a+b":        ([1, 2], {},                3),
        "a: ceil(a)":       ([.5], {},                  1.),
        "a: sin(a)":        ([0.], {},                  0.),
        "*_: cos(pi)":      ([], {},                    -1.),
        "*_: abs(cos(pi))": ([], {},                    +1.),
        "a: np.mean(a)":    ([[1,2,3]], {},             2),
        "a: xr.DataArray(a).mean()": ([[1,2,3]], {},    2),
        "**ks: {k for k in ks}": ([], dict(a=1, b=2),   {"a", "b"}),
        "*a: range(*a)":    ([0, 10, 2], {},            range(0, 10, 2)),
        "*a: slice(*a)":    ([0, 10, 2], {},            slice(0, 10, 2)),
    }
    for expr, (args, kwargs, expected) in valid.items():
        expr = "lambda " + expr
        print("\nexpr:   ", repr(expr))
        print("args:   ", args)
        print("kwargs: ", kwargs)
        assert gen_and_call(expr, *args, **kwargs) == expected

    # Invalid patterns
    invalid = (
        "",
        "not a lambda a, b: foo",
        "LAMBDA: foo",
    )
    for expr in invalid:
        with pytest.raises(SyntaxError, match="not a valid lambda expression"):
            print("\nexpr: ", repr(expr))
            gen_and_call(expr)

    # Restricted
    restricted = (
        "lambda a: (lambda b: b)(a)",
        "lambda a: __import__('os').system()",
    )
    for expr in restricted:
        with pytest.raises(SyntaxError, match="one or more disallowed"):
            print("\nexpr: ", repr(expr))
            gen_and_call(expr)

    # Syntax error: passes the above but makes some other syntax error or uses
    # an undefined symbol
    bad_syntax = (
        "lambda x: x+",
        "lambda x: x+(x+1))",
        "lambda x: x : x",
    )
    for expr in bad_syntax:
        with pytest.raises(SyntaxError, match="Failed generating"):
            print("\nexpr: ", repr(expr))
            gen_and_call(expr)

    # Missing names (e.g. from restricted parts of builtins)
    bad_names = (
        "lambda *_: eval('_'+'_builtins_'+'_')",
        "lambda *_: dir(dict())",
        "lambda *_: exec('exit')",
    )
    for expr in bad_names:
        with pytest.raises(NameError):
            print("\nexpr: ", repr(expr))
            gen_and_call(expr)
