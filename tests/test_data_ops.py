"""Tests the utils.data_ops module"""

import builtins
import copy
import sys
from typing import Callable, Union

import numpy as np
import pytest
import sympy as sym
import xarray as xr

import dantro
import dantro.data_ops.db as dops
from dantro.containers import ObjectContainer, XrDataContainer
from dantro.data_ops import _OPERATIONS as OPERATIONS
from dantro.data_ops import apply_operation, is_operation, register_operation
from dantro.exceptions import *
from dantro.groups import OrderedDataGroup

# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_operations():
    """A temporary operations database; reset to its previous state after test
    teardown.
    """
    original_ops = copy.copy(OPERATIONS)
    yield
    OPERATIONS.clear()
    OPERATIONS.update(original_ops)


@pytest.fixture
def darrs() -> np.ndarray:
    da000 = xr.DataArray(
        name="foo",
        data=np.random.randint(0, 10, size=(1, 2, 3)),
        dims=("a", "x", "y"),
        coords=dict(a=[0], x=[0, 1], y=[0, 1, 2]),
    )
    da100 = xr.DataArray(
        name="foo",
        data=np.random.randint(0, 10, size=(1, 2, 3)),
        dims=("a", "x", "y"),
        coords=dict(a=[1], x=[0, 1], y=[0, 1, 2]),
    )

    a = np.empty((2, 1, 1), dtype=object)
    a[0, 0, 0] = da000
    a[1, 0, 0] = da100

    return a


# Interface tests -------------------------------------------------------------


def test_OPERATIONS():
    """Test the operations database"""
    assert isinstance(OPERATIONS, dict)

    # Make sure some basics are in there
    assert "print" in OPERATIONS
    assert "getitem" in OPERATIONS
    assert "getattr" in OPERATIONS
    assert "increment" in OPERATIONS


def test_register_operation(tmp_operations):
    """Test operation registration"""
    # Can add something
    assert "op_foobar" not in OPERATIONS
    func_foobar = lambda: "foobar"
    register_operation(func_foobar, "op_foobar")
    assert "op_foobar" in OPERATIONS
    assert OPERATIONS["op_foobar"] is func_foobar

    # Need not specify a name
    def my_test_func():
        pass

    register_operation(my_test_func)
    assert OPERATIONS["my_test_func"] is my_test_func

    # Lambdas need to be named explicitly
    with pytest.raises(
        ValueError,
        match="Could not automatically deduce an operation name .* lambda",
    ):
        register_operation(lambda: 0)

    # Cannot overwrite it ...
    with pytest.raises(ValueError, match="already exists!"):
        register_operation(func=lambda: "some_other_func", name="op_foobar")

    # No error if it's to be skipped
    register_operation(
        lambda: "some_other_func", name="op_foobar", skip_existing=True
    )
    assert OPERATIONS["op_foobar"] is func_foobar

    # ... unless explicitly allowed
    func_foobar2 = lambda: "foobar2"
    register_operation(
        name="op_foobar", func=func_foobar2, overwrite_existing=True
    )
    assert OPERATIONS["op_foobar"] is not func_foobar
    assert OPERATIONS["op_foobar"] is func_foobar2

    # Needs be a callable
    with pytest.raises(TypeError, match="is not callable!"):
        register_operation(name="some_invalid_func", func=123)

    # Name needs be a string
    with pytest.raises(TypeError, match="need be a string, was"):
        register_operation(name=123, func=func_foobar)

    # Remove the test entry again
    del OPERATIONS["op_foobar"]
    assert "op_foobar" not in OPERATIONS

    # Can also use a custom operations database
    my_ops = dict()
    register_operation(func_foobar, name="my_foobar", _ops=my_ops)
    register_operation(func=func_foobar2, name="my_foobar2", _ops=my_ops)
    assert "my_foobar" in my_ops
    assert "my_foobar2" in my_ops
    assert "my_foobar" not in OPERATIONS
    assert "my_foobar2" not in OPERATIONS


def test_is_operation_decorator(tmp_operations):
    """Tests the `is_operation` decorator"""
    assert "some_func" not in OPERATIONS
    assert "another_func" not in OPERATIONS
    assert "yet_another_func" not in OPERATIONS

    # Without parentheses
    @is_operation
    def some_func():
        pass

    assert OPERATIONS["some_func"] is some_func

    # With parentheses
    @is_operation()
    def another_func():
        pass

    assert OPERATIONS["another_func"] is another_func

    # With custom name
    @is_operation("yet_another_func")
    def one_more_func():
        pass

    assert "one_more_func" not in OPERATIONS
    assert OPERATIONS["yet_another_func"] is one_more_func

    # Passing of arguments: overwrite existing operation
    @is_operation("yet_another_func", overwrite_existing=True)
    def yet_one_more_func():
        pass

    assert OPERATIONS["yet_another_func"] is yet_one_more_func

    # Skip an existing operation
    @is_operation("some_func", skip_existing=True)
    def some_other_func():
        pass

    assert OPERATIONS["some_func"] is some_func


def test_is_operation_decorator_overwriting(tmp_operations):
    """Test whether the decorator allows overwriting in an easy way, e.g. to
    specify different defaults.
    """
    my_ops = dict()

    def my_reg_func(*args, skip_existing=True, _ops=None, **kwargs):
        _ops = _ops if _ops is not None else my_ops
        return register_operation(
            *args, skip_existing=skip_existing, _ops=_ops, **kwargs
        )

    def my_decorator(arg: Union[str, Callable] = None, /, **kws):
        return is_operation(arg, _reg_func=my_reg_func, **kws)

    @my_decorator
    def my_func():
        pass

    assert my_ops["my_func"] is my_func

    @my_decorator("my_func")  # now with `skip_existing = True` as default
    def also_my_func():
        pass

    assert my_ops["my_func"] is my_func


def test_apply_operation():
    """Test operation application"""
    assert apply_operation("add", 1, 2) == 3

    # Test the "did you mean" feature
    with pytest.raises(BadOperationName, match="Did you mean: add ?"):
        apply_operation("addd")

    # ... and check that a list of available operations is posted
    with pytest.raises(BadOperationName, match="getitem"):
        apply_operation("addd")

    # Test application failure error message
    with pytest.raises(
        DataOperationFailed, match="Operation 'add' failed with a TypeError"
    ):
        apply_operation("add", 1, foo="bar")

    # Check again if kwargs are part of the error message
    with pytest.raises(
        DataOperationFailed, match=r"foo:  \['barbazspam', 123\]"
    ):
        apply_operation("add", 1, foo=["barbazspam", 123])

    # Can also have a custom operations database (where there is no add)
    with pytest.raises(BadOperationName, match="No operation 'add'"):
        apply_operation("add", 1, 2, _ops=dict()) == 3


# -----------------------------------------------------------------------------
# Tests of specific operations ------------------------------------------------
# -----------------------------------------------------------------------------


def test_make_passthrough():
    """Tests how functions are made passthrough"""
    pt_setitem = OPERATIONS["setitem"]
    d = dict()
    rv = pt_setitem(d, "foo", 123)
    assert d["foo"] == 123
    assert rv is d

    class Foo:
        bar = None

    pt_setattr = OPERATIONS["setattr"]
    o = Foo()
    assert o.bar is None
    rv = pt_setattr(o, "bar", 123)
    assert o.bar == 123
    assert rv is o


def test_op_print_data(capsys):
    """Tests the print_data operation

    Does not test the print output; but that should be ok.
    """
    # Test passthrough
    d = dict(foo="bar")
    assert dops.print_data(d) is d

    # dantro opjects are shown differently
    dops.print_data(OrderedDataGroup(name="spam"))
    dops.print_data(ObjectContainer(name="my_dict", data=d))

    # Can also format more complex objects
    da = xr.DataArray(
        [1, 2, 3], dims=("some_dim",), coords=dict(some_dim=range(3))
    )
    da.attrs["some_attr"] = "this is an attribute"
    dops.print_data(XrDataContainer(name="my_xr_array", data=da))

    # And use an fstr for more control
    dops.print_data(
        ObjectContainer(name="my_list", data=["FOO", "BAR", "SPAM"]),
        fstr="my custom fstr: {data[2]}",
    )

    # Check the output
    captured = capsys.readouterr()
    print(captured.out)

    assert str(d) in captured.out

    assert "Tree of OrderedDataGroup 'spam'" in captured.out

    assert "<ObjectContainer 'my_dict'" in captured.out
    assert f"with data:\n{repr(d)}" in captured.out

    assert "xarray.DataArray" in captured.out
    assert "some_dim" in captured.out
    assert "some_attr" in captured.out

    assert "FOO" not in captured.out
    assert "my custom fstr" in captured.out
    assert "SPAM" in captured.out


def test_op_import_module_or_object():
    """Test the import_module_or_object operation"""
    _import = dops.import_module_or_object

    # Module import
    assert _import() is builtins
    assert _import(".data_ops") is dantro.data_ops
    assert _import(".data_ops.db") is dantro.data_ops.db
    assert _import("numpy.random") is np.random
    assert _import("numpy.random") is _import(module="numpy.random")

    # Name import, including traversal
    assert _import(name="abs") is builtins.abs is abs
    assert _import(name="abs.__name__") == "abs"
    assert (
        _import(".data_ops", "register_operation")
        is dantro.data_ops.register_operation
    )
    assert _import("numpy", "pi") is np.pi
    assert _import("numpy.random", "randint.__name__") == "randint"

    # Errors
    with pytest.raises(ModuleNotFoundError, match="foobar"):
        _import("foobar")

    with pytest.raises(AttributeError, match="abs> has no attribute 'has'!"):
        _import(name="abs.has.no.attributes")

    with pytest.raises(AttributeError, match="has no attribute 'not_pi_but_"):
        _import("numpy", "not_pi_but_something_else")


def test_op_create_mask():
    """Tests the create_mask operation"""
    da = xr.DataArray(
        name="foo",
        data=np.random.random((2, 3, 4)),
        dims=("x", "y", "z"),
        coords=dict(x=[1, 2], y=[1, 2, 3], z=[1, 2, 3, 4]),
    )

    da_masked = dops.create_mask(da, "<", 0.5)
    assert isinstance(da_masked, xr.DataArray)
    assert all(da_masked.coords["x"] == da.coords["x"])
    assert all(da_masked.coords["y"] == da.coords["y"])
    assert all(da_masked.coords["z"] == da.coords["z"])
    assert da_masked.dims == da.dims

    da_mask_neg = dops.create_mask(da, ">", 0.0)
    assert da_mask_neg.all()
    assert "(masked by '> 0.0')" in da_mask_neg.name

    da_mask_larger1 = dops.create_mask(da, ">", 1.0)
    assert not da_mask_larger1.any()
    assert "(masked by '> 1.0')" in da_mask_larger1.name

    # Should also work with unnamed arrays
    da_noname = da.copy()
    da_noname.name = None
    da_noname_masked = dops.create_mask(da_noname, "<", 0.5)
    assert isinstance(da_noname_masked, xr.DataArray)
    assert all(da_noname_masked.coords["x"] == da_noname.coords["x"])
    assert all(da_noname_masked.coords["y"] == da_noname.coords["y"])
    assert all(da_noname_masked.coords["z"] == da_noname.coords["z"])
    assert da_noname_masked.dims == da_noname.dims

    # Error messages
    with pytest.raises(KeyError, match="No boolean operator '123' available!"):
        dops.create_mask(da, "123", 0.5)


def test_op_where():
    """Tests the where operation"""
    da = xr.DataArray(
        name="foo",
        data=np.random.random((2, 3, 4)),
        dims=("x", "y", "z"),
        coords=dict(x=[1, 2], y=[1, 2, 3], z=[1, 2, 3, 4]),
    )

    da_all_nan = dops.where(da, "<", 0.0)
    assert np.isnan(da_all_nan).all()

    da_no_nan = dops.where(da, "<=", 1.0)
    assert not np.isnan(da_no_nan).any()


def test_op_count_unique():
    """Test the count_unique operation"""
    da = xr.DataArray(name="foo", data=np.random.randint(0, 6, size=(20, 20)))
    da_noname = xr.DataArray(data=np.random.randint(0, 6, size=(20, 20)))

    # introduce some NaN values which are not to be counted
    da = dops.where(da, "<", 5)
    da_noname = dops.where(da_noname, "<", 5)

    for data in [da, da.data, da_noname, da_noname.data]:
        cu = dops.count_unique(data)
        assert isinstance(cu, xr.DataArray)
        assert cu.dims == ("unique",)
        print(cu)
        assert (cu.coords["unique"] == [0, 1, 2, 3, 4]).all()
        assert "unique counts" in cu.name

    with pytest.raises(
        TypeError, match="Data needs to be of type xr.DataArray"
    ):
        dops.count_unique(da.data, dims=["dim_1"])

    cu_along_dim_1 = dops.count_unique(da, dims=["dim_1"])

    assert isinstance(cu_along_dim_1, xr.DataArray)
    assert cu_along_dim_1.dims == ("unique", "dim_0")
    assert (cu_along_dim_1.coords["unique"] == [0, 1, 2, 3, 4]).all()
    assert (cu_along_dim_1.coords["dim_0"] == da.coords["dim_0"]).all()
    assert "(unique counts)" in cu_along_dim_1.name

    cu_along_dims = dops.count_unique(da, dims=["dim_0", "dim_1"])

    assert isinstance(cu_along_dims, xr.DataArray)
    assert cu_along_dims.dims == ("unique",)
    assert (cu_along_dims.coords["unique"] == [0, 1, 2, 3, 4]).all()
    assert "(unique counts)" in cu_along_dims.name


def test_op_populate_ndarray():
    """Test np.ndarray population from a sequence of objects"""
    a0 = dops.populate_ndarray([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype=object)

    # Shape and dtype as requested
    assert (a0 == np.arange(1, 7).reshape((2, 3))).all()
    assert a0.dtype == object

    # Specifying a different order should have an effect
    a0f = dops.populate_ndarray(
        [1, 2, 3, 4, 5, 6], shape=(2, 3), dtype=float, order="F"
    )
    assert (a0f == np.arange(1, 7).reshape((2, 3), order="F")).all()

    # Argument mismatch should raise
    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray([1, 2, 3, 4, 5], shape=(2, 3))

    with pytest.raises(ValueError, match="Mismatch between array size"):
        dops.populate_ndarray([1, 2, 3, 4, 5, 6, 7], shape=(2, 3))

    with pytest.raises(TypeError, match="Without an output array given"):
        dops.populate_ndarray([1, 2, 3, 4, 5, 6])

    # Can also work directly on an output array
    out = np.zeros((2, 3))
    assert (out == 0).all()

    dops.populate_ndarray([1, 2, 3, 4, 5, 6], out=out)
    assert (out.flat == [1, 2, 3, 4, 5, 6]).all()


def test_op_build_object_array():
    """Test the build_object_array operation"""
    boa = dops.build_object_array

    # Works for list-like iterables
    arr = boa(["some", 1.23, {}, None])
    assert arr.ndim == 1
    assert arr.dtype == np.dtype("object")
    assert list(arr.data.flat) == ["some", 1.23, {}, None]
    assert (arr.coords["label"] == range(4)).all()

    # And for dicts
    arr = boa({1: "foo", 2: "bar", 4.5: "baz"})
    assert arr.ndim == 1
    assert arr.dtype == np.dtype("object")
    assert list(arr.data.flat) == ["foo", "bar", "baz"]
    assert (arr.coords["label"] == [1.0, 2.0, 4.5]).all()

    # Can also go multi-dimensional
    arr = boa({(0, 0): "foo", (2, 3): "baz", (2, 0): "bar"}, dims=("x", "y"))
    assert arr.ndim == 2
    assert arr.shape == (2, 2)
    assert arr.dtype == np.dtype("object")
    assert list(arr.data.flat)[0] == "foo"
    assert np.isnan(list(arr.data.flat)[1])
    assert list(arr.data.flat)[2:] == ["bar", "baz"]

    # Can also specify the fill value
    arr = boa(
        {(0, 0): "foo", (2, 3): "baz", (2, 0): "bar"},
        dims=("x", "y"),
        fillna="",
    )
    assert list(arr.data.flat) == ["foo", "", "bar", "baz"]

    # Errors
    # ... bad dimensionality
    with pytest.raises(ValueError, match="Can only create one-dimensional"):
        boa(["some", "1d", "iterable"], dims=("x", "y"))

    # ... scalar coordinate for multi-dimensional shape
    with pytest.raises(ValueError, match="Got scalar coordinate '1' but have"):
        boa({1: "foo", 2: "bar", 4.5: "baz"}, dims=("x", "y"))

    # ... bad coordinate tuple size
    with pytest.raises(ValueError, match="could not be matched to the specif"):
        boa({(0, 0): "foo", (1, 2, 3): "bar", 4.5: "baz"}, dims=("x", "y"))


def test_op_multi_concat(darrs):
    """Test dantro specialization of xr.concat"""
    c1 = dops.multi_concat(darrs, dims=("a", "x", "y"))
    print(c1)

    assert isinstance(c1, xr.DataArray)
    assert c1.dims == ("a", "x", "y")
    assert c1.shape == (2, 2, 3)

    # dtype is maintained
    assert c1.dtype == int

    # The case is different when aggregating over a new dimension, which is
    # interpreted as missing data, thus leading to a dtype change to allow NaNs
    c2 = dops.multi_concat(darrs.squeeze(), dims=("b",))
    print(c2)
    assert c2.dtype == float

    # When the number of to-be-concatenated dimensions does not match the
    # number of array dimensions, an error should be raised
    with pytest.raises(ValueError, match="did not match the number of dimens"):
        dops.multi_concat(darrs.squeeze(), dims=("b", "c", "d"))

    with pytest.raises(ValueError, match="did not match the number of dimens"):
        dops.multi_concat(darrs, dims=("b",))


def test_op_merge(darrs):
    """Test dantro specialization of xr.merge"""
    m1 = dops.merge(list(darrs.flat))
    assert isinstance(m1, xr.Dataset)
    assert m1["foo"].shape == (2, 2, 3)
    assert m1["foo"].dtype == float
    assert m1["foo"].dims == ("a", "x", "y")

    # Can also pass it as an object array
    m2 = dops.merge(darrs)
    assert isinstance(m2, xr.Dataset)
    assert (m1 == m2).all()

    # Can also reduce it to a DataArray
    m3 = dops.merge(darrs, reduce_to_array=True)
    assert isinstance(m3, xr.DataArray)

    # Cannot reduce it if there are multiple data variables
    darrs[0, 0, 0].name = "bar"
    with pytest.raises(ValueError, match="one and only one data variable"):
        dops.merge(darrs, reduce_to_array=True)


def test_op_expand_dims():
    """Tests dantro specialization of xarray's expand_dims method"""
    data = np.random.randint(0, 5, size=(20, 20))
    da = xr.DataArray(
        data=data,
        dims=("x", "y"),
        coords=dict(x=list(range(20)), y=list(range(20))),
    )

    da_e1 = dops.expand_dims(da, dim=dict(a=[0]))
    print(da_e1)
    assert da_e1.dims == ("a", "x", "y")
    assert (da_e1.coords["a"] == [0]).all()

    # Also works directly on the data, just without coordinates
    da_e2 = dops.expand_dims(data, dim=dict(a=[0]))
    print(da_e2)
    assert da_e2.dims == ("a", "dim_0", "dim_1")
    assert (da_e2.coords["a"] == [0]).all()


def test_op_expand_object_array():
    """Tests the expand_object_array operation"""
    expand = dops.expand_object_array

    arr = dops.populate_ndarray(
        [i * np.ones((4, 5), dtype=int) for i in range(6)],
        shape=(2, 3),
        dtype=object,
    )
    da = xr.DataArray(
        arr, dims=("a", "b"), coords=dict(a=range(10, 12), b=range(20, 23))
    )
    assert da.shape == (2, 3)
    assert da.dtype == np.dtype("O")

    # Test expansion with minimal arguments
    eda = expand(da)
    print(eda)

    assert eda.shape == (2, 3, 4, 5)  # expanded in the back
    assert eda.dims == ("a", "b", "inner_dim_0", "inner_dim_1")
    assert eda.dtype == int  # no coercion

    assert (eda.coords["a"] == [10, 11]).all()
    assert (eda.coords["b"] == [20, 21, 22]).all()
    assert (eda.coords["inner_dim_0"] == range(4)).all()
    assert (eda.coords["inner_dim_1"] == range(5)).all()

    # Explicitly coerce to float
    eda2 = expand(da, astype=float)
    assert (eda2 == eda).all()
    assert eda2.dtype == float

    # Again, now via merge
    mda = expand(da, combination_method="merge")
    assert (mda == eda).all()
    assert mda.dtype == float  # will always fall back to this

    # ... which also supports missing values, if configured to do so
    da2 = da.copy()
    da2[1, 1] = "foo"

    with pytest.raises(ValueError, match="Failed reshaping"):
        expand(da2)

    mda2 = expand(da2, allow_reshaping_failure=True)
    print(mda2)

    assert np.isnan(mda2[1, 1]).all()

    # Check error messages
    with pytest.raises(TypeError, match="Failed extracting a shape from the"):
        da2 = da.copy()
        da2[0, 0] = "some scalar non-array"
        expand(da2)

    with pytest.raises(ValueError, match="Number of dimension names.*match"):
        expand(da, dims=("foo",))

    with pytest.raises(ValueError, match="Mismatch between dimension names"):
        expand(da, coords=dict(bad_name="trivial"))

    with pytest.raises(TypeError, match="needs to be a dict or str, but was"):
        expand(da, coords=["foo"])

    with pytest.raises(ValueError, match="Invalid combination method"):
        expand(da, combination_method="bad method")


def test_op_transform_coords():
    """Tests the dantro transform_coords operation"""
    data = np.random.randint(0, 5, size=(20, 20))
    da = xr.DataArray(
        data=data,
        dims=("x", "y"),
        coords=dict(x=list(range(20)), y=list(range(20))),
    )

    da_e1 = dops.transform_coords(da, "x", lambda c: c)
    assert da_e1 is not da
    assert (da_e1 == da).all()
    assert (da_e1.coords["x"] == da.coords["x"]).all()

    da_e2 = dops.transform_coords(da, "x", lambda c: 2 * c)
    assert (da_e2.coords["x"].data == da.coords["x"].data * 2).all()

    da_e3 = dops.transform_coords(da, ["x"], lambda c: 2 * c)
    assert (da_e3.coords["x"].data == da.coords["x"].data * 2).all()

    da_e4 = dops.transform_coords(da, ["x", "y"], lambda c: 2 * c)
    assert (da_e4.coords["x"].data == da.coords["x"].data * 2).all()
    assert (da_e4.coords["y"].data == da.coords["y"].data * 2).all()

    # Can also pass function arguments
    da_e4 = dops.transform_coords(
        da, ["x", "y"], lambda c, exp: c**exp, func_kwargs=dict(exp=3)
    )
    assert (da_e4.coords["x"].data == da.coords["x"].data ** 3).all()
    assert (da_e4.coords["y"].data == da.coords["y"].data ** 3).all()


def test_op_expression():
    """Tests the ``expression`` data operation"""
    expr = dops.expression

    # Basics
    assert expr("1 + 2*3 / (4 - 5)") == -5.0
    assert expr("1 + 2*3 / (4 - five)", symbols=dict(five=5)) == -5.0
    assert expr("1 + 2*3 / (4 - 5) + .1", astype="float32") == np.float32(-4.9)
    assert expr("1 + 2*3 / (4 - 5) + .1", astype=int) == -4
    assert expr("1 + 2*3 / (4 - 5) + .1", astype="int8") == -4

    if sys.version_info >= (3, 9):
        with pytest.raises(
            TypeError, match="OverflowError:.*-4 out of bounds for uint8"
        ):
            expr("1 + 2*3 / (4 - 5) + .1", astype="uint8")

    # ... also works with expr evaluating to a literal type
    assert expr("foo - foo") == 0
    assert expr("(foo - bar)*0 + 1") == 1
    assert expr("(foo - bar)*0 + 1", evaluate=False) == 1

    # XOR operator ^ is _not_ converted to exponentiation
    assert expr("2**3") == 8
    assert expr("true^false", astype=bool) in (True, np.True_)

    # Expressions that retain free symbols cannot be fully evaluated
    with pytest.raises(TypeError, match="Failed casting.*free symbols.*"):
        expr("foo + bar", symbols=dict(foo=1))

    # ... unless they do _not_ specify a dtype
    assert expr(
        "foo + bar", symbols=dict(foo=1), astype=None
    ).free_symbols == {sym.Symbol("bar")}

    # --- Array support ---
    arrs = dict()
    a1 = arrs["a1"] = np.array([1, 2, 3])
    a2 = arrs["a2"] = np.array([0.1, 0.2, 0.3])
    a3 = arrs["a3"] = np.array([[1, 2], [3, 4]])

    assert (expr("a1 + a2 / 2", symbols=arrs) == a1 + a2 / 2).all()
    assert np.allclose(expr("a1 ** exp(3)", symbols=arrs), a1 ** np.exp(3))

    # --- Advanced features ---
    # List definition ...
    assert (expr("[1,2,3]") == [1, 2, 3]).all()
    assert (expr("[1,2,3] * 2") == [1, 2, 3, 1, 2, 3]).all()
    assert (
        expr("[1,2,3] * foo", symbols=dict(foo=2)) == [1, 2, 3, 1, 2, 3]
    ).all()
    assert (expr("[1,2,3]", evaluate=False) == [1, 2, 3]).all()

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
        "a, b: a+b": ([1, 2], {}, 3),
        "a: ceil(a)": ([0.5], {}, 1.0),
        "a: sin(a)": ([0.0], {}, 0.0),
        "*_: cos(pi)": ([], {}, -1.0),
        "*_: abs(cos(pi))": ([], {}, +1.0),
        "a: np.mean(a)": ([[1, 2, 3]], {}, 2),
        "a: xr.DataArray(a).mean()": ([[1, 2, 3]], {}, 2),
        "**ks: {k for k in ks}": ([], dict(a=1, b=2), {"a", "b"}),
        "*a: range(*a)": ([0, 10, 2], {}, range(0, 10, 2)),
        "*a: slice(*a)": ([0, 10, 2], {}, slice(0, 10, 2)),
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
