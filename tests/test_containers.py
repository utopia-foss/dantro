"""Test the BaseDataContainer-derived classes"""

import math
import operator
import sys

import dask.array as da
import h5py as h5
import numpy as np
import pytest
import xarray as xr

from dantro.base import BaseDataContainer, CheckDataMixin, ItemAccessMixin
from dantro.containers import (
    CONTAINERS,
    LinkContainer,
    MutableSequenceContainer,
    NumpyDataContainer,
    ObjectContainer,
    PathContainer,
    StringContainer,
    XrDataContainer,
    is_container,
    register_container,
)
from dantro.groups import OrderedDataGroup
from dantro.mixins import ForwardAttrsToDataMixin
from dantro.mixins.base import UnexpectedTypeWarning
from dantro.mixins.proxy_support import Hdf5ProxySupportMixin
from dantro.proxy import Hdf5DataProxy
from dantro.utils import Link


class DummyContainer(ItemAccessMixin, BaseDataContainer):
    """A dummy container that fulfills all the requirements of the abstract
    BaseDataContainer class."""

    def _format_info(self):
        return "dummy"


# Fixtures --------------------------------------------------------------------
from .test_base import pickle_roundtrip
from .test_proxy import tmp_h5file


@pytest.fixture
def tmp_h5_dset(tmp_h5file) -> h5.Dataset:
    """Creates a temporary hdf5 dataset"""
    dset = tmp_h5file.create_dataset(
        "init", data=np.zeros(shape=(1, 2, 3), dtype=int)
    )
    return dset


# Tests -----------------------------------------------------------------------


def test_basics():
    """Tests initialisation of the DummyContainer class"""
    # Simple init
    dc = DummyContainer(name="dummy", data="foo")

    # Assert the name is a string
    assert isinstance(dc.name, str)

    # Check invalid name arguments
    with pytest.raises(TypeError, match="Name for DummyContainer needs"):
        DummyContainer(name=123, data="foo")

    # Invalid names
    for bad_name in (
        "a/path/that/contains/the/PATH_JOIN_CHAR",
        "some_(weird):characters!",
        "backslash\\",
        "name*",
        "name?",
        "bra]cke[ts",
    ):
        with pytest.raises(ValueError, match="Invalid name"):
            DummyContainer(name=bad_name, data="foo")


def test_registration():
    """Tests registration via the is_container decorator. Only tests cases
    that are not already covered during general import of dantro"""

    # Invalid type
    with pytest.raises(TypeError, match="needs to be a subclass"):

        @is_container
        class NotAContainer:
            pass

    assert "NotAContainer" not in CONTAINERS

    # Custom name
    @is_container("object")
    class MyObjectContainer(ObjectContainer):
        pass

    assert "object" in CONTAINERS
    assert "MyObjectContainer" in CONTAINERS

    register_container(MyObjectContainer, "foo")
    assert "foo" in CONTAINERS

    register_container(MyObjectContainer, "foo", overwrite_existing=True)
    assert "foo" in CONTAINERS


# -----------------------------------------------------------------------------


def test_CheckDataMixin():
    """Checks the CheckDataMixin class."""
    # Define some test classes ................................................

    class TestContainerA(CheckDataMixin, DummyContainer):
        """All types allowed"""

    class TestContainerB(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""

        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = "raise"

    class TestContainerC(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""

        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = "warn"

    class TestContainerD(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""

        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = "ignore"

    class TestContainerE(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""

        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = "invalid"

    # Tests ...................................................................
    # Run tests for A
    TestContainerA(name="foo", data="bar")

    # Run tests for B
    TestContainerB(name="foo", data=["my", "list"])
    TestContainerB(name="foo", data=("my", "tuple"))

    with pytest.raises(TypeError, match="Unexpected type <class 'str'> for.*"):
        TestContainerB(name="foo", data="bar")

    # Run tests for C
    with pytest.warns(
        UnexpectedTypeWarning, match="Unexpected type <class 'str'> for.*"
    ):
        TestContainerC(name="foo", data="bar")

    # Run tests for D
    TestContainerD(name="foo", data="bar")

    # Run tests for E
    with pytest.raises(ValueError, match="Illegal value 'invalid' for class"):
        TestContainerE(name="foo", data="bar")


# -----------------------------------------------------------------------------
# General containers


def test_MutableSequenceContainer():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation of sequence-like data
    msc1 = MutableSequenceContainer(name="foo", data=["bar", "baz"])
    msc2 = MutableSequenceContainer(
        name="foo", data=["bar", "baz"], attrs=dict(one=1, two="two")
    )

    # There will be warnings for other data types:
    with pytest.warns(UnexpectedTypeWarning):
        msc3 = MutableSequenceContainer(name="bar", data=("hello", "world"))

    with pytest.warns(UnexpectedTypeWarning):
        msc4 = MutableSequenceContainer(name="baz", data=None)

    mscs = (msc1, msc2, msc3)

    # Basic assertions ........................................................
    # Data access
    assert msc1.data == ["bar", "baz"] == msc1[:]
    assert msc2.data == ["bar", "baz"] == msc2[:]

    # Attribute access
    assert msc2.attrs == dict(one=1, two="two")
    assert msc2.attrs["one"] == 1

    # this will still work, as it is a sequence
    assert msc3.data == ("hello", "world") == msc3[:]
    with pytest.raises(TypeError):
        msc4[:]

    # Test insertion into the list ............................................
    msc1.insert(0, "foo")
    assert msc1.data == ["foo", "bar", "baz"]

    # This should not work:
    with pytest.raises(AttributeError):
        msc3.insert(len(msc3), ("!",))

    # Properties ..............................................................
    # strings
    for msc in mscs:
        str(msc)
        f"{msc:info,cls_name,name}"
        f"{msc}"

        with pytest.raises(ValueError):
            f"{msc:illegal_formatspec}"

    # Pickling ................................................................
    for msc in mscs:
        assert pickle_roundtrip(msc) == msc


def test_LinkContainer():
    """Test the behaviour of a LinkContainer inside a hierarchy"""
    root = OrderedDataGroup(name="root")
    group = root.new_group("group")
    links = root.new_group("links")
    data = root.new_container("data", Cls=StringContainer, data="some_string")

    links.new_container(
        "group", Cls=LinkContainer, data=Link(anchor=root, rel_path="group")
    )
    assert links["group"].path == "/root/links/group"
    assert links["group"].data.path == "/root/group"

    links.new_container(
        "data", Cls=LinkContainer, data=Link(anchor=root, rel_path="data")
    )
    assert links["data"].path == "/root/links/data"
    assert links["data"].data.path == "/root/data"
    assert links["data"].upper() == "SOME_STRING"

    print(root.tree)

    # Test extended formatting information
    assert "root -> data" in str(links["data"])

    # Pickling, testing via full tree
    # NOTE Comparison has to happen via the tree, as comparison via data may
    #      lead to infinite recursion in this case
    assert pickle_roundtrip(data) == data
    assert pickle_roundtrip(links).tree == links.tree
    assert pickle_roundtrip(root).tree == root.tree


def test_StringContainer():
    """Test the behaviour of a StringContainer"""
    # Basic initialization of a string container
    test_data = "This is a test string."
    sc = StringContainer(name="oof", data=test_data)

    # Assert that the data is of type string
    assert isinstance(sc.data, str)

    # Test PaththroughContainer functionality here
    assert len(sc.data) == len(test_data)
    assert sc.data.upper() == test_data.upper()

    # Pickling
    assert pickle_roundtrip(sc) == sc


# -----------------------------------------------------------------------------
# PathContainer


def test_PathContainer():
    p1 = PathContainer(name="p1", data="some/path")

    # Have additional fs_path attribute
    assert str(p1.fs_path) == "some/path"

    # … but can also access the underlying Path object interface (passthrough)
    assert str(p1.joinpath("foo")) == "some/path/foo"

    # item interface is not implemented
    with pytest.raises(NotImplementedError):
        p1["foo"]
    with pytest.raises(NotImplementedError):
        del p1["foo"]
    with pytest.raises(NotImplementedError):
        p1["foo"] = "bar"

    # Can also create a new path from a parent group
    from dantro.groups import DirectoryGroup

    grp = DirectoryGroup(name="root", dirpath="root")
    assert str(grp.fs_path) == "root"

    p2 = grp.new_container("some.file")
    assert str(p2.fs_path) == "root/some.file"

    # … but the parent needs to be present and of correct type
    with pytest.raises(TypeError, match="need a parent.*None"):
        PathContainer(name="will fail")

    with pytest.raises(TypeError, match="need a parent.*OrderedDataGroup"):
        PathContainer(name="will fail", parent=OrderedDataGroup(name="foo"))


# -----------------------------------------------------------------------------
# Numeric containers


def test_NumpyDataContainer():
    """Tests whether the __init__method behaves as desired"""
    # Basic initialization of Numpy ndarray-like data
    ndc1 = NumpyDataContainer(name="oof", data=np.array([1, 2, 3]))
    ndc2 = NumpyDataContainer(name="zab", data=np.array([2, 4, 6]))

    # Initialisation with lists should also work
    NumpyDataContainer(name="rab", data=[3, 6, 9])

    # Ensure that the CheckDataMixin does its job
    with pytest.raises(TypeError, match="Unexpected type"):
        NumpyDataContainer(name="zab", data=("not", "a", "valid", "type"))

    # Test the ForwardAttrsToDataMixin on a selection of numpy functions
    l1 = [1, 2, 3]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    npa1 = np.array(l1)
    assert ndc1.size == npa1.size
    assert ndc1.ndim == npa1.ndim
    assert ndc1.max() == npa1.max()
    assert ndc1.mean() == npa1.mean()
    assert ndc1.cumsum()[-1] == npa1.cumsum()[-1]

    # Check __len__ magic method behaves same as numpy.ndarray.__len__
    assert len(ndc1) == len(ndc1.data) == 3
    assert len(ndc2) == len(ndc2.data) == 3

    # Test the NumbersMixin
    ndc1 = NumpyDataContainer(name="oof", data=np.array([1, 2, 3]))
    ndc2 = NumpyDataContainer(name="zab", data=np.array([2, 4, 6]))
    add = ndc1 + ndc2
    sub = ndc1 - ndc2
    mult = ndc1 * ndc2
    div = ndc1 / ndc2
    floordiv = ndc1 // ndc2
    mod = ndc1 % ndc2
    div_mod = divmod(ndc1, ndc2)
    power = ndc1**ndc2

    # Test NumbersMixin function for operations on two numpy arrays
    l1 = [1, 2, 3]
    l2 = [2, 4, 6]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2 = NumpyDataContainer(name="zab", data=np.array(l2))
    npa1 = np.array(l1)
    npa2 = np.array(l2)

    add_npa = npa1 + npa2
    sub_npa = npa1 - npa2
    mult_npa = npa1 * npa2
    div_npa = npa1 / npa2
    floordiv_npa = npa1 // npa2
    mod_npa = npa1 % npa2
    power_npa = npa1**npa2

    assert add.data.all() == add_npa.all()
    assert sub.data.all() == sub_npa.all()
    assert mult.data.all() == mult_npa.all()
    assert div.data.all() == div_npa.all()
    assert floordiv.data.all() == floordiv_npa.all()
    assert mod.data.all() == mod_npa.all()
    assert div_mod.data[0].all() == floordiv_npa.all()
    assert div_mod.data[1].all() == mod_npa.all()
    assert power.data.all() == power_npa.all()

    assert ndc1.data.all() == npa1.all()

    # Test NumbersMixin function for operations on one numpy array and a number
    add_number = ndc1 + 4.2
    sub_number = ndc1 - 4.2
    mult_number = ndc1 * 4.2
    div_number = ndc1 / 4.2
    floordiv_number = ndc1 // 4.2
    divmod_number = divmod(ndc1, 4.2)
    mod_number = ndc1 % 4.2
    power_number = ndc1**4.2

    add_npa_number = npa1 + 4.2
    sub_npa_number = npa1 - 4.2
    mult_npa_number = npa1 * 4.2
    div_npa_number = npa1 / 4.2
    floordiv_npa_number = npa1 // 4.2
    divmod_npa_number = divmod(npa1, 4.2)
    mod_npa_number = npa1 % 4.2
    power_npa_number = npa1**4.2

    assert add_number.data.all() == add_npa_number.all()
    assert sub_number.data.all() == sub_npa_number.all()
    assert mult_number.data.all() == mult_npa_number.all()
    assert div_number.data.all() == div_npa_number.all()
    assert floordiv_number.data.all() == floordiv_npa_number.all()
    assert mod_number.data.all() == mod_npa_number.all()
    assert divmod_number.data[0].all() == floordiv_npa_number.all()
    assert divmod_number.data[1].all() == mod_npa_number.all()
    assert power_number.data.all() == power_number.all()

    # Test inplace operations
    l1 = [1.0, 2.0, 3.0]
    l2 = [2.0, 4.0, 6.0]
    ndc1_inplace = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2_inplace = NumpyDataContainer(name="zab", data=np.array(l2))
    npa1_inplace = np.array(l1)
    npa2_inplace = np.array(l2)

    ndc1_inplace += ndc2_inplace
    npa1_inplace += npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace -= ndc2_inplace
    npa1_inplace -= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace *= ndc2_inplace
    npa1_inplace *= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace /= ndc2_inplace
    npa1_inplace /= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace //= ndc2_inplace
    npa1_inplace //= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace %= ndc2_inplace
    npa1_inplace %= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    ndc1_inplace **= ndc2_inplace
    npa1_inplace **= npa2_inplace
    assert ndc1_inplace.all() == npa1_inplace.all()

    # Test unary operations
    l1 = [1.0, -2.0, 3.0]
    l2 = [-2.0, 4.0, 6.0]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2 = NumpyDataContainer(name="zab", data=np.array(l2))
    npa1 = np.array(l1)
    npa2 = np.array(l2)

    assert (-ndc1).all() == (-npa1).all()
    assert (+ndc1).all() == (+npa1).all()
    assert abs(ndc1).all() == abs(npa1).all()
    assert ~ndc1.all() == ~npa1.all()

    # Test ComparisonMixin
    l1 = [1, 2, 3]
    l2 = [0, 2, 4]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2 = NumpyDataContainer(name="tada", data=np.array(l2))
    npa1 = np.array(l1)
    npa2 = np.array(l2)

    eq = ndc1 == ndc2
    ne = ndc1 != ndc2
    lt = ndc1 < ndc2
    le = ndc1 <= ndc2
    gt = ndc1 > ndc2
    ge = ndc1 >= ndc2

    eq_npa = npa1 == npa2
    ne_npa = npa1 != npa2
    lt_npa = npa1 < npa2
    le_npa = npa1 <= npa2
    gt_npa = npa1 > npa2
    ge_npa = npa1 >= npa2

    assert eq.all() == eq_npa.all()
    assert ne.all() == ne_npa.all()
    assert lt.all() == lt_npa.all()
    assert le.all() == le_npa.all()
    assert gt.all() == gt_npa.all()
    assert ge.all() == ge_npa.all()

    with pytest.raises(ValueError):
        bool(ndc1)

    # Test copy
    ndc = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc_copy = ndc.copy()
    assert ndc_copy.data is not ndc.data
    assert np.all(ndc_copy.data == ndc.data)

    # Test string representation
    assert ndc._format_info().startswith(str(ndc.dtype))

    # Test that the size of the container's data is taken into account
    ndc = NumpyDataContainer(name="some_zeros", data=np.zeros((100, 100, 100)))
    assert sys.getsizeof(ndc) > sys.getsizeof(ndc.data)


def test_XrDataContainer():
    """Tests the XrDataContainer"""

    # Basic initialization of Numpy ndarray-like data
    xrdc = XrDataContainer(name="xrdc", data=np.array([1, 2, 3]))

    # Initialisation with lists and xr.DataArrays should also work
    XrDataContainer(name="rab", data=[2, 4, 6])
    XrDataContainer(name="zab", data=xr.DataArray([2, 4, 6]))

    # Initialisation with dimension names .....................................
    xrdc = XrDataContainer(
        name="xrdc_dims_1", data=[3, 6, 9], attrs=dict(dims=["first_dim"])
    )
    assert "first_dim" in xrdc.data.dims

    xrdc = XrDataContainer(
        name="xrdc_dims_2",
        data=[[1, 2, 3], [4, 5, 6]],
        attrs=dict(dims=["first_dim", "second_dim"]),
    )
    assert "first_dim" in xrdc.data.dims
    assert "second_dim" in xrdc.data.dims

    xrdc = XrDataContainer(
        name="xrdc_dims_prefix_1",
        data=[1, 2, 4],
        attrs=dict(dim_name__0="first_dim"),
    )
    assert "first_dim" in xrdc.data.dims

    xrdc = XrDataContainer(
        name="xrdc_dims_prefix_2",
        data=[[1, 2, 4], [2, 4, 8]],
        attrs=dict(dim_name__0="first_dim"),
    )
    assert "first_dim" in xrdc.data.dims
    assert "dim_1" in xrdc.data.dims

    with pytest.raises(
        ValueError, match="Number of given dimension names does not match"
    ):
        xrdc = XrDataContainer(
            name="xrdc_dims_mismatch",
            data=[3, 6, 9],
            attrs=dict(dims=["first_dim", "second_dim"]),
        )

    with pytest.raises(
        ValueError, match="Could not extract the dimension number from"
    ):
        xrdc = XrDataContainer(
            name="xrdc_dims_mismatch",
            data=[1, 2, 4],
            attrs=dict(dim_name__mismatch="first_dim"),
        )

    # With a dimension name list given, the other attributes overwrite the
    # ones given in the list
    xrdc = XrDataContainer(
        name="xrdc_dims_3",
        data=[[1, 2, 3], [4, 5, 6]],
        attrs=dict(dims=["first_dim", "second_dim"], dim_name__0="foo"),
    )
    assert xrdc.data.dims == ("foo", "second_dim")

    # Pathological cases: attribute is an iterable of scalar numpy arrays
    xrdc = XrDataContainer(
        name="xrdc_dims_3",
        data=[[1, 2, 3], [4, 5, 6]],
        attrs=dict(
            dims=[np.array("first_dim", dtype="U"), np.array(["second_dim"])]
        ),
    )
    assert xrdc.data.dims == ("first_dim", "second_dim")

    xrdc = XrDataContainer(
        name="xrdc_dims_3",
        data=[[1, 2, 3], [4, 5, 6]],
        attrs=dict(
            dim_name__0=np.array("first_dim", dtype="U"),
            dim_name__1=np.array(["second_dim"]),
        ),
    )
    assert xrdc.data.dims == ("first_dim", "second_dim")

    # Bad dimension name attribute type
    with pytest.raises(TypeError, match="sequence of strings, but not"):
        XrDataContainer(
            name="xrdc", data=[[1, 2, 3], [4, 5, 6]], attrs=dict(dims="13")
        )

    with pytest.raises(TypeError, match="needs to be an iterable"):
        XrDataContainer(
            name="xrdc", data=[[1, 2, 3], [4, 5, 6]], attrs=dict(dims=123)
        )

    with pytest.raises(TypeError, match="need to be strings, got"):
        XrDataContainer(
            name="xrdc",
            data=[[1, 2, 3], [4, 5, 6]],
            attrs=dict(dims=["foo", 123]),
        )

    with pytest.raises(TypeError, match="need be strings, but the attribute"):
        XrDataContainer(
            name="xrdc",
            data=[[1, 2, 3], [4, 5, 6]],
            attrs=dict(dim_name__0=123),
        )

    # Bad dimension number
    with pytest.raises(ValueError, match="exceeds the given rank 2!"):
        XrDataContainer(
            name="xrdc_dims_3",
            data=[[1, 2, 3], [4, 5, 6]],
            attrs=dict(dim_name__10="foo"),
        )

    # Initialisation with coords ..............................................
    # Explicitly given coordinates . . . . . . . . . . . . . . . . . . . . . .
    coords__time = ["Jan", "Feb", "Mar"]
    xrdc = XrDataContainer(
        name="xrdc",
        data=xr.DataArray([1, 2, 3]),
        attrs=dict(dims=["time"], coords__time=coords__time),
    )
    assert "time" in xrdc.data.dims
    assert np.all(coords__time == xrdc.data.coords["time"])

    coords__time = ["Jan", "Feb", "Mar", "Apr"]
    coords__space = ["IA", "IL", "IN"]

    xrdc = XrDataContainer(
        name="xrdc",
        data=np.random.rand(4, 3),
        attrs=dict(
            dims=["time", "space"],
            coords__time=coords__time,
            coords__space=coords__space,
        ),
    )
    assert "time" in xrdc.data.dims
    assert np.all(coords__time == xrdc.data.coords["time"])

    with pytest.raises(ValueError, match="Could not associate coordinates"):
        xrdc = XrDataContainer(
            name="xrdc",
            data=xr.DataArray([1, 2, 3]),
            attrs=dict(dims=["time"], coords__time=["Jan", "Feb"]),
        )

    with pytest.raises(ValueError, match="Got superfluous attribute 'coords_"):
        xrdc = XrDataContainer(
            name="xrdc_coord_mismatch",
            data=[1, 2, 3, 4],
            attrs=dict(dims=["time"], coords__space=["IA", "IL", "IN"]),
        )

    # Coordinates and data don't match in length
    with pytest.raises(ValueError, match="Could not associate coordinates"):
        XrDataContainer(
            name="xrdc",
            data=np.arange(10),
            attrs=dict(dims=["time"], coords__time=[1, 2, 3]),
        )

    # Coordinates as range expression . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"], coords__time=[10], coords_mode__time="arange"
        ),
    )
    assert np.all(np.arange(10) == xrdc.data.coords["time"])

    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"], coords__time=[3, 13], coords_mode__time="arange"
        ),
    )
    assert np.all(np.arange(3, 13) == xrdc.data.coords["time"])

    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"],
            coords__time=[0, 100, 10],
            coords_mode__time="arange",
        ),
    )
    assert np.all(np.arange(0, 100, 10) == xrdc.data.coords["time"])

    # start and step values . . . . . . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"],
            coords__time=[0, 2],
            coords_mode__time="start_and_step",
        ),
    )
    assert np.all(list(range(0, 20, 2)) == xrdc.data.coords["time"])

    # trivial . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"], coords__time=[0, 2], coords_mode__time="trivial"
        ),
    )
    assert np.all(list(range(10)) == xrdc.data.coords["time"])

    # scalar . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=[0],
        attrs=dict(
            dims=["time"], coords__time=[42], coords_mode__time="scalar"
        ),
    )
    assert xrdc.data.coords["time"] == [42]

    # linked mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(
            dims=["time"],
            coords__time="",  # refer to itself
            coords_mode__time="linked",
        ),
    )
    # NOTE This works properly only when using proxies; tested seperately

    # array-like mode value (as frequently produced by hdf5 data) . . . . . . .
    xrdc = XrDataContainer(
        name="xrdc",
        data=np.arange(10),
        attrs=dict(dims=["time"], coords_mode__time=np.array(["trivial"])),
    )
    assert (xrdc.coords["time"] == list(range(10))).all()

    # Error messages . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Invalid coordinate mode
    with pytest.raises(ValueError, match="Invalid mode 'invalid' to interpre"):
        XrDataContainer(
            name="invalid_coord_type",
            data=[1, 2, 3],
            attrs=dict(
                dims=["time"],
                coords__time=[0, 1, 2],
                coords_mode__time="invalid",
            ),
        )

    with pytest.raises(
        ValueError, match="Failed extracting coordinates .* 'scalar'.*"
    ):
        XrDataContainer(
            name="bad_coord_val",
            data=[1, 2, 3],
            attrs=dict(
                dims=["time"],
                coords__time=[0, 1, 2],
                coords_mode__time="scalar",
            ),
        )

    # without strict attribute checking . . . . . . . . . . . . . . . . . . . .
    class TolerantXrDataContainer(XrDataContainer):
        _XRC_STRICT_ATTR_CHECKING = False

    xrdc = TolerantXrDataContainer(
        name="tolerant_xrdc",
        data=[1, 2, 3],
        attrs=dict(
            dims=["time"],
            coords__time=[0, 1, 2],
            coords__foo="bar",  # throws not
        ),
    )

    # Carrying over attributes ................................................
    # Attributes are carried over as expected; prefixed attributes are not!
    xrdc = XrDataContainer(
        name="xrdc",
        data=[1, 2, 3],
        attrs=dict(
            foo="bar", dims=["time"], coords__time=["Jan", "Feb", "Mar"]
        ),
    )
    assert ("foo", "bar") in xrdc.data.attrs.items()
    assert "dims" not in xrdc.data.attrs
    assert "coords__time" not in xrdc.data.attrs

    # Copying .................................................................
    xrdc_copy = xrdc.copy()
    assert xrdc_copy.data is not xrdc.data
    assert np.all(xrdc_copy.data == xrdc.data)

    # __len__ .................................................................
    assert len(xrdc) == len(xrdc.data) == 3

    # String representation ...................................................
    assert xrdc._format_info().startswith(str(xrdc.dtype))

    # Format info contains dimension names
    assert all([d in xrdc._format_info() for d in xrdc.dims])

    xrdc = XrDataContainer(
        name="format_info",
        data=np.zeros((2, 3, 4)),
        attrs=dict(dim_name__0="first", dim_name__2="third"),
    )
    assert "first" in xrdc.dims
    assert "third" in xrdc.dims
    assert all([d in xrdc._format_info() for d in xrdc.dims])


def test_XrDataContainer_proxy_support(tmp_h5_dset):
    """Test proxy support for XrDataContainer"""

    # Specialize a class with proxy support
    class Hdf5ProxyXrDC(Hdf5ProxySupportMixin, XrDataContainer):
        pass

    # Some attributes to initialize containers
    attrs = dict(
        foo="bar",
        dims=["x", "y", "z"],
        coords__x=["1 m"],
        coords__z=["1 cm", "2 cm", "3 cm"],
    )

    # Check that proxy support is enabled
    assert Hdf5ProxyXrDC.DATA_ALLOW_PROXY

    # Create a proxy
    proxy = Hdf5DataProxy(obj=tmp_h5_dset)

    # Create a XrDataContainer with proxy support
    pxrdc = Hdf5ProxyXrDC(name="xrdc", data=proxy, attrs=attrs)

    # Initialize another one directly without using the proxy
    pxrdc_direct = Hdf5ProxyXrDC(
        name="xrdc", data=tmp_h5_dset[()], attrs=attrs
    )

    # Check that the _data member is now a proxy
    assert isinstance(pxrdc._data, Hdf5DataProxy)

    # Support mixin should also give the same result
    assert pxrdc.data_is_proxy

    # And the info string should contain the word "proxy"
    assert "proxy" in pxrdc._format_info()

    # Make a copy of the container
    pxrdc_copy = pxrdc.copy()

    # Check that after copying it is still a proxy
    assert isinstance(pxrdc_copy._data, Hdf5DataProxy)

    # Check wether the fundamental attributes are correct
    assert pxrdc.shape == (1, 2, 3)
    assert pxrdc.dtype == int
    assert pxrdc.ndim == 3
    assert pxrdc.size == 6
    assert pxrdc.chunks is None

    # TODO Would be awesome to have access to extracted metadata here

    # ... should not have resolved the proxy
    assert isinstance(pxrdc._data, Hdf5DataProxy)

    # Resolve the proxy by calling the data property
    pxrdc.data
    assert not pxrdc.data_is_proxy

    # Now the data should be an xarray
    assert isinstance(pxrdc._data, xr.DataArray)

    # Format string should not contain proxy now
    assert "proxy" not in pxrdc._format_info()

    # ... and check that it is the same as the XrDataContainer initialized
    # without proxy
    assert np.all(pxrdc.data == pxrdc_direct.data)
    assert pxrdc.attrs == pxrdc_direct.attrs

    # All properties should also be the same
    assert pxrdc.shape == pxrdc_direct.shape
    assert pxrdc.dtype == pxrdc_direct.dtype
    assert pxrdc.ndim == pxrdc_direct.ndim
    assert pxrdc.size == pxrdc_direct.size
    assert pxrdc.chunks == pxrdc_direct.chunks

    # Test re-instatement .....................................................

    # Create a new proxy object
    proxy = Hdf5DataProxy(obj=tmp_h5_dset)

    # With retainment enabled, it should be retained after resolution
    pxrdc = Hdf5ProxyXrDC(name="xrdc", data=proxy, attrs=attrs)
    pxrdc.PROXY_RETAIN = True
    assert pxrdc.data_is_proxy
    assert pxrdc._retained_proxy is None

    pxrdc.data
    assert not pxrdc.data_is_proxy
    assert pxrdc._retained_proxy is proxy

    # ... allowing to reinstate the proxy, releasing the data
    pxrdc.reinstate_proxy()
    assert pxrdc.data_is_proxy

    # If it is already a proxy, nothing happens
    pxrdc.reinstate_proxy()
    assert pxrdc.data_is_proxy

    # With retainment disabled (default), it should not be retained
    pxrdc = Hdf5ProxyXrDC(name="xrdc", data=proxy, attrs=attrs)
    assert not pxrdc.PROXY_RETAIN
    assert pxrdc.data_is_proxy
    assert pxrdc._retained_proxy is None

    pxrdc.data
    assert not pxrdc.data_is_proxy
    assert pxrdc._retained_proxy is None

    # Thus, reinstatement cannot work
    with pytest.raises(ValueError, match="Could not reinstate a proxy for"):
        pxrdc.reinstate_proxy()

    # Test all the different fail actions.
    # NOTE These will generate warnings in the logs, but that's intended.
    pxrdc.PROXY_REINSTATE_FAIL_ACTION = "warn"
    with pytest.warns(RuntimeWarning, match="Could not reinstate"):
        pxrdc.reinstate_proxy()

    pxrdc.PROXY_REINSTATE_FAIL_ACTION = "log_warn"
    pxrdc.reinstate_proxy()

    pxrdc.PROXY_REINSTATE_FAIL_ACTION = "log_warning"
    pxrdc.reinstate_proxy()

    pxrdc.PROXY_REINSTATE_FAIL_ACTION = "log_debug"
    pxrdc.reinstate_proxy()

    with pytest.raises(ValueError, match="Invalid PROXY_REINSTATE_FAIL_"):
        pxrdc.PROXY_REINSTATE_FAIL_ACTION = "bad_value"
        pxrdc.reinstate_proxy()

    # String representation ...................................................
    # Without dimension info and with proxy still in place, the shape is given
    pxrdc = Hdf5ProxyXrDC(name="format_info_test", data=proxy, attrs=attrs)

    assert pxrdc.data_is_proxy
    assert not pxrdc._metadata_was_applied

    assert all([d in pxrdc._format_info() for d in pxrdc._dim_names])

    assert pxrdc.data_is_proxy
    assert not pxrdc._metadata_was_applied

    # For case without extracted metadata, expect "shape"
    pxrdc = Hdf5ProxyXrDC(
        name="format_info_test", data=proxy, extract_metadata=False
    )  # no attributes

    assert pxrdc.data_is_proxy
    assert not pxrdc._metadata_was_applied

    assert "shape" in pxrdc._format_info()
    assert "dim_0" not in pxrdc._format_info()

    assert pxrdc.data_is_proxy
    assert not pxrdc._metadata_was_applied


def test_XrDataContainer_dask_integration(tmp_h5file):
    """Tests dask integration in proxy-supporting XrDataContainer"""

    class XrDC(Hdf5ProxySupportMixin, XrDataContainer):
        PROXY_RETAIN = True

    # Build some proxy objects
    proxy_dask = Hdf5DataProxy(
        obj=tmp_h5file["chunked/zeros"], resolve_as_dask=True
    )
    proxy_nodask = Hdf5DataProxy(obj=tmp_h5file["chunked/zeros"])

    # Prepare the coordinates that are to be used as attributes
    attrs = dict(
        dims=("x", "y", "z", "t"),
        coords__x=["1km", "2km", "3km"],
        coords__y=["1m", "2m", "3m", "4m"],
        coords__z=["1mm", "2mm", "3mm", "4mm", "5mm"],
        coords__t=["1s", "2s", "3s", "4s", "5s", "6s"],
    )

    # Construct the data containers
    xrdc = XrDC(name="with_dask", data=proxy_dask, attrs=attrs)
    xrdc_nodask = XrDC(name="without_dask", data=proxy_nodask, attrs=attrs)

    # Work on them without resolving, and see that they behave in the same way
    assert xrdc.data_is_proxy
    assert xrdc_nodask.data_is_proxy

    assert xrdc.proxy.chunks == xrdc.chunks
    assert xrdc.proxy.chunks == (3, 4, 5, 1)

    assert xrdc_nodask.proxy.chunks == xrdc_nodask.chunks
    assert xrdc_nodask.proxy.chunks == (3, 4, 5, 1)

    assert xrdc.shape == xrdc_nodask.shape

    # Both still proxy
    assert xrdc.data_is_proxy
    assert xrdc_nodask.data_is_proxy

    # Now, resolve them
    xrdc.data
    xrdc_nodask.data

    assert not xrdc.data_is_proxy
    assert not xrdc_nodask.data_is_proxy

    # Check that it is still an xarray
    assert isinstance(xrdc.data, xr.DataArray)
    assert isinstance(xrdc_nodask.data, xr.DataArray)

    # And dimension labels and coordinates were applied
    assert xrdc.dims == ("x", "y", "z", "t")
    assert xrdc_nodask.dims == xrdc.dims

    assert (xrdc.coords["x"] == ["1km", "2km", "3km"]).all()
    assert (xrdc.coords["x"] == xrdc_nodask.coords["x"]).all()

    # ... but has a dask array beneath it, unlike the other one
    assert xrdc.__dask_keys__()
    assert not xrdc_nodask.__dask_keys__()

    # How about calculating with them? This should be possible without any
    # calls to compute or persist; same interface in both cases...
    some_ints = np.random.randint(10, size=xrdc.shape)
    xrdc += some_ints
    xrdc_nodask += some_ints
    assert (xrdc == xrdc_nodask).all()

    # Also, compute and persist calls should work on both, regardless of the
    # underlying data being dask or not
    assert (xrdc.compute() == xrdc_nodask.compute()).all()
    assert (xrdc.persist() == xrdc_nodask.persist()).all()

    # Check that reinstatement of the proxy also works
    xrdc.reinstate_proxy()
    xrdc_nodask.reinstate_proxy()

    assert xrdc.data_is_proxy
    assert xrdc_nodask.data_is_proxy


def test_XrDataContainer_linked_coordinates(tmp_h5_dset):
    """Test 'linked' and 'from_path' coordinate modes for XrDataContainer"""

    class Hdf5ProxyXrDC(Hdf5ProxySupportMixin, XrDataContainer):
        pass

    # Check that proxy support is enabled
    assert Hdf5ProxyXrDC.DATA_ALLOW_PROXY

    # Create a proxy
    proxy = Hdf5DataProxy(obj=tmp_h5_dset)  # shape: (1,2,3)

    # Create a XrDataContainer with proxy support and linked coordinates
    xrdc = Hdf5ProxyXrDC(
        name="xrdc",
        data=proxy,
        attrs=dict(
            dims=["x", "y", "z"],
            coords_mode__x="linked",
            coords__x="some_other_data",
            coords_mode__y="linked",
            coords__y="../coords/y",
            coords_mode__z="linked",
            coords__z="../coords/more/z",
        ),
    )

    # Should have succeeded and be a proxy now
    assert xrdc.data_is_proxy

    # Now, incorporate it into a tree
    root = OrderedDataGroup(name="root")

    g_data = root.new_group("data")
    g_data.add(xrdc)
    g_data.new_container("some_other_data", Cls=XrDataContainer, data=[3.14])

    g_coords = root.new_group("coords")
    g_coords.new_container("y", Cls=XrDataContainer, data=[23, 42])

    g_more_coords = g_coords.new_group("more")
    g_more_coords.new_container("z", Cls=XrDataContainer, data=[2, 4, 8])

    # The original data should still be proxy
    assert xrdc.data_is_proxy

    # Now, resolving it should lead to link resolution ... which happens in the
    # backround and just leads to the data being available as coordiantes
    assert (xrdc.coords["x"] == [3.14]).all()
    assert (xrdc.coords["y"] == [23, 42]).all()
    assert (xrdc.coords["z"] == [2, 4, 8]).all()

    # Link resolution should fail if not embedded in a data tree
    lone_xrdc = Hdf5ProxyXrDC(
        name="xrdc",
        data=proxy,
        attrs=dict(
            dims=["x", "y", "z"],
            coords_mode__x="linked",
            coords__x="some_other_data",
            coords_mode__y="linked",
            coords__y="../coords/y",
            coords_mode__z="linked",
            coords__z="../coords/more/z",
        ),
    )

    with pytest.raises(ValueError, match="'xrdc' is not embedded into a data"):
        lone_xrdc.coords["x"]
