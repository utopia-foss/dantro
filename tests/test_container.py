"""Test the BaseDataContainer-derived classes"""

import math
import operator

import numpy as np
import xarray as xr
import h5py as h5

import pytest

from dantro.base import BaseDataContainer, CheckDataMixin
from dantro.base import ItemAccessMixin
from dantro.mixins.base import UnexpectedTypeWarning
from dantro.mixins.proxy_support import Hdf5ProxyMixin
from dantro.containers import MutableSequenceContainer
from dantro.containers import NumpyDataContainer, XrDataContainer
from dantro.proxy import Hdf5DataProxy

# Local constants

class DummyContainer(ItemAccessMixin, BaseDataContainer):
    """A dummy container that fulfills all the requirements of the abstract
    BaseDataContainer class.

    NOTE: the methods have not the correct functionality!
    """
    def _format_info(self):
        return "dummy"

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def tmp_h5_dset(tmpdir) -> h5.Dataset:
    """Creates a temporary hdf5 dataset"""

    test_file = h5.File(tmpdir.join("test_h5_file.hdf5"))
    # Create a h5 dataset
    dset = test_file.create_dataset("init", data=np.zeros(shape=(1, 2, 3), 
                                                          dtype=int))
    return dset

# Tests -----------------------------------------------------------------------

def test_init():
    """Tests initialisation of the DummyContainer class"""
    # Simple init
    dc = DummyContainer(name="dummy", data="foo")

    # Assert the name is a string
    assert isinstance(dc.name, str)

    # Check invalid name arguments
    with pytest.raises(TypeError, match="Name for DummyContainer needs"):
        DummyContainer(name=123, data="foo")

    with pytest.raises(ValueError, match="Name for DummyContainer cannot "):
        DummyContainer(name="a/name/with/the/PATH_JOIN_CHAR", data="foo")


def test_check_data_mixin():
    """Checks the CheckDataMixin class."""
    # Define some test classes ................................................

    class TestContainerA(CheckDataMixin, DummyContainer):
        """All types allowed"""
    
    class TestContainerB(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""
        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = 'raise'
    
    class TestContainerC(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""
        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = 'warn'
    
    class TestContainerD(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""
        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = 'ignore'
    
    class TestContainerE(CheckDataMixin, DummyContainer):
        """Only list or tuple allowed, raising if not correct"""
        DATA_EXPECTED_TYPES = (list, tuple)
        DATA_ALLOW_PROXY = True
        DATA_UNEXPECTED_ACTION = 'invalid'

    # Tests ...................................................................
    # Run tests for A
    TestContainerA(name="foo", data="bar")

    # Run tests for B
    TestContainerB(name="foo", data=["my", "list"])
    TestContainerB(name="foo", data=("my", "tuple"))

    with pytest.raises(TypeError, match="Unexpected type <class 'str'> for.*"):
        TestContainerB(name="foo", data="bar")

    # Run tests for C
    with pytest.warns(UnexpectedTypeWarning,
                      match="Unexpected type <class 'str'> for.*"):
        TestContainerC(name="foo", data="bar")
    
    # Run tests for D
    TestContainerD(name="foo", data="bar")
    
    # Run tests for E
    with pytest.raises(ValueError, match="Illegal value 'invalid' for class"):
        TestContainerE(name="foo", data="bar")


def test_mutuable_sequence_container():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation of sequence-like data
    msc1 = MutableSequenceContainer(name="foo", data=["bar", "baz"])
    msc2 = MutableSequenceContainer(name="foo", data=["bar", "baz"],
                                    attrs=dict(one=1, two="two"))

    # There will be warnings for other data types:
    with pytest.warns(UnexpectedTypeWarning):
        msc3 = MutableSequenceContainer(name="bar", data=("hello", "world"))

    with pytest.warns(UnexpectedTypeWarning):
        msc4 = MutableSequenceContainer(name="baz", data=None)
    
    # Basic assertions ........................................................
    # Data access
    assert msc1.data == ["bar", "baz"] == msc1[:]
    assert msc2.data == ["bar", "baz"] == msc2[:]

    # Attribute access
    assert msc2.attrs == dict(one=1, two="two")
    assert msc2.attrs['one'] == 1

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
    for msc in [msc1, msc2, msc3]:
        str(msc)
        "{:info,cls_name,name}".format(msc)
        "{}".format(msc)

        with pytest.raises(ValueError):
            "{:illegal_formatspec}".format(msc)


def test_numpy_data_container():
    """Tests whether the __init__method behaves as desired"""
    # Basic initialization of Numpy ndarray-like data
    ndc1 = NumpyDataContainer(name="oof", data=np.array([1,2,3]))
    ndc2 = NumpyDataContainer(name="zab", data=np.array([2,4,6]))

    # Initialisation with lists should also work
    NumpyDataContainer(name="rab", data=[3,6,9])

    # Ensure that the CheckDataMixin does its job
    with pytest.raises(TypeError, match="Unexpected type"):
        NumpyDataContainer(name="zab", data=("not", "a", "valid", "type"))

    # Test the ForwardAttrsToDataMixin on a selection of numpy functions
    l1 = [1,2,3]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    npa1 = np.array(l1)
    assert ndc1.size == npa1.size
    assert ndc1.ndim == npa1.ndim
    assert ndc1.max() == npa1.max()
    assert ndc1.mean() == npa1.mean()
    assert ndc1.cumsum()[-1] == npa1.cumsum()[-1]

    # Test the NumbersMixin
    ndc1 = NumpyDataContainer(name="oof", data=np.array([1,2,3]))
    ndc2 = NumpyDataContainer(name="zab", data=np.array([2,4,6]))
    add = ndc1 + ndc2
    sub = ndc1 - ndc2
    mult = ndc1 * ndc2
    div = ndc1 / ndc2
    floordiv = ndc1 // ndc2
    mod = ndc1 % ndc2
    div_mod = divmod(ndc1, ndc2)
    power = ndc1 ** ndc2

    # Test NumbersMixin function for operations on two numpy arrays 
    l1 = [1,2,3]
    l2 = [2,4,6]
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
    power_npa = npa1 ** npa2
        
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
    power_number = ndc1 ** 4.2

    add_npa_number = npa1 + 4.2
    sub_npa_number = npa1 - 4.2
    mult_npa_number = npa1 * 4.2
    div_npa_number = npa1 / 4.2
    floordiv_npa_number = npa1 // 4.2
    divmod_npa_number = divmod(npa1, 4.2)
    mod_npa_number = npa1 % 4.2
    power_npa_number = npa1 ** 4.2

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
    l1 = [1.,2.,3.]
    l2 = [2.,4.,6.]
    ndc1_inplace = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2_inplace = NumpyDataContainer(name="zab", data=np.array(l2))
    npa1_inplace = np.array(l1)
    npa2_inplace = np.array(l2)

    ndc1_inplace += ndc2_inplace
    npa1_inplace += npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace -= ndc2_inplace
    npa1_inplace -= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace *= ndc2_inplace
    npa1_inplace *= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace /= ndc2_inplace
    npa1_inplace /= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace //= ndc2_inplace
    npa1_inplace //= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace %= ndc2_inplace
    npa1_inplace %= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())
    
    ndc1_inplace **= ndc2_inplace
    npa1_inplace **= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    # Test unary operations
    l1 = [1.,-2.,3.]
    l2 = [-2.,4.,6.]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2 = NumpyDataContainer(name="zab", data=np.array(l2))
    npa1 = np.array(l1)
    npa2 = np.array(l2)

    assert (-ndc1).all() == (-npa1).all()
    assert (+ndc1).all() == (+npa1).all()
    assert abs(ndc1).all() == abs(npa1).all()
    assert ~ndc1.all() == ~npa1.all()

    # Test ComparisonMixin
    l1 = [1,2,3]
    l2 = [0,2,4]
    ndc1 = NumpyDataContainer(name="oof", data=np.array(l1))
    ndc2 = NumpyDataContainer(name="tada", data=np.array(l2))
    npa1 = np.array(l1)
    npa2 = np.array(l2)

    eq = (ndc1 == ndc2)
    ne = (ndc1 != ndc2)
    lt = (ndc1 < ndc2)
    le = (ndc1 <= ndc2)
    gt = (ndc1 > ndc2)
    ge = (ndc1 >= ndc2)

    eq_npa = (npa1 == npa2)
    ne_npa = (npa1 != npa2)
    lt_npa = (npa1 < npa2)
    le_npa = (npa1 <= npa2)
    gt_npa = (npa1 > npa2)
    ge_npa = (npa1 >= npa2)

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

def test_xr_data_container(tmp_h5_dset):
    """Tests whether the __init__method behaves as desired"""
    
    # Basic initialization of Numpy ndarray-like data
    xrdc = XrDataContainer(name="xrdc", data=np.array([1,2,3]))

    # Initialisation with lists and xr.DataArrays should also work
    XrDataContainer(name="rab", data=[2,4,6])
    XrDataContainer(name="zab", data=xr.DataArray([2,4,6]))

    # Initialisation with dimension names .....................................
    xrdc = XrDataContainer(name="xrdc_dims_1", data=[3,6,9],
                           attrs=dict(dims=['first_dim']))
    assert 'first_dim' in xrdc.data.dims

    xrdc = XrDataContainer(name="xrdc_dims_2", data=[[1,2,3], [4,5,6]],
                           attrs=dict(dims=['first_dim', 'second_dim']))
    assert 'first_dim' in xrdc.data.dims
    assert 'second_dim' in xrdc.data.dims

    xrdc = XrDataContainer(name="xrdc_dims_prefix_1", data=[1,2,4],
                           attrs=dict(dim_name__0='first_dim'))
    assert 'first_dim' in xrdc.data.dims

    xrdc = XrDataContainer(name="xrdc_dims_prefix_2", data=[[1,2,4], [2,4,8]],
                           attrs=dict(dim_name__0='first_dim'))
    assert 'first_dim' in xrdc.data.dims
    assert 'dim_1' in xrdc.data.dims

    with pytest.raises(ValueError,
                       match="Number of given dimension names does not match"):
        xrdc = XrDataContainer(name="xrdc_dims_mismatch", data=[3,6,9],
                               attrs=dict(dims=['first_dim', 'second_dim']))

    with pytest.raises(ValueError,
                       match="Could not extract the dimension number from"):
        xrdc = XrDataContainer(name="xrdc_dims_mismatch", data=[1,2,4],
                               attrs=dict(dim_name__mismatch='first_dim'))

    # With a dimension name list given, the other attributes overwrite the
    # ones given in the list
    xrdc = XrDataContainer(name="xrdc_dims_3", data=[[1,2,3], [4,5,6]],
                           attrs=dict(dims=['first_dim', 'second_dim'],
                                      dim_name__0="foo"))
    assert xrdc.data.dims == ('foo', 'second_dim')

    # Pathological cases: attribute is an iterable of scalar numpy arrays
    xrdc = XrDataContainer(name="xrdc_dims_3", data=[[1,2,3], [4,5,6]],
                           attrs=dict(dims=[np.array('first_dim', dtype='U'),
                                            np.array(['second_dim'])]))
    assert xrdc.data.dims == ('first_dim', 'second_dim')

    xrdc = XrDataContainer(name="xrdc_dims_3", data=[[1,2,3], [4,5,6]],
                           attrs=dict(dim_name__0=np.array('first_dim',
                                                           dtype='U'),
                                      dim_name__1=np.array(['second_dim'])))
    assert xrdc.data.dims == ('first_dim', 'second_dim')

    # Bad dimension name attribute type
    with pytest.raises(TypeError, match="sequence of strings, but not"):
        XrDataContainer(name="xrdc", data=[[1,2,3], [4,5,6]],
                        attrs=dict(dims="13"))
    
    with pytest.raises(TypeError, match="needs to be an iterable"):
        XrDataContainer(name="xrdc", data=[[1,2,3], [4,5,6]],
                        attrs=dict(dims=123))

    with pytest.raises(TypeError, match="need to be strings, got"):
        XrDataContainer(name="xrdc", data=[[1,2,3], [4,5,6]],
                        attrs=dict(dims=["foo", 123]))

    with pytest.raises(TypeError, match="need be strings, but the attribute"):
        XrDataContainer(name="xrdc", data=[[1,2,3], [4,5,6]],
                        attrs=dict(dim_name__0=123))

    # Bad dimension number
    with pytest.raises(ValueError, match="exceeds the rank \(2\)"):
        XrDataContainer(name="xrdc_dims_3", data=[[1,2,3], [4,5,6]],
                        attrs=dict(dim_name__10="foo"))


    # Initialisation with coords ..............................................
    # Explicitly given coordinates . . . . . . . . . . . . . . . . . . . . . .
    coords__time = ['Jan', 'Feb', 'Mar']
    xrdc = XrDataContainer(name="xrdc", data=xr.DataArray([1,2,3]),
                           attrs=dict(dims=['time'],
                                      coords__time=coords__time))
    assert 'time' in xrdc.data.dims
    assert np.all(coords__time == xrdc.data.coords['time'])

    coords__time = ['Jan', 'Feb', 'Mar', 'Apr']
    coords__space = ['IA', 'IL', 'IN']

    xrdc = XrDataContainer(name="xrdc", data=np.random.rand(4, 3),
                           attrs=dict(dims=['time', 'space'],
                                      coords__time=coords__time,
                                      coords__space=coords__space))
    assert 'time' in xrdc.data.dims
    assert np.all(coords__time == xrdc.data.coords['time'])

    with pytest.raises(ValueError, match="Could not associate coordinates"):
        xrdc = XrDataContainer(name="xrdc", data=xr.DataArray([1,2,3]),
                               attrs=dict(dims=['time'],
                                          coords__time=['Jan', 'Feb']))
    
    with pytest.raises(ValueError, match="Got superfluous container attr"):
        xrdc = XrDataContainer(name="xrdc_coord_mismatch",
                               data=[1,2,3,4],
                               attrs=dict(dims=['time'],
                                          coords__space=['IA', 'IL', 'IN']))
    
    # Coordinates and data don't match in length
    with pytest.raises(ValueError, match="Could not associate coordinates"):
        XrDataContainer(name="xrdc", data=np.arange(10),
                        attrs=dict(dims=['time'],
                                   coords__time= [1, 2, 3]))

    # Coordinates as range expression . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(name="xrdc", data=np.arange(10),
                           attrs=dict(dims=['time'],
                                      coords__time=[10],
                                      coords_mode__time='range'))
    assert np.all(np.arange(10) == xrdc.data.coords['time'])
    
    xrdc = XrDataContainer(name="xrdc", data=np.arange(10),
                           attrs=dict(dims=['time'],
                                      coords__time=[3, 13],
                                      coords_mode__time='range'))
    assert np.all(np.arange(3, 13) == xrdc.data.coords['time'])

    xrdc = XrDataContainer(name="xrdc", data=np.arange(10),
                           attrs=dict(dims=['time'],
                                      coords__time=[0, 100, 10],
                                      coords_mode__time='range'))
    assert np.all(np.arange(0, 100, 10) == xrdc.data.coords['time'])

    # start and step values . . . . . . . . . . . . . . . . . . . . . . . . . .
    xrdc = XrDataContainer(name="xrdc", data=np.arange(10),
                           attrs=dict(dims=['time'],
                                      coords__time=[0, 2],
                                      coords_mode__time='start_and_step'))
    assert np.all(list(range(0, 20, 2)) == xrdc.data.coords['time'])


    # linked mapping . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    with pytest.raises(NotImplementedError):
        XrDataContainer(name="xrdc", data=np.arange(10),
                        attrs=dict(dims=['time'],
                                   coords__time="../foo",
                                   coords_mode__time='linked'))

    # Invalid coordinate mode . . . . . . . . . . . . . . . . . . . . . . . . .
    with pytest.raises(ValueError, match="Invalid mode 'invalid' to interpre"):
        XrDataContainer(name="invalid_coord_type", data=[1,2,3],
                        attrs=dict(dims=['time'],
                                   coords__time=[0, 1, 2],
                                   coords_mode__time='invalid'))

    # without strict attribute checking . . . . . . . . . . . . . . . . . . . .
    class TolerantXrDataContainer(XrDataContainer):
        _XRC_STRICT_ATTR_CHECKING = False

    xrdc = TolerantXrDataContainer(name="tolerant_xrdc", data=[1,2,3],
                                   attrs=dict(dims=['time'],
                                              coords__time=[0, 1, 2],
                                              coords__foo="bar"  # throws not
                                              ))
    
    # Carrying over attributes ................................................
    # Attributes are carried over as expected; prefixed attributes are not!
    xrdc = XrDataContainer(name="xrdc", data=[1,2,3],
                           attrs=dict(foo="bar",
                                      dims=['time'],
                                      coords__time=['Jan', 'Feb', 'Mar']))
    assert ('foo', 'bar') in xrdc.data.attrs.items()
    assert 'dims' not in xrdc.data.attrs
    assert 'coords__time' not in xrdc.data.attrs

    # Copying .................................................................
    xrdc_copy = xrdc.copy()
    assert xrdc_copy.data is not xrdc.data
    assert np.all(xrdc_copy.data == xrdc.data)

    # String representation ...................................................
    assert xrdc._format_info().startswith(str(xrdc.dtype))


    # Proxysupport ............................................................

    # class with hdf5proxysupport (proxies need to have ndim, shape, dtype)
    class Hdf5ProxyXrDC(Hdf5ProxyMixin, XrDataContainer):
        pass
    
    # Check that proxy support isenabled now
    assert Hdf5ProxyXrDC.DATA_ALLOW_PROXY == True

    # Create a proxy
    proxy = Hdf5DataProxy(obj=tmp_h5_dset)

    # Create a XrDataContainer with proxy support
    pxrdc = Hdf5ProxyXrDC(name="xrdc", data=proxy, 
                          attrs=dict(foo="bar", dims=['x','y', 'z'],
                                     coords__x=['1 m'], 
                                     coords__z=['1 cm', '2 cm', '3 cm']))
    
    # Initialize another one directly without using the proxy
    pxrdc_direct = Hdf5ProxyXrDC(name="xrdc", data=tmp_h5_dset[()], 
                                 attrs=dict(foo="bar", dims=['x', 'y', 'z'],
                                            coords__x=['1 m'], 
                                            coords__z=['1 cm', '2 cm', '3 cm'])
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
