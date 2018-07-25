"""Test the BaseDataContainer-derived classes"""

import math

import numpy as np

import pytest

from dantro.base import BaseDataContainer, CheckDataMixin, ItemAccessMixin, UnexpectedTypeWarning
from dantro.container import MutableSequenceContainer, NumpyDataContainer

# Local constants

class DummyContainer(ItemAccessMixin, BaseDataContainer):
    """A dummy container that fulfills all the requirements of the abstract
    BaseDataContainer class.

    NOTE: the methods have not the correct functionality!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_to(self, TargetCls, **target_init_kwargs):
        return

    def _format_info(self):
        return "dummy"

# Fixtures --------------------------------------------------------------------


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

    # Conversion ..............................................................
    # To itself
    msc1c = msc1.convert_to(MutableSequenceContainer)

    # Ensure that it is a shallow copy
    assert msc1c.data is msc1.data

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

    # Assert that some functions are not available in the NumpyDataContainer
    # as they make no sense with (most of the possible) np.ndarray-data
    with pytest.raises(NotImplementedError):
        complex(ndc1)
    with pytest.raises(NotImplementedError):
        int(ndc1)
    with pytest.raises(NotImplementedError):
        float(ndc1)
    with pytest.raises(NotImplementedError):
        round(ndc1)
    with pytest.raises(NotImplementedError):
        math.ceil(ndc1)
    with pytest.raises(NotImplementedError):
        math.floor(ndc1)
    with pytest.raises(NotImplementedError):
        math.trunc(ndc1)

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
