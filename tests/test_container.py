"""Test the BaseDataContainer-derived classes"""

import pytest

from dantro.container import MutableSequenceContainer, NumpyDataContainer
import numpy as np
# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_mutuable_sequence_container():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation of sequence-like data
    msc1 = MutableSequenceContainer(name="foo", data=["bar", "baz"])
    msc2 = MutableSequenceContainer(name="foo", data=["bar", "baz"],
                                    attrs=dict(one=1, two="two"))

    # There will be warnings for other data types:
    with pytest.warns(UserWarning):
        msc3 = MutableSequenceContainer(name="bar", data=("hello", "world"))

    with pytest.warns(UserWarning):
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

    # Test the ForwardAttrsToDataMixin 
    npa1 = np.array([1,2,3])
    assert ndc1.size == npa1.size
    assert ndc1.ndim == npa1.ndim
    assert ndc1.max() == npa1.max()
    assert ndc1.mean() == npa1.mean()
    assert ndc1.cumsum()[-1] == npa1.cumsum()[-1]

    # Test the NumbersMixin
    add = ndc1 + ndc2
    sub = ndc1 - ndc2
    mult = ndc1 * ndc2
    div = ndc1 / ndc2
    floordiv = ndc1 // ndc2
    mod = ndc1 % ndc2
    div_mod = divmod(ndc1, ndc2)

    # Test NumbersMixin function for operations on two numpy arrays 
    npa1 = np.array([1,2,3])
    npa2 = np.array([2,4,6])
    
    add_npa = npa1 + npa2
    sub_npa = npa1 - npa2
    mult_npa = npa1 * npa2
    div_npa = npa1 / npa2
    floordiv_npa = npa1 // npa2
    mod_npa = npa1 % npa2
        
    for i in [0,1,2]:
        assert add.data[i] == add_npa[i] 
        assert sub.data[i] == sub_npa[i] 
        assert mult.data[i] == mult_npa[i] 
        assert div.data[i] == div_npa[i] 
        assert floordiv.data[i] == floordiv_npa[i] 
        assert mod.data[i] == mod_npa[i] 
        assert div_mod.data[0][i] == floordiv_npa[i] 
        assert div_mod.data[1][i] == mod_npa[i]

        assert ndc1.data[i] == npa1[i]

    # Test NumbersMixin function for operations on one numpy array and a number
    add_number = npa1 + 4.2
    sub_number = npa1 - 4.2
    mult_number = npa1 * 4.2
    div_number = npa1 / 4.2
    floordiv_number = npa1 // 4.2
    mod_number = npa1 % 4.2

    for i in [0,1,2]:
        assert add.data[i] == add_npa[i] 
        assert sub.data[i] == sub_npa[i] 
        assert mult.data[i] == mult_npa[i] 
        assert div.data[i] == div_npa[i] 
        assert floordiv.data[i] == floordiv_npa[i] 
        assert mod.data[i] == mod_npa[i] 
        assert div_mod.data[0][i] == floordiv_npa[i] 
        assert div_mod.data[1][i] == mod_npa[i]

        assert ndc1.data[i] == npa1[i]
