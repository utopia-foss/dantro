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
        
    
    for i in [0,1,2]:
        assert add.data[i] == add_npa[i] 
        assert sub.data[i] == sub_npa[i] 
        assert mult.data[i] == mult_npa[i] 
        assert div.data[i] == div_npa[i] 
        assert floordiv.data[i] == floordiv_npa[i] 
        assert mod.data[i] == mod_npa[i] 
        assert div_mod.data[0][i] == floordiv_npa[i] 
        assert div_mod.data[1][i] == mod_npa[i]
        assert power.data[i] == power_npa[i] 

        assert ndc1.data[i] == npa1[i]

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

    for i in [0,1,2]:
        assert add_number.data[i] == add_npa_number[i] 
        assert sub_number.data[i] == sub_npa_number[i] 
        assert mult_number.data[i] == mult_npa_number[i] 
        assert div_number.data[i] == div_npa_number[i] 
        assert floordiv_number.data[i] == floordiv_npa_number[i] 
        assert mod_number.data[i] == mod_npa_number[i] 
        assert divmod_number.data[0][i] == floordiv_npa_number[i] 
        assert divmod_number.data[1][i] == mod_npa_number[i]
        assert power_number.data[i] == power_number[i]

    # Test inplace operations
    l1 = [1,2,3]
    l2 = [2,4,6]
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

    # ndc1_inplace /= ndc2_inplace
    # npa1_inplace /= npa2_inplace
    # assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace //= ndc2_inplace
    npa1_inplace //= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

    ndc1_inplace %= ndc2_inplace
    npa1_inplace %= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())
    
    ndc1_inplace **= ndc2_inplace
    npa1_inplace **= npa2_inplace
    assert (ndc1_inplace.all() == npa1_inplace.all())

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

    for i in [0,1,2]:
        assert eq[i] == eq_npa[i]
        assert ne[i] == ne_npa[i]
        assert lt[i] == lt_npa[i]
        assert le[i] == le_npa[i]
        assert gt[i] == gt_npa[i]
        assert ge[i] == ge_npa[i]
