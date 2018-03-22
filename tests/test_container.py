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
    ndc = NumpyDataContainer(name="oof", data=np.array([2,3,4]))
    dasd = ndc.copy()
    print(dasd)