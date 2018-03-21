"""Test the BaseDataContainer-derived classes"""

import pytest

from dantro.container import MutableSequenceContainer

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation
    ic = MutableSequenceContainer(name="foo", data=["bar", "baz"])
    assert ic.data == ["bar", "baz"]

    # Without data
    ic = MutableSequenceContainer(name="foo", data=None)
    assert ic.data is None

    # Basic 
