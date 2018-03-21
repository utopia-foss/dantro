"""Test BaseDataGroup-derived classes"""

import pytest

import dantro.group as grp
from dantro.container import MutableSequenceContainer

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation, without containers
    dg = grp.OrderedDataGroup(name="foo")

    # Passing some containers
    conts = [MutableSequenceContainer(name=i, data=list(range(i)))
             for i in range(10)]
    dg2 = grp.OrderedDataGroup(name="bar", containers=conts)
    
    # Nest these together
    grp.OrderedDataGroup(name="root", containers=[dg, dg2])

    # If a non-container object is passed to a group, this should fail.
    with pytest.raises(TypeError):
        grp.OrderedDataGroup(name="bar", containers=["foo", "bar"])
