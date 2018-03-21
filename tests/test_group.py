"""Test BaseDataGroup-derived classes"""

import pytest

from dantro.group import OrderedDataGroup
from dantro.container import MutableSequenceContainer

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_ordered_data_group():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation, without containers
    dg1 = OrderedDataGroup(name="foo")

    # Passing some containers
    conts = [MutableSequenceContainer(name=i, data=list(range(i)))
             for i in range(10)]
    dg2 = OrderedDataGroup(name="bar", containers=conts)
    
    # Nest these together
    OrderedDataGroup(name="root", containers=[dg1, dg2])

    # If a non-container object is passed to a group, this should fail.
    with pytest.raises(TypeError):
        OrderedDataGroup(name="bar", containers=["foo", "bar"])
