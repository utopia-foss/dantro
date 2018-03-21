"""Test the DataGroup and derived classes"""

import pytest

import dantro.group as grp
import dantro.container as cont

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation, without containers
    dg = grp.DataGroup(name="foo")

    # Passing some containers
    conts = [cont.ItemContainer(name=i, data=list(range(i)))
             for i in range(10)]
    dg2 = grp.DataGroup(name="bar", containers=conts)
    
    # Nest these together
    root = grp.DataGroup(name="root", containers=[dg, dg2])

    # If a non-container object is passed to a group, this should fail.
    # with pytest.raises(TypeError):
        
