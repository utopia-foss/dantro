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

    # Names need be string
    with pytest.raises(TypeError, match="Name for OrderedDataGroup needs to"):
        OrderedDataGroup(name=123)

    # Passing some containers
    conts = [MutableSequenceContainer(name=str(i), data=list(range(i)))
             for i in range(10)]
    dg2 = OrderedDataGroup(name="bar", containers=conts)
    
    # Nest these together
    root = OrderedDataGroup(name="root", containers=[dg1, dg2])

    # If a non-container object is passed to a group, this should fail.
    with pytest.raises(TypeError):
        OrderedDataGroup(name="bar", containers=["foo", "bar"])

    # Try to access them
    assert 'foo' in root
    assert 'bar' in root
    assert 'baz' not in root

    # There should be a custom key error if accessing something not available
    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root['i_am_a_ghost']
    
    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root['foo/is/a/ghost']

    # Test adding a new group in that group
    subgroup = root.new_group("subgroup")

    # Test it was added
    assert "subgroup" in root
    assert root["subgroup"] is subgroup

    # Adding it again should fail
    with pytest.raises(ValueError, match="has a member with name 'subgroup'"):
        root.new_group("subgroup")

    # Should also work when explicitly giving the class
    sg2 = root.new_group("sg2", Cls=OrderedDataGroup)
    assert isinstance(sg2, OrderedDataGroup)
    # TODO pass another class here

    # Should _not_ work with something that is not a class or not a group
    with pytest.raises(TypeError,
                       match="Argument `Cls` needs to be a class"):
        root.new_group("foobar", Cls="not_a_class")

    with pytest.raises(TypeError,
                       match="Argument `Cls` needs to be a subclass"):
        root.new_group("foobar", Cls=MutableSequenceContainer)

def test_group_creation():
    """Tests whether groups and containers can be created as desired."""
    root = OrderedDataGroup(name="root")

    # Add a group by name and check it was added
    foo = root.new_group("foo")
    assert foo in root

    # Add a container
    msc = root.new_container("spam", Cls=MutableSequenceContainer,
                             data=[1, 2, 3])
    assert msc in root

    # Now test adding groups by path
    bar = root.new_group("foo/bar")
    assert "foo/bar" in root
    assert "bar" in foo

    # Check that intermediate parts not existing leads to errors
    with pytest.raises(KeyError, match="Could not create OrderedDataGroup at"):
        root.new_group("some/longer/path")
