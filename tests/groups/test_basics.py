"""Test basic properties of groups.

NOTE This uses OrderedDataGroup as that is the simplest non-abstract group
     that is regularly implemented in dantro.
"""

import pytest

import numpy as np

from dantro.groups import OrderedDataGroup
from dantro.containers import MutableSequenceContainer, NumpyDataContainer

# -----------------------------------------------------------------------------


def test_basics():
    """Tests whether the basic group interface behaves as desired."""
    # Basic initialisation, without containers
    dg1 = OrderedDataGroup(name="foo")

    # Names need be string
    with pytest.raises(TypeError, match="Name for OrderedDataGroup needs to"):
        OrderedDataGroup(name=123)

    # Passing some containers
    conts = [
        MutableSequenceContainer(name=str(i), data=list(range(i)))
        for i in range(10)
    ]
    dg2 = OrderedDataGroup(name="bar", containers=conts)

    # Nest these together
    root = OrderedDataGroup(name="root", containers=[dg1, dg2])

    # If a non-container object is passed to a group, this should fail.
    with pytest.raises(TypeError):
        OrderedDataGroup(name="bar", containers=["foo", "bar"])

    # Try to access them
    assert "foo" in root
    assert "bar" in root
    assert "baz" not in root

    # There should be a custom key error if accessing something not available
    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root["i_am_a_ghost"]

    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root["foo/is/a/ghost"]

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
    with pytest.raises(TypeError, match="Argument `Cls` needs to be a class"):
        root.new_group("foobar", Cls="not_a_class")

    with pytest.raises(
        TypeError, match="Argument `Cls` needs to be a subclass"
    ):
        root.new_group("foobar", Cls=MutableSequenceContainer)


def test_group_creation():
    """Tests whether groups and containers can be created as desired."""
    root = OrderedDataGroup(name="root")

    # Add a group by name and check it was added
    foo = root.new_group("foo")
    assert foo in root

    # Add a container
    msc = root.new_container(
        "spam", Cls=MutableSequenceContainer, data=[1, 2, 3]
    )
    assert msc in root

    # Should raise an error withou Cls given
    with pytest.raises(ValueError, match="Got neither argument `Cls` nor"):
        root.new_container("spam2", Cls=None, data=[1, 2, 3])

    # Set the class variable and try again
    root._NEW_CONTAINER_CLS = MutableSequenceContainer
    msc2 = root.new_container("spam2", data=[1, 2, 3])
    assert msc2 in root
    assert isinstance(msc2, MutableSequenceContainer)

    # Now test adding groups by path
    bar = root.new_group("foo/bar")
    assert "foo/bar" in root
    assert "bar" in foo

    # Check that intermediate parts not existing leads to errors
    with pytest.raises(KeyError, match="Could not create OrderedDataGroup at"):
        root.new_group("some/longer/path")

    # Set the allowed container types of the bar group differently
    bar._ALLOWED_CONT_TYPES = (MutableSequenceContainer,)

    # ... this should now fail
    with pytest.raises(TypeError, match="Can only add objects derived from"):
        bar.new_group("baz")

    # While adding a MutableSequenceContainer should work
    bar.new_container("eggs", Cls=MutableSequenceContainer, data=[1, 2, 3])


def test_list_item_access():
    """Tests that passing lists with arbitrary content along __getitem__ works
    as desired ...
    """

    root = OrderedDataGroup(name="root")
    one = root.new_group("one")
    two = one.new_group("two")
    two.add(NumpyDataContainer(name="arr", data=np.zeros((2, 3, 4))))
    # Path created: root/one/two/arr

    # Test that regular item access is possible
    arr = root["one/two/arr"]

    # Test that access via a list-type path is possible
    sliced_arr = root[["one", "two", "arr", slice(None, 2)]]
    assert sliced_arr.shape == arr[slice(None, 2)].shape
