"""Tests the mixin classes in the base and mixin modules."""

import copy

import pytest

import dantro as dtr
import dantro.base
import dantro.mixins
import dantro.container
import dantro.groups

from .test_data_mngr import NumpyTestDC

# Class definitions -----------------------------------------------------------


# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_PathMixin():
    """Tests the PathMixin class using the OrderedDataGroup"""
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")

    # Test correct parent association
    assert root.parent is None
    assert foo.parent is root
    assert bar.parent is foo

    # Path creation
    assert root.path == "root"
    assert foo.path == "root/foo"
    assert bar.path == "root/foo/bar"

    # Format function
    assert root._format_path() == root.path
    assert foo._format_path() == foo.path
    assert bar._format_path() == bar.path

    # Trying to set a parent if it is currently set should not work
    with pytest.raises(ValueError, match="A parent was already associated"):
        bar.parent = root

def test_ItemAccessMixin():
    """Tests the ItemAccessMixin using the ObjectContainer"""
    obj = dtr.container.ObjectContainer(name="obj", data=dict())

    # As ObjectContainer uses the ItemAccessMixin, it should forward all the
    # corresponding syntactic sugar to the dict data

    # Set an element of the dict
    obj['foo'] = "bar"
    assert obj.data.get('foo') == "bar"

    # Get the value
    assert obj['foo'] == "bar"

    # Delete the value
    del obj['foo']
    with pytest.raises(KeyError, match="foo"):
        obj['foo']


    # Passing on to the object works
    root = dtr.groups.OrderedDataGroup(name="root", containers=[obj])
    root['obj/foo'] = "bar"
    assert root['obj/foo'] == "bar"

    del root['obj/foo']
    with pytest.raises(KeyError, match="No key or key sequence"):
        root['obj/foo']

    # List arguments
    root[['obj', 'foo']] = "bar"
    assert root[['obj', 'foo']] == "bar"

    del root[['obj', 'foo']]
    with pytest.raises(KeyError, match="No key or key sequence"):
        root[['obj', 'foo']]

    # Too long lists
    with pytest.raises(KeyError, match="No key or key sequence"):
        root[['obj', 'foo', 'spam']]    


def test_MappingAccessMixin():
    """Tests the MappingAccessMixin using MutableMappingContainer"""
    mmc = dtr.container.MutableMappingContainer(name="map",
                                                data=dict(foo="bar",
                                                          spam="eggs"))

    assert list(mmc.keys()) == ['foo', 'spam']
    assert list(mmc.values()) == ["bar", "eggs"]
    assert list(mmc.items()) == [('foo', "bar"), ('spam', "eggs")]

    assert mmc.get('foo') == "bar"
    assert mmc.get('spam') == "eggs"
    assert mmc.get('baz', "buz") == "buz"


@pytest.mark.skip("Does not currently work")
def test_numeric_mixins():
    """Tests UnaryOperationsMixin and NumbersMixin"""
    # Define a test class using the NumbersMixin, which inherits the
    # UnaryOperationsMixin
    class Num(dtr.mixins.NumbersMixin):
        def __init__(self, data):
            self._data = data

        @property
        def data(self):
            return self._data

        def copy(self):
            return Num(copy.deepcopy(self.data))

    num = Num(1.23)
    assert num.data == 1.23

    # Test each function
    assert num.__neg__() == -1.23
