"""Tests the base classes, as far as possible.

NOTE This test module merely complements the other, already existing tests of
     the base classes that are made implicitly through testing the derivatives.
"""

import pytest

import dantro as dtr
import dantro.base
import dantro.container

# Class definitions -----------------------------------------------------------


# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_BaseDataAttrs():
    """Test the BaseDataAttrs class"""
    bda = dtr.base.BaseDataAttrs(name="test")

    assert isinstance(bda.data, dict)
    assert bda.data == dict()
    assert bda._format_info() == "0 attribute(s)"

    bda['foo'] = "bar"
    assert bda.data == dict(foo="bar")

    assert bda._format_info() == "1 attribute(s)"

def test_BaseDataGroup():
    """Tests the BaseDataGroup using OrderedDataGroup"""
    root = dtr.group.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")
    bar.add(dtr.container.ObjectContainer(name="obj", data=dict(test=123)))

    # Test item interface
    assert root['foo'] is foo
    assert root['foo/bar'] is bar

    # Setting should not work
    with pytest.raises(ValueError, match="cannot carry out __setitem__"):
        root['baz'] = dtr.group.OrderedDataGroup(name="baz")

    # Put passing on the __item__ calls to the object beneath should work
    assert root['foo/bar/obj/test'] == 123
    root['foo/bar/obj/test'] = 234
    assert root['foo/bar/obj/test'] == 234

    # 
