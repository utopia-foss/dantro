"""Tests the base classes, as far as possible.

NOTE This test module merely complements the other, already existing tests of
     the base classes that are made implicitly through testing the derivatives.
"""

import pytest

import dantro as dtr
import dantro.base
import dantro.groups
import dantro.containers

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
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")
    bar.add(dtr.containers.ObjectContainer(name="obj", data=dict(test=123)))

    # Test item interface
    assert root['foo'] is foo
    assert root['foo/bar'] is bar

    # Setting should not work
    with pytest.raises(ValueError, match="cannot carry out __setitem__"):
        root['baz'] = dtr.groups.OrderedDataGroup(name="baz")

    # Put passing on the __item__ calls to the object beneath should work
    assert root['foo/bar/obj/test'] == 123
    root['foo/bar/obj/test'] = 234
    assert root['foo/bar/obj/test'] == 234

    # Test the `add` method
    baz1 = foo.new_group('baz')
    assert foo['baz'] is baz1
    
    baz2 = dtr.groups.OrderedDataGroup(name="baz")
    foo.add(baz2, overwrite=True)
    assert foo['baz'] is baz2


    # And adding new containers that are not of the correct type
    with pytest.raises(TypeError, match="needs to be a subclass of BaseData"):
        root.new_container("testpath", Cls=dict, foo="bar")


    # Recursive update
    root2 = dtr.groups.OrderedDataGroup(name="root")
    foo2 = root2.new_group("foo")
    spam = foo2.new_group("spam")
    sth = foo2.new_container("sth", Cls=dtr.containers.ObjectContainer,
                             data=dict(foo="bar"))

    root.recursive_update(root2)
    assert 'foo/spam' in root
    assert 'foo/sth' in root
    assert root['foo/sth'].data == dict(foo="bar")

    with pytest.raises(TypeError, match="Can only update (.+) with objects"):
        root.recursive_update(dict(foo="bar"))


    # Linking and unlinking
    # When the object is not a member
    with pytest.raises(ValueError, match="needs to be a child of"):
        root._link_child(new_child=root2)
    
    with pytest.raises(ValueError, match="was not linked to"):
        root._unlink_child(root2)

    # When it is already a member
    root._unlink_child(foo)
    assert foo.parent is None
    
    root._link_child(new_child=foo)
    assert foo.parent is root


    # __contains__
    assert 'foo' in root
    assert foo in root
    assert [] not in root

    with pytest.raises(TypeError, match="Can only check content of"):
        ('foo', 'bar') in root


    # iteration and dict access
    assert [k for k in root] == ['foo']
    assert [k for k in root.keys()] == ['foo']
    assert [v for v in root.values()] == [foo]
    assert [v for v in root.items()] == [('foo', foo)]

    assert root.get('FOO') is None
    assert root.get('FOO', foo) is foo

    with pytest.raises(NotImplementedError, match="is not supported"):
        root.setdefault(foo)