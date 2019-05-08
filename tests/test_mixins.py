"""Tests the mixin classes in the base and mixin modules."""

import copy

import pytest

import dantro as dtr
import dantro.base
import dantro.mixins
import dantro.containers
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


def test_LockDataMixin():
    """Test the LockDataMixin using an OrderedDataGroup"""
    class ODG(dantro.groups.OrderedDataGroup):
        def _lock_hook(self):
            # Should be locked now
            assert self.locked

        def _unlock_hook(self):
            # Should be unlocked now
            assert not self.locked

    types_to_check = (dantro.groups.OrderedDataGroup, ODG)

    for ODGType in types_to_check:
        odg = ODGType(name="lock_data_test")

        # Add some data groups
        assert not odg.locked
        foo = odg.new_group("foo")
        odg.new_group("bar")

        # Lock it and make sure that addition does no longer work
        assert not odg.locked
        odg.lock()
        assert odg.locked

        with pytest.raises(RuntimeError, match="Cannot modify"):
            odg.new_group("can't add this")
        assert "can't add this" not in odg
        
        with pytest.raises(RuntimeError, match="Cannot modify"):
            odg.add(dantro.containers.ObjectContainer(name="foo", data="bar"))
        assert odg["foo"] is foo

        # Can unlock
        odg.unlock()
        assert "baz" not in odg
        odg.new_group("baz")
        assert "baz" in odg


def test_ForwardAttrsMixin():
    """Tests the ForwardAttrsMixin"""
    class MyObjectContainer(dantro.mixins.ForwardAttrsMixin,
                            dantro.containers.ObjectContainer):
        # The name of the existing attribute to forward to.
        FORWARD_ATTR_TO = None

        # If set, the only attributes to be forwarded
        FORWARD_ATTR_ONLY = None

        # Attributes to not forward
        FORWARD_ATTR_EXCLUDE = ()

    class MyObject:
        foo = 123
        bar = 2.34
        baz = None

    moc = MyObjectContainer(name="foo", data=MyObject())

    assert moc.data.foo == 123
    with pytest.raises(AttributeError, match="foo"):
        moc.foo
    
    with pytest.raises(AttributeError, match="invalid_attr_name"):
        moc.invalid_attr_name
    
    moc.FORWARD_ATTR_TO = 'data'
    assert moc.foo is moc.data.foo
    assert moc.bar is moc.data.bar
    assert moc.baz is moc.data.baz
    
    with pytest.raises(AttributeError, match="invalid_attr_name"):
        assert moc.invalid_attr_name
    
    with pytest.raises(AttributeError, match="invalid_attr_name"):
        assert moc.invalid_attr_name

    moc.FORWARD_ATTR_EXCLUDE = ('bar',)
    assert moc.foo is moc.data.foo
    assert moc.baz is moc.data.baz
    with pytest.raises(AttributeError, match="bar"):
        moc.bar
    
    moc.FORWARD_ATTR_ONLY = ('foo',)
    assert moc.foo is moc.data.foo
    with pytest.raises(AttributeError, match="bar"):
        moc.bar
    with pytest.raises(AttributeError, match="baz"):
        moc.baz

    # Test hooks
    class HookedObjectContainer(MyObjectContainer):
        FORWARD_ATTR_TO = 'data'

        def _forward_attr_pre_hook(self, attr_name: str=None):
            """Invoked before attribute forwarding occurs"""
            assert attr_name is not None
            self.__last_attr_name = attr_name
        
        def _forward_attr_post_hook(self, attr):
            """Invoked before attribute forwarding occurs"""
            assert hasattr(getattr(self, self.FORWARD_ATTR_TO),
                           self.__last_attr_name)
            return attr

    hoc = HookedObjectContainer(name="foo_hooked", data=MyObject())
    assert hoc.foo is hoc.data.foo


def test_ItemAccessMixin():
    """Tests the ItemAccessMixin using the ObjectContainer"""
    obj = dtr.containers.ObjectContainer(name="obj", data=dict())

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
    mmc = dtr.containers.MutableMappingContainer(name="map",
                                                 data=dict(foo="bar",
                                                           spam="eggs"))

    assert list(mmc.keys()) == ['foo', 'spam']
    assert list(mmc.values()) == ["bar", "eggs"]
    assert list(mmc.items()) == [('foo', "bar"), ('spam', "eggs")]

    assert mmc.get('foo') == "bar"
    assert mmc.get('spam') == "eggs"
    assert mmc.get('baz', "buz") == "buz"


def test_numeric_mixins():
    """Tests UnaryOperationsMixin and NumbersMixin"""
    # Define a test class using the NumbersMixin, which inherits the
    # UnaryOperationsMixin
    class Num(dtr.mixins.NumbersMixin, dtr.mixins.ComparisonMixin,
              dtr.mixins.ItemAccessMixin, dtr.base.BaseDataContainer):
        
        def copy(self):
            return type(self)(name=self.name + "_copy",
                              data=copy.deepcopy(self.data))

    num = Num(name="foo", data=1.23)
    assert num.data == 1.23

    # Test each function
    assert num.__neg__() == -1.23 == -num
    assert abs(-num) == 1.23
    assert round(num) == 1.
    assert num.__ceil__() == 2.
    assert num.__floor__() == 1.
    assert num.__trunc__() == 1.

    # Make sure functions are called, even if raising errors
    with pytest.raises(TypeError, match="bad operand type for unary ~"):
        num.__invert__()


def test_IntegerItemAccessMixin():
    """Test the IntegerItemAccessMixin"""

    class IntItemAccessGroup(dtr.mixins.IntegerItemAccessMixin,
                             dtr.groups.OrderedDataGroup):
        pass

    grp = IntItemAccessGroup(name="test_group")
    grp.new_group("0")
    grp.new_group("42")
    grp.new_group("132")
    grp.new_group("-12")

    assert grp[0] is grp["0"]
    assert grp[42] is grp["42"]
    assert grp[132] is grp["132"]
    assert grp[-12] is grp["-12"]

    assert 0 in grp
    assert 42 in grp
    assert 132 in grp
    assert -12 in grp


def test_PaddedIntegerItemAccessMixin():
    """Test PaddedIntegerItemAccessMixin"""

    class PaddedIntItemAccessGroup(dtr.mixins.PaddedIntegerItemAccessMixin,
                                   dtr.groups.OrderedDataGroup):
        pass

    grp = PaddedIntItemAccessGroup(name="test_group")
    assert grp.padded_int_key_width is None

    # Add a group
    grp.new_group("00")

    # Now, key width should have been deduced
    assert grp.padded_int_key_width == 2

    # Add more groups
    grp.new_group("01")
    grp.new_group("04")
    grp.new_group("42")

    # Check that access works
    for int_key, str_key in [(0, "00"), (1, "01"), (4, "04"), (42, "42")]:
        assert int_key in grp
        assert str_key in grp
        assert grp[int_key] is grp[str_key]

    # Check bad access values
    with pytest.raises(IndexError, match=r'out of range \[0, 99\]'):
        grp[-1]
    
    with pytest.raises(IndexError, match=r'out of range \[0, 99\]'):
        grp[100]

    # Check bad container names
    with pytest.raises(ValueError, match="need names of the same length"):
        grp.new_group("123")

    # Already set
    with pytest.raises(ValueError, match="already set"):
        grp.padded_int_key_width = 123

    # Non-positive value
    with pytest.raises(ValueError, match="needs to be positive"):
        PaddedIntItemAccessGroup(name="baaad").padded_int_key_width = 0
    
    with pytest.raises(ValueError, match="needs to be positive"):
        PaddedIntItemAccessGroup(name="baaad").padded_int_key_width = -42
