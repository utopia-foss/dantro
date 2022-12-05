"""Tests the base classes, as far as possible.

NOTE This test module merely complements the other, already existing tests of
     the base classes that are made implicitly through testing the derivatives.
"""

import logging
import sys

import dill as pkl
import numpy as np
import pytest

import dantro as dtr
import dantro.base
import dantro.containers
import dantro.groups
from dantro.exceptions import *

log = logging.getLogger()


# Class definitions -----------------------------------------------------------


# Fixtures and tools ----------------------------------------------------------


def pickle_roundtrip(
    original_obj,
    *,
    protocols: tuple = (pkl.HIGHEST_PROTOCOL, pkl.DEFAULT_PROTOCOL),
):
    """Makes pickling roundtrips with the given object. It does so multiple
    times with different protocols and returns the _last_ protocol's result.
    """
    for protocol in protocols:
        s = pkl.dumps(original_obj, protocol=protocol)
        log.debug("Pickled '%s'. (Protocol: %s)", type(original_obj), protocol)

        loaded_obj = pkl.loads(s)
        log.debug(
            "Unpickled '%s'. (Protocol: %s", type(original_obj), protocol
        )

    assert loaded_obj is not original_obj

    return loaded_obj


# Tests -----------------------------------------------------------------------


def test_BaseDataAttrs():
    """Test the BaseDataAttrs class"""
    bda = dtr.base.BaseDataAttrs(name="test")

    assert isinstance(bda.data, dict)
    assert bda.data == dict()
    assert bda._format_info() == "0 attribute(s)"

    bda["foo"] = "bar"
    assert bda.data == dict(foo="bar")
    assert bda.as_dict() == dict(foo="bar")

    assert bda._format_info() == "1 attribute(s)"


def test_BaseDataAttrs_pickling():
    """Makes sure that these types are pickleable"""
    bda = dtr.base.BaseDataAttrs(name="test")
    bda["foo"] = "bar"
    bda["bar"] = dict(foofoo="barbar")

    assert pickle_roundtrip(bda) == bda


# .............................................................................


def test_BaseDataGroup():
    """Tests the BaseDataGroup using OrderedDataGroup"""
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")
    bar.add(dtr.containers.ObjectContainer(name="obj", data=dict(test=123)))

    # Test item interface
    assert root["foo"] is foo
    assert root["foo/bar"] is bar

    # Setting should not work
    with pytest.raises(ValueError, match="cannot carry out __setitem__"):
        root["baz"] = dtr.groups.OrderedDataGroup(name="baz")

    # String representation
    assert "foo" in str(foo)
    assert foo.logstr in str(foo)
    assert repr(foo) == str(foo)
    assert foo._format_tree() == foo.tree == foo._tree_repr()
    assert foo._format_tree_condensed() == foo.tree_condensed

    # Accessing data directly
    with pytest.raises(AttributeError, match="Cannot directly access group"):
        foo.data

    # Put passing on the __item__ calls to the object beneath should work
    assert root["foo/bar/obj/test"] == 123
    root["foo/bar/obj/test"] = 234
    assert root["foo/bar/obj/test"] == 234

    # Test the `add` method
    baz1 = foo.new_group("baz")
    assert foo["baz"] is baz1

    baz2 = dtr.groups.OrderedDataGroup(name="baz")
    foo.add(baz2, overwrite=True)
    assert foo["baz"] is baz2

    # Test clearing, then reset to previous state
    assert len(root) == 1
    root.clear()
    assert len(root) == 0
    assert foo.parent is None

    root.add(foo)
    assert len(root) == 1
    assert foo.parent is root

    # And adding new containers that are not of the correct type
    with pytest.raises(TypeError, match="Expected a subclass of BaseData"):
        root.new_container("testpath", Cls=dict, foo="bar")

    # Recursive update
    root2 = dtr.groups.OrderedDataGroup(name="root")
    foo2 = root2.new_group("foo")
    spam = foo2.new_group("spam")
    sth = foo2.new_container(
        "sth", Cls=dtr.containers.ObjectContainer, data=dict(foo="bar")
    )

    root.recursive_update(root2)
    assert "foo/spam" in root
    assert "foo/sth" in root
    assert root["foo/sth"].data == dict(foo="bar")

    with pytest.raises(TypeError, match="Can only update (.+) with objects"):
        root.recursive_update(dict(foo="bar"))

    with pytest.raises(
        ExistingDataError, match="already has a member with name"
    ):
        root.recursive_update(root2, overwrite=False)

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

    # __contains__  (more tests below)
    assert "foo" in root
    assert foo in root  # is-comparison
    assert log not in root

    # iteration and dict access
    assert [k for k in root] == ["foo"]
    assert [k for k in root.keys()] == ["foo"]
    assert [v for v in root.values()] == [foo]
    assert [v for v in root.items()] == [("foo", foo)]

    assert root.get("FOO") is None
    assert root.get("FOO", foo) is foo

    with pytest.raises(NotImplementedError, match="is not supported"):
        root.setdefault(foo)

    # IPython key completions
    assert root._ipython_key_completions_() == list(root.keys())

    # SizeOfMixin
    # Groups with the same number of members take up the same number of bytes
    assert len(root) == len(root2)
    assert sys.getsizeof(root) == sys.getsizeof(root2)

    # Groups with larger number of members take up more bytes
    assert sys.getsizeof(foo) > sys.getsizeof(root)


def test_BaseDataGroup_new_group():
    """Tests adding new groups"""
    root = dtr.groups.OrderedDataGroup(name="root")
    assert not len(root)

    # Simple case
    foo = root.new_group("foo")
    assert "foo" in root
    assert foo is root["foo"]

    # Intermediate groups
    baz = root.new_group("foo/bar/baz")
    assert "bar" in root["foo"]
    assert "baz" in root["foo/bar"]
    assert "bar" in foo
    assert "baz" in foo["bar"]

    # Create object in PARENT
    also_foo = foo.new_group("../also_foo")
    assert "also_foo" in root
    assert also_foo in root

    # Errors
    with pytest.raises(ValueError, match="may not be empty"):
        root.new_group("")


def test_BaseDataGroup_getitem():
    """Tests the item access interface"""
    root = dtr.groups.OrderedDataGroup(name="root")

    # Create a deep hierarchy
    root.new_group("alpha/beta/gamma/delta/epsilon")

    gamma = root["alpha/beta/gamma"]
    assert gamma.name == "gamma"
    assert gamma.parent.name == "beta"
    assert "delta" in gamma

    # Walk through the tree, up and down
    for path, expected_name in (
        # Access self
        (".", "gamma"),
        ("./.", "gamma"),
        ([".", "."], "gamma"),
        #
        # Access via parent
        ("..", "beta"),
        ([".."], "beta"),
        ("../.", "beta"),
        (["..", "."], "beta"),
        (["..", ".."], "alpha"),
        ("../../", "alpha"),
        ("../../.", "alpha"),
        #
        # Downstream access
        ("delta", "delta"),
        ("delta/", "delta"),
        ("./delta/", "delta"),
        ("delta/epsilon", "epsilon"),
        ("delta/epsilon/", "epsilon"),
        ("./delta/epsilon", "epsilon"),
        #
        # Mixed
        (".//delta/epsilon/../", "delta"),
        (".//delta/./epsilon/////../epsilon/", "epsilon"),
    ):
        print("path:         ", path)
        print("expected_name:", expected_name)
        assert gamma[path].name == expected_name

    # Errors
    with pytest.raises(ItemAccessError, match="No parent associated."):
        root[".."]

    with pytest.raises(ItemAccessError, match="No item .*Available.*alpha"):
        root["i_do_not_exist"]

    with pytest.raises(ItemAccessError, match="No item .*Available.*gamma"):
        root["alpha/beta/i_do_not_exist"]

    with pytest.raises(ItemAccessError, match="Can only do relative lookups!"):
        gamma["/delta"]

    with pytest.raises(ItemAccessError, match="Can only do relative lookups!"):
        gamma[()]

    # Hints if there are many keys
    for i in range(100):
        root.new_group(f"group_with_many_members/{i:02d}")
    grp = root["group_with_many_members"]
    assert len(grp) == 100

    with pytest.raises(ItemAccessError, match="Did you mean.*42.*?"):
        grp["042"]


def test_BaseDataGroup_contains():
    """Tests the __contains__ method"""
    root = dtr.groups.OrderedDataGroup(name="root")
    baz = root.new_group("foo/bar/baz")

    # Via object lookup
    assert baz in root["foo/bar"]
    assert baz not in root["foo"]

    # Via path
    assert "baz" in root["foo/bar"]
    assert "bar/baz" in root["foo"]
    assert "foo/bar/baz" in root
    assert "foo//bar/./baz" in root
    assert "../foo/bar/baz/../baz" in root["foo"]

    assert "baz" not in root
    assert "baz" not in root["foo"]
    assert "./baz" not in root["foo"]

    assert "something else" not in root["foo"]


def test_BaseDataGroup_tree_repr():
    """Tests the tree representation algorithm.

    NOTE This is only a coverage test, it does not ascertain the correct string
         output! For that, add an assert False and inspect the output.
    """
    # Generate test data
    root = dtr.groups.OrderedDataGroup(name="root")
    grps = [root.new_group(f"grp_{i:d}") for i in range(13)]
    for n, grp in zip(range(13), grps):
        for i in range(n + 1):
            grp.add(
                dtr.containers.ObjectContainer(
                    name=f"obj_{i:d}", data=dict(foo=i)
                )
            )

    def ct_func_half(*, num_items, **_) -> int:
        return num_items // 2

    def ct_func_by_level(*, level, **_) -> int:
        return 10 - 3 * level

    def ct_func_tot(*, total_item_count, **_) -> int:
        if total_item_count <= 13:
            return None
        return 3

    # Test the condense_thresh feature
    for ct in (
        None,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        ct_func_half,
        ct_func_tot,
        ct_func_by_level,
    ):
        print("condense_thresh:", ct, root._tree_repr(condense_thresh=ct))

    # Test the max_level feature
    for ml in (None, 0, 1, 2):
        print("max_level:", ml, root._tree_repr(max_level=ml))

    # Invoke again using properties
    assert root.tree == root._format_tree() == root._tree_repr()
    assert (
        root.tree_condensed
        == root._format_tree_condensed()
        == root._tree_repr(
            max_level=root._COND_TREE_MAX_LEVEL,
            condense_thresh=root._COND_TREE_CONDENSE_THRESH,
        )
    )

    # With a bad member, where formatting fails, there is an error message
    root._data["baaad_member"] = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Failed formatting.*ndarray"):
        root.tree


def test_BaseDataGroup_path_behaviour():
    """Test path capabilities using the OrderedDataGroup"""
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")

    # Test correct parent association
    assert root.parent is None
    assert foo.parent is root
    assert bar.parent is foo

    # Path creation
    assert root.path == "/root"
    assert foo.path == "/root/foo"
    assert bar.path == "/root/foo/bar"

    # Format function
    assert root._format_path() == root.path
    assert foo._format_path() == foo.path
    assert bar._format_path() == bar.path

    # Trying to set a parent if it is currently set should not work
    with pytest.raises(ValueError, match="A parent was already associated"):
        bar.parent = root


def test_BaseDataGroup_renaming():
    """Test that renaming works only if no parent is associated"""
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")

    assert root.path == "/root"
    assert foo.path == "/root/foo"

    # Cannot rename foo
    with pytest.raises(
        ValueError,
        match="Cannot rename .* because a parent was already associated",
    ):
        foo.name = "bar"

    # Can rename root
    root.name = "new_root"

    assert root.path == "/new_root"
    assert foo.path == "/new_root/foo"


def test_BaseDataGroup_pickling():
    """Tests pickling of the BaseDataGroup (using OrderedDataGroup as a simple
    implementation of that base class
    """
    root = dtr.groups.OrderedDataGroup(name="root")
    foo = root.new_group("foo")
    bar = foo.new_group("bar")

    assert pickle_roundtrip(root) == root

    sth = foo.new_container(
        "sth", Cls=dtr.containers.ObjectContainer, data=dict(foo="bar")
    )

    assert pickle_roundtrip(sth) == sth
    root_pkld = pickle_roundtrip(root)
    assert root_pkld == root

    # Attributes matter
    sth.attrs["foo"] = "some new attributes"
    assert pickle_roundtrip(root) == root
    assert pickle_roundtrip(root) != root_pkld
