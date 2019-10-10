"""Test the _utils subpackage"""

import sys
import random
import uuid
import collections
from typing import Callable

import pytest

import numpy as np

import dantro
import dantro.utils
import dantro.utils.coords
import dantro.groups

from dantro.containers import ObjectContainer, XrDataContainer
from dantro.mixins import ForwardAttrsToDataMixin

# Fixtures --------------------------------------------------------------------

def random_kv_pairs(max_num: int=100, *, key_kind="int",
                    key_sort_func: Callable=None) -> tuple:
    """Returns a randomly shuffled list of key-value pairs and a sequence
    of the ordered keys.
    """
    # First, build a set of random keys, i.e.: without collisions
    if key_kind == "int":
        keys = {random.randint(0, max_num) for _ in range(max_num)}

    elif key_kind == "str":
        keys = {random.randint(0, max_num) for _ in range(max_num)}
        keys = [str(k) for k in keys]

    else:
        raise ValueError(key_kind)

    # Now, build the key-value pair list and shuffle it (in place)
    l = [(k, uuid.uuid4().hex) for k in keys]
    random.shuffle(l)

    # Also generate a sequence of ordered keys and return it alongside
    return l, sorted([k for k, _ in l], key=key_sort_func)


# Tests -----------------------------------------------------------------------

def test_KeyOrderedDict():
    """Tests the KeyOrderedDict, a subclass of OrderedDict maintaining key
    order rather than insertion order
    """

    KOD = dantro.utils.KeyOrderedDict

    # Simple test
    kod = KOD()

    # Insert elements one by one
    kv_pairs, sorted_keys = random_kv_pairs()

    for k, v in kv_pairs:
        kod[k] = v

    print("\n--- Initial test")
    print("Length: {}, expected {}".format(len(kod), len(sorted_keys)))
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))

    # Check that keys are ordered correctly
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Reverse iteration should also work
    assert all([k1 == k2 for k1, k2 in zip(reversed(kod),
                                           reversed(sorted_keys))])

    # Custom comparator, here: for reverse ordering
    comp_reversed = lambda k: -k
    kv_pairs, sorted_keys = random_kv_pairs(key_sort_func=comp_reversed)
    kod = KOD(kv_pairs, key=comp_reversed)
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Custom insert method
    with pytest.raises(NotImplementedError):
        kod.insert("key", "value")

    # With str-cast keys but integer sorting
    comp_reversed = lambda k: int(k)
    kv_pairs, sorted_keys = random_kv_pairs(key_kind="str",
                                            key_sort_func=comp_reversed)
    kod = KOD(kv_pairs, key=comp_reversed)
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Custom insert method
    with pytest.raises(NotImplementedError):
        kod.insert("key", "value")

    # Test remaining OrderedDict functionality ................................
    # Do not use a custom comparator for that
    kv_pairs, sorted_keys = random_kv_pairs()
    kod = KOD(kv_pairs)

    print("\n--- Dict functionality")
    print("Length: {}, expected {}".format(len(kod), len(sorted_keys)))
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))
    print("Items:\n ", "\n  ".join([str(i) for i in kod.items()]))

    # String representation, pickling, copy
    KOD.__repr__(None)
    repr(kod)

    kod.__reduce__()

    kod.copy()

    # Comparison
    assert kod == kod.copy()
    assert kod == {k: v for k, v in kod.items()}

    # Iteration methods
    assert all([p1 == p2 for p1, p2 in zip(kod.items(), sorted(kv_pairs))])
    assert all([v == kod[k] for v, k in zip(kod.values(), sorted_keys)])

    # Size and any kind of delete operations
    size = sys.getsizeof
    s1 = size(kod)

    del kod[sorted_keys[0]]
    del kod[sorted_keys[1]]
    s2 = size(kod)
    assert s2 < s1

    kv3 = kod[sorted_keys[2]]
    assert kv3 is kod.pop(sorted_keys[2])
    s3 = size(kod)
    assert None is kod.pop(sorted_keys[2], None)
    assert size(kod) == s3
    assert s3 < s2

    kod.clear()
    assert size(kod) < s3

    # Fill it again, this time from the classmethod and without values
    kod = KOD.fromkeys([k for k, _ in kv_pairs], value="foobar")

    # Set with default
    print("\n--- setdefault")
    print("Length: {}, expected {}".format(len(kod), len(sorted_keys)))
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))
    assert sorted_keys[3] in kod
    assert kod.setdefault(sorted_keys[3]) == "foobar"
    assert -42 not in kod
    assert kod.setdefault(-42) is None
    assert kod[-42] is None

    # Exceptions
    with pytest.raises(TypeError, match="expected at most 1 arguments, got"):
        KOD("foo", "bar")

    with pytest.raises(KeyError, match="-123"):
        kod.pop(-123)

    kod._key = lambda k: int(k)
    with pytest.raises(ValueError, match="Could not apply key transformation"):
        kod["foo"] = None

    kod._key = lambda k: None
    with pytest.raises(ValueError, match="Failed comparing 'None'"):
        kod["foo"] = "bar"

# -----------------------------------------------------------------------------

def test_Link():
    """Test the Link class"""
    Link = dantro.utils.Link

    class StringContainer(ForwardAttrsToDataMixin, ObjectContainer):
        pass

    # Build a hierarchy of groups and containers
    root = dantro.groups.OrderedDataGroup(name="root")
    group = root.new_group("group")
    subgroup = group.new_group("subgroup")

    c0 = StringContainer(name="c0", data="at_root")
    root.add(c0)

    c1 = StringContainer(name="c1", data="at_level1")
    group.add(c1)

    c2 = StringContainer(name="c2", data="at_level2")
    subgroup.add(c2)

    detached = StringContainer(name="detached", data="detached_data")
    assert detached.parent is None

    # Test paths, property storage, and attribute forwarding
    assert root.path == "/root"
    assert group.path == "/root/group"
    assert subgroup.path == "/root/group/subgroup"
    assert c0.path == "/root/c0"
    assert c1.path == "/root/group/c1"
    assert c2.path == "/root/group/subgroup/c2"
    assert detached.path == "/detached"

    link_root2c2 = Link(anchor=root, rel_path="group//subgroup/c2")  # sic

    assert link_root2c2.anchor_weakref() is root
    assert link_root2c2.anchor_object is root
    assert link_root2c2.target_rel_path is "group//subgroup/c2"
    assert link_root2c2.target_object is c2
    assert link_root2c2.upper() == "AT_LEVEL2"

    # Scenario1: Link from one object to one further down the tree
    link_c0c2 = Link(anchor=c0, rel_path="group/subgroup/c2")
    assert link_c0c2.target_object is c2
    assert link_c0c2.upper() == "AT_LEVEL2"

    # Scenario2: Link from one object to one further _up_ the tree
    link_c2c0 = Link(anchor=c2, rel_path="../../c0")
    assert link_c2c0.target_object is c0
    assert link_c2c0.upper() == "AT_ROOT"
    
    # Scenario3: Link from a group to another group
    link_gsg = Link(anchor=group, rel_path="subgroup")
    assert link_gsg.target_object is subgroup
    assert link_gsg.name == "subgroup"
    
    link_sgg = Link(anchor=subgroup, rel_path="../")
    assert link_sgg.target_object is group
    assert link_sgg.name == "group"
    
    # Scenario4: Link from a container to a group
    link_c0sg = Link(anchor=c0, rel_path="group/subgroup")
    assert link_c0sg.target_object is subgroup
    assert link_c0sg.name == "subgroup"

    link_c2root = Link(anchor=c2, rel_path="../../")
    assert link_c2root.target_object is root
    assert link_c2root.name == "root"

    # Scenario5: Link from a group to a container
    link_rootc2 = Link(anchor=root, rel_path="group/subgroup/c2")
    assert link_rootc2.target_object is c2
    assert link_rootc2.upper() == "AT_LEVEL2"

    link_sgc0 = Link(anchor=subgroup, rel_path="../../c0")
    assert link_sgc0.target_object is c0
    assert link_sgc0.upper() == "AT_ROOT"
    
    # Scenario6: Link with empty segments, i.e. with repeated `/` in the path
    # This should have the same behaviour as with just a single `/`.
    link_c0c2_mod = Link(anchor=c0, rel_path="group///subgroup//c2")
    assert link_c0c2_mod.target_rel_path is "group///subgroup//c2"  # sic
    assert link_c0c2_mod.target_object is link_c0c2.target_object

    # Scenario7: Link to itself
    link_rootroot = Link(anchor=root, rel_path="")
    assert link_rootroot.target_object is root
    
    link_c0c0 = Link(anchor=c0, rel_path="")
    assert link_c0c0.target_object is c0

    link_detdet = Link(anchor=detached, rel_path="")
    assert link_detdet.target_object is detached

    # Scenario8: Circular link between two side-by-side objects
    foo = StringContainer(name="foo", data="foo_data")
    bar = StringContainer(name="bar", data="bar_data")
    root.add(foo, bar)

    assert foo.path == "/root/foo"
    assert bar.path == "/root/bar"

    link2bar = Link(anchor=foo, rel_path="bar")
    link2foo = Link(anchor=bar, rel_path="foo")

    assert link2bar.anchor_weakref() is foo
    assert link2foo.anchor_weakref() is bar

    assert link2bar.anchor_object is foo
    assert link2foo.anchor_object is bar
    
    assert link2bar.target_rel_path == "bar"
    assert link2foo.target_rel_path == "foo"

    assert link2bar.target_object is bar
    assert link2foo.target_object is foo

    assert link2bar.upper() == "BAR_DATA"
    assert link2foo.upper() == "FOO_DATA"

    # Test error messages . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Scenario E1: Detached container as anchor
    # A link can still be formed, but fails upon resolving because the anchor
    # is a container and is not attached anywhere
    assert detached.parent is None
    det_link = Link(anchor=detached, rel_path="some/imaginary/../path")
    
    with pytest.raises(ValueError, match="is not embedded into a data tree;"):
        det_link.target_object

    # ... a different error is raised when having a group as anchor, see E2:

    # Scenario E2: Target path cannot be found
    bad_path = Link(anchor=root, rel_path="some/imaginary/../path")

    with pytest.raises(ValueError, match="Failed resolving target of link"):
        bad_path.target_object


# -----------------------------------------------------------------------------
# NOTE Partly tested by containers using this, e.g. XrDataContainer ...

def test_coord_extractor_functions():
    """Tests the coordinate extractor functions"""
    extr = dantro.utils.coords.COORD_EXTRACTORS

    # Values; just returns the given ones, no type change
    assert extr['values']([1,2,3]) == [1,2,3]
    assert isinstance(extr['values']((1,2,3)), tuple)

    # Range
    assert extr['range']([10]) == list(range(10))
    assert extr['range']([2, 10, 2]) == list(range(2, 10, 2))

    # np.arange
    assert (extr['arange']([0, 10]) == np.arange(0, 10)).all()
    assert (extr['arange']([2, 10, 2]) == np.arange(2, 10, 2)).all()
    
    # np.linspace
    assert (extr['linspace']([0, 10, 10]) == np.linspace(0, 10, 10)).all()
    assert (extr['linspace']([2, 10, 2]) == np.linspace(2, 10, 2)).all()
    
    # np.logspace
    assert (extr['logspace']([0, 10, 11]) == np.logspace(0, 10, 11)).all()
    assert (extr['logspace']([2, 10, 2]) == np.logspace(2, 10, 2)).all()

    # start and step
    assert extr['start_and_step']([0, 1], data_shape=(2,3,4),
                                  dim_num=2) == [0, 1, 2, 3]
    assert extr['start_and_step']([10, 2], data_shape=(5,),
                                  dim_num=0) == [10, 12, 14, 16, 18]

    # trivial
    assert extr['trivial'](None, data_shape=(2,3,4), dim_num=2) == [0, 1, 2, 3]
    assert extr['trivial'](123, data_shape=(40,), dim_num=0) == list(range(40))

    # scalar
    assert extr['scalar'](1) == [1]
    assert extr['scalar']([1]) == [1]
    assert extr['scalar'](np.array([1])) == [1]
    assert extr['scalar']((1,)) == [1]
    assert isinstance(extr['scalar']((1,)), list)

    with pytest.raises(ValueError, match="Expected scalar coordinate"):
        extr['scalar']([1, 2, 3])

    # linked
    class C:
        """A Mock class for creating a Link object"""
        logstr = "object of class C"

    assert isinstance(extr['linked']("foo/bar",
                                     link_anchor_obj=C()),
                      dantro.utils.coords.Link)
    assert isinstance(extr['linked'](np.array(["foo/bar"]),
                                     link_anchor_obj=C()),
                      dantro.utils.coords.Link)

def test_extract_coords_from_name():
    """Tests .utils.coords.extract_coords_from_name"""
    extract = dantro.utils.coords.extract_coords_from_name
    Cont = lambda name: ObjectContainer(name=name, data=None)

    assert extract(Cont('123;456;789'), dims=('foo', 'bar', 'baz'),
                   separator=';'
                   ) == dict(foo=[123], bar=[456], baz=[789])
    assert extract(Cont('123;456;789'), dims=('foo', 'bar', 'baz'),
                   attempt_conversion=False, separator=';'
                   ) == dict(foo=['123'], bar=['456'], baz=['789'])

    # Conversion
    kws = dict(dims=('foo',), separator=';')
    assert extract(Cont('1'), **kws)['foo'] == [1]
    assert extract(Cont('1.'), **kws)['foo'] == [1.]
    assert isinstance(extract(Cont('1.'), **kws)['foo'][0], float)
    assert extract(Cont('1.+1j'), **kws)['foo'] == [1.+1j]
    assert extract(Cont('stuff'), **kws)['foo'] == ['stuff']
    assert isinstance(extract(Cont('stuff'), **kws)['foo'][0], str)

    # Error messages
    with pytest.raises(ValueError,
                       match="Number of coordinates .* does not match"):
        extract(Cont('1;2'), **kws)

    with pytest.raises(ValueError, match="One or more .* were empty!"):
        extract(Cont('1;;3'), dims=('foo', 'bar', 'baz'), separator=';')


def test_extract_coords():
    """This is mostly tested already by the XrDataContainer test"""
    extract = dantro.utils.coords.extract_coords

    # Invalid mode
    with pytest.raises(ValueError, match="Invalid extraction mode 'bad_mode'"):
        extract(XrDataContainer(name='test', data=[1,2,3]),
                mode='bad_mode', dims=('foo',))

    # Caching
    with pytest.raises(NotImplementedError):
        extract(XrDataContainer(name='test', data=[1,2,3]),
                mode='name', dims=('foo',), use_cache=True)
