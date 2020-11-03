"""Test the utils.link module"""

import pytest

import numpy as np

import dantro
import dantro.utils.coords

from dantro.utils import Link
from dantro.containers import ObjectContainer, StringContainer
from dantro.mixins import ForwardAttrsToDataMixin
from dantro.groups import OrderedDataGroup


# Fixtures and Tools ----------------------------------------------------------

from ..test_base import pickle_roundtrip

@pytest.fixture
def root() -> OrderedDataGroup:
    """A fixture returning a group with some content"""
    root = OrderedDataGroup(name="root")
    group = root.new_group("group")
    subgroup = group.new_group("subgroup")

    c0 = StringContainer(name="c0", data="at_root")
    root.add(c0)

    c1 = StringContainer(name="c1", data="at_level1")
    group.add(c1)

    c2 = StringContainer(name="c2", data="at_level2")
    subgroup.add(c2)

    return root

# -----------------------------------------------------------------------------

def test_Link(root):
    """Test the Link class"""
    # The hierarchy
    group = root["group"]
    subgroup = root["group/subgroup"]
    c0 = root["c0"]
    c1 = root["group/c1"]
    c2 = root["group/subgroup/c2"]

    # A detached container
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

    # Test equality
    assert link_root2c2 == link_root2c2
    assert link_root2c2 == Link(anchor=root, rel_path="group//subgroup/c2")
    assert link_root2c2 != Link(anchor=root, rel_path="group/subgroup/c2")
    assert link_root2c2 != 1
    assert link_root2c2 != dict(anchor=root, rel_path="group//subgroup/c2")

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


def test_Link_pickling(root):
    """Tests pickling of Link objects"""
    link_root2c2 = Link(anchor=root, rel_path="group/subgroup/c2")

    assert pickle_roundtrip(link_root2c2) == link_root2c2
