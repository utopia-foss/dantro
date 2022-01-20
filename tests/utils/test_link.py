"""Test the utils.link module"""

import numpy as np
import pytest

import dantro
import dantro.utils.coords
from dantro.containers import ObjectContainer, StringContainer
from dantro.groups import OrderedDataGroup
from dantro.mixins import ForwardAttrsToDataMixin
from dantro.utils import Link, StrongLink
from dantro.utils.link import _strongref

from ..test_base import pickle_roundtrip

# Fixtures and Tools ----------------------------------------------------------


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


def test_Link_and_StrongLink(root):
    """Test the Link and StrongLink classes"""
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

    for _Link in (Link, StrongLink):
        name = _Link.__name__  # To use for distinguishing cases

        link_root2c2 = _Link(anchor=root, rel_path="group//subgroup/c2")  # sic

        assert link_root2c2.anchor_weakref() is root
        assert link_root2c2.anchor_object is root
        assert link_root2c2.target_rel_path == "group//subgroup/c2"
        assert link_root2c2.target_object is c2
        assert link_root2c2.upper() == "AT_LEVEL2"

        # Test equality
        assert link_root2c2 == link_root2c2
        assert link_root2c2 == _Link(
            anchor=root, rel_path="group//subgroup/c2"
        )
        assert link_root2c2 != _Link(anchor=root, rel_path="group/subgroup/c2")
        assert link_root2c2 != 1
        assert link_root2c2 != dict(anchor=root, rel_path="group//subgroup/c2")

        # Scenario1: Link from one object to one further down the tree
        link_c0c2 = _Link(anchor=c0, rel_path="group/subgroup/c2")
        assert link_c0c2.target_object is c2
        assert link_c0c2.upper() == "AT_LEVEL2"

        # Scenario2: Link from one object to one further _up_ the tree
        link_c2c0 = _Link(anchor=c2, rel_path="../../c0")
        assert link_c2c0.target_object is c0
        assert link_c2c0.upper() == "AT_ROOT"

        # Scenario3: Link from a group to another group
        link_gsg = _Link(anchor=group, rel_path="subgroup")
        assert link_gsg.target_object is subgroup
        assert link_gsg.name == "subgroup"

        link_sgg = _Link(anchor=subgroup, rel_path="../")
        assert link_sgg.target_object is group
        assert link_sgg.name == "group"

        # Scenario4: Link from a container to a group
        link_c0sg = _Link(anchor=c0, rel_path="group/subgroup")
        assert link_c0sg.target_object is subgroup
        assert link_c0sg.name == "subgroup"

        link_c2root = _Link(anchor=c2, rel_path="../../")
        assert link_c2root.target_object is root
        assert link_c2root.name == "root"

        # Scenario5: Link from a group to a container
        link_rootc2 = _Link(anchor=root, rel_path="group/subgroup/c2")
        assert link_rootc2.target_object is c2
        assert link_rootc2.upper() == "AT_LEVEL2"

        link_sgc0 = _Link(anchor=subgroup, rel_path="../../c0")
        assert link_sgc0.target_object is c0
        assert link_sgc0.upper() == "AT_ROOT"

        # Scenario6: Link with empty segments, i.e. with repeated `/` in the
        # path. This should have the same behaviour as with just a single `/`.
        link_c0c2_mod = _Link(anchor=c0, rel_path="group///subgroup//c2")
        assert link_c0c2_mod.target_rel_path == "group///subgroup//c2"  # sic
        assert link_c0c2_mod.target_object is link_c0c2.target_object

        # Scenario7: Link to itself
        link_rootroot = _Link(anchor=root, rel_path="")
        assert link_rootroot.target_object is root

        link_c0c0 = _Link(anchor=c0, rel_path="")
        assert link_c0c0.target_object is c0

        link_detdet = _Link(anchor=detached, rel_path="")
        assert link_detdet.target_object is detached

        # Scenario8: Circular link between two side-by-side objects
        foo = StringContainer(name=f"foo_{name}", data="foo_data")
        bar = StringContainer(name=f"bar_{name}", data="bar_data")
        root.add(foo, bar)

        assert foo.path == f"/root/foo_{name}"
        assert bar.path == f"/root/bar_{name}"

        link2bar = _Link(anchor=foo, rel_path=bar.name)
        link2foo = _Link(anchor=bar, rel_path=foo.name)

        assert link2bar.anchor_weakref() is foo
        assert link2foo.anchor_weakref() is bar

        assert link2bar.anchor_object is foo
        assert link2foo.anchor_object is bar

        assert link2bar.target_rel_path == bar.name
        assert link2foo.target_rel_path == foo.name

        assert link2bar.target_object is bar
        assert link2foo.target_object is foo

        assert link2bar.upper() == "BAR_DATA"
        assert link2foo.upper() == "FOO_DATA"

        # Test error messages . . . . . . . . . . . . . . . . . . . . . . . . .
        # Scenario E1: Detached container as anchor
        # A link can still be formed, but fails upon resolving because the
        # anchor is a container and is not attached anywhere
        assert detached.parent is None
        det_link = _Link(anchor=detached, rel_path="some/imaginary/../path")

        with pytest.raises(ValueError, match="not embedded into a data tree;"):
            det_link.target_object

        # ... a different error is raised when having a group as anchor, see E2

        # Scenario E2: Target path cannot be found
        bad_path = _Link(anchor=root, rel_path="some/imaginary/../path")

        with pytest.raises(
            ValueError, match="Failed resolving target of link"
        ):
            bad_path.target_object


def test_Link_pickling(root):
    """Tests pickling of Link objects"""
    link_root2c2 = Link(anchor=root, rel_path="group/subgroup/c2")
    pickle_roundtrip(link_root2c2) == link_root2c2

    stronglink_root2c2 = StrongLink(anchor=root, rel_path="group/subgroup/c2")
    pickle_roundtrip(stronglink_root2c2) == stronglink_root2c2


def test__strongref():
    foo = "foo"
    bar = "bar"
    ref = _strongref(foo)

    # Call method
    assert ref() is foo
    assert ref() is not bar

    # Equality
    assert ref == ref
    assert ref != foo
    assert ref != "foo"
    assert ref != _strongref(bar)
