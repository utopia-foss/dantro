"""Test the utils.link module"""

import pytest

import numpy as np

import dantro
import dantro.utils
import dantro.utils.coords

from dantro.containers import ObjectContainer
from dantro.mixins import ForwardAttrsToDataMixin


# -----------------------------------------------------------------------------

def test_Link():
    """Test the Link class"""
    Link = dantro.utils.Link

    class StringContainer(ForwardAttrsToDataMixin, ObjectContainer):
        pass

    # Build up a circular scenario
    foo = StringContainer(name="foo", data="foo_data")
    bar = StringContainer(name="bar", data="bar_data")
    
    some_grp = dantro.groups.OrderedDataGroup(name="root")
    some_grp.add(foo, bar)

    link2bar = Link(anchor=foo, rel_path="../bar")
    link2foo = Link(anchor=bar, rel_path="../foo")

    assert link2bar.anchor_weakref() is foo
    assert link2foo.anchor_weakref() is bar

    assert link2bar.anchor_object is foo
    assert link2foo.anchor_object is bar
    
    assert link2bar.target_rel_path == "../bar"
    assert link2foo.target_rel_path == "../foo"

    assert link2bar.target_object is bar
    assert link2foo.target_object is foo

    # Test attribute forwarding
    assert link2bar.upper() == "BAR_DATA"
    assert link2foo.upper() == "FOO_DATA"
