"""Test the ordered groups"""

import pytest

from dantro.containers import ObjectContainer, PathContainer
from dantro.exceptions import *
from dantro.groups import DirectoryGroup, OrderedDataGroup

# -----------------------------------------------------------------------------


def test_DirectoryGroup(tmpdir):
    """Test the DirectoryGroup"""
    testdir = tmpdir.mkdir("testdir")
    grp = DirectoryGroup(name="test", dirpath=str(testdir))
    p1 = grp.fs_path
    assert p1 == testdir

    # is stored as copy
    grp2 = DirectoryGroup(name="test", dirpath=testdir)
    assert grp2.fs_path is not testdir

    # Can deduce path from parent
    subgrp = grp.new_group("sub")
    assert subgrp.fs_path == grp.fs_path.joinpath("sub")

    # ... but it needs to be of the correct type
    with pytest.raises(TypeError, match="need a parent"):
        DirectoryGroup(
            name="need_DirectoryGroup", parent=OrderedDataGroup(name="foo")
        )

    with pytest.raises(TypeError, match="need a parent"):
        DirectoryGroup(name="need_DirectoryGroup", parent=None)


def test_DirectoryGroup_strictness(tmpdir):
    """Test that new groups or containers can only be of a certain type"""
    # new containers can only be PathContainers or DirectoryGroups
    testdir = tmpdir.mkdir("testdir")

    grp = DirectoryGroup(name="test", dirpath=testdir)

    subgrp = grp.new_group("sub", dirpath=testdir.mkdir("sub"))
    assert isinstance(subgrp, DirectoryGroup)

    file = grp.new_container(path="file", data=testdir.join("foo.bar"))
    assert isinstance(file, PathContainer)

    with pytest.raises(TypeError, match="Can only add objects derived from"):
        grp.new_group("foo", Cls=OrderedDataGroup)

    with pytest.raises(TypeError, match="Can only add objects derived from"):
        grp.new_container("bar", data="", Cls=ObjectContainer)

    grp = DirectoryGroup(name="test", dirpath=testdir, strict=False)
    grp.new_group("foo", Cls=OrderedDataGroup)
    grp.new_container("bar", data="", Cls=ObjectContainer)
