"""Test the tools module"""

import sys

import pytest

import dantro.tools as t


# Local constants


# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_tmp_sys_path():
    """Tests the tmp_sys_path context manager"""

    # Define some test paths
    test_paths = ["/foo/bar", "/bar/baz"]

    # Store the old paths -- should return to this config afterwards
    old_paths = sys.path

    # Check that they are not empty; testing makes no sense otherwise
    assert sys.path

    # As default, the paths should be appended
    with t.tmp_sys_path(*test_paths):
        assert sys.path == old_paths + test_paths

    # Everything should be the same as before
    assert sys.path == old_paths
