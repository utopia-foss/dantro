"""Test the DataGroup and derived classes"""

import pytest

import dantro.group as grp

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """ """
    dg = grp.DataGroup(name="foo", containers=[], parent=None)
