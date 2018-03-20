"""Test the DataGroup and derived classes"""

import pytest

import dantro.group as grp

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation
    dg = grp.DataGroup(name="foo", containers=[])

    # Passing some container
