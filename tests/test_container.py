"""Test the BaseDataContainer-derived classes"""

import pytest

import dantro.container as cont

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_init():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation
    ic = cont.ItemContainer(name="foo", data=[])

    # Passing some containers
