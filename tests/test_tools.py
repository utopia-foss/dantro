"""Test the tools module"""

import pytest

import dantro.tools as t


# Local constants


# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------

def test_fill_line():
    """Tests the fill_line and center_in_line methods"""
    # Shortcut for setting number of columns
    fill = lambda *args, **kwargs: t.fill_line(*args, num_cols=10, **kwargs)

    # Check that the expected number of characters are filled at the right spot
    assert fill("foo") == "foo" + 7*" "
    assert fill("foo", fill_char="-") == "foo" + 7*"-"
    assert fill("foo", align='r') == 7*" " + "foo"
    assert fill("foo", align='c') == "   foo    "
    assert fill("foob", align='c') == "   foob   "

    with pytest.raises(ValueError, match="length 1"):
        fill("foo", fill_char="---")

    with pytest.raises(ValueError, match="align argument 'bar' not supported"):
        fill("foo", align='bar')

    # The center_in_line method has a fill_char given and adds a spacing
    assert t.center_in_line("foob", num_cols=10) == "·· foob ··"  # cdot!
    assert t.center_in_line("foob", num_cols=10, spacing=2) == "·  foob  ·"
