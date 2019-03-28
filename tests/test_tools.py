"""Test the tools module"""

import pytest
import numpy as np

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

def test_decode_bytestrings():
    """Tests the decode bytestrings function"""
    decode = t.decode_bytestrings
    
    foob = bytes("foo", "utf8")
    barb = bytes("bar", "utf8")

    # Nothing happens for regular data
    assert decode(1) == 1
    assert decode("foo") == "foo"
    assert (decode(np.array([123, 345])) == np.array([123, 345])).all()

    # Bytes get decoded to utf8
    assert decode(foob) == "foo"

    # Numpy string arrays get their dtype changed
    assert decode(np.array(["foo", "bar"], dtype="S")).dtype == np.dtype("<U3")
    assert decode(np.array(["foo", "bar"], dtype="a")).dtype == np.dtype("<U3")

    # Object arrays get their items converted
    assert decode(np.array([foob, barb], dtype="O"))[0] == "foo"
    assert decode(np.array([foob, barb], dtype="O"))[1] == "bar"
    assert decode(np.array([foob, "bar"], dtype="O"))[1] == "bar"
