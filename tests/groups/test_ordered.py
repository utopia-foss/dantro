"""Test the ordered groups"""

import pytest

from dantro.groups import IndexedDataGroup

# -----------------------------------------------------------------------------

def test_IndexedDataGroup():
    """Test the IndexedDataGroup"""
    IDG = IndexedDataGroup

    idg = IDG(name="test")
    keys = ['4', '41', '5', '51', '50']
    keys_ordered = sorted(keys, key=int)
    
    for k in keys:
        idg.new_group(k)
    
    assert [k for k in idg.keys()] == keys_ordered

    # Test key access
    assert [k for k in idg.keys_as_int()] == [int(k) for k in keys_ordered]

    assert idg.key_at_idx(0) == '4'
    assert idg.key_at_idx(-1) == '51'

    for i in ("foo", dict(), [], ()):
        with pytest.raises(TypeError, match="Expected integer, got"):
            idg.key_at_idx(i)

    for i in (-6, -100, 5, 100):
        with pytest.raises(IndexError, match="out of range"):
            idg.key_at_idx(i)
