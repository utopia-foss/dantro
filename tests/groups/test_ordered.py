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

    assert idg.min_key == '4'
    assert idg.max_key == '51'
    assert [k for k in idg.keys_as_int()] == [int(k) for k in keys_ordered]
