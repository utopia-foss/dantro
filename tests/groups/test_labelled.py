"""Test the LabelledDataGroups"""

import pytest

from dantro.groups import TimeSeriesGroup

# -----------------------------------------------------------------------------

def test_TimeSeriesGroup():
    """Test the TimeSeriesGroup"""
    TSG = TimeSeriesGroup

    # Build and populate
    tsg = TSG(name="test")
    keys = ['4', '41', '5', '51', '50']
    keys_ordered = sorted(keys, key=int)
    
    for k in keys:
        tsg.new_group(k)
        
    # Test properties
    assert tsg.dims == ('time',)
    assert tsg.coords == dict(time=[int(k) for k in keys_ordered])

    # Test selection methods
    # By value
    for c, k in ((4, '4'), (41, '41'), (51, '51'), (5, '5'), (50, '50')):
        assert tsg.sel(time=c) is tsg[k]
    
    # By index
    for i, k in ((4, '51'), (0, '4'), (-1, '51'), (2, 41), (0, 4)):
        assert tsg.isel(time=i) is tsg[k]

    # Errors
    with pytest.raises(TypeError):
        tsg.sel()
    
    with pytest.raises(TypeError):
        tsg.isel()

    for c in (dict(), 1.23, []):
        with pytest.raises(NotImplementedError, match="Cannot yet"):
            tsg.sel(time=c)

    for c in (dict(), 1.23, []):
        with pytest.raises(NotImplementedError, match="Cannot yet"):
            tsg.isel(time=c)
