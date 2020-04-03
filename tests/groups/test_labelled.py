"""Test the LabelledDataGroup and specializations of it, e.g. TimeSeriesGroup
"""

from itertools import product

import pytest

import numpy as np

from dantro.groups import (LabelledDataGroup,
                           TimeSeriesGroup, HeterogeneousTimeSeriesGroup)
from dantro.containers import XrDataContainer

# -----------------------------------------------------------------------------

def test_LabelledDataGroup_basics():
    """Test the basics of the LabelledDataGroup, e.g. reading the coordinates,
    building the member map, basic selection calls, ...
    """
    # Specify some names and data, intentionally unordered
    names_and_data = [("{};{}".format(f, b), np.ones((2,2)) * b + f)
                      for f, b in product((.1, .2, .5, .3), (2, 8, 4))]

    # Test initialization
    ldg = LabelledDataGroup(name="test", dims=('foo', 'bar'))
    ldg.LDG_EXTRACT_COORDS_FROM = 'name'  # can do this here because no members
                                          # were added yet

    # Add some members (in 2D space)
    for name, data in names_and_data:
        ldg.new_container(name, data=data,
                          attrs=dict(dims=('subfoo', 'subbar')))

    print(ldg.tree)

    # It's also possible to directly add members during initialization
    class LDGnames(LabelledDataGroup):
        LDG_EXTRACT_COORDS_FROM = 'name'

    LDGnames(name="test2", dims=('foo', 'bar'),
             containers=[XrDataContainer(name=name, data=data)
                         for name, data in names_and_data])

    # Access coordinates and other properties prior to member map creation
    assert not ldg.member_map_available
    dims = ldg.dims
    ndim = ldg.ndim
    coords = ldg.coords
    shape = ldg.shape

    assert not ldg.member_map_available
    assert dims == ('foo', 'bar')
    assert ndim == 2
    assert shape == (4, 3)
    assert dict(coords) == dict(foo=[.1, .2, .3, .5], bar=[2, 4, 8]) # sorted!

    # Now, compute the member map
    mm = ldg.member_map
    assert ldg.member_map_available

    # Property values should be the same as before it was available
    assert dims == ldg.dims
    assert ndim == ldg.ndim
    assert (coords['foo'] == ldg.coords['foo']).all()
    assert (coords['bar'] == ldg.coords['bar']).all()
    assert shape == ldg.shape

    # Result should be the same when reading coordiantes from attributes
    ldga = LabelledDataGroup(name="test", dims=('foo', 'bar'))
    ldga.LDG_EXTRACT_COORDS_FROM = 'attrs'

    # Add some members (in 2D space), with names intentionally unordered
    for f, b in product((.1, .2, .5, .3), (2, 8, 4)):
        ldga.new_container("{};{}".format(f, b),
                           data=np.ones((2,2)) * b + f,
                           attrs=dict(dims=('subfoo', 'subbar'),
                                      ext_coords__foo=[f],
                                      ext_coords__bar=[b]))

    print(ldga.tree)

    assert (ldga.member_map == ldg.member_map).all()

    # Should be able to perform selection with both
    assert (ldg.sel(bar=[2, 8]) == ldga.isel(bar=[0, 2])).all()

    # ... and even do this via concatenation
    assert (   ldg.sel(bar=[2, 8], combination_method='concat')
            == ldg.sel(bar=[2, 8], combination_method='merge')).all()

    # When a new container is added, the member map _can_ be kept
    assert ldga.member_map_available
    ldga.new_container("new_cont".format(.1, 2), data=np.zeros((2, 2)),
                       attrs=dict(ext_coords__foo=[.1],
                                  ext_coords__bar=[2]))
    print(ldga.tree)
    assert ldga.member_map_available
    assert 'new_cont' in ldga.member_map
    assert '0.1;2' not in ldga.member_map

    # ... but for containers for which no coordinates are available, it cannot:
    assert ldga.member_map_available
    ldga.new_container("outside".format(0., 0), data=np.zeros((2, 2)),
                       attrs=dict(ext_coords__foo=[0.],
                                  ext_coords__bar=[0]))
    print(ldga.tree)
    assert not ldga.member_map_available

    # This can now no longer be selected via concatenation
    with pytest.raises(KeyError, match="need it for concatenation"):
        ldga.sel(bar=0, combination_method='concat')

    # ... but via merge
    ldga.sel(bar=0, combination_method='merge')

    # Invalid combination method argument
    with pytest.raises(ValueError, match="Invalid combination_method argumen"):
        ldga.sel(bar=0, combination_method='foo')


def test_LabelledDataGroup_deep_selection():
    """Tests the deep selection interface"""
    # Deep selection control
    ldg = LabelledDataGroup(name="test", allow_deep_selection=True)
    assert ldg.allow_deep_selection
    ldg.allow_deep_selection = False
    assert not ldg.allow_deep_selection

    # There is no dimension associated, so every selection is deep
    with pytest.raises(ValueError, match="Deep indexing is not allowed for"):
        ldg.sel(spam=23)

    # TODO Add some data and deep-select it

# -----------------------------------------------------------------------------

def test_TimeSeriesGroup():
    """Test the TimeSeriesGroup, a specialization of LabelledDataGroup"""
    TSG = TimeSeriesGroup

    # Can initialize it with containers
    TSG(name="with_containers",
        containers=[XrDataContainer(name=str(i), data=np.zeros((2,3,4)))
                    for i in range(5)])

    # Build and populate
    tsg = TSG(name="test")
    keys = ['4', '41', '5', '51', '50']
    keys_ordered = sorted(keys, key=int)

    for k in keys:
        tsg.new_container(k, data=np.ones((13,)) * int(k),
                          attrs=dict(dims=('some_dim_name',)))

    assert len(tsg) == 5

    # Test dimensions
    assert tsg.dims == ('time',)

    # Test coordinates
    coords = tsg.coords
    print("Coordinates: ", coords)

    assert len(coords) == 1
    assert list(coords.keys()) == ['time']
    assert tsg.coords['time'] == [int(k) for k in keys_ordered]

    # Test selection methods
    # By value
    for c, k in ((4, '4'), (41, '41'), (51, '51'), (5, '5'), (50, '50')):
        assert tsg.sel(time=c) is tsg[k]

    # By index
    for i, k in ((4, '51'), (0, '4'), (-1, '51'), (2, 41), (0, 4)):
        assert tsg.isel(time=i) is tsg[k]

    # Can also select multiple time labels or indices
    t550 = tsg.sel(time=[5, 50])
    assert 'time' in t550.dims
    assert t550.sizes == dict(time=2, some_dim_name=13)
    assert t550.shape == (2, 13)
    assert (t550.coords['time'] == [5, 50]).all()

    ti13 = tsg.isel(time=[1, 3])
    assert 'time' in ti13.dims
    assert ti13.sizes == dict(time=2, some_dim_name=13)
    assert ti13.shape == (2, 13)
    assert (ti13.coords['time'] == [5, 50]).all()

    # Check that the selected data is correct
    assert (t550 == ti13).all()
    assert (t550.sel(time=5) == 5).all()
    assert (t550.sel(time=50) == 50).all()

def test_HeterogeneousTimeSeriesGroup_continuous():
    """Test the HeterogeneousTimeSeriesGroup for continous data"""
    itsg = HeterogeneousTimeSeriesGroup(name="test")

    # Construct some data with varying time and ID coordinates and varying
    # sizes of the data contained in each group
    coord_attrs = dict(dims=('time', 'id'), coords__id=[0, 2, 3, 4, 10])
    dt = np.dtype('int8')

    itsg.new_container('0', data=np.ones((3, 5), dtype=dt) * 0,
                       attrs=dict(coords__time=[0, 1, 2],
                                  **coord_attrs))

    itsg.new_container('3', data=np.ones((5, 5), dtype=dt) * 3,
                       attrs=dict(coords__time=[3, 4, 5, 6, 7],
                                  **coord_attrs))

    itsg.new_container('8', data=np.ones((2, 5), dtype=dt) * 8,
                       attrs=dict(coords__time=[8, 9],
                                  **coord_attrs))

    itsg.new_container('10', data=np.ones((5, 5), dtype=dt) * 10,
                       attrs=dict(coords__time=[10, 12, 14, 16, 18],
                                  **coord_attrs))

    print(itsg.tree)

    # Expected data (integer!)
    #             id ->  0   2   3   4  10     time
    expctd = np.array([[ 0,  0,  0,  0,  0], #   0
                       [ 0,  0,  0,  0,  0], #   1
                       [ 0,  0,  0,  0,  0], #   2
                       [ 3,  3,  3,  3,  3], #   3
                       [ 3,  3,  3,  3,  3], #   4
                       [ 3,  3,  3,  3,  3], #   5
                       [ 3,  3,  3,  3,  3], #   6
                       [ 3,  3,  3,  3,  3], #   7
                       [ 8,  8,  8,  8,  8], #   8
                       [ 8,  8,  8,  8,  8], #   9
                       [10, 10, 10, 10, 10], #  10
                       [10, 10, 10, 10, 10], #  12
                       [10, 10, 10, 10, 10], #  14
                       [10, 10, 10, 10, 10], #  16
                       [10, 10, 10, 10, 10]]) # 18

    # Now, select all data and compare to the selected one
    all_data = itsg.sel()

    # Should be a large, NaN-including array now
    assert all_data.dims == ('time', 'id')
    assert all_data.shape == (15, 5)
    assert all_data.dtype == 'int8'
    np.testing.assert_equal(all_data, expctd)

    # Check the coordinates
    assert (   all_data.coords['time'].values
            == list(range(10)) + list(range(10, 20, 2))).all()
    assert (all_data.coords['id'].values == [0, 2, 3, 4, 10]).all()

    # Select a single time step
    time18 = itsg.sel(time=18)

    # How about selecting it by index?
    time_last = itsg.isel(time=-1)
    assert (time18 == time_last).all()

    # Select multiple and compare
    teens = itsg.sel(time=slice(13, 19))  # pandas indexing, including last
    last_three = itsg.isel(time=slice(-3, -1))  # pandas indexing again
    assert (teens == last_three).all()

    # Ok, how about doing some deep selections
    id2 = itsg.sel(id=2)
    assert id2.dtype == 'int8'
    assert id2.sizes == dict(time=15)
    assert (id2 == itsg.isel(id=1)).all()

    # Can also select multiple
    assert dict(itsg.sel(id=[2, 10]).sizes) == dict(time=15, id=2)
    assert dict(itsg.isel(id=[1, 3]).sizes) == dict(time=15, id=2)

    # Select a single value
    item_via_sel = itsg.sel(time=16, id=10)
    print(item_via_sel)

    item_via_isel = itsg.isel(time=-2, id=-1)
    print(item_via_isel)

    assert (item_via_isel == item_via_sel).all()

    # With deep selection disabled, an error is raised
    itsg.allow_deep_selection = False
    with pytest.raises(ValueError, match="Deep indexing is not allowed for"):
        itsg.sel(id=2)


def test_HeterogeneousTimeSeriesGroup_discontinuous():
    """Test the HeterogeneousTimeSeriesGroup, i.e. where the data and the
    """
    itsg = HeterogeneousTimeSeriesGroup(name="test")

    # Construct some data with varying time and ID coordinates and varying
    # sizes of the data contained in each group
    coord_attrs = dict(dims=('time', 'idx'))
    dt = np.dtype('int8')

    itsg.new_container('0', data=np.ones((3, 5), dtype=dt) * 0,
                       attrs=dict(coords__time=[0, 1, 2],
                                  coords__idx=[1, 2, 3, 4, 5],
                                  **coord_attrs))

    itsg.new_container('3', data=np.ones((5, 4), dtype=dt) * 3,
                       attrs=dict(coords__time=[3, 4, 5, 6, 7],
                                  coords__idx=[1, 3, 4, 5],
                                  **coord_attrs))

    itsg.new_container('8', data=np.ones((2, 1), dtype=dt) * 8,
                       attrs=dict(coords__time=[8, 9],
                                  coords__idx=[6],
                                  **coord_attrs))

    itsg.new_container('10', data=np.ones((5, 4), dtype=dt) * 10,
                       attrs=dict(coords__time=[10, 12, 14, 16, 18],
                                  coords__idx=[0, 8, 7, 9],
                                  **coord_attrs))

    print(itsg.tree)

    # Expected data
    nan = np.nan
    #            idx -> 0     1    2    3    4    5   6    7    8    9   time v
    expctd = np.array([[nan,  0.,  0.,  0.,  0.,  0., nan, nan, nan, nan], #  0
                       [nan,  0.,  0.,  0.,  0.,  0., nan, nan, nan, nan], #  1
                       [nan,  0.,  0.,  0.,  0.,  0., nan, nan, nan, nan], #  2
                       [nan,  3., nan,  3.,  3.,  3., nan, nan, nan, nan], #  3
                       [nan,  3., nan,  3.,  3.,  3., nan, nan, nan, nan], #  4
                       [nan,  3., nan,  3.,  3.,  3., nan, nan, nan, nan], #  5
                       [nan,  3., nan,  3.,  3.,  3., nan, nan, nan, nan], #  6
                       [nan,  3., nan,  3.,  3.,  3., nan, nan, nan, nan], #  7
                       [nan, nan, nan, nan, nan, nan,  8., nan, nan, nan], #  8
                       [nan, nan, nan, nan, nan, nan,  8., nan, nan, nan], #  9
                       [10., nan, nan, nan, nan, nan, nan, 10., 10., 10.], # 10
                       [10., nan, nan, nan, nan, nan, nan, 10., 10., 10.], # 12
                       [10., nan, nan, nan, nan, nan, nan, 10., 10., 10.], # 14
                       [10., nan, nan, nan, nan, nan, nan, 10., 10., 10.], # 16
                       [10., nan, nan, nan, nan, nan, nan, 10., 10., 10.]]) #18

    # Now, select all data and compare to the selected one
    all_data = itsg.sel()

    # Should be a large, NaN-including array now
    assert all_data.dims == ('time', 'idx')
    assert all_data.shape == (15, 10)
    assert all_data.dtype == 'float64'
    np.testing.assert_equal(all_data, expctd)

    # Check the coordinates
    assert (   all_data.coords['time'].values
            == list(range(10)) + list(range(10, 20, 2))).all()
    assert (all_data.coords['idx'].values == list(range(10))).all()
