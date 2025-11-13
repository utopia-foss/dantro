"""Test the LabelledDataGroup and specializations of it, e.g. TimeSeriesGroup"""

from itertools import product

import numpy as np
import pytest

from dantro.containers import XrDataContainer
from dantro.exceptions import *
from dantro.groups import (
    HeterogeneousTimeSeriesGroup,
    LabelledDataGroup,
    TimeSeriesGroup,
)

# -----------------------------------------------------------------------------


def test_LabelledDataGroup_basics():
    """Test the basics of the LabelledDataGroup, e.g. reading the coordinates,
    building the member map, basic selection calls, ...
    """
    # Specify some names and data, intentionally unordered
    names_and_data = [
        (f"{f};{b}", np.ones((2, 2)) * b + f)
        for f, b in product((0.1, 0.2, 0.5, 0.3), (2, 8, 4))
    ]

    # Test initialization
    ldg = LabelledDataGroup(name="test", dims=("foo", "bar"), mode="name")

    # Add some members (in 2D space)
    for name, data in names_and_data:
        ldg.new_container(
            name, data=data, attrs=dict(dims=("subfoo", "subbar"))
        )

    print(ldg.tree)

    # It's also possible to directly add members during initialization
    class LDGnames(LabelledDataGroup):
        LDG_EXTRACT_COORDS_FROM = "name"

    LDGnames(
        name="test2",
        dims=("foo", "bar"),
        containers=[
            XrDataContainer(name=name, data=data)
            for name, data in names_and_data
        ],
    )

    # There was no need to calculate the member map so far
    assert not ldg.member_map_available

    # Access some group properties; for that the member map is calculated
    dims = ldg.dims
    ndim = ldg.ndim
    coords = ldg.coords
    shape = ldg.shape

    assert ldg.member_map_available
    assert dims == ("foo", "bar")
    assert ndim == 2
    assert shape == (4, 3)
    assert len(coords) == 2
    np.testing.assert_equal(coords["foo"].values, [0.1, 0.2, 0.3, 0.5])
    np.testing.assert_equal(coords["bar"].values, [2, 4, 8])  # sorted!

    # Result should be the same when reading coordiantes from attributes
    ldga = LabelledDataGroup(name="test", dims=("foo", "bar"), mode="attrs")

    # Add some members (in 2D space), with names intentionally unordered
    for f, b in product((0.1, 0.2, 0.5, 0.3), (2, 8, 4)):
        ldga.new_container(
            f"{f};{b}",
            data=np.ones((2, 2)) * b + f,
            attrs=dict(
                dims=("subfoo", "subbar"),
                ext_coords__foo=[f],
                ext_coords__bar=[b],
            ),
        )

    print(ldga.tree)

    assert (ldga.member_map == ldg.member_map).all()

    # Should be able to perform selection with both
    assert (ldg.sel(bar=[2, 8]) == ldga.isel(bar=[0, 2])).all()

    # ... and even do this via concatenation
    assert (
        ldg.sel(bar=[2, 8], combination_method="concat")
        == ldg.sel(bar=[2, 8], combination_method="merge")
    ).all()

    # When a new container is added, the member map _can_ be kept
    assert ldga.member_map_available
    ldga.new_container(
        f"new_cont",
        data=np.zeros((2, 2)),
        attrs=dict(ext_coords__foo=[0.1], ext_coords__bar=[2]),
    )
    print(ldga.tree)
    assert ldga.member_map_available
    assert "new_cont" in ldga.member_map
    assert "0.1;2" not in ldga.member_map

    # ... but for containers for which no coordinates are available, it cannot:
    assert ldga.member_map_available
    ldga.new_container(
        f"outside",
        data=np.zeros((2, 2)),
        attrs=dict(ext_coords__foo=[0.0], ext_coords__bar=[0]),
    )
    print(ldga.tree)
    assert not ldga.member_map_available

    # This can now no longer be selected via concatenation
    with pytest.raises(ItemAccessError, match="change the combination method"):
        ldga.sel(bar=[0, 2], combination_method="concat")

    # ... but via merge
    ldga.sel(bar=0, combination_method="merge")

    # Deep selection control
    ldg = LabelledDataGroup(name="test", allow_deep_selection=True)
    assert ldg.allow_deep_selection
    ldg.allow_deep_selection = False
    assert not ldg.allow_deep_selection

    # There is no dimension associated, so every selection is deep
    with pytest.raises(ValueError, match="Deep indexing is not allowed for"):
        ldg.sel(spam=23)

    # Invalid combination method argument
    with pytest.raises(ValueError, match="Invalid combination_method argumen"):
        ldga.sel(bar=[0, 2], combination_method="foo")


def test_LabelledDataGroup_selection():
    """Tests the selection interface in detail, i.e., deep selection,
    scalar selection, overlapping dimensions, drop argument, ...
    """
    # Test selection for LabelledDataGroup with mode="name".
    # Construct some data with subdimensions that partly overlap with the group
    # dimensions, also containing scalar coordinates.
    ldg = LabelledDataGroup(name="test", dims=("foo", "bar"), mode="name")

    ldg.new_container(
        "0;10",
        data=np.ones((3, 2)) * 0,
        dims=("foo", "baz"),
        coords=dict(foo=[0, 2, 4], bar="bad", baz=[10, 11]),
    )

    ldg.new_container(
        "2;12",
        data=np.ones((3, 3)) * 2,
        dims=("foo", "baz"),
        coords=dict(foo=[0, 2, 4], bar=12.0 + 1e-16, baz=[10, 11, 12]),
    )

    ldg.new_container(
        "4;12",
        data=np.ones((3, 4)) * 4,
        dims=("foo", "baz"),
        coords=dict(foo=[0, 2, 4], bar=12, baz=[10, 11, 12, 13]),
    )

    ldg.new_container(
        "4;14",
        data=np.ones((3, 4)) * 4,
        dims=("foo", "baz"),
        coords=dict(foo=[0, 2, 4], bar=14, baz=[10, 11, 12, 13]),
    )

    print(ldg.tree)

    # Test the case where a single container is selected on the group level,
    # and thus no recombination is needed (i.e. `_combine` is not called).
    # First, without deep selection
    for cont in [ldg.isel(foo=1, bar=1), ldg.sel(foo=2, bar=12)]:
        assert cont.sizes == dict(baz=3)
        assert cont.coords["foo"] == 2
        assert cont.coords["bar"] == 12
        np.testing.assert_equal(cont.values, np.array([2, 2, 2]))

    # # Now, with selection along the deep `baz` dimension
    for cont in [
        ldg.isel(foo=1, bar=1, baz=-1),
        ldg.sel(foo=2, bar=12, baz=12),
    ]:
        assert cont.ndim == 0
        assert cont.coords["foo"] == 2
        assert cont.coords["bar"] == 12
        assert cont.coords["baz"] == 12
        np.testing.assert_equal(cont.values, 2)

    # Test a case where concatenation fails due to missing entries in the
    # member map, thus falling back to `merge`.
    ldg.sel(foo=[2, 4], baz=11, combination_method="try_concat")

    # Test the selection of single coordinates together with the drop argument.
    # In particular, test selection via `.sel(foo=bar)` vs. `.sel(foo=[bar])`.
    # The former yields a non-dimension `foo` coordinate (without dimension),
    # the latter preserves the `foo` dimension (of size 1 after selection).
    # First, with drop=False...
    drop = dict(drop=False)

    # ...passing scalars:
    for cont in [
        ldg.isel(foo=-1, baz=-1, **drop),
        ldg.sel(foo=4, baz=13, **drop),
    ]:
        assert cont.dims == ("bar",)
        assert cont.coords["foo"] == 4
        assert cont.coords["baz"] == 13
        np.testing.assert_equal(cont.coords["bar"].values, [12, 14])
        np.testing.assert_equal(cont.values, [4, 4])

    # ...passing lists of size 1:
    for cont in [
        ldg.isel(foo=[1], baz=[-1], **drop),
        ldg.sel(foo=[2], baz=[12], **drop),
    ]:
        assert cont.sizes == dict(foo=1, bar=1, baz=1)
        assert cont.coords["foo"] == 2
        assert cont.coords["bar"] == 12
        assert cont.coords["baz"] == 12
        np.testing.assert_equal(cont.values, [[[2]]])

    # ...mixing both:
    for cont in [
        ldg.isel(foo=[-1], baz=-1, **drop),
        ldg.sel(foo=[4], baz=13, **drop),
    ]:
        assert cont.sizes == dict(foo=1, bar=2)
        assert cont.coords["foo"] == 4
        assert cont.coords["baz"] == 13
        np.testing.assert_equal(cont.coords["bar"].values, [12, 14])
        np.testing.assert_equal(cont.values, [[4, 4]])

    # Now, with drop=True...
    drop = dict(drop=True)

    # ...passing scalars:
    for cont in [
        ldg.isel(foo=-1, baz=-1, **drop),
        ldg.sel(foo=4, baz=13, **drop),
    ]:
        assert cont.dims == ("bar",)
        assert "foo" not in cont.coords
        assert "baz" not in cont.coords
        np.testing.assert_equal(cont.coords["bar"].values, [12, 14])
        np.testing.assert_equal(cont.values, [4, 4])

    # ...passing lists of size 1:
    for cont in [
        ldg.isel(foo=[1], baz=[-1], **drop),
        ldg.sel(foo=[2], baz=[12], **drop),
    ]:
        assert cont.sizes == dict(foo=1, bar=1, baz=1)
        assert cont.coords["foo"] == 2
        assert cont.coords["bar"] == 12
        assert cont.coords["baz"] == 12
        np.testing.assert_equal(cont.values, [[[2]]])

    # ...mixing both:
    for cont in [
        ldg.isel(foo=[-1], baz=-1, **drop),
        ldg.sel(foo=[4], baz=13, **drop),
    ]:
        assert cont.sizes == dict(foo=1, bar=2)
        assert cont.coords["foo"] == 4
        assert "baz" not in cont.coords
        np.testing.assert_equal(cont.coords["bar"].values, [12, 14])
        np.testing.assert_equal(cont.values, [[4, 4]])

    # Test error being raised on conflicting scalar coordinates.
    # The container '0;10' contains a scalar 'bar' coordinate that does not
    # match the group-level coordinate '10'.
    with pytest.raises(ValueError, match="Conflicting non-dimension coord"):
        ldg.sel(foo=0)

    # Test basic selection for data with missing coordinate information
    ldgm = LabelledDataGroup(name="test", dims=["dim"])
    ldgm.new_container(
        "foo",
        data=[[["11", "12"], ["21", "22"], ["31", "32"]]],
        dims=["dim", "subdim1", "subdim2"],
        coords=dict(subdim2=["coord1", "coord2"]),
    )

    selctd = ldgm.sel(subdim2=["coord2"])

    assert selctd.sizes == dict(dim=1, subdim1=3, subdim2=1)
    np.testing.assert_equal(selctd.values, [[["12"], ["22"], ["32"]]])

    selctd = ldgm.isel(dim=[0], subdim1=-1, drop=True)

    assert selctd.sizes == dict(dim=1, subdim2=2)
    np.testing.assert_equal(selctd.values, [["31", "32"]])
    assert "subdim1" not in selctd.coords

    # Test "all"-selection
    # In this case the *fast* all-selection (via merging all group members)
    # should work.
    ldg = LabelledDataGroup(name="test", dims=("foo", "bar"), mode="name")

    names_and_data = [
        (f"{f};{b}", np.ones((2, 2)) * b + f)
        for f, b in product((0.1, 0.2, 0.5, 0.3), (2, 8, 4))
    ]

    for name, data in names_and_data:
        ldg.new_container(
            name, data=data, attrs=dict(dims=("subfoo", "subbar"))
        )

    print(ldg.tree)

    sel_all = ldg.sel(combination_method="merge")

    assert sel_all.ndim == 4
    assert sel_all.size == 48
    assert sel_all.sizes == dict(foo=4, bar=3, subfoo=2, subbar=2)

    # Here, fall back to the (costly) ALL-selection via the member map. This is
    # because the containers have overlapping coordinates but different data.
    # When selecting via the member map only the most recently added container
    # is taken into account, while merging the containers directly fails.
    ldg = LabelledDataGroup(name="test", dims=("time", "foo"))

    coord_attrs = dict(
        dims=("time", "foo", "id"), coords__id=[0, 2, 4, 6], coords__foo=[42]
    )

    ldg.new_container(
        "foo",
        data=np.ones((3, 1, 4)) * 1,
        attrs=dict(coords__time=[0, 1, 2], **coord_attrs),
    )

    ldg.new_container(
        "bar",
        data=np.ones((3, 1, 4)) * 3,
        attrs=dict(coords__time=[0, 1, 2], **coord_attrs),
    )

    print(ldg.tree)

    sel_all = ldg.sel(combination_method="merge")

    assert sel_all.ndim == 3
    assert sel_all.size == 12
    assert sel_all.sizes == dict(time=3, id=4, foo=1)


def test_TimeSeriesGroup():
    """Test the TimeSeriesGroup, a specialization of LabelledDataGroup"""
    # Can initialize it with containers
    TimeSeriesGroup(
        name="with_containers",
        containers=[
            XrDataContainer(name=str(i), data=np.zeros((2, 3, 4)))
            for i in range(5)
        ],
    )

    # Build and populate
    tsg = TimeSeriesGroup(name="test")
    keys = ["4", "41", "5", "51", "50"]
    keys_ordered = sorted(keys, key=int)

    for k in keys:
        tsg.new_container(
            k,
            data=np.ones((13,)) * int(k),
            attrs=dict(dims=("some_dim_name",)),
        )

    assert len(tsg) == 5

    # Test dimensions
    assert tsg.dims == ("time",)

    # Test coordinates
    assert len(tsg.coords) == 1
    assert list(tsg.coords.keys()) == ["time"]
    assert (tsg.coords["time"] == [int(k) for k in keys_ordered]).all()

    # Test selection methods
    # By value
    for c, k in ((4, "4"), (41, "41"), (51, "51"), (5, "5"), (50, "50")):
        # pass value as scalar
        cont = tsg.sel(time=c)
        assert (cont.values == tsg[k].values).all()
        assert cont.dims == tsg[k].dims
        assert len(cont.coords) == len(tsg[k].coords) + 1
        assert "time" in cont.coords
        assert all([cont.coords[d] == tsg[k].coords[d] for d in tsg[k].coords])

        # pass value as list, thus keeping the time dimension
        cont = tsg.sel(time=[c])
        assert (cont.values == tsg[k].values).all()
        assert cont.ndim == tsg[k].ndim + 1
        assert "time" in cont.dims
        assert len(cont.coords) == len(tsg[k].coords) + 1
        assert "time" in cont.coords
        assert all([cont.coords[d] == tsg[k].coords[d] for d in tsg[k].coords])

    # By index
    for i, k in ((4, "51"), (0, "4"), (-1, "51"), (2, 41), (0, 4)):
        # pass index as scalar
        cont = tsg.isel(time=i)
        assert (cont.values == tsg[k].values).all()
        assert cont.dims == tsg[k].dims
        assert len(cont.coords) == len(tsg[k].coords) + 1
        assert "time" in cont.coords
        assert all([cont.coords[d] == tsg[k].coords[d] for d in tsg[k].coords])

        # pass index as list, thus keeping the time dimension
        cont = tsg.isel(time=[i])
        assert (cont.values == tsg[k].values).all()
        assert cont.ndim == tsg[k].ndim + 1
        assert "time" in cont.dims
        assert len(cont.coords) == len(tsg[k].coords) + 1
        assert "time" in cont.coords
        assert all([cont.coords[d] == tsg[k].coords[d] for d in tsg[k].coords])

    # Can also select multiple time labels or indices
    t550 = tsg.sel(time=[5, 50])
    assert "time" in t550.dims
    assert t550.sizes == dict(time=2, some_dim_name=13)
    assert t550.shape == (2, 13)
    assert (t550.coords["time"] == [5, 50]).all()

    ti13 = tsg.isel(time=[1, 3])
    assert "time" in ti13.dims
    assert ti13.sizes == dict(time=2, some_dim_name=13)
    assert ti13.shape == (2, 13)
    assert (ti13.coords["time"] == [5, 50]).all()

    # Check that the selected data is correct
    assert (t550 == ti13).all()
    assert (t550.sel(time=5) == 5).all()
    assert (t550.sel(time=50) == 50).all()


def test_HeterogeneousTimeSeriesGroup_continuous():
    """Test the HeterogeneousTimeSeriesGroup for continous data"""
    itsg = HeterogeneousTimeSeriesGroup(name="test")

    # Construct some data with varying time and ID coordinates and varying
    # sizes of the data contained in each group
    coord_attrs = dict(dims=("time", "id"), coords__id=[0, 2, 3, 4, 10])
    dt = np.dtype("int8")

    itsg.new_container(
        "0",
        data=np.ones((3, 5), dtype=dt) * 0,
        attrs=dict(coords__time=[0, 1, 2], **coord_attrs),
    )

    itsg.new_container(
        "3",
        data=np.ones((5, 5), dtype=dt) * 3,
        attrs=dict(coords__time=[3, 4, 5, 6, 7], **coord_attrs),
    )

    itsg.new_container(
        "8",
        data=np.ones((2, 5), dtype=dt) * 8,
        attrs=dict(coords__time=[8, 9], **coord_attrs),
    )

    itsg.new_container(
        "10",
        data=np.ones((5, 5), dtype=dt) * 10,
        attrs=dict(coords__time=[10, 12, 14, 16, 18], **coord_attrs),
    )

    print(itsg.tree)

    # Expected data (integer!)
    # id -> 0   2   3   4  10     time
    expctd = np.array(
        [
            [0, 0, 0, 0, 0],  #   0
            [0, 0, 0, 0, 0],  #   1
            [0, 0, 0, 0, 0],  #   2
            [3, 3, 3, 3, 3],  #   3
            [3, 3, 3, 3, 3],  #   4
            [3, 3, 3, 3, 3],  #   5
            [3, 3, 3, 3, 3],  #   6
            [3, 3, 3, 3, 3],  #   7
            [8, 8, 8, 8, 8],  #   8
            [8, 8, 8, 8, 8],  #   9
            [10, 10, 10, 10, 10],  #  10
            [10, 10, 10, 10, 10],  #  12
            [10, 10, 10, 10, 10],  #  14
            [10, 10, 10, 10, 10],  #  16
            [10, 10, 10, 10, 10],  #  18
        ]
    )

    # Now, select all data and compare to the selected one
    all_data = itsg.sel(combination_method="auto")

    # Should be a large, NaN-including array now
    assert all_data.dims == ("time", "id")
    assert all_data.shape == (15, 5)
    assert all_data.dtype == "int8"
    np.testing.assert_equal(all_data, expctd)

    # Check the coordinates
    assert (
        all_data.coords["time"].values
        == list(range(10)) + list(range(10, 20, 2))
    ).all()
    assert (all_data.coords["id"].values == [0, 2, 3, 4, 10]).all()

    # Select a single time step...
    # ...with drop=False
    time18 = itsg.sel(time=18, drop=False)

    assert time18.dims == ("id",)
    assert time18.coords["time"].values == 18
    assert time18.shape == (5,)
    np.testing.assert_equal(time18, expctd[-1])

    # ...and with drop=True
    time18_drop = itsg.sel(time=18, drop=True)

    assert time18_drop.dims == ("id",)
    assert "time" not in time18_drop.coords
    assert time18_drop.shape == (5,)
    np.testing.assert_equal(time18_drop, expctd[-1])

    # How about selecting it by index?...
    # ...with drop=False
    time_last = itsg.isel(time=-1)

    assert time_last.dims == ("id",)
    assert time_last.coords["time"].values == 18
    assert time_last.shape == (5,)
    np.testing.assert_equal(time_last, expctd[-1])
    assert (time18 == time_last).all()

    # ...and with drop=True
    time_last_drop = itsg.sel(time=18, drop=True)

    assert time_last_drop.dims == ("id",)
    assert "time" not in time_last_drop.coords
    assert time_last_drop.shape == (5,)
    np.testing.assert_equal(time_last_drop, expctd[-1])

    # Select multiple and compare
    # pandas indexing, including last when value-based
    teens = itsg.sel(time=slice(13, 18))

    assert teens.dims == ("time", "id")
    assert teens.shape == (3, 5)
    np.testing.assert_equal(teens, expctd[-3:])

    # pandas indexing again, not including last when index-based
    last_three = itsg.isel(time=slice(-3, None))

    assert last_three.dims == ("time", "id")
    assert last_three.shape == (3, 5)
    np.testing.assert_equal(teens, expctd[-3:])
    assert (teens == last_three).all()

    # Ok, how about doing some deep selections...
    # ...with scalar selection
    id2 = itsg.sel(id=2)
    assert id2.dtype == "int8"
    assert id2.sizes == dict(time=15)
    assert id2.coords["id"].values == 2
    assert (id2 == itsg.isel(id=1)).all()

    # ...preserving the dimension
    id2_dim = itsg.sel(id=[2])
    assert id2_dim.dtype == "int8"
    assert id2_dim.sizes == dict(time=15, id=1)
    assert id2_dim.coords["id"].values == 2
    assert (id2_dim == itsg.isel(id=[1])).all()

    # ...with scalar selection and drop=True
    id2_drop = itsg.sel(id=2, drop=True)
    assert id2_drop.dtype == "int8"
    assert id2_drop.sizes == dict(time=15)
    assert "id" not in id2_drop.coords
    assert (id2_drop == itsg.isel(id=1, drop=True)).all()

    # Can also select multiple
    assert dict(itsg.sel(id=[2, 10]).sizes) == dict(time=15, id=2)
    assert dict(itsg.isel(id=[1, 3]).sizes) == dict(time=15, id=2)

    # Select a single value
    item_via_sel = itsg.sel(time=16, id=10)
    item_via_isel = itsg.isel(time=-2, id=-1)
    assert (item_via_isel == item_via_sel).all()

    # With deep selection disabled, an error is raised
    itsg.allow_deep_selection = False
    with pytest.raises(ValueError, match="Deep indexing is not allowed for"):
        itsg.sel(id=2)


def test_HeterogeneousTimeSeriesGroup_discontinuous():
    """Test the HeterogeneousTimeSeriesGroup for discontinuous data."""
    itsg = HeterogeneousTimeSeriesGroup(name="test")

    # Construct some data with varying time and ID coordinates and varying
    # sizes of the data contained in each group
    coord_attrs = dict(dims=("time", "idx"))
    dt = np.dtype("int8")

    itsg.new_container(
        "0",
        data=np.ones((3, 5), dtype=dt) * 0,
        attrs=dict(
            coords__time=[0, 1, 2], coords__idx=[1, 2, 3, 4, 5], **coord_attrs
        ),
    )

    itsg.new_container(
        "3",
        data=np.ones((5, 4), dtype=dt) * 3,
        attrs=dict(
            coords__time=[3, 4, 5, 6, 7],
            coords__idx=[1, 3, 4, 5],
            **coord_attrs,
        ),
    )

    itsg.new_container(
        "8",
        data=np.ones((2, 1), dtype=dt) * 8,
        attrs=dict(coords__time=[8, 9], coords__idx=[6], **coord_attrs),
    )

    itsg.new_container(
        "10",
        data=np.ones((5, 4), dtype=dt) * 10,
        attrs=dict(
            coords__time=[10, 12, 14, 16, 18],
            coords__idx=[0, 8, 7, 9],
            **coord_attrs,
        ),
    )

    print(itsg.tree)

    # Check the coordinates
    assert (
        itsg.coords["time"].values == list(range(10)) + list(range(10, 20, 2))
    ).all()

    # Expected data
    nan = np.nan
    # idx -> 0     1    2    3    4    5   6    7    8    9   time v
    expctd = np.array(
        [
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],  #  0
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],  #  1
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],  #  2
            [nan, 3.0, nan, 3.0, 3.0, 3.0, nan, nan, nan, nan],  #  3
            [nan, 3.0, nan, 3.0, 3.0, 3.0, nan, nan, nan, nan],  #  4
            [nan, 3.0, nan, 3.0, 3.0, 3.0, nan, nan, nan, nan],  #  5
            [nan, 3.0, nan, 3.0, 3.0, 3.0, nan, nan, nan, nan],  #  6
            [nan, 3.0, nan, 3.0, 3.0, 3.0, nan, nan, nan, nan],  #  7
            [nan, nan, nan, nan, nan, nan, 8.0, nan, nan, nan],  #  8
            [nan, nan, nan, nan, nan, nan, 8.0, nan, nan, nan],  #  9
            [10.0, nan, nan, nan, nan, nan, nan, 10.0, 10.0, 10.0],  # 10
            [10.0, nan, nan, nan, nan, nan, nan, 10.0, 10.0, 10.0],  # 12
            [10.0, nan, nan, nan, nan, nan, nan, 10.0, 10.0, 10.0],  # 14
            [10.0, nan, nan, nan, nan, nan, nan, 10.0, 10.0, 10.0],  # 16
            [10.0, nan, nan, nan, nan, nan, nan, 10.0, 10.0, 10.0],  # 18
        ]
    )

    # Try some deep and shallow selection
    deep = itsg.isel(idx=0)
    assert deep.dims == ("time",)
    assert len(deep.coords["idx"]) == deep.sizes["time"]

    shallow = itsg.sel(time=[8, 10])
    assert shallow.dims == ("time", "idx")

    # Now, select all data and compare to the selected one
    all_data = itsg.sel()

    # Should be a large, NaN-including array now
    assert all_data.dims == ("time", "idx")
    assert all_data.shape == (15, 10)
    assert all_data.dtype in ("float32", "float64")
    np.testing.assert_equal(all_data, expctd)

    # Check the coordinates
    assert (
        all_data.coords["time"].values
        == list(range(10)) + list(range(10, 20, 2))
    ).all()
    assert (all_data.coords["idx"].values == list(range(10))).all()
