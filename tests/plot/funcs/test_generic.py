"""Tests the generic plotting functions, mostly the facet_grid method and the
related functions and decorators.
"""

import logging
import os
from builtins import *  # to have Exception types available in globals
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from dantro._import_tools import get_resource_path
from dantro.containers import PassthroughContainer, XrDataContainer
from dantro.exceptions import *
from dantro.plot import PlotHelper, PyPlotCreator
from dantro.plot.funcs._utils import plot_errorbar
from dantro.plot.funcs.generic import (
    _FACET_GRID_FUNCS,
    _FACET_GRID_KINDS,
    determine_encoding,
    determine_plot_kind,
    facet_grid,
    make_facet_grid_plot,
    map_dims_to_encoding,
)
from dantro.tools import DoNothingContext, load_yml

# Local variables and configuration ...........................................
# If True, runs all tests. If False, runs only the basics (much faster)
# TODO Make this controllable via environment variables
FULL_TEST = False
skip_if_not_full = pytest.mark.skipif(
    not FULL_TEST, reason="Will only run with FULL_TEST."
)

# Test configurations
PLOTS_CFG_EBAR = load_yml(get_resource_path("tests", "cfg/plots_ebar.yml"))
PLOTS_CFG_FG = load_yml(get_resource_path("tests", "cfg/plots_facet_grid.yml"))

# Disable matplotlib logger (much too verbose)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# -- Helpers ------------------------------------------------------------------


def create_nd_data(
    n: int,
    *,
    shape=None,
    with_coords: bool = False,
    extra_coords: dict = None,
    size_offset: int = 1,
    **da_kwargs,
) -> xr.DataArray:
    """Creates n-dimensional random data of a certain shape. If no shape is
    given, will use ``(2, 3, 4, ..)``. Can also add coords to the data.
    """

    def gen_coords(i: int, s: int, step: int = 1) -> list:
        return range((i + 10 * i), (i + 10 * i) + s, step)

    if shape is None:
        shape = tuple(i + size_offset for i in range(1, n + 1))

    dims = tuple(f"dim_{i}" for i, _ in enumerate(shape))

    # We want to control whether all or no dimensions have coordinates
    dims_iter = enumerate(zip(dims, shape))
    if with_coords is True:
        # coordinates for all dimensions --> all dims are indexed
        coords = {dim: gen_coords(i, s) for i, (dim, s) in dims_iter}

    elif with_coords is False:
        # no coordinates --> no indexed dimensions
        coords = None

    elif isinstance(with_coords, int):
        # coordinates for every second dimension --> not all dims are indexed
        coords = {
            dim: gen_coords(i, s)
            for i, (dim, s) in dims_iter
            if i % with_coords == 0
        }

    elif isinstance(with_coords, (list, tuple)):
        # coordinates for specific dimension --> none, some or all dims indexed
        coords = {
            dim: gen_coords(i, s)
            for i, (dim, s) in dims_iter
            if dim in with_coords
        }

    else:
        raise TypeError(
            f"Unexpected type {type(with_coords)} for argument `with_coords`! "
            f"Should be boolean, int, list or tuple. Value was: {with_coords}"
        )

    da = xr.DataArray(
        data=np.random.random(shape), dims=dims, coords=coords, **da_kwargs
    )

    # May want additional, non-indexed coordinates
    if extra_coords:
        da = da.assign_coords(**extra_coords)

    return da


def associate_specifiers(
    data, *, specifiers: Tuple[str] = None, exclude: tuple = None
) -> dict:
    """Associates representation specifiers with data dimension, i.e. returns a
    mapping from one of the ``specifiers`` to a ``data`` dimension name.

    Specifiers start with the last data dimension and only label the minimum
    of (number of specifiers specifiers, number of data dimensions).

    Args:
        data (TYPE): The data
        specifiers (Tuple[str], optional): The available specifiers. If None,
            will use: (x, y, row, col, hue, frames)
        exclude (tuple, optional): Which ones to exclude from ``specifiers``
    """
    if specifiers is None:
        specifiers = ("x", "y", "row", "col", "hue", "frames")

    if exclude is not None:
        specifiers = [s for s in specifiers if s not in exclude]

    if isinstance(data, (xr.DataArray, XrDataContainer)):
        dim_names = data.dims[::-1]
    else:
        # Probably a dataset or something similar
        dim_names = list(data.sizes.keys())[::-1]

    # Drop names that are not also a dimension
    dim_names = [d for d in dim_names if d in data.sizes]

    return {spec: dim_name for spec, dim_name in zip(specifiers, dim_names)}


def invoke_facet_grid(*, dm, out_dir, to_test: dict, max_num_figs: int = 1):
    """Repeatedly invokes the facet_grid plot function and checks whether it
    runs through as expected or generates an exception as expected.

    After each invocation, if the number of open figures is checked, which can
    be used to detect figure leakage.
    """
    ppc = PyPlotCreator("test_facet_grid", dm=dm, plot_func=facet_grid)
    ppc._exist_ok = True

    # Shortcuts
    animation = dict(
        enabled=False,
        writer="frames",
        writer_kwargs=dict(frames=dict(saving=(dict(dpi=36)))),
    )
    shared_kwargs = dict(animation=animation)
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Now, the generic testing, combinations of: kinds, specifier, data
    for case_name, cfg in to_test.items():
        kinds = cfg.get("kinds")
        specifiers = cfg["specifiers"]
        min_dims = cfg.get("min_dims", 0)
        max_dims = cfg["max_dims"]
        raises = cfg.get("raises", {})
        warns = cfg.get("warns", {})
        plot_kwargs = cfg.get("plot_kwargs", {})
        test_data_path = cfg.get("test_data_path", "ndim_da")

        print(
            "\n\nTesting scenario '{}' with {}…{}-dimensional data ...".format(
                case_name, min_dims, max_dims
            )
        )
        print("Kinds:          ", ", ".join(kinds) if kinds else "auto")
        print("All specifiers: ", specifiers)

        # Restrict the container iteration to the maximum dimensionality
        conts_it = [
            (name, cont)
            for name, cont in dm[test_data_path].items()
            if len(cont.sizes) in range(min_dims, max_dims + 1)
        ]
        kinds_it = kinds if kinds else [None]

        # Now, iterate over these combinations
        for kind, (cont_name, cont) in product(kinds_it, conts_it):
            aspecs = associate_specifiers(cont, specifiers=specifiers)
            print("\n... with data:    ", cont)
            print("    plot_kwargs:  ", plot_kwargs)
            print("    kind:         ", kind)
            print("    and spec map: ", aspecs)

            # Determine a context to allow to test for failing cases
            context = DoNothingContext()
            ndim = len(cont.sizes)
            if ndim in raises:
                # These are expected to fail with a specific type and message
                raise_spec = raises[ndim]
                exc_type, match = raise_spec
                print("    expct. raise: ", exc_type, f": '{match}'")
                exc_type = globals()[exc_type]
                context = pytest.raises(exc_type, match=match)
            elif ndim in warns:
                warn_spec = warns[ndim]
                warn_type, match = warn_spec
                print("    expct. warn:  ", warn_type, f": '{match}'")
                warn_type = globals()[warn_type]
                context = pytest.warns(warn_type, match=match)

            # Now, run the plot function in that context
            with context:
                ppc(
                    **out_path(
                        "{case:}__kind_{kind:}_{data:}_{specs:}".format(
                            kind=kind if kind else "None",
                            case=case_name,
                            data=cont.name,
                            specs="-".join(aspecs),
                        )
                    ),
                    **shared_kwargs,
                    **aspecs,
                    **plot_kwargs,
                    kind=kind,
                    select=dict(
                        data=dict(
                            path=f"{test_data_path}/{cont_name}",
                            transform=[".data"],
                        ),
                    ),
                )

            # Check plot figure count
            fignums = plt.get_fignums()
            print("    Plot finished as expected.")
            print("    Open figures: ", fignums)
            assert len(fignums) <= max_num_figs

        print(f"Scenario '{case_name}' succeeded.\n")
    print("All scenarios tested successfully.")


class MockDataArray:
    """A Mock class for a data array, simply with the ndim attribute."""

    def __init__(self, ndim: int):
        self.ndim = ndim


from ..._fixtures import *

# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests
from ...test_plot_mngr import dm as _dm

# Use the fixture that's controlled via global variables
out_dir = tmpdir_or_local_dir

# .. Generate test data .......................................................


@pytest.fixture
def dm(_dm):
    """Returns a data manager populated with some high-dimensional test data"""
    # Add xr.Datasets for testing
    grp_dataset = _dm.new_group("datasets")

    ds = xr.Dataset(
        dict(
            foo=xr.DataArray(np.random.rand(5, 4, 3)),
            bar=xr.DataArray(np.random.rand(5, 4, 3)),
        )
    )
    grp_dataset.add(PassthroughContainer(name="foobar3D", data=ds))

    ds = xr.Dataset(
        dict(
            mean=xr.DataArray(
                np.random.rand(5, 4, 3, 2),
                dims=("foo", "bar", "baz", "spam"),
                coords=dict(
                    foo=range(5), bar=range(4), baz=range(3), spam=range(2)
                ),
            ),
            std=xr.DataArray(
                np.random.rand(5, 4, 3, 2),
                dims=("foo", "bar", "baz", "spam"),
                coords=dict(
                    foo=range(5), bar=range(4), baz=range(3), spam=range(2)
                ),
            ),
        )
    )
    grp_dataset.add(PassthroughContainer(name="mean_and_std4D", data=ds))

    ds = xr.Dataset(
        dict(
            mean=xr.DataArray(
                np.random.rand(5, 4),
                dims=("foo", "bar"),
                coords=dict(foo=range(5), bar=range(4)),
            ),
            std=xr.DataArray(
                np.random.rand(5, 4),
                dims=("foo", "bar"),
                coords=dict(foo=range(5), bar=range(4)),
            ),
        )
    )
    grp_dataset.add(PassthroughContainer(name="mean_and_std2D", data=ds))

    ds = xr.Dataset(
        dict(
            mean=xr.DataArray(
                np.random.rand(5),
                dims=("foo",),
                coords=dict(foo=range(5)),
            ),
            std=xr.DataArray(
                np.random.rand(5),
                dims=("foo",),
                coords=dict(foo=range(5)),
            ),
        )
    )
    grp_dataset.add(PassthroughContainer(name="mean_and_std1D", data=ds))

    # Add ndim random data for DataArrays, going from 0 to 7 dimensions
    grp_ndim_da = _dm.new_group("ndim_da")
    grp_ndim_da.add(
        *[
            XrDataContainer(name=f"{n:d}D", data=create_nd_data(n))
            for n in range(1, 7)
        ]
    )

    grp_labelled = _dm.new_group("labelled")
    grp_labelled.add(
        *[
            XrDataContainer(
                name=f"{n:d}D",
                data=create_nd_data(n, with_coords=True),
            )
            for n in range(1, 7)
        ]
    )

    grp_labelled_scalar = _dm.new_group("labelled_scalar")
    grp_labelled_scalar.add(
        *[
            XrDataContainer(
                name=f"{n:d}D",
                data=create_nd_data(n, with_coords=True, size_offset=0),
            )
            for n in range(1, 7)
        ]
    )

    grp_labelled_large = _dm.new_group("labelled_large")
    grp_labelled_large.add(
        *[
            XrDataContainer(
                name=f"{n:d}D",
                data=create_nd_data(n, with_coords=True, size_offset=4),
            )
            for n in range(1, 7)
        ]
    )

    grp_labelled_alternating = _dm.new_group("labelled_alternating")
    grp_labelled_alternating.add(
        *[
            XrDataContainer(
                name=f"{n:d}D",
                data=create_nd_data(n, with_coords=2, size_offset=4),
            )
            for n in range(1, 7)
        ]
    )

    grp_labelled_extra_coords = _dm.new_group("labelled_extra_coords")
    extra_coords = dict(foo=123, bar=1.23, baz="spam")
    grp_labelled_extra_coords.add(
        *[
            XrDataContainer(
                name=f"{n:d}D",
                data=create_nd_data(
                    n, with_coords=2, extra_coords=extra_coords, size_offset=4
                ),
            )
            for n in range(1, 7)
        ]
    )

    grp_labelled_with_size_one = _dm.new_group("labelled_with_size_one")
    grp_labelled_with_size_one.add(
        *[
            XrDataContainer(
                name=f"{n+2:d}D",
                data=create_nd_data(
                    n,
                    shape=(1,) + tuple(range(3, n + 3)) + (1,),
                    extra_coords=extra_coords,
                ),
            )
            for n in range(0, 7)
        ]
    )

    grp_ds_labelled = _dm.new_group("ds_labelled")
    grp_ds_labelled.add(
        *[
            PassthroughContainer(
                name=f"{n:d}D",
                data=xr.Dataset(
                    dict(
                        foo=create_nd_data(n, with_coords=True),
                        bar=create_nd_data(n, with_coords=True),
                        baz=create_nd_data(n, with_coords=True),
                        spam=create_nd_data(n, with_coords=True),
                        fish=create_nd_data(n, with_coords=True),
                    )
                ),
            )
            for n in range(7)
        ]
    )

    grp_ds_labelled_scalar = _dm.new_group("ds_labelled_scalar")
    _ds_scalar_kws = dict(with_coords=True, size_offset=0)
    grp_ds_labelled_scalar.add(
        *[
            PassthroughContainer(
                name=f"{n:d}D",
                data=xr.Dataset(
                    dict(
                        foo=create_nd_data(n, **_ds_scalar_kws),
                        bar=create_nd_data(n, **_ds_scalar_kws),
                        baz=create_nd_data(n, **_ds_scalar_kws),
                        spam=create_nd_data(n, **_ds_scalar_kws),
                        fish=create_nd_data(n, **_ds_scalar_kws),
                    )
                ),
            )
            for n in range(7)
        ]
    )

    grp_ds_labelled_scalar_sqz = _dm.new_group("ds_labelled_scalar_squeezed")
    grp_ds_labelled_scalar_sqz.add(
        *[
            PassthroughContainer(
                name=f"{n:d}D",
                data=xr.Dataset(
                    dict(
                        foo=create_nd_data(n, **_ds_scalar_kws).squeeze(),
                        bar=create_nd_data(n, **_ds_scalar_kws).squeeze(),
                        baz=create_nd_data(n, **_ds_scalar_kws).squeeze(),
                        spam=create_nd_data(n, **_ds_scalar_kws).squeeze(),
                        fish=create_nd_data(n, **_ds_scalar_kws).squeeze(),
                    )
                ),
            )
            for n in range(7)
        ]
    )

    return _dm


@pytest.fixture
def anim_disabled() -> dict:
    """Returns a dict with default (disabled) animation kwargs"""
    return dict(
        enabled=False,
        writer="frames",
        writer_kwargs=dict(frames=dict(saving=(dict(dpi=36)))),
    )


# -- Tests --------------------------------------------------------------------

# .. Isolated helper function tests ...........................................


def test_determine_plot_kind():
    dpk = determine_plot_kind
    DA = MockDataArray

    # Not like a data array -> return dset_default_kind
    assert dpk("no_ndim", kind="auto") == "scatter"
    assert (
        dpk("no_ndim", kind="auto", default_kind_map=dict(dataset="foo"))
        == "foo"
    )

    # Like a data array -> depends on (mocked) dimensionality, fallback to hist
    assert dpk(DA(0), kind="auto") == "hist"
    assert dpk(DA(1), kind="auto") == "line"
    for i in range(2, 6):
        assert dpk(DA(i), kind="auto") == "pcolormesh"
    assert dpk(DA(6), kind="auto") == "hist"
    assert dpk(DA(23), kind="auto") == "hist"

    # With custom mapping
    custom_map = {1: "one", 2: "two", "fallback": "some_default"}
    assert dpk(DA(1), kind=custom_map) == "one"
    assert dpk(DA(2), kind=custom_map) == "two"
    assert dpk(DA(42), kind=custom_map) == "some_default"

    # With partly specified layout encoding: fixes line or pcolormesh
    for i in range(6):
        assert dpk(DA(i), kind="auto", hue="foo") == "line"
    for i in range(6):
        assert dpk(DA(i), kind="auto", x="foo", y="bar") == "pcolormesh"

    # Without a fallback or dataset, should get KeyErrors
    with pytest.raises(KeyError, match="fallback"):
        dpk(DA(123), kind=dict(foo="bar"), default_kind_map=dict())
    with pytest.raises(KeyError, match="dataset"):
        dpk("not_an_array", kind="auto", default_kind_map=dict())


def test_map_dims_to_encoding():
    # Some shortcuts to make test definitions shorter
    mp = map_dims_to_encoding
    tp = tuple
    dt = dict

    # Simple assignments
    #            encodings  dimensions      (mapped, free specs, free dims)
    assert mp(tp("ABC"), tp("abc")) == (dt(A="a", B="b", C="c"), [], [])
    assert mp(tp("ABC"), tp("cab")) == (dt(A="c", B="a", C="b"), [], [])
    assert mp(tp("A"), tp("cab")) == (dt(A="c"), [], ["a", "b"])
    assert mp(tp("ABC"), tp("c")) == (dt(A="c"), [("B", 1), ("C", 1)], [])
    assert mp(tp(), tp()) == (dt(), [], [])

    # Multi-dim encodings by naming an encoding multiple times
    assert mp(tp("ABA"), tp("abcd")) == (dt(A=tp("ac"), B="b"), [], ["d"])
    assert mp(tp("AAA"), tp("a")) == (dt(A=tp("a")), [("A", 1), ("A", 1)], [])

    # Ellipses
    assert mp(("A", "B", ("C", ...)), tp("abcdefgh")) == (
        dt(A="a", B="b", C=tp("cdefgh")),
        [],
        [],
    )

    assert mp((("A", ...), "B", "C", "D"), tp("abcdefgh")) == (
        dt(A=tp("abcde"), B="f", C="g", D="h"),
        [],
        [],
    )

    assert mp(("A", "B", "A", ("C", ...), "D"), tp("abcdefgh")) == (
        dt(A=tp("ac"), B="b", C=tp("defg"), D="h"),
        [],
        [],
    )

    assert mp(
        (("A", ...), "B", "C", "D", "E"), tp("bcde")  # zero-sized ellipsis
    ) == (dt(B="b", C="c", D="d", E="e"), [], [])

    # With partly-specified encoding
    assert mp(tp("ABC"), tp("cab"), encoding=dict(B="b")) == (
        dt(A="c", B="b", C="a"),
        [],
        [],
    )

    assert mp(tp("ABC"), tp("cab"), encoding=dict(A="a", B="b", C="c")) == (
        dt(A="a", B="b", C="c"),
        [],
        [],
    )

    assert mp(tp("ABC"), tp("abcd"), encoding=dict(B="b")) == (
        dt(A="a", B="b", C="c"),
        [],
        ["d"],
    )

    assert mp(tp("ABCD"), tp("abc"), encoding=dict(B="b")) == (
        dt(A="a", B="b", C="c"),
        [("D", 1)],
        [],
    )

    assert mp(tp("ABBC"), tp("abcd"), encoding=dict(B=["c"])) == (
        dt(A="a", B=tp("cb"), C="d"),
        [],
        [],
    )

    assert mp(tp("ABCBDE"), tp("abcdefg"), encoding=dict(B=["d"], D="g")) == (
        dt(A="a", B=tp("dc"), C="b", D="g", E="e"),
        [],
        ["f"],
    )

    # Dropping missing pre-specified dimensions
    assert mp(
        tp("ABC"),
        tp("abc"),
        encoding=dict(C="a", B="d"),
        drop_missing_dims=False,
    ) == (
        dt(A="b", B="d", C="a"),
        [],
        ["c"],
    )  # -> downstream errors

    assert mp(
        tp("ABC"),
        tp("abc"),
        encoding=dict(C="a", B="d"),
        drop_missing_dims=True,
    ) == (
        dt(A="b", B="c", C="a"),
        [],
        [],
    )  # -> unavailable d dim dropped

    assert mp(
        tp("ABBC"),
        tp("abcdef"),
        encoding=dict(C="a", B=tp("gfj"), A="k"),
        drop_missing_dims=True,
    ) == (dt(C="a", B=tp("fc"), A="b"), [], ["d", "e"])

    # Dropping encodings
    assert mp(
        tp("ABCBDE"),
        tp("abcdefg"),
        encoding=dict(B=["d"], D="g"),
        drop_encodings=tp("BC"),  # -> only ADE remain
    ) == (dt(A="a", D="g", E="b"), [], ["c", "d", "e", "f"])

    # None-valued encodings lead to specifier being blocked
    assert mp(
        tp("ABCBDE"), tp("abcdefg"), encoding=dict(B=None, D="g", E=None)
    ) == (dt(A="a", B=None, C="b", D="g", E=None), [], ["c", "d", "e", "f"])

    assert mp(
        tp("ABCBDE"), tp("abcdefg"), encoding=dict(B=[None], D="g", E=None)
    ) == (
        dt(
            A="a",
            C="b",
            B=(
                None,
                "c",
            ),
            D="g",
            E=None,
        ),
        [],
        ["d", "e", "f"],
    )

    assert mp(
        tp("ABCBDBEB"), tp("abcdefg"), encoding=dict(B=[None, None], D=None)
    ) == (
        dt(A="a", C="b", D=None, B=(None, None, "c", "e"), E="d"),
        [],
        ["f", "g"],
    )

    # Errors
    with pytest.raises(
        ValueError, match="need to be unique.* duplicate.* a, b, b, c"
    ):
        mp(tp("ABCDE"), tp("abbcde"))

    with pytest.raises(
        ValueError, match="Only one encoding can be an Ellipsis.* foo, bar"
    ):
        mp(("A", ("foo", ...), "B", ("bar", ...)), tp("abcd"))


def test_determine_encoding():
    """Test the determine_encoding helper function

    NOTE Most parts are covered by the config-based plot function tests.
    """
    # Bind some defaults for the interface tests (and make it shorter for the
    # purpose of this test)
    default_kws = dict(
        kind="line", auto_encoding=True, default_encodings=_FACET_GRID_KINDS
    )

    def detenc(d, **plot_kwargs):
        return determine_encoding(d, **default_kws, plot_kwargs=plot_kwargs)

    # Input can be a sizes dict or a sequence of dimension names. If size info
    # is available, will sort it descendingly and use that order
    kws = detenc(["foo", "bar", "baz", "spam", "fish"])
    assert kws["x"] == "foo"
    assert kws["hue"] == "bar"
    assert kws["col"] == "baz"
    assert kws["row"] == "spam"
    assert kws["frames"] == "fish"

    kws = detenc(dict(foo=10, bar=12, baz=15))
    assert kws["x"] == "baz"
    assert kws["hue"] == "bar"
    assert kws["col"] == "foo"
    assert "row" not in kws
    assert "frames" not in kws

    # Automatic column wrapping
    kws = detenc(dict(foo=10, bar=12, baz=15), col_wrap="auto")
    assert kws["col_wrap"] == 5  # from foo column, ceil(sqrt(10)) +/- optim

    kws = detenc(dict(foo=10, bar=12, baz=15), col_wrap="square")
    assert kws["col_wrap"] == 4  # from foo column, ceil(sqrt(10))

    # ... deactivated if no sizes are given
    kws = detenc(["foo", "bar", "baz"], col_wrap="auto")
    assert "col_wrap" not in kws

    # ... or if rows are assigned
    kws = detenc(["foo", "bar", "baz", "spam", "fish"], col_wrap="auto")
    assert "col_wrap" not in kws

    # Can pre-assign an encoding and that will not be touched
    kws = detenc(
        dict(foo=10, bar=12, baz=15, spam=21), frames="bar", col_wrap="auto"
    )
    assert kws["x"] == "spam"
    assert kws["hue"] == "baz"
    assert kws["col"] == "foo"
    assert kws["frames"] == "bar"
    assert kws["col_wrap"] == 5

    # Allow mappings between x and y
    kws = determine_encoding(
        dict(foo=10, bar=12, baz=15, spam=21),
        **default_kws,
        plot_kwargs=dict(y="bar"),
        allow_y_for_x=["line"],
    )
    assert kws["y"] == "bar"
    assert "x" not in kws
    assert kws["hue"] == "spam"
    assert kws["col"] == "baz"
    assert kws["row"] == "foo"

    # Can ignore missing dims
    kws = determine_encoding(
        dict(foo=10, bar=12, baz=15, spam=21),
        **default_kws,
        plot_kwargs=dict(hue="foo", col="fish"),
        drop_missing_dims=True,
    )
    assert kws["x"] == "spam"
    assert kws["hue"] == "foo"
    assert kws["col"] == "baz"
    assert kws["row"] == "bar"

    # Can have catch-all dimensions where free dimensions are mapped to
    kws = determine_encoding(
        dict(foo=10, bar=12, baz=15, spam=21, fish=25, chips=30, salt=33),
        **default_kws,
        plot_kwargs=dict(hue="bar", col="bad_dim_name"),
        drop_missing_dims=True,
        map_free_dims_to="files",
    )
    assert kws["x"] == "salt"
    assert kws["hue"] == "bar"
    assert kws["col"] == "chips"
    assert kws["row"] == "fish"
    assert kws["frames"] == "spam"
    assert kws["files"] == ("baz", "foo")

    # Can drop encodings from the defaults to not have them filled
    kws = determine_encoding(
        dict(foo=10, bar=12, baz=15, spam=21, fish=25, chips=30, salt=33),
        **default_kws,
        plot_kwargs=dict(hue="bar", col="bad_dim_name"),
        drop_missing_dims=True,
        drop_encodings=("col",),
        map_free_dims_to="files",
    )
    assert kws["x"] == "salt"
    assert kws["hue"] == "bar"
    assert "col" not in kws
    assert kws["row"] == "chips"
    assert kws["frames"] == "fish"
    assert kws["files"] == ("spam", "baz", "foo")


# -----------------------------------------------------------------------------
# -- FacetGrid tests ----------------------------------------------------------
# -----------------------------------------------------------------------------

# .. make_facet_grid_plot decorator ...........................................


def test_make_facet_grid_plot():
    mfg = make_facet_grid_plot
    FGF = _FACET_GRID_FUNCS

    # Needs arguments
    with pytest.raises(TypeError):

        @mfg
        def foo():
            pass

    # Only supports specific mappings
    for m in ("dataset", "dataarray", "dataarray_line"):

        @mfg(map_as=m, encodings=("foo",), register_as_kind="foo_" + m)
        def foo():
            pass

        try:  # NOTE Can't use `with pytest.raises` here

            @mfg(map_as=m + "foo", encodings=("foo",))
            def foo():
                pass

        except ValueError as err:
            assert "Unsupported value" in str(err)

    # Custom registration name
    assert "my_foo" not in FGF

    @mfg(map_as="dataset", encodings=("foo",), register_as_kind="my_foo")
    def foo():
        pass

    assert "my_foo" in FGF

    # Skip registration
    assert "my_bar" not in FGF

    @mfg(map_as="dataset", encodings=("bar",), register_as_kind=False)
    def bar():
        pass

    assert "my_bar" not in FGF

    # Overwrite existing
    prev_my_foo = FGF["my_foo"]

    @mfg(
        map_as="dataset",
        encodings=("bar",),
        register_as_kind="my_foo",
        overwrite_existing=True,
    )
    def foo():
        pass

    new_my_foo = FGF["my_foo"]
    assert new_my_foo is not prev_my_foo

    # Error without overwrite
    try:

        @mfg(map_as="dataset", encodings=("bar",))
        def my_foo():
            pass

    except ValueError as err:
        assert "is already used" in str(err)
    else:
        assert FGF["my_foo"] is new_my_foo


# .. facet_grid itself ........................................................


def test_facet_grid(dm, out_dir, anim_disabled):
    """Tests the basic features and special cases of the facet_grid plot"""
    ppc = PyPlotCreator("test_facet_grid", dm=dm, plot_func=facet_grid)
    ppc._exist_ok = True

    # Shortcuts
    shared_kwargs = dict(animation=anim_disabled)
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Make sure there are no figures currently open, in order to be able to
    # track whether any figures leak from the plot function ...
    plt.close("all")
    assert len(plt.get_fignums()) == 0

    # Invoke the plotting function with data of different dimensionality.
    # This should succeed even for high-dimensional data, because a plot kind
    # is not explicitly given, thus always falling back to `hist`.
    for cont_name in dm["ndim_da"]:
        ppc(
            **out_path("auto__no_specs_" + cont_name),
            **shared_kwargs,
            select=dict(data="ndim_da/" + cont_name),
        )

    # The last figure should survive from this.
    assert len(plt.get_fignums()) == 1

    # Error message upon invalid kind. There should be no figure surviving from
    # such an invocation ...
    plt.close("all")

    for cont_name in dm["ndim_da"]:
        with pytest.raises(AttributeError, match="seems not to be available"):
            ppc(
                **out_path("bad_kind__" + cont_name),
                **shared_kwargs,
                select=dict(data="ndim_da/" + cont_name),
                kind="some_invalid_plot_kind",
            )

    assert len(plt.get_fignums()) == 0

    # Special cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # scatter: Is only valid for dataset data
    ppc(
        **out_path("scatter_ds"),
        **shared_kwargs,
        kind="scatter",
        x="foo",
        y="bar",
        select=dict(data="datasets/foobar3D"),
    )

    # errorbars: also requires dataset
    ppc(
        **out_path("errorbars"),
        **shared_kwargs,
        kind="errorbars",
        y="mean",
        yerr="std",
        x="foo",
        col="bar",
        hue="spam",
        frames="baz",
        select=dict(data="datasets/mean_and_std4D"),
    )

    ppc(
        **out_path("errorbars_single"),
        **shared_kwargs,
        kind="errorbars",
        y="mean",
        yerr="std",
        # x="foo",  # auto-deduced
        select=dict(data="datasets/mean_and_std1D"),
    )

    # ... will fail with unlabelled dimensions
    with pytest.raises(PlottingError, match="coordinates associated"):
        ppc(
            **out_path("errorbars_unlabelled"),
            **shared_kwargs,
            kind="errorbars",
            y="foo",
            yerr="bar",
            x="dim_0",
            col="dim_1",
            row="dim_1",
            select=dict(data="datasets/foobar3D"),
        )


def test_facet_grid_no_kind(dm, out_dir):
    """Tests the facet_grid without a ``kind`` specified"""
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["no_kind"])


def test_facet_grid_auto_encoding(dm, out_dir):
    """Tests the facet_grid with auto-encoding of kind and specifiers"""
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["auto"])


def test_facet_grid_kinds(dm, out_dir):
    """Very briefly tests the different facet_grid ``kind``s. More extended
    tests are part of the full test suite, see ``FULL_TEST``.
    """
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["kinds"])


def test_facet_grid_errorbars(dm, out_dir):
    """Tests the facet_grid for ``kind == 'errorbars'``"""
    invoke_facet_grid(
        dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["errorbars"]
    )


def test_facet_grid_scatter3d(dm, out_dir):
    """Tests the facet_grid for ``kind == 'scatter3d'``"""
    invoke_facet_grid(
        dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["scatter3d"]
    )


def test_facet_grid_cm_support(dm, out_dir):
    """Test ColorManager support for facet grid kinds"""
    invoke_facet_grid(
        dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["cm_support"]
    )


def test_facet_grid_fig_creation(dm, out_dir):
    """Test figure creation for faceting and non-faceting plots"""
    invoke_facet_grid(
        dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["fig_creation"]
    )


# -- Extended tests -----------------------------------------------------------


@skip_if_not_full
def test_facet_grid_line(dm, out_dir):
    """Tests the facet_grid for ``kind == 'line'``"""
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["line"])


@skip_if_not_full
def test_facet_grid_2d(dm, out_dir):
    """Tests the facet_grid for 2D ``kind``s"""
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["2d"])


@skip_if_not_full
def test_facet_grid_hist(dm, out_dir):
    """Tests the facet_grid for ``kind == 'hist'``"""
    invoke_facet_grid(dm=dm, out_dir=out_dir, to_test=PLOTS_CFG_FG["hist"])
