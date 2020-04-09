"""Tests the generic external plot functions."""

import os
import copy
import builtins
import logging
from itertools import product
from typing import Tuple
from pkg_resources import resource_filename

import pytest

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from dantro.tools import load_yml, DoNothingContext
from dantro.plot_creators.ext_funcs.generic import facet_grid
from dantro.plot_creators import PlotHelper, ExternalPlotCreator
from dantro.containers import XrDataContainer, PassthroughContainer


# Local variables .............................................................
# Whether to write test output to a temporary directory
# NOTE When manually debugging, it's useful to set this to False, such that the
#      output can be inspected in TEST_OUTPUT_PATH
USE_TMPDIR = False

# If not using a temporary directory, the desired output directory
TEST_OUTPUT_PATH = os.path.abspath("test_output")

# Test configuration
PLOTS_CFG = load_yml(resource_filename("tests", "cfg/plots_facet_grid.yml"))

# The facet_grid plot kinds that can be created with the generic plot function
KINDS_1D = ('line', 'step')
KINDS_2D = ('contourf', 'contour', 'imshow', 'pcolormesh')

# Disable matplotlib logger
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# -- Helpers ------------------------------------------------------------------

def create_nd_data(n: int, *, shape=None, **data_array_kwargs) -> xr.DataArray:
    """Creates n-dimensional random data of a certain shape. If no shape is
    given, will use ``(3, 4, 5, ..)``.
    """
    if shape is None:
        shape = tuple([i+3 for i in range(n)])

    return xr.DataArray(data=np.random.random(shape), **data_array_kwargs)

def associate_specifiers(data, *,
                         specifiers: Tuple[str]=None,
                         exclude: tuple=None) -> dict:
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
        specifiers = ('x', 'y', 'row', 'col', 'hue', 'frames')

    if exclude is not None:
        specifiers = [s for s in specifiers if s not in exclude]

    dim_names = data.dims[::-1]
    return {spec: dim_name for spec, dim_name in zip(specifiers, dim_names)}


def invoke_facet_grid(*, dm, out_dir, to_test: dict, max_num_figs: int=5):
    """Repeatedly invokes the facet_grid plot function"""
    epc = ExternalPlotCreator("test_facet_grid", dm=dm)

    # Shortcuts
    animation = dict(enabled=False, writer='frames',
                     writer_kwargs=dict(frames=dict(saving=(dict(dpi=36)))))
    shared_kwargs = dict(plot_func=facet_grid,
                         animation=animation,
                         helpers=dict(save_figure=dict(dpi=36)))
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Now, the generic testing, combinations of: kinds, specifier, data
    for case_name, cfg in to_test.items():
        kinds = cfg.get('kinds')
        specifiers = cfg['specifiers']
        max_dims_to_test = cfg['max_dims_to_test']
        raises = cfg.get('raises', {})
        plot_kwargs = cfg.get('plot_kwargs', {})

        print("\nTesting scenario '{}' up to {}-dimensional data ..."
              "".format(case_name, max_dims_to_test))
        print("Kinds:          ", ", ".join(kinds) if kinds else "auto")
        print("All specifiers: ", specifiers)

        # Restrict the container iteration to the maximum dimensionality
        conts_it = [(name, cont) for name, cont in dm['ndim_da'].items()
                    if cont.ndim <= max_dims_to_test]
        kinds_it = kinds if kinds else [None]

        # Now, iterate over these combinations
        for kind, (cont_name, cont) in product(kinds_it, conts_it):
            aspecs = associate_specifiers(cont, specifiers=specifiers)
            print("... with data:    ", cont)
            print("    kind:         ", kind)
            print("    and spec map: ", aspecs)

            # Determine a context to allow to test for failing cases
            context = DoNothingContext()
            if cont.ndim in raises:
                # These are expected to fail with a specific type and message
                raise_spec = raises[cont.ndim]
                exc_type, match = raise_spec
                print("    expct. raise: ", exc_type, ": '{}'".format(match))
                exc_type = getattr(builtins, exc_type)
                context = pytest.raises(exc_type, match=match)

            # Now, run the plot function in that context
            with context:
                epc.plot(**out_path("{case:}__{kind:}_{data:}_{specs:}"
                                    "".format(kind=kind if kind else 'auto',
                                              case=case_name, data=cont.name,
                                              specs="-".join(aspecs))),
                         **shared_kwargs, **aspecs, **plot_kwargs,
                         kind=kind,
                         select=dict(data="ndim_da/" + cont_name))

            # Check plot figure count
            fignums = plt.get_fignums()
            print("    Succeeded.")
            print("    Open figures: ", fignums)
            assert len(fignums) <= max_num_figs


# -- Fixtures -----------------------------------------------------------------
# Import fixtures from other tests
from ...test_plot_mngr import dm as _dm

@pytest.fixture
def data_dataarray() -> dict:
    """Create a test xarray.DaraArray"""
    return xr.DataArray(np.random.rand(5, 4, 3))


@pytest.fixture
def data_dataset() -> dict:
    """Create a test xarray.Daraset"""
    d0 = xr.DataArray(np.random.rand(5, 4, 3))
    d1 = xr.DataArray(np.random.rand(5, 4, 3))

    ds = xr.Dataset()
    ds['d0'] = d0
    ds['d1'] = d1

    return ds


@pytest.fixture
def dm(_dm, data_dataarray, data_dataset):
    """Returns a data manager populated with some high-dimensional test data"""
    # Add a test xr.DataArray
    grp_dataarray = _dm.new_group("dataarray")
    grp_dataarray.add(XrDataContainer(name="data", data=data_dataarray))

    # Add a test xr.Dataset
    grp_dataset = _dm.new_group("dataset")
    grp_dataset.add(PassthroughContainer(name="data", data=data_dataset))

    # Add ndim random data for DataArrays
    grp_ndim_da = _dm.new_group("ndim_da")
    grp_ndim_da.add(*[XrDataContainer(name="{:d}D".format(n),
                                      data=create_nd_data(n))
                      for n in range(8)])

    return _dm


@pytest.fixture
def anim_disabled() -> dict:
    """Returns a dict with default (disabled) animation kwargs"""
    return dict(enabled=False, writer='frames',
                writer_kwargs=dict(frames=dict(saving=(dict(dpi=36)))))


@pytest.fixture
def anim_enabled(anim_disabled) -> dict:
    """Returns a dict with default (enabled) animation kwargs"""
    d = copy.deepcopy(anim_disabled)
    d['enabled'] = True
    return d


@pytest.fixture
def out_dir(tmpdir) -> str:
    if USE_TMPDIR:
        return str(tmpdir)

    # else: Create an output path if it does not yet exist
    if not os.path.exists(TEST_OUTPUT_PATH):
        os.mkdir(TEST_OUTPUT_PATH)

    return TEST_OUTPUT_PATH


# -- Tests --------------------------------------------------------------------

def test_facet_grid(dm, out_dir, anim_disabled):
    """Tests the basic features and special cases of the facet_grid plot"""
    epc = ExternalPlotCreator("test_facet_grid", dm=dm)

    # Shortcuts
    shared_kwargs = dict(plot_func=facet_grid, animation=anim_disabled)
    out_path = lambda name: dict(out_path=os.path.join(out_dir, name + ".pdf"))

    # Some simple, explicit invocation
    epc.plot(**out_path("manual_2d"), **shared_kwargs,
             select=dict(data='ndim_da/2D'))

    # More systematically invoke the plotting function with data of different
    # dimensionality. This should succeed even for high-dimensional data.
    for cont_name in dm['ndim_da']:
        epc.plot(**out_path("auto__no_specs_" + cont_name), **shared_kwargs,
                 select=dict(data="ndim_da/" + cont_name))

    # Error message upon invalid kind
    for cont_name in dm['ndim_da']:
        with pytest.raises(AttributeError, match="seems not to be available"):
            epc.plot(**out_path("bad_kind__" + cont_name), **shared_kwargs,
                     select=dict(data="ndim_da/" + cont_name),
                     kind='some_invalid_plot_kind')

    # Special cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # scatter: Is only valid for dataset data
    epc.plot(**out_path("scatter_ds"), **shared_kwargs,
             kind='scatter', x='d0', y='d1',
             select=dict(data='dataset/data'))


def test_facet_grid_auto(dm, out_dir):
    """Tests the facet_grid without a ``kind`` specified"""
    invoke_facet_grid(dm=dm, out_dir=out_dir,
                      to_test=PLOTS_CFG['test_facet_grid_auto'])

def test_facet_grid_line(dm, out_dir):
    """Tests the facet_grid without a ``kind`` specified"""
    invoke_facet_grid(dm=dm, out_dir=out_dir,
                      to_test=PLOTS_CFG['test_facet_grid_line'])

def test_facet_grid_2d(dm, out_dir):
    """Tests the facet_grid without a ``kind`` specified"""
    invoke_facet_grid(dm=dm, out_dir=out_dir,
                      to_test=PLOTS_CFG['test_facet_grid_2d'])

def test_facet_grid_hist(dm, out_dir):
    """Tests the facet_grid without a ``kind`` specified"""
    invoke_facet_grid(dm=dm, out_dir=out_dir,
                      to_test=PLOTS_CFG['test_facet_grid_hist'])


@pytest.mark.skip()
def test_facet_grid_animation(dm, anim_disabled, anim_enabled, out_dir):
    """Test the FacetGrid animation; requires invocation via a plot creator"""
    epc = ExternalPlotCreator("test_facet_grid_anim", dm=dm)

    # The data paths to the test xr.DataArray and xr.Dataset
    DATAARRAY_PATH = dict(select=dict(data='dataarray/data'))
    DATASET_PATH = dict(select=dict(data='dataset/data'))

    # Dictionary to collect passing test cases
    test_cases = {}

    # Dictionary to collect from dimension cases
    test_dim_error_cases = {}

    # Add general cases without kind specification to the test cases
    test_cases['default_1'] = dict(col='dim_1')
    test_cases['default_2'] = dict(row='dim_1', hue='dim_2',)
    test_cases['default_3'] = dict(row='dim_1', frames='dim_0')
    test_cases['default_anim'] = dict(frames='dim_1', col='dim_2', )

    # Add 1D plot kinds to the test cases
    for k in KINDS_1D:
        test_dim_error_cases['_'.join([k, '0'])] = dict(kind=k)
        test_dim_error_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')
        test_cases['_'.join([k, '2'])] = dict(kind=k,
                                              row='dim_1',
                                              hue='dim_2',
                                              col='dim_0', )

    # Add 2D plot cases to the test cases
    for k in KINDS_2D:
        test_cases['_'.join([k, '1'])] = dict(kind=k, col='dim_1')
        test_cases['_'.join([k, 'anim_1'])] = dict(
            kind=k, frames='dim_1')

    # .. Tests ................................................................
    for name, plot_kwargs in test_dim_error_cases.items():
        with pytest.raises(Exception):
            # Invoke plotting function via plot creator
            epc.plot(out_path=None,
                     plot_func=facet_grid,
                     animation=anim_disabled, **plot_kwargs, **DATAARRAY_PATH)

    for name, plot_kwargs in test_cases.items():
        # Invoke plotting function via plot creator
        epc.plot(out_path=os.path.join(out_dir, "test_{}".format(name)),
                 plot_func=facet_grid,
                 animation=anim_disabled, **plot_kwargs, **DATAARRAY_PATH)

    # .. Special Cases ........................................................
    # hist
    # Invoke plotting function via plot creator
    epc.plot(out_path="/".join([out_dir, "test_hist"]),
             plot_func=facet_grid,
             animation=anim_disabled, kind='hist', **DATAARRAY_PATH)

    # scatter: Is only valid for dataset data
    # Invoke plotting function via plot creator
    epc.plot(out_path="/".join([out_dir, "test_scatter"]),
             plot_func=facet_grid,
             animation=anim_disabled, kind='scatter', x='d0', y='d1',
             **DATASET_PATH)

    # .. Errors ...............................................................
    with pytest.raises(ValueError, match="Got an unknown plot kind"):
        # Invoke plotting function via plot creator
        epc.plot(out_path="/".join([out_dir, "test_{}".format(name)]),
                 plot_func=facet_grid,
                 animation=anim_disabled, kind='wrong', **DATAARRAY_PATH)
