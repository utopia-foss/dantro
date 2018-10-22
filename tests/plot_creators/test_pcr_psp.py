"""Tests the ParamSpacePlotCreator classes."""

import pytest

import numpy as np

from dantro.data_mngr import DataManager
from dantro.plot_creators import UniversePlotCreator, MultiversePlotCreator


# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..test_group import psp_grp, pspace

@pytest.fixture
def init_kwargs(psp_grp, tmpdir) -> dict:
    """Default initialisation kwargs for both plot creators"""
    dm = DataManager(tmpdir)
    dm.add(psp_grp)
    return dict(dm=dm, psgrp_path='mv')

# Tests -----------------------------------------------------------------------

def test_MultiversePlotCreator(init_kwargs):
    """Assert the MultiversePlotCreator behaves correctly"""
    # Initialization works
    mpc = MultiversePlotCreator("init", **init_kwargs)

    # Properties work
    assert mpc.psgrp == init_kwargs['dm']['mv']

    # Check the error messages
    with pytest.raises(ValueError, match="Missing class variable PSGRP_PATH"):
        _mpc = MultiversePlotCreator("init", **init_kwargs)
        _mpc.PSGRP_PATH = None
        _mpc.psgrp

    # Check that the select function is called as expected
    selector = dict(field="testdata/fixedsize/state",
                    subspace=dict(p0=slice(3), p1=slice(2.5, 3.5)))
    args, kwargs = mpc._prepare_plot_func_args(select=selector)
    assert 'mv_data' in kwargs
    mv_data = kwargs['mv_data']
    print("Selected Multiverse data:", mv_data)

    # Check the lengths are correct
    assert mv_data.dims['p0'] == 2
    assert mv_data.dims['p1'] == 1
    assert mv_data.dims['p2'] == 5

    # Check the coordinates are correct
    assert np.all(mv_data.coords['p0'] == [1, 2])
    assert np.all(mv_data.coords['p1'] == [3])
    assert np.all(mv_data.coords['p2'] == [1, 2, 3, 4, 5])


def test_UniversePlotCreator(init_kwargs):
    """Assert the UniversePlotCreator behaves correctly"""
    # Initialization works
    upc = UniversePlotCreator("init", **init_kwargs)

    # Properties work
    assert upc.psgrp == init_kwargs['dm']['mv']

    # Check the error messages
    with pytest.raises(ValueError, match="Missing class variable PSGRP_PATH"):
        _upc = UniversePlotCreator("init", **init_kwargs)
        _upc.PSGRP_PATH = None
        _upc.psgrp

    with pytest.raises(RuntimeError, match="No state mapping was stored yet"):
        UniversePlotCreator("init", **init_kwargs).state_map
