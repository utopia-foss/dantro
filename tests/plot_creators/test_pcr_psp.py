"""Tests the ParamSpacePlotCreator classes."""

import pytest

import numpy as np
import xarray as xr

from paramspace import ParamDim, ParamSpace

from dantro.data_mngr import DataManager
from dantro.groups import ParamSpaceGroup
from dantro.plot_creators import UniversePlotCreator, MultiversePlotCreator


# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..groups.test_pspgrp import psp_grp, psp_grp_default

@pytest.fixture()
def pspace():
    """Used to setup a small pspace object to be tested on."""
    # NOTE This is the same as in ..test_group module, but with an order value
    #      set for the first dimension.
    return ParamSpace(dict(foo="bar",
                           p0=ParamDim(default=0, values=[0, 1], order=0),
                           p1=ParamDim(default=0, values=[0, 1, 2]),
                           p2=ParamDim(default=0, values=[0, 1, 2, 3, 4]),
                           a=dict(a0=ParamDim(default=0, range=[4]))))

@pytest.fixture
def init_kwargs(psp_grp, tmpdir) -> dict:
    """Default initialisation kwargs for both plot creators"""
    dm = DataManager(tmpdir)
    dm.add(psp_grp)
    return dict(dm=dm, psgrp_path='mv')

# Some mock callables .........................................................
def mock_pfunc(*_, **__):
    """... does nothing"""
    pass

# Tests -----------------------------------------------------------------------

def test_MultiversePlotCreator(init_kwargs, psp_grp_default, tmpdir):
    """Assert the MultiversePlotCreator behaves correctly"""
    # Initialization works
    mpc = MultiversePlotCreator("test", **init_kwargs)

    # Properties work
    assert mpc.psgrp == init_kwargs['dm']['mv']

    # Check the error messages
    with pytest.raises(ValueError, match="Missing class variable PSGRP_PATH"):
        _mpc = MultiversePlotCreator("failing", **init_kwargs)
        _mpc.PSGRP_PATH = None
        _mpc.psgrp

    # Check that the select function is called as expected
    selector = dict(field="testdata/fixedsize/state",
                    subspace=dict(p0=slice(2), p1=slice(1.5, 2.5)))
    args, kwargs = mpc._prepare_plot_func_args(mock_pfunc, select=selector)
    assert 'mv_data' in kwargs
    mv_data = kwargs['mv_data']
    print("Selected Multiverse data:", mv_data)

    # Check the lengths are correct
    assert mv_data.dims['p0'] == 2
    assert mv_data.dims['a0'] == 4
    assert mv_data.dims['p1'] == 1
    assert mv_data.dims['p2'] == 5

    # Check the coordinates are correct
    assert np.all(mv_data.coords['p0'] == [0, 1])
    assert np.all(mv_data.coords['a0'] == [0, 1, 2, 3])
    assert np.all(mv_data.coords['p1'] == [2])
    assert np.all(mv_data.coords['p2'] == [0, 1, 2, 3, 4])


    # Check again with a zero-volume parameter space
    dm = DataManager(tmpdir.join("default"))
    dm.add(psp_grp_default)
    assert dm[psp_grp_default.name].pspace.volume == 0

    mpcd = MultiversePlotCreator("default", dm=dm,
                                 psgrp_path=psp_grp_default.name)

    selector = dict(field="testdata/fixedsize/state")
    args, kwargs = mpcd._prepare_plot_func_args(mock_pfunc, select=selector)
    assert 'mv_data' in kwargs


def test_UniversePlotCreator(init_kwargs):
    """Assert the UniversePlotCreator behaves correctly"""
    # Initialization works
    upc = UniversePlotCreator("test", **init_kwargs)

    # Properties work
    assert upc.psgrp == init_kwargs['dm']['mv']

    # Check the error messages
    with pytest.raises(ValueError, match="Missing class variable PSGRP_PATH"):
        _upc = UniversePlotCreator("test", **init_kwargs)
        _upc.PSGRP_PATH = None
        _upc.psgrp


    # Check that prepare_cfg, where most of the actions happen, does what it
    # should do.
    # `uni` argument not given
    with pytest.raises(ValueError, match="Missing required keyword-argument"):
        upc.prepare_cfg(plot_cfg=dict(), pspace=None)

    # wrong `uni` type
    with pytest.raises(TypeError, match="Need parameter `universes` to be "):
        upc.prepare_cfg(plot_cfg=dict(universes=[1,2,3]), pspace=None)

    # wrong `uni` string value
    with pytest.raises(ValueError, match="Invalid value for `universes` arg"):
        upc.prepare_cfg(plot_cfg=dict(universes='invalid'), pspace=None)


    # Now, with valid arguments
    # "all" universes
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=None)

    assert not cfg
    assert isinstance(psp, ParamSpace)
    assert [k for k in psp.dims.keys()] == ['p0', 'a0', 'p1', 'p2']
    assert psp.volume == 2 * 4 * 3 * 5     # in this order
    assert psp.num_dims == 4
    assert '_coords' in psp.default
    assert len(psp.default['_coords']) == 4

    # Parameter dimensions have very negative order values
    assert all([pdim.order <= -1e6 for pdim in psp.dims.values()])

    # Having given a parameter space, those dimensions are used as well
    pspace = ParamSpace(dict(foo="bar", v0=ParamDim(default=-1, range=[5])))
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=pspace)

    assert psp.num_dims == 4 + 1
    assert psp.default['foo'] == "bar"
    assert psp.default['v0'] == -1

    # Give custom coordinates
    unis = dict(p0=[1], p1=slice(0.5, 2.5))
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes=unis), pspace=None)

    assert psp.num_dims == 4
    assert psp.volume == 1 * 4 * 2 * 5

    # Now check the plot function arguments are correctly parsed
    for cfg in psp:
        assert '_coords' in cfg
        print("Latest config: ", cfg)
        print(upc.psgrp.pspace.state_map)
        print(upc.psgrp.pspace.active_state_map)
        args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)

        assert 'uni' in kwargs
        assert 'coords' not in kwargs
        assert '_coords' not in kwargs


    # Check the more elaborate string arguments
    # 'single'/'first' universe
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='first'), pspace=None)

    assert psp.num_dims == 4
    assert psp.volume == 1 * 1 * 1 * 1
    
    for cfg in psp:
        args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)
        assert kwargs['uni'].name == '151'  # first non-default
    

    # 'random'/'any' universe
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='any'), pspace=None)

    assert psp.num_dims == 4
    assert psp.volume == 1 * 1 * 1 * 1
    
    for cfg in psp:
        args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)
        # ID is >= first possible ID and smaller than maximum ID
        assert 151 <= int(kwargs['uni'].name) < (3 * 4 * 5 * 6)


    # Assert correct error messages
    # Fails with invalid dimension name
    with pytest.raises(ValueError, match="No parameter dimension 'foo'"):
        upc.prepare_cfg(plot_cfg=dict(universes=dict(p0=[2],
                                                     p1=slice(1.5, 3.5),
                                                     foo=slice(10))),
                         pspace=None)

    # Fails (in paramspace) if such a coordinate is not available
    with pytest.raises(KeyError, match="is not available as coordinate of"):
        upc.prepare_cfg(plot_cfg=dict(universes=dict(p0=[-1],
                                                     p1=slice(1.5, 3.5))),
                         pspace=None)
    
    # Fails (in paramspace) if dimension would be squeezed away
    with pytest.raises(ValueError, match="'p0' would be totally masked"):
        upc.prepare_cfg(plot_cfg=dict(universes=dict(p0=slice(10, None),
                                                     p1=slice(1.5, 3.5))),
                         pspace=None)

    # Fails with a _coords already part of plots config
    with pytest.raises(ValueError, match="may _not_ contain the key '_coords"):
        upc.prepare_cfg(plot_cfg=dict(universes='all', _coords="foo"),
                         pspace=None)


def test_UniversePlotCreator_default_only(init_kwargs):
    """Check that a ParamSpaceGroup with only the default data or only a zero-
    dimensional parameter space still behaves as expected"""

    dm = init_kwargs['dm']
    pspace = ParamSpace(dict(foo="bar"))
    mvd = dm.new_group("mv_default", Cls=ParamSpaceGroup, pspace=pspace)
    mvd.new_group("00")
    upc = UniversePlotCreator("test2", dm=dm, psgrp_path="mv_default")

    assert upc._without_pspace is False

    # 'all' universes
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=None)
    assert psp is None
    assert upc._without_pspace

    args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)
    assert kwargs['uni'].name == "00"

    # 'single'/'first' universe
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='first'), pspace=None)
    assert psp is None
    assert upc._without_pspace

    args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)
    assert kwargs['uni'].name == "00"

    # Giving a pspace also works
    pspace = ParamSpace(dict(foo=ParamDim(default=0, range=[5])))
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=pspace)

    assert not cfg
    assert isinstance(psp, ParamSpace)
    assert psp.num_dims == 1

    # Doing anything more fancy would fail ...
    with pytest.raises(ValueError, match="Could not select a universe for"):
        upc.prepare_cfg(plot_cfg=dict(universes=dict(p0=[-1])), pspace=None)


    # One more possibility: associated paramspace has many dimensions but only
    # data for the default exists in the associated group.
    dm = init_kwargs['dm']
    pspace = ParamSpace(dict(p0=ParamDim(default=-1, range=[5])))
    mvd = dm.new_group("mv_default_only", Cls=ParamSpaceGroup, pspace=pspace)
    mvd.new_group("0")
    upc = UniversePlotCreator("test3", dm=dm, psgrp_path="mv_default_only")

    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=None)
    assert psp is None
    assert upc._without_pspace

    args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)
    assert kwargs['uni'].name == "0"

    # Should also be able to pass a parameter space additionally
    pspace = ParamSpace(dict(foo=ParamDim(default=0, range=[5])))
    cfg, psp = upc.prepare_cfg(plot_cfg=dict(universes='all'), pspace=pspace)
    
    assert not cfg
    assert isinstance(psp, ParamSpace)
    assert psp.num_dims == 1

    # Fails when trying to do something more fancy, e.g. selecting subspace
    with pytest.raises(ValueError, match="Could not select a universe for"):
        upc.prepare_cfg(plot_cfg=dict(universes=dict(p0=[-1])), pspace=None)

def test_UniversePlotCreator_DAG_usage(init_kwargs):
    """Tests DAG feature integration into the UniversePlotCreator.

    The integration works completely via the _prepare_plot_func_args, thus it
    is sufficient to test that dag_options are set properly.
    """
    upc = UniversePlotCreator("test_DAG_usage", **init_kwargs)

    # "all" universes
    _, psp = upc.prepare_cfg(pspace=None,
                             plot_cfg=dict(universes='all', use_dag=True,
                                           select=dict(randints='labelled/randints'))
                             )

    # Now check the plot function arguments are correctly parsed
    for cfg in psp:
        assert '_coords' in cfg
        print("Latest config: ", cfg)
        args, kwargs = upc._prepare_plot_func_args(mock_pfunc, **cfg)

        assert 'coords' not in kwargs
        assert '_coords' not in kwargs
        
        assert 'uni' in kwargs
        assert 'data' in kwargs

        assert 'randints' in kwargs['data']
        assert kwargs['data']['randints'] is kwargs['uni']['labelled/randints']
