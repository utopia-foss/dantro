"""Test the plot helper module"""

from pkg_resources import resource_filename

import pytest

from dantro.tools import load_yml
from dantro.plot_creators.pcr_ext_modules.plot_helper import PlotHelper, PlotHelperWarning
from dantro.plot_creators.pcr_ext import ExternalPlotCreator, is_plot_func
from dantro.data_mngr import DataManager

# Local constants
# Paths
CFG_HELPER_PATH = resource_filename("tests", "cfg/helper_cfg.yml")

# Configurations
CFG_HELPER = load_yml(CFG_HELPER_PATH)
ENABLED = ['set_title']


# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..test_plot_mngr import dm

@pytest.fixture
def init_kwargs(dm) -> dict:
    """Default initialisation kwargs"""
    return dict(dm=dm, default_ext="pdf")

@pytest.fixture
def ph_init(tmpdir) -> PlotHelper:
    """Plot Helper for testing methods directly"""
    ph = PlotHelper(out_path=tmpdir, enabled_helpers_defaults=ENABLED,
                    **CFG_HELPER)
    return ph

@pytest.fixture
def epc_init(init_kwargs) -> ExternalPlotCreator:
    """External Plot Creator for integration tests"""
    epc = ExternalPlotCreator("ph_test", **init_kwargs)

    return epc

# Plot functions --------------------------------------------------------------

def plot1(dm: DataManager, *, out_path: str):
    """Test plot that does nothing"""
    pass

@is_plot_func(creator_name='external')
def plot2(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot that uses different PlotHelper methods.
    
    Args:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
    """
    # Assemble the arguments
    args = [[1,2], [1,2]]

    # Call the plot function
    hlpr.ax.plot(*args)

@is_plot_func(creator_name='external', enabled_helpers=['set_title'],
              helper_defaults={'set_title': {'title': "Title"}})
def plot3(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot with helper defaults in decorator.
    
    Args:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
    """
    # Assemble the arguments
    args = [[1,2], [1,2]]

    # Call the plot function
    hlpr.ax.plot(*args)


# Tests -----------------------------------------------------------------------

def test_plot_helper(ph_init, epc_init):
    """Tests the Plot Helper"""
    # test PlotHelper methods directly ........................................

    # test init
    # trying to configure helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        PlotHelper(out_path="test_path", **{'foo': {}})

    # This should initialize correctly
    hlpr = ph_init
    assert hlpr.cfg == CFG_HELPER

    # trying to get figure instance before initialization
    with pytest.raises(ValueError, match="No figure initialized!"):
        hlpr.fig()

    # setup_figure should find configured figsize in helper config
    hlpr.setup_figure()
    assert hlpr.fig.get_size_inches()[0] == 5
    assert hlpr.fig.get_size_inches()[1] == 5

    # trying to call setup function twice
    with pytest.raises(ValueError, match="Figure is already initialized!"):
        hlpr.setup_figure()

    hlpr.provide_cfg('set_title', **{'title': "overwritten"})
    assert hlpr.cfg['set_title']['title'] == "overwritten"
    hlpr.provide_cfg('save_figure', **{'foo': 42})
    assert hlpr.cfg['save_figure']['foo'] == 42
    hlpr.provide_cfg('save_figure', **{'foo': 24})
    assert hlpr.cfg['save_figure']['foo'] == 24

    # trying to provide cfg for unavailable helper
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.provide_cfg('foo')

    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_enabled('foo')

    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_disabled('foo')

    # check mark_disabled
    hlpr.mark_disabled('set_title')
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers

    # check mark_enabled
    hlpr.mark_enabled('set_title')
    hlpr.mark_enabled('set_title')
    assert 'set_title' in hlpr.enabled_helpers

    # invoke already enabled helper without disabling
    with pytest.raises(PlotHelperWarning, match="The already enabled helper "
                                                "'set_title' was invoked."):
        hlpr.invoke_helpers('set_title', mark_disabled_after_use=False,
                            **{'set_title': {'size': 10}})
    assert 'set_title' in hlpr.enabled_helpers

    # trying to invoke helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.invoke_helpers(*['set_title', 'foo'])

    hlpr.invoke_all()
    hlpr.save_figure()

    # test PlotHelper in ExternalPlotCreator ..................................
    # call the plot method of the External Plot Creator using plot functions
    # with different decorators 'is_plot_func'
    epc = epc_init
    epc.plot(out_path=hlpr.out_path, plot_func=plot1)
    epc.plot(out_path=hlpr.out_path, plot_func=plot2)
    epc.plot(out_path=hlpr.out_path, plot_func=plot3)



