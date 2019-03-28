"""Test the plot helper module"""

import builtins
from pkg_resources import resource_filename

import pytest

from dantro.tools import load_yml
from dantro.data_mngr import DataManager
from dantro.plot_creators import ExternalPlotCreator
from dantro.plot_creators import PlotHelper, PlotHelperWarning, is_plot_func

# Local constants
# Paths
CFG_HELPER_PATH = resource_filename("tests", "cfg/helper_cfg.yml")
CFG_HELPER_FUNCS_PATH = resource_filename("tests", "cfg/helper_funcs.yml")

# Configurations
CFG_HELPER = load_yml(CFG_HELPER_PATH)
CFG_HELPER_FUNCS = load_yml(CFG_HELPER_FUNCS_PATH)


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
    ph = PlotHelper(out_path=tmpdir, helper_defaults=CFG_HELPER)
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

@is_plot_func(creator_name='external',
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
        PlotHelper(out_path="test_path", update_helper_cfg={'foo': {}})

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

    hlpr.provide_defaults('set_title', title="overwritten")
    assert hlpr.cfg['set_title']['title'] == "overwritten"
    hlpr.provide_defaults('save_figure', foo=42)
    assert hlpr.cfg['save_figure']['foo'] == 42
    hlpr.provide_defaults('save_figure', foo=24)
    assert hlpr.cfg['save_figure']['foo'] == 24

    # trying to provide cfg for unavailable helper
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.provide_defaults('foo')

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

    # invoke a helper that might be disabled
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers

    hlpr.invoke_helper('set_title')  # will do nothing
    hlpr.invoke_helper('set_title', enabled=True)
    assert 'set_title' not in hlpr.enabled_helpers

    # invoke already enabled helper without disabling
    hlpr.mark_enabled('set_title')
    hlpr.invoke_helpers('set_title', mark_disabled_after_use=False,
                        set_title=dict(size=10))
    assert 'set_title' in hlpr.enabled_helpers

    # trying to invoke helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.invoke_helpers('set_title', 'foo')

    hlpr.invoke_all()
    hlpr.save_figure()

    # test PlotHelper in ExternalPlotCreator ..................................
    # call the plot method of the External Plot Creator using plot functions
    # with different decorators 'is_plot_func'
    epc = epc_init
    epc.plot(out_path=hlpr.out_path, plot_func=plot1)
    epc.plot(out_path=hlpr.out_path, plot_func=plot2)
    epc.plot(out_path=hlpr.out_path, plot_func=plot3)


def test_helper_functions(tmpdir):
    """Test all helper functions directly"""
    ph = PlotHelper(out_path=tmpdir.join("test_plot.pdf"))
    ph.setup_figure()

    # Keep track of tested helpers
    tested_helpers = set()

    # Go over the helpers to be tested
    for i, test_cfg in enumerate(CFG_HELPER_FUNCS):
        # There is a single key that is used to identify the helper
        helper_name = [k for k in test_cfg if not k.startswith("_")][0]

        # Get helper function and kwargs to test
        hlpr_func = getattr(ph, '_hlpr_' + helper_name)
        hlpr_kwargs = test_cfg[helper_name]

        print("Testing '{}' helper with the following parameters:\n  {}"
              "".format(helper_name, hlpr_kwargs))

        # Find out if it was set to raise
        if not test_cfg.get('_raises', None):
            # Should pass
            hlpr_func(**hlpr_kwargs)

        else:
            # Should fail
            # Determine exception type
            exc_type = Exception
            if isinstance(test_cfg['_raises'], str):
                exc_type = getattr(builtins, test_cfg['_raises'])

            # Test
            with pytest.raises(exc_type, match=test_cfg.get('_match')):
                hlpr_func(**hlpr_kwargs)

        tested_helpers.add(helper_name)
        print("  Test successful.\n")

    # Make sure all available helpers were tested
    assert all([h in tested_helpers for h in ph._AVAILABLE_HELPERS])
