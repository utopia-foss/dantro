"""Test the plot helper module"""

import builtins
from pkg_resources import resource_filename

import pytest
import os

from dantro.tools import load_yml
from dantro.data_mngr import DataManager
from dantro.plot_creators import ExternalPlotCreator
from dantro.plot_creators import PlotHelper, is_plot_func

# Local constants
# Paths
CFG_HELPER_PATH = resource_filename("tests", "cfg/helper_cfg.yml")
CFG_HELPER_FUNCS_PATH = resource_filename("tests", "cfg/helper_funcs.yml")
CFG_ANIM_PATH = resource_filename("tests", "cfg/anim_cfg.yml")

# Configurations
CFG_HELPER = load_yml(CFG_HELPER_PATH)
CFG_HELPER_FUNCS = load_yml(CFG_HELPER_FUNCS_PATH)
CFG_ANIM = load_yml(CFG_ANIM_PATH)

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

@is_plot_func(creator_name='external', supports_animation=True)
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
              helper_defaults={'set_title': {'title': "Title"}},
              supports_animation=True)
def plot3(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot with helper defaults in decorator.
    
    Args:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
    """
    x_data = dm['vectors/times']
    y_data = dm['vectors/values']

    # Assemble the arguments
    args = [x_data[-1], y_data[-1]]

    # Call the plot function
    hlpr.ax.plot(*args)

    # define update generator for possible animations
    def update():
        for i in range(5):
            hlpr.ax.clear()
            hlpr.ax.plot(x_data[:i+1], y_data[:i+1])
            yield

    hlpr.register_animation_update(update)

@is_plot_func(creator_name='external')
def plot4(dm: DataManager, *, hlpr:PlotHelper):
    """Test plot that does nothing"""
    pass


# Tests -----------------------------------------------------------------------

def test_plot_helper(ph_init, epc_init, tmpdir):
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
    with pytest.raises(ValueError, match="No figure initialized or already"):
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
    epc.plot(out_path=hlpr.out_path, plot_func=plot4)

    # Check errors and warnings
    with pytest.raises(ValueError, match="'animation' was found"):
        epc.plot(out_path=hlpr.out_path, plot_func=plot1,
                 animation=dict(foo="bar"))

    with pytest.raises(ValueError, match="'helpers' was found in the"):
        epc.plot(out_path=hlpr.out_path, plot_func=plot1,
                 helpers=dict(foo="bar"))

    
def test_animation(epc_init, tmpdir):
    """Test the animation feature"""
    epc = epc_init

    # Test error messages . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # plot function does not support animation
    with pytest.raises(ValueError, match="'plot4' was not marked as "
                                         "supporting an animation!"):
        epc.plot(out_path=tmpdir, plot_func=plot4,
                 animation=CFG_ANIM['complete'])

    # no generator defined in plot function
    with pytest.raises(ValueError, match="No animation update generator"):
        epc.plot(out_path=tmpdir, plot_func=plot2,
                 animation=CFG_ANIM['complete'])

    # missing writer
    with pytest.raises(TypeError,
                       match="missing 1 required keyword-only argument: 'wri"):
        epc.plot(out_path=tmpdir, plot_func=plot3,
                 animation=CFG_ANIM['missing_writer'])
    
    # unavailable writer
    with pytest.raises(ValueError, match="'foo' is not available"):
        epc.plot(out_path=tmpdir, plot_func=plot3,
                 animation=CFG_ANIM['unavailable_writer'])
    
    # Test behaviour . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # this should work correctly
    epc.plot(out_path=tmpdir.join("basic.pdf"), plot_func=plot3,
             animation=CFG_ANIM['complete'])

    # check if all frames were saved
    files_in_plot_dir = os.listdir(tmpdir.join("basic"))
    assert '0000000.pdf' in files_in_plot_dir
    assert '0000001.pdf' in files_in_plot_dir
    assert '0000002.pdf' in files_in_plot_dir
    assert '0000003.pdf' in files_in_plot_dir
    assert '0000004.pdf' in files_in_plot_dir

    # test that no animation is created when marked as disabled
    # this should work correctly
    epc.plot(out_path=tmpdir.join("not_enabled.pdf"), plot_func=plot3,
             animation=CFG_ANIM['not_enabled'])
    assert tmpdir.join("not_enabled.pdf").isfile()


    # test some more in an automated fashion
    for i, anim_cfg in enumerate(CFG_ANIM['should_work']):
        print("Testing 'should_work' animation config #{} ...\n  {}"
              "".format(i, anim_cfg))
        epc.plot(out_path=tmpdir.join(str(i) + ".pdf"), plot_func=plot3,
                 animation=dict(enabled=True, **anim_cfg))
    
    for i, anim_cfg in enumerate(CFG_ANIM['should_not_work']):
        print("Testing 'should_not_work' animation config #{} ...\n  {}"
              "".format(i, anim_cfg))
        with pytest.raises(Exception):
            epc.plot(out_path=tmpdir.join(str(i) + ".pdf"), plot_func=plot3,
                     animation=dict(enabled=True, **anim_cfg))

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
