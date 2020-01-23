"""Test the plot helper module"""

import copy
import builtins
from itertools import chain
from pkg_resources import resource_filename

import pytest
import os
import matplotlib.pyplot as plt

from dantro.tools import load_yml
from dantro.data_mngr import DataManager
from dantro.plot_creators import ExternalPlotCreator
from dantro.plot_creators import PlotHelper, is_plot_func
from dantro.plot_creators._pcr_ext_modules.plot_helper import temporarily_changed_axis, coords_match

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
def hlpr(tmpdir) -> PlotHelper:
    """Plot Helper for testing methods directly"""
    return PlotHelper(out_path=tmpdir.join("test.pdf"),
                      helper_defaults=CFG_HELPER)

@pytest.fixture
def epc(dm) -> ExternalPlotCreator:
    """External Plot Creator for integration tests"""
    return ExternalPlotCreator("ph_test", dm=dm, default_ext="pdf")

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
    # Call the plot function
    hlpr.ax.plot([1, 2], [-1, -2])

@is_plot_func(creator_name='external',
              helper_defaults={'set_title': {'title': "Title"}},
              supports_animation=True)
def plot3(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot with helper defaults in decorator.
    
    Args:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
    """
    # Get the data
    x_data = dm['vectors/times']
    y_data = dm['vectors/values']

    # Call the plot function, only using the last value
    hlpr.ax.plot(x_data[-1], y_data[-1])

    # define update generator for possible animations
    def update():
        for i in range(5):
            hlpr.ax.clear()
            hlpr.ax.plot(x_data[:i+1], y_data[:i+1])
            yield

    # register it
    hlpr.register_animation_update(update)

@is_plot_func(creator_name='external')
def plot4(dm: DataManager, *, hlpr:PlotHelper):
    """Test plot that does nothing"""
    pass




# Tests -----------------------------------------------------------------------

def test_init(hlpr):
    """Tests the PlotHelper's basic functionality"""
    # Initialisation
    # trying to configure helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        PlotHelper(out_path="test_path", update_helper_cfg={'foo': {}})

    # Test that the base config was carried over correctly, i.e. without the
    # axis-specific configuration
    expected_base_cfg = copy.deepcopy(CFG_HELPER)
    axis_specific_updates = expected_base_cfg.pop('axis_specific')
    assert hlpr.base_cfg == expected_base_cfg

    # Test the axis-specific configuration is loaded
    assert isinstance(hlpr._axis_specific_updates, dict)
    assert hlpr._axis_specific_updates == axis_specific_updates

    # The axis-specific config should still be empty
    assert hlpr._cfg is None
    with pytest.raises(RuntimeError, match="while a figure is associated"):
        hlpr.axis_cfg

    # Coordinates are not available
    with pytest.raises(RuntimeError, match="while a figure is associated"):
        hlpr.ax_coords

    # No figure associated yet
    with pytest.raises(ValueError, match="No figure initialized or already"):
        hlpr.fig

def test_epc_integration(epc, tmpdir):
    """Test integration into external plot creator"""
    # call the plot method of the External Plot Creator using plot functions
    # with different decorators 'is_plot_func'
    epc.plot(out_path=tmpdir.join("plot1.pdf"), plot_func=plot1)
    epc.plot(out_path=tmpdir.join("plot2.pdf"), plot_func=plot2)
    epc.plot(out_path=tmpdir.join("plot3.pdf"), plot_func=plot3)
    epc.plot(out_path=tmpdir.join("plot4.pdf"), plot_func=plot4)

    # Check errors and warnings
    with pytest.raises(ValueError, match="'animation' was found"):
        epc.plot(out_path=tmpdir.join("anim_found.pdf"), plot_func=plot1,
                 animation=dict(foo="bar"))

    with pytest.raises(ValueError, match="'helpers' was found in the"):
        epc.plot(out_path=tmpdir.join("helpers_found.pdf"), plot_func=plot1,
                 helpers=dict(foo="bar"))

def test_figure_setup_simple(hlpr):
    """Test simple figure setup, i.e. without subplots"""

    # setup_figure should find configured figsize in helper config
    hlpr.setup_figure()
    assert hlpr.fig.get_size_inches()[0] == 5
    assert hlpr.fig.get_size_inches()[1] == 6

    # config should be set up now and match the passed one + updates
    assert hlpr.axis_cfg['set_title']['title'] == "bottom right hand"
    assert hlpr.axis_cfg['set_title']['color'] == "green"
    assert hlpr.axis_cfg['set_title']['size'] == 42

    # trying to call setup function twice should not lead to an error but to
    # a new figure being created
    old_fig = hlpr.fig
    hlpr.setup_figure()
    assert old_fig is not hlpr.fig

    # When closing the figure, the associated attributes are deleted
    hlpr.close_figure()
    assert hlpr._fig is None
    assert hlpr._axes is None
    assert hlpr._cfg is None

def test_figure_setup_subplots(hlpr):
    """Test plot helper figure setup with subplots"""
    # Set up a figure with subplots
    hlpr.setup_figure(ncols=2, nrows=3)
    assert hlpr.axes.shape == (2, 3)

    # Current axis is the very first one, unlike matplotlib does it
    assert hlpr.ax is hlpr.axes[0, 0]
    assert hlpr.ax_coords == (0, 0)

    # Axis-specific updates should have been applied
    assert hlpr._cfg[(0, 0)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 0)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 0)]['set_title']['size'] == 42

    assert hlpr._cfg[(1, 0)]['set_title']['title'] == "last column"
    assert hlpr._cfg[(1, 0)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 0)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(1, 1)]['set_title']['title'] == "last column"
    assert hlpr._cfg[(1, 1)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 1)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(1, 2)]['set_title']['title'] == "bottom right hand"
    assert hlpr._cfg[(1, 2)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 2)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(0, 1)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 1)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 1)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(0, 2)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 2)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 2)]['set_title']['size'] == 5

    # Create a new figure and test the scaling feature
    hlpr.setup_figure(ncols=2, nrows=4, scale_figsize_with_subplots_shape=True)
    assert (hlpr.fig.get_size_inches() == (2*5, 4*6)).all()

    # Axis-specific updates should have been applied again, despite different
    # shape
    assert hlpr._cfg[(0, 0)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 0)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 0)]['set_title']['size'] == 42

    assert hlpr._cfg[(1, 0)]['set_title']['title'] == "last column"
    assert hlpr._cfg[(1, 0)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 0)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(1, 1)]['set_title']['title'] == "last column"
    assert hlpr._cfg[(1, 1)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 1)]['set_title']['size'] == 5

    assert hlpr._cfg[(1, 3)]['set_title']['title'] == "bottom right hand"
    assert hlpr._cfg[(1, 3)]['set_title']['color'] == "green"
    assert hlpr._cfg[(1, 3)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(0, 1)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 1)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 1)]['set_title']['size'] == 5
    
    assert hlpr._cfg[(0, 2)]['set_title']['title'] == "Test Title"
    assert hlpr._cfg[(0, 2)]['set_title']['color'] == "green"
    assert hlpr._cfg[(0, 2)]['set_title']['size'] == 5

def test_figure_attachment(hlpr):
    """Test the attach_figure function"""
    # Define a new figure a single axis and replace the existing
    fig = plt.figure()
    ax = fig.gca()
    hlpr.attach_figure(fig, ax)

    # Same with multiple axes
    fig, axes = plt.subplots(2, 2)
    hlpr.attach_figure(fig, axes)
    hlpr.select_axis(1, 1)

    with pytest.raises(ValueError, match="must be passed as a 2d array-like!"):
        hlpr.attach_figure(fig, axes.flatten())

def test_cfg_manipulation(hlpr):
    """Test manipulation of the configuration"""
    # Setup the figure, needed for all below
    hlpr.setup_figure()

    # Marking as enabled/disabled . . . . . . . . . . . . . . . . . . . . . . .
    # 'enabled' not specified -> regarded as enabled
    assert 'set_title' in hlpr.enabled_helpers

    # check mark_disabled
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers

    # can also call it again and it is still disabled
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers

    # check mark_enabled
    hlpr.mark_enabled('set_title')
    assert 'set_title' in hlpr.enabled_helpers

    # can also call it again
    hlpr.mark_enabled('set_title')
    assert 'set_title' in hlpr.enabled_helpers

    # also works for titles without any entries
    assert 'set_labels' not in hlpr.axis_cfg
    hlpr.mark_enabled('set_labels')
    assert 'set_labels' in hlpr.axis_cfg

    assert 'set_limits' not in hlpr.axis_cfg
    hlpr.mark_disabled('set_limits')
    assert 'set_limits' in hlpr.axis_cfg

    # bad helper names raise
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_enabled('foo')

    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_disabled('foo')

    # trying to mark a special key enabled or disabled is not allowed
    with pytest.raises(ValueError, match="'save_figure' (.*) are NOT allowed"):
        hlpr.mark_enabled('save_figure')

    with pytest.raises(ValueError, match="'save_figure' (.*) are NOT allowed"):
        hlpr.mark_disabled('save_figure')

    # Providing defaults . . . . . . . . . . . . . . . . . . . . . . . . . . .
    hlpr.provide_defaults('set_title', title="overwritten")
    assert hlpr.axis_cfg['set_title']['title'] == "overwritten"

    # Also possible for special config keys, but stored in base config
    hlpr.provide_defaults('save_figure', foo=42)
    assert hlpr.base_cfg['save_figure']['foo'] == 42

    hlpr.provide_defaults('save_figure', foo=24)
    assert hlpr.base_cfg['save_figure']['foo'] == 24

    # For a helper that has no key yet
    assert 'set_suptitle' not in hlpr.axis_cfg
    hlpr.provide_defaults('set_suptitle', title="The Figure Title")
    assert hlpr.axis_cfg['set_suptitle']['title'] == "The Figure Title"

    # trying to provide defaults for an unavailable helper raises
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.provide_defaults('foo')

    # This also marks them enabled
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers
    hlpr.provide_defaults('set_title', title="something")
    assert 'set_title' in hlpr.enabled_helpers

    # ...unless specified otherwise
    hlpr.mark_disabled('set_title')
    assert 'set_title' not in hlpr.enabled_helpers
    hlpr.provide_defaults('set_title', title="something", mark_enabled=False)
    assert 'set_title' not in hlpr.enabled_helpers

def test_invocation(hlpr):
    """Test helper invocation"""
    # Setup the figure
    hlpr.setup_figure()

    # Invoke a helper directly
    hlpr.invoke_helper('set_title', title="manually enabled")
    assert 'set_title' not in hlpr.enabled_helpers

    # Mark it as enabled, but not let the invocation disable it
    hlpr.mark_enabled('set_title')
    hlpr.invoke_helpers('set_title', mark_disabled_after_use=False,
                        set_title=dict(size=10))
    assert 'set_title' in hlpr.enabled_helpers

    # trying to invoke helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.invoke_helpers('set_title', 'foo')

    # Invoke all enabled on the current axis
    hlpr.invoke_enabled()

    # All disabled now
    assert not hlpr.enabled_helpers

    # Invocation via private method takes into account enabled state
    hlpr._invoke_helper('set_title') # nothing happens

    # Check that a helpful error message is generated
    with pytest.raises(ValueError, match="was raised during invocation of"):
        hlpr.invoke_helper('set_title', invalid_key="something")

    # But it can also be just logged
    hlpr._raise_on_error = False
    hlpr.invoke_helper('set_title', another_invalid_key="something else")

    # Finally, save and close the figure
    hlpr.save_figure()


def test_coords_match():
    """Test the coords_match function"""
    full_shape = (5, 4)
    match = lambda c, m: coords_match(c, match=m, full_shape=full_shape)
    
    assert match((0, 0), (0, 0))
    assert match((0, 0), (None, None))
    assert match((0, 0), (Ellipsis, Ellipsis))
    assert match((0, 0), (None, 0))
    assert match((0, 0), (0, None))

    assert not match((0, 0), (1, 2))
    assert not match((0, 0), (1, 0))
    assert not match((0, 0), (0, 1))
    assert not match((0, 0), (None, 1))
    assert not match((0, 0), (1, None))

    # May also be a list
    assert match((0, 0), [0, 0])

    # Or a string
    assert match((0, 0), 'all')

    # But nothing else
    with pytest.raises(TypeError, match="needs to be a 2-tuple, list, or a s"):
        match((0, 0), "hi")

    # And lengths always have to match
    with pytest.raises(ValueError, match="Need 2-tuples for arguments"):
        match((0, 0), (0, 0, 0))
    
    # And values should not exceed the full shape 
    with pytest.raises(ValueError, match="exceeding the shape"):
        match((0, 0), (5, 3))
    
    with pytest.raises(ValueError, match="exceeding the shape"):
        match((0, 0), (4, 4))


def test_tmp_axis_context_manager(hlpr):
    """Test the temporarily_changed_axis context manager"""
    hlpr.setup_figure(ncols=5, nrows=4)
    shape = hlpr.axes.shape

    # Current axis should be the first one
    assert hlpr.ax_coords == (0, 0)

    with temporarily_changed_axis(hlpr, tmp_ax_coords=(0, 1)):
        # Should be on (0, 1) now
        assert hlpr.ax_coords == (0, 1)

    # Should be back on (0, 0)
    assert hlpr.ax_coords == (0, 0)

    # Can also select the last axis via negative indices
    with temporarily_changed_axis(hlpr, tmp_ax_coords=(-1, -1)):
        # Should be on (4, 3) now
        assert hlpr.ax_coords == (4, 3)
    assert hlpr.ax_coords == (0, 0)

    # An error should still lead to a reset
    with pytest.raises(Exception, match="Some Error"):
        with temporarily_changed_axis(hlpr, tmp_ax_coords=(0, 1)):
            # Should be on (0, 1) now
            assert hlpr.ax_coords == (0, 1)

            # manually let an exception occur
            raise Exception("Some Error")

    # Should still be back on (0, 0), despite the exception
    assert hlpr.ax_coords == (0, 0)

    # If exceeding the valid axis, this should of course propagate the error
    with pytest.raises(ValueError, match=r"Could not select axis \(23, 42"):
        with temporarily_changed_axis(hlpr, tmp_ax_coords=(23, 42)):
            # Should not reach this point
            pass

    # Should still be on the old axis
    assert hlpr.ax_coords == (0, 0)

def test_axis_specificity(hlpr):
    """Test that configurations and invocations are only applied to the
    specified axis"""
    # Setup some figure without subplots
    hlpr.setup_figure()

    # Need a subplots figure for that, so overwrite the existing one
    hlpr.setup_figure(ncols=4, nrows=3)

    # Test selecting axes
    hlpr.select_axis(3, 2)
    assert hlpr.ax_coords == (3, 2)

    hlpr.select_axis(-1, -1)
    assert hlpr.ax_coords == (3, 2)
    
    hlpr.select_axis(0, 0)
    assert hlpr.ax_coords == (0, 0)

    # Providing defaults by itself only changes the current axis
    hlpr.provide_defaults('set_title', title="some title")
    assert hlpr.axis_cfg['set_title']['title'] == "some title"
    assert hlpr._cfg[(0, 1)]['set_title']['title'] != "some title"

    # Supply defaults for all axes
    hlpr.provide_defaults('set_title', title="default", axes='all')

    for ax_coords, params in hlpr._cfg.items():
        assert params['set_title']['title'] == "default"

    # Set the title for the current axis
    assert hlpr.ax_coords == (0, 0)
    hlpr.invoke_helper('set_title', title="Axis (0, 0)")

    # Check that the title is the specified one
    assert hlpr.ax.title.get_text() == "Axis (0, 0)"

    # Now, try to specify it for multiple axes
    hlpr.invoke_helper('set_title', title="Top Row", axes=(Ellipsis, 0))
    for ax in hlpr.axes[:, 0]:
        assert ax.title.get_text() == "Top Row"

    # The above are all disabled now. Only the remaining should still be set
    # when invoking all enabled
    hlpr.invoke_enabled(axes="all")
    for ax in hlpr.axes[:, 0]:
        assert ax.title.get_text() == "Top Row"

    for ax in chain(hlpr.axes[:, 1], hlpr.axes[:, 1]):
        assert ax.title.get_text() == "default"
    
def test_animation(epc, tmpdir):
    """Test the animation feature"""
    # Test error messages . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # plot function does not support animation
    with pytest.raises(ValueError, match="'plot4' was not marked as "
                                         "supporting an animation!"):
        epc.plot(out_path=tmpdir.join("test.pdf"), plot_func=plot4,
                 animation=CFG_ANIM['complete'])

    # no generator defined in plot function
    with pytest.raises(ValueError, match="No animation update generator"):
        epc.plot(out_path=tmpdir.join("test.pdf"), plot_func=plot2,
                 animation=CFG_ANIM['complete'])

    # missing writer
    with pytest.raises(TypeError,
                       match="missing 1 required keyword-only argument: 'wri"):
        epc.plot(out_path=tmpdir.join("test.pdf"), plot_func=plot3,
                 animation=CFG_ANIM['missing_writer'])
    
    # unavailable writer
    with pytest.raises(ValueError, match="'foo' is not available"):
        epc.plot(out_path=tmpdir.join("test.pdf"), plot_func=plot3,
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

def test_helper_functions(hlpr):
    """Test all helper functions directly"""
    def call_helper(helper_name, test_cfg: dict):
        hlpr_func = getattr(hlpr, '_hlpr_' + helper_name)
        hlpr_kwargs = test_cfg[helper_name]

        hlpr_func(**hlpr_kwargs)

    # Keep track of tested helpers
    tested_helpers = set()

    # Go over the helpers to be tested
    for i, test_cfg in enumerate(CFG_HELPER_FUNCS):
        # Always work on a new figure
        hlpr.setup_figure()

        # There is a single key that is used to identify the helper
        helper_names = [k for k in test_cfg if not k.startswith("_")]

        print("Testing helper(s) {} with the following parameters:\n  {}"
              "".format(helper_names, test_cfg))

        # Make sure the axis is empty
        hlpr.ax.clear()

        # Can do some plotting beforehand ...
        if test_cfg.get('_plot_values'):
            hlpr.ax.plot(test_cfg['_plot_values'])

        # Find out if this config was set to raise
        if not test_cfg.get('_raises', None):
            # Should pass
            for helper_name in helper_names:
                call_helper(helper_name, test_cfg)

        else:
            # Should fail
            # Determine exception type
            exc_type = Exception
            if isinstance(test_cfg['_raises'], str):
                exc_type = getattr(builtins, test_cfg['_raises'])

            # Test
            with pytest.raises(exc_type, match=test_cfg.get('_match')):
                for helper_name in helper_names:
                    call_helper(helper_name, test_cfg)

        for helper_name in helper_names:
            tested_helpers.add(helper_name)
        print("  Test successful.\n")

    # Make sure all available helpers were tested
    assert all([h in tested_helpers for h in hlpr.available_helpers])

    # Close the figure
    hlpr.close_figure()
