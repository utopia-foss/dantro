"""Test the plot helper module"""

import builtins
import copy
import os
from itertools import chain

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pkg_resources import resource_filename

from dantro.data_mngr import DataManager
from dantro.plot import PlotHelper, PyPlotCreator, is_plot_func
from dantro.plot.plot_helper import (
    EnterAnimationMode,
    ExitAnimationMode,
    PlotConfigError,
    PlotHelperErrors,
)
from dantro.plot.plot_helper import _coords_match as coords_match
from dantro.plot.plot_helper import temporarily_changed_axis
from dantro.tools import load_yml

# Paths
CFG_HELPER_PATH = resource_filename("tests", "cfg/helper_cfg.yml")
CFG_HELPER_FUNCS_PATH = resource_filename("tests", "cfg/helper_funcs.yml")
CFG_ANIM_PATH = resource_filename("tests", "cfg/anim_cfg.yml")

# Configurations
CFG_HELPER = load_yml(CFG_HELPER_PATH)
CFG_HELPER_FUNCS = load_yml(CFG_HELPER_FUNCS_PATH)
CFG_ANIM = load_yml(CFG_ANIM_PATH)

from .._fixtures import *

# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..test_plot_mngr import dm


@pytest.fixture
def hlpr(out_dir) -> PlotHelper:
    """Plot Helper for testing methods directly"""
    test_path = os.path.join(out_dir, "test.pdf")
    return PlotHelper(out_path=test_path, helper_defaults=CFG_HELPER)


@pytest.fixture
def ppc(dm) -> PyPlotCreator:
    """PyPlotCreator for integration tests; the ``_plot_func`` needs to be set
    manually, though!
    """
    return PyPlotCreator(
        "ph_test", plot_func=lambda: 0, dm=dm, default_ext="pdf"
    )


# Plot functions --------------------------------------------------------------


def plot1(dm: DataManager, *, out_path: str):
    """Test plot that does nothing"""
    pass


@is_plot_func(creator="pyplot", supports_animation=True)
def plot2(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot that uses different PlotHelper methods.

    Args:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
    """
    # Call the plot function
    hlpr.ax.plot([1, 2], [-1, -2])


@is_plot_func(
    creator="pyplot",
    helper_defaults={"set_title": {"title": "Title"}},
    supports_animation=True,
)
def plot3(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot with helper defaults in decorator"""
    # Get the data
    x_data = dm["vectors/times"]
    y_data = dm["vectors/values"]

    # Call the plot function, only using the last value
    hlpr.ax.plot(x_data[-1], y_data[-1])

    # define update generator for possible animations
    def update():
        for i in range(5):
            hlpr.ax.clear()
            hlpr.ax.plot(x_data[: i + 1], y_data[: i + 1])
            yield

    # register it
    hlpr.register_animation_update(update)


@is_plot_func(creator="pyplot", supports_animation=True)
def plot3_mode_switching(
    dm: DataManager,
    *,
    hlpr: PlotHelper,
    should_exit: bool = None,
    should_enter: bool = None,
):
    """Test a plot function that exits animation mode"""
    if should_exit:
        hlpr.disable_animation()
    if should_enter:
        hlpr.enable_animation()
    # NOTE NEVER do this if-if construct in production code; it should never be
    #      possible to repeatedly change the animation mode. Ideally, use only
    #      one of the two indicators and make sure that it cannot flip-flop.

    plot3(dm, hlpr=hlpr)


@is_plot_func(creator="pyplot")
def plot4(dm: DataManager, *, hlpr: PlotHelper):
    """Test plot that does nothing"""
    pass


# Tests -----------------------------------------------------------------------


def test_init(hlpr):
    """Tests the PlotHelper's basic functionality"""
    # Initialisation
    # trying to configure helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        PlotHelper(out_path="test_path", update_helper_cfg={"foo": {}})

    # Test that the base config was carried over correctly, i.e. without the
    # axis-specific configuration
    expected_base_cfg = copy.deepcopy(CFG_HELPER)
    axis_specific_updates = expected_base_cfg.pop("axis_specific")
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


def test_ppc_integration(ppc, tmpdir):
    """Test integration into external plot creator"""
    # call the plot method of the External Plot Creator using plot functions
    # with different decorators 'is_plot_func'
    ppc._plot_func = plot1
    ppc.plot(out_path=tmpdir.join("plot1.pdf"))

    ppc._plot_func = plot2
    ppc.plot(out_path=tmpdir.join("plot2.pdf"))

    ppc._plot_func = plot3
    ppc.plot(out_path=tmpdir.join("plot3.pdf"))

    ppc._plot_func = plot4
    ppc.plot(out_path=tmpdir.join("plot4.pdf"))

    # Check errors and warnings
    with pytest.raises(ValueError, match="'animation' was found"):
        ppc._plot_func = plot1
        ppc.plot(
            out_path=tmpdir.join("anim_found.pdf"), animation=dict(foo="bar")
        )

    with pytest.raises(ValueError, match="'helpers' was found in the"):
        ppc._plot_func = plot1
        ppc.plot(
            out_path=tmpdir.join("helpers_found.pdf"),
            helpers=dict(foo="bar"),
        )


def test_figure_setup_simple(hlpr):
    """Test simple figure setup, i.e. without subplots"""

    # setup_figure should find configured figsize in helper config
    hlpr.setup_figure()
    assert hlpr.fig.get_size_inches()[0] == 5
    assert hlpr.fig.get_size_inches()[1] == 6

    # config should be set up now and match the passed one + updates
    assert hlpr.axis_cfg["set_title"]["title"] == "bottom right hand"
    assert hlpr.axis_cfg["set_title"]["color"] == "green"
    assert hlpr.axis_cfg["set_title"]["size"] == 42

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
    assert hlpr._cfg[(0, 0)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 0)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 0)]["set_title"]["size"] == 42

    assert hlpr._cfg[(1, 0)]["set_title"]["title"] == "last column"
    assert hlpr._cfg[(1, 0)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 0)]["set_title"]["size"] == 5

    assert hlpr._cfg[(1, 1)]["set_title"]["title"] == "last column"
    assert hlpr._cfg[(1, 1)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 1)]["set_title"]["size"] == 5

    assert hlpr._cfg[(1, 2)]["set_title"]["title"] == "bottom right hand"
    assert hlpr._cfg[(1, 2)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 2)]["set_title"]["size"] == 5

    assert hlpr._cfg[(0, 1)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 1)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 1)]["set_title"]["size"] == 5

    assert hlpr._cfg[(0, 2)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 2)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 2)]["set_title"]["size"] == 5

    # Create a new figure and test the scaling feature
    hlpr.setup_figure(ncols=2, nrows=4, scale_figsize_with_subplots_shape=True)
    assert (hlpr.fig.get_size_inches() == (2 * 5, 4 * 6)).all()

    # Axis-specific updates should have been applied again, despite different
    # shape
    assert hlpr._cfg[(0, 0)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 0)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 0)]["set_title"]["size"] == 42

    assert hlpr._cfg[(1, 0)]["set_title"]["title"] == "last column"
    assert hlpr._cfg[(1, 0)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 0)]["set_title"]["size"] == 5

    assert hlpr._cfg[(1, 1)]["set_title"]["title"] == "last column"
    assert hlpr._cfg[(1, 1)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 1)]["set_title"]["size"] == 5

    assert hlpr._cfg[(1, 3)]["set_title"]["title"] == "bottom right hand"
    assert hlpr._cfg[(1, 3)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(1, 3)]["set_title"]["size"] == 5

    assert hlpr._cfg[(0, 1)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 1)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 1)]["set_title"]["size"] == 5

    assert hlpr._cfg[(0, 2)]["set_title"]["title"] == "Test Title"
    assert hlpr._cfg[(0, 2)]["set_title"]["color"] == "green"
    assert hlpr._cfg[(0, 2)]["set_title"]["size"] == 5


def test_figure_attachment(hlpr):
    """Test the attach_figure function"""

    assert hlpr._fig is None
    assert not hlpr.axes

    # Define a new figure a single axis and replace the existing
    fig = plt.figure()
    ax = fig.gca()  # matplotlib.axes.Axes object
    hlpr.attach_figure_and_axes(fig=fig, axes=ax)

    assert hlpr.fig is fig
    assert (hlpr.axes == np.array([[ax]])).all()

    # Same with multiple axes
    fig, axes = plt.subplots(2, 2)
    hlpr.attach_figure_and_axes(fig=fig, axes=axes)

    assert hlpr.fig is fig
    assert (hlpr.axes == axes.T).all()

    # Can also pass a manually constructed nested list (in (y,x) format, as
    # given as return value from plt.subplots)
    hlpr.attach_figure_and_axes(
        fig=fig, axes=[[axes[0, 0], axes[0, 1]], [axes[1, 0], axes[1, 1]]]
    )
    assert hlpr.fig is fig
    assert (hlpr.axes == axes.T).all()

    # Test the selection
    hlpr.select_axis(0, 1)
    hlpr.select_axis(1, 0)
    hlpr.select_axis(1, 1)

    with pytest.raises(ValueError, match="Could not select axis"):
        hlpr.select_axis(1, 2)

    # Now, when passing multiple axes in 1d array-like, it should throw
    with pytest.raises(ValueError, match="must be passed as a 2d array-like"):
        hlpr.attach_figure_and_axes(fig=fig, axes=axes.flatten())

    # If attaching again but having `skip_if_identical` set, no error should
    # be raised
    hlpr.attach_figure_and_axes(
        fig=fig, axes="bad argument, but ignored", skip_if_identical=True
    )
    assert hlpr.fig is fig


def test_select_axis(hlpr):
    """Tests the select_axis method"""
    hlpr.setup_figure(ncols=2, nrows=3)
    assert hlpr.ax_coords == (0, 0)

    # Basic interface
    hlpr.select_axis(0, 1)
    assert hlpr.ax_coords == (0, 1)

    # Negative values are wrapped around ...
    hlpr.select_axis(col=-1, row=-2)
    assert hlpr.ax_coords == (1, 1)

    # Axis syncing
    # ... this one does nothing: already working on the current axis
    hlpr.select_axis(ax=hlpr.ax)

    # ... this one should change the axis
    hlpr.select_axis(ax=hlpr.axes[1, 1])
    assert hlpr.ax is hlpr.axes[1, 1]
    assert hlpr.ax_coords == (1, 1)

    # ... this one should sync to the currently selected axis
    hlpr.fig.sca(hlpr.axes[0, 0])
    assert hlpr.ax_coords == (1, 1)  # not changed --> out of sync
    hlpr.select_axis(ax=None)
    assert hlpr.ax_coords == (0, 0)  # in sync again
    hlpr.select_axis()
    assert hlpr.ax_coords == (0, 0)  # nothing changed

    # -- Error messages
    # bad axes coordinates
    with pytest.raises(ValueError, match=r"Could not select.*shape \(2, 3\)"):
        hlpr.select_axis(5, 6)

    # bad argument combinations
    with pytest.raises(ValueError, match="Need both `col` and `row`"):
        hlpr.select_axis(row=0)
    with pytest.raises(ValueError, match="Need both `col` and `row`"):
        hlpr.select_axis(col=0)
    with pytest.raises(ValueError, match="Need both `col` and `row`"):
        hlpr.select_axis(col=0, ax=hlpr.axes[1, 1])
    with pytest.raises(ValueError, match="Need both `col` and `row`"):
        hlpr.select_axis(row=0, ax=hlpr.axes[1, 1])
    with pytest.raises(ValueError, match="Cannot specify.*if also setting"):
        hlpr.select_axis(col=0, row=0, ax=hlpr.axes[1, 1])

    # ... with plt.gca() not in the currently associated axes
    _, _ = plt.subplots()
    assert plt.gca() not in hlpr.axes.flat
    with pytest.raises(ValueError, match="Could not find the given axis"):
        hlpr.select_axis()


def test_cfg_manipulation(hlpr):
    """Test manipulation of the configuration"""
    # Setup the figure, needed for all below
    hlpr.setup_figure()

    # Marking as enabled/disabled . . . . . . . . . . . . . . . . . . . . . . .
    # 'enabled' not specified -> regarded as enabled
    assert "set_title" in hlpr.enabled_helpers

    # check mark_disabled
    hlpr.mark_disabled("set_title")
    assert "set_title" not in hlpr.enabled_helpers

    # can also call it again and it is still disabled
    hlpr.mark_disabled("set_title")
    assert "set_title" not in hlpr.enabled_helpers

    # check mark_enabled
    hlpr.mark_enabled("set_title")
    assert "set_title" in hlpr.enabled_helpers

    # can also call it again
    hlpr.mark_enabled("set_title")
    assert "set_title" in hlpr.enabled_helpers

    # also works for titles without any entries
    assert "set_labels" not in hlpr.axis_cfg
    hlpr.mark_enabled("set_labels")
    assert "set_labels" in hlpr.axis_cfg

    assert "set_limits" not in hlpr.axis_cfg
    hlpr.mark_disabled("set_limits")
    assert "set_limits" in hlpr.axis_cfg

    # bad helper names raise
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_enabled("foo")

    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.mark_disabled("foo")

    # trying to mark a special key enabled or disabled is not allowed
    with pytest.raises(ValueError, match="'save_figure' (.*) are NOT allowed"):
        hlpr.mark_enabled("save_figure")

    with pytest.raises(ValueError, match="'save_figure' (.*) are NOT allowed"):
        hlpr.mark_disabled("save_figure")

    # Providing defaults . . . . . . . . . . . . . . . . . . . . . . . . . . .
    hlpr.provide_defaults("set_title", title="overwritten")
    assert hlpr.axis_cfg["set_title"]["title"] == "overwritten"

    # Also possible for special config keys, but stored in base config
    hlpr.provide_defaults("save_figure", foo=42)
    assert hlpr.base_cfg["save_figure"]["foo"] == 42

    hlpr.provide_defaults("save_figure", foo=24)
    assert hlpr.base_cfg["save_figure"]["foo"] == 24

    # For a helper that has no key yet
    assert "set_scales" not in hlpr.axis_cfg
    hlpr.provide_defaults("set_scales", x="linear")
    assert hlpr.axis_cfg["set_scales"]["x"] == "linear"

    # For a figure-level helper
    assert "set_suptitle" not in hlpr.base_cfg
    hlpr.provide_defaults("set_suptitle", title="The Figure Title")
    assert hlpr.base_cfg["set_suptitle"]["title"] == "The Figure Title"

    # trying to provide defaults for an unavailable helper raises
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.provide_defaults("foo")

    # This also marks them enabled
    hlpr.mark_disabled("set_title")
    assert "set_title" not in hlpr.enabled_helpers
    hlpr.provide_defaults("set_title", title="something")
    assert "set_title" in hlpr.enabled_helpers

    # ...unless specified otherwise
    hlpr.mark_disabled("set_title")
    assert "set_title" not in hlpr.enabled_helpers
    hlpr.provide_defaults("set_title", title="something", mark_enabled=False)
    assert "set_title" not in hlpr.enabled_helpers

    # Mark a figure-level helper (without defaults) as disabled
    hlpr.mark_disabled("set_figlegend")


def test_invocation(hlpr):
    """Test helper invocation"""
    # Setup the figure and draw something on it such that it is not empty
    hlpr.setup_figure()
    hlpr.ax.plot([1, 2, 3], [1, 2, 3], label="foo")

    # Invoke a helper directly
    hlpr.invoke_helper("set_title", title="manually enabled")
    assert "set_title" not in hlpr.enabled_helpers

    # Mark it as enabled, but not let the invocation disable it
    hlpr.mark_enabled("set_title")
    hlpr.invoke_helpers(
        "set_title", mark_disabled_after_use=False, set_title=dict(size=10)
    )
    assert "set_title" in hlpr.enabled_helpers

    # trying to invoke helper that is not available
    with pytest.raises(ValueError, match="No helper with name 'foo'"):
        hlpr.invoke_helpers("set_title", "foo")

    # Mark a figure-level helper as enabled
    hlpr.mark_enabled("set_suptitle")

    # Invoke all enabled on the current axis
    hlpr.invoke_enabled()

    # All disabled now
    assert not hlpr.enabled_helpers

    # Invocation via private method takes into account enabled state
    hlpr._invoke_helper("set_title")  # nothing happens

    # Check that a helpful error message is generated and includes docstring
    with pytest.raises(PlotHelperErrors, match="invalid_key"):
        hlpr.invoke_helper("set_title", invalid_key="something")

    with pytest.raises(PlotHelperErrors, match=r"Encountered 2 error\(s\)"):
        hlpr.invoke_helpers(
            "set_title",
            "set_suptitle",
            set_title=dict(invalid_key="something"),
            set_suptitle=dict(foobar="barbaz"),
        )

    with pytest.raises(PlotHelperErrors, match="Relevant Docstrings"):
        hlpr.invoke_helper("set_title", invalid_key="something")

    # Errors are gathered also when invoking only
    hlpr.provide_defaults("set_title", invalid_key="something")
    hlpr.provide_defaults("set_suptitle", invalid_key="something else")
    hlpr.provide_defaults("set_figlegend", title="this is ok")  # not an error
    hlpr.mark_enabled("set_title", "set_suptitle")
    with pytest.raises(PlotHelperErrors, match=r"Encountered 2 error\(s\)"):
        hlpr.invoke_enabled()

    # But it can also be just logged
    assert hlpr.raise_on_error
    hlpr._raise_on_error = False
    assert not hlpr.raise_on_error
    hlpr.invoke_helper("set_title", another_invalid_key="something else")

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
    assert match((0, 0), "all")

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

    # Draw something on each axis, such that they are not empty
    for tmp_ax_coords in hlpr.coords_iter(match="all"):
        with temporarily_changed_axis(hlpr, tmp_ax_coords):
            hlpr.ax.plot([1, 2, 3], [1, 2, 3], label=str(hlpr.ax_coords))

    # Test selecting axes
    hlpr.select_axis(3, 2)
    assert hlpr.ax_coords == (3, 2)

    hlpr.select_axis(-1, -1)
    assert hlpr.ax_coords == (3, 2)

    hlpr.select_axis(0, 0)
    assert hlpr.ax_coords == (0, 0)

    # Providing defaults by itself only changes the current axis
    hlpr.provide_defaults("set_title", title="some title")
    assert hlpr.axis_cfg["set_title"]["title"] == "some title"
    assert hlpr._cfg[(0, 1)]["set_title"]["title"] != "some title"

    # Supply defaults for all axes
    hlpr.provide_defaults("set_title", title="default", axes="all")

    for ax_coords, params in hlpr._cfg.items():
        assert params["set_title"]["title"] == "default"

    # Set the title for the current axis
    assert hlpr.ax_coords == (0, 0)
    hlpr.invoke_helper("set_title", title="Axis (0, 0)")

    # Check that the title is the specified one
    assert hlpr.ax.title.get_text() == "Axis (0, 0)"

    # Now, try to specify it for multiple axes
    hlpr.invoke_helper("set_title", title="Top Row", axes=(Ellipsis, 0))
    for ax in hlpr.axes[:, 0]:
        assert ax.title.get_text() == "Top Row"

    # The above are all disabled now. Only the remaining should still be set
    # when invoking all enabled
    hlpr.invoke_enabled(axes="all")

    # ... check top row was not changed (because was disabled)
    for ax in hlpr.axes[:, 0]:
        assert ax.title.get_text() == "Top Row"

    # ... check the two bottom rows were changed (were not disabled)
    for ax in chain(hlpr.axes[:, 1], hlpr.axes[:, 2]):
        assert ax.title.get_text() == "default"

    # A base config entry with figure-level helper will fail
    hlpr._axis_specific_updates = dict(foo=dict(set_suptitle=dict(foo="bar")))
    with pytest.raises(PlotConfigError, match="figure-level helper"):
        hlpr._compile_axis_specific_cfg()

    # Need either `axis` key or a 2-tuple as update key
    hlpr._axis_specific_updates = {
        (1, 2, 3): dict(set_title=dict(foo="bar"))  # not a 2-tuple
    }
    with pytest.raises(PlotConfigError, match="No axis could be determined"):
        hlpr._compile_axis_specific_cfg()

    hlpr._axis_specific_updates = {
        "foo": dict(set_title=dict(foo="bar"))  # `axis` entry missing
    }
    with pytest.raises(PlotConfigError, match="No axis could be determined"):
        hlpr._compile_axis_specific_cfg()


def test_animation(ppc, tmpdir):
    """Test the animation feature"""
    # Test error messages . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # plot function does not support animation
    with pytest.raises(
        ValueError, match="'plot4' was not marked as supporting an animation!"
    ):
        ppc._plot_func = plot4
        ppc.plot(
            out_path=tmpdir.join("test.pdf"),
            animation=CFG_ANIM["complete"],
        )

    # no generator defined in plot function
    with pytest.raises(ValueError, match="No animation update generator"):
        ppc._plot_func = plot2
        ppc.plot(
            out_path=tmpdir.join("test.pdf"),
            animation=CFG_ANIM["complete"],
        )

    # missing writer
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'wri"
    ):
        ppc._plot_func = plot3
        ppc.plot(
            out_path=tmpdir.join("test.pdf"),
            animation=CFG_ANIM["missing_writer"],
        )

    # unavailable writer
    with pytest.raises(ValueError, match="'foo' is not available"):
        ppc._plot_func = plot3
        ppc.plot(
            out_path=tmpdir.join("test.pdf"),
            animation=CFG_ANIM["unavailable_writer"],
        )

    # Test behaviour . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # this should work correctly
    ppc._plot_func = plot3
    ppc.plot(
        out_path=tmpdir.join("basic.pdf"),
        animation=CFG_ANIM["complete"],
    )

    # check if all frames were saved
    files_in_plot_dir = os.listdir(tmpdir.join("basic"))
    assert "0000000.pdf" in files_in_plot_dir
    assert "0000001.pdf" in files_in_plot_dir
    assert "0000002.pdf" in files_in_plot_dir
    assert "0000003.pdf" in files_in_plot_dir
    assert "0000004.pdf" in files_in_plot_dir
    assert "0000005.pdf" not in files_in_plot_dir

    # test that no animation is created when marked as disabled
    # this should work correctly
    ppc._plot_func = plot3
    ppc.plot(
        out_path=tmpdir.join("not_enabled.pdf"),
        animation=CFG_ANIM["not_enabled"],
    )
    assert tmpdir.join("not_enabled.pdf").isfile()

    # test some more in an automated fashion
    for i, anim_cfg in enumerate(CFG_ANIM["should_work"]):
        print(
            "Testing 'should_work' animation config #{} ...\n  {}".format(
                i, anim_cfg
            )
        )
        ppc.plot(
            out_path=tmpdir.join(str(i) + ".pdf"),
            animation=dict(enabled=True, **anim_cfg),
        )

    for i, anim_cfg in enumerate(CFG_ANIM["should_not_work"]):
        print(
            "Testing 'should_not_work' animation config #{} ...\n  {}".format(
                i, anim_cfg
            )
        )
        with pytest.raises(Exception):
            ppc.plot(
                out_path=tmpdir.join(str(i) + ".pdf"),
                animation=dict(enabled=True, **anim_cfg),
            )


def test_animation_mode_switching(hlpr, ppc, tmpdir):
    """Tests the feature that allows entering and exiting animation mode"""
    # -- Part 1: Helper raises the right control exceptions . . . . . . . . . .
    # Animation mode disabled
    assert not hlpr.animation_enabled
    hlpr.disable_animation()  # no exception
    with pytest.raises(EnterAnimationMode):
        hlpr.enable_animation()
    assert hlpr.animation_enabled

    # Animation mode now enabled
    assert hlpr.animation_enabled
    hlpr.enable_animation()
    with pytest.raises(ExitAnimationMode):
        hlpr.disable_animation()
    assert not hlpr.animation_enabled

    # -- Part 2: Switching between modes within PyPlotCreator . . . . . .
    ppc._plot_func = plot3_mode_switching

    # Animation-enabled plot --> NOT exiting --> directory with multiple plots
    plot_name = "not_exiting"
    ppc.plot(
        out_path=tmpdir.join(plot_name + ".pdf"),
        should_exit=False,
        animation=CFG_ANIM["complete"],
    )
    assert not tmpdir.join(plot_name + ".pdf").isfile()
    assert tmpdir.join(plot_name).isdir()
    assert len(tmpdir.join(plot_name).listdir()) == 5

    # Animation-enabled plot --> exiting --> single plot
    plot_name = "exiting"
    ppc.plot(
        out_path=tmpdir.join(plot_name + ".pdf"),
        should_exit=True,
        animation=CFG_ANIM["complete"],
    )
    assert tmpdir.join(plot_name + ".pdf").isfile()
    assert not tmpdir.join(plot_name).isdir()

    # Animation-disabled plot --> entering --> directory with multiple plots
    plot_name = "entering"
    anim_cfg = copy.deepcopy(CFG_ANIM["complete"])
    anim_cfg["enabled"] = False
    ppc.plot(
        out_path=tmpdir.join(plot_name + ".pdf"),
        should_enter=True,
        animation=anim_cfg,
    )
    assert not tmpdir.join(plot_name + ".pdf").isfile()
    assert tmpdir.join(plot_name).isdir()
    assert len(tmpdir.join(plot_name).listdir()) == 5

    plot_name = "not_entering"
    ppc.plot(
        out_path=tmpdir.join(plot_name + ".pdf"),
        should_enter=False,
        animation=anim_cfg,
    )
    assert tmpdir.join(plot_name + ".pdf").isfile()
    assert not tmpdir.join(plot_name).isdir()

    # -- Part 3: Error messages . . . . . . . . . . . . . . . . . . . . . . . .
    plot_name = "entering_with_missing_kwargs"
    with pytest.raises(ValueError, match="Cannot dynamically enter animation"):
        ppc.plot(
            out_path=tmpdir.join(plot_name + ".pdf"),
            should_enter=True,
            animation=None,
        )  # <-- missing animation kwargs here
    assert not tmpdir.join(plot_name + ".pdf").isfile()
    assert not tmpdir.join(plot_name).isdir()

    plot_name = "repeatedly_switching"
    with pytest.raises(RuntimeError, match="Cannot repeatedly enter or exit"):
        ppc.plot(
            out_path=tmpdir.join(plot_name + ".pdf"),
            should_enter=True,
            should_exit=True,  # <-- repeated switching
            animation=anim_cfg,
        )
    assert not tmpdir.join(plot_name + ".pdf").isfile()
    assert not tmpdir.join(plot_name).isdir()


def test_legend_handle_tracking(hlpr):
    """Tests the tracking of legend handles and labels"""
    hlpr.setup_figure(ncols=3, nrows=2)
    hlpr.select_axis(0, 0)
    hlpr.track_handles_labels([hlpr.ax_coords], [str(hlpr.ax_coords)])

    hlpr.select_axis(1, 1)
    hlpr.track_handles_labels([hlpr.ax_coords], [str(hlpr.ax_coords)])

    # Have these entries associated to the axis
    h, l = hlpr.axis_handles_labels
    assert h == [hlpr.ax_coords]
    assert l == [str(hlpr.ax_coords)]

    hlpr.select_axis(0, 0)
    h, l = hlpr.axis_handles_labels
    assert h == [hlpr.ax_coords]
    assert l == [str(hlpr.ax_coords)]

    # An axis without tracked handles still returns something, but empty
    hlpr.select_axis(0, 1)
    h, l = hlpr.axis_handles_labels
    assert not h
    assert not l

    # How about all handles and labels?
    all_h, all_l = hlpr.all_handles_labels
    assert all_h == [(0, 0), (1, 1)]
    assert all_l == ["(0, 0)", "(1, 1)"]

    with pytest.raises(ValueError, match="need to be of the same size"):
        hlpr.track_handles_labels([1, 2, 3], [2, 3])


def test_helper_functions(hlpr):
    """Test all helper functions directly"""

    def call_helper(helper_name, test_cfg: dict):
        hlpr_func = getattr(hlpr, "_hlpr_" + helper_name)
        hlpr_kwargs = test_cfg[helper_name]

        hlpr_func(**hlpr_kwargs)

    # Keep track of tested helpers
    tested_helpers = set()

    # Go over the helpers to be tested
    for i, test_cfg in enumerate(CFG_HELPER_FUNCS):
        test_cfg = copy.deepcopy(test_cfg)

        # Always work on a new figure
        hlpr.setup_figure(**test_cfg.pop("setup_figure", {}))

        # There is a single key that is used to identify the helper
        helper_names = [k for k in test_cfg if not k.startswith("_")]

        print(
            "Testing helper(s) {} with the following parameters:\n  {}".format(
                helper_names, test_cfg
            )
        )

        # Make sure the axis is empty
        hlpr.ax.clear()

        # Can do some plotting beforehand ...
        if test_cfg.get("_plot_values"):
            hlpr.ax.plot(
                test_cfg["_plot_values"], **test_cfg.get("_plot_kwargs", {})
            )

        if test_cfg.get("_invoke_legend"):
            hlpr.ax.legend(("foo", "bar"))
        if test_cfg.get("_invoke_figlegend"):
            hlpr.fig.legend(("foo", "bar"))

        # Find out if this config was set to raise
        if not test_cfg.get("_raises", None):
            # Should pass
            for helper_name in helper_names:
                call_helper(helper_name, test_cfg)

        else:
            # Should fail
            # Determine exception type
            exc_type = Exception
            if isinstance(test_cfg["_raises"], str):
                exc_type = getattr(builtins, test_cfg["_raises"])

            # Test
            with pytest.raises(exc_type, match=test_cfg.get("_match")):
                for helper_name in helper_names:
                    call_helper(helper_name, test_cfg)

        for helper_name in helper_names:
            tested_helpers.add(helper_name)
        print("  Test successful.\n")

    # Make sure all available helpers were tested
    assert all([h in tested_helpers for h in hlpr.available_helpers])

    # Close the figure
    hlpr.close_figure()


# .............................................................................


def test_helper_set_legend(hlpr):
    """Tests the `set_legend` helper directly"""
    get_legends = lambda obj: obj.findobj(mpl.legend.Legend)
    get_texts = lambda lg: [t.get_text() for t in lg.texts]
    get_len = lambda lg: len(lg.legend_handles)
    is_empty = lambda lg: get_len(lg) == 0

    # Set up a more involved figure
    hlpr.setup_figure(ncols=3, nrows=2)
    assert not get_legends(hlpr.fig)

    # Draw something on a subset of axes: (1, 1) and (2, 0)
    hlpr.select_axis(1, 1)
    hlpr.ax.plot([1, 2, 3], label="one")
    hlpr.ax.plot([2, 3, 4], label="two")
    assert not get_legends(hlpr.fig)

    hlpr.select_axis(2, 0)
    hlpr.ax.plot([3, 4, 5], label="three")
    hlpr.ax.plot([4, 5, 6], label="four")
    hlpr.ax.plot([5, 6, 7], label="five")
    assert not get_legends(hlpr.fig)

    # Invoke 'set_legend' on this axis and check that that legend is drawn
    hlpr.invoke_helper("set_legend")
    assert len(get_legends(hlpr.fig)) == 1

    legend = get_legends(hlpr.fig)[0]
    assert all([t in get_texts(legend) for t in ("three", "four", "five")])

    # Now invoke it on all axes; should yield two legend objects
    hlpr.invoke_helper("set_legend", axes="all")
    legends = get_legends(hlpr.fig)
    assert len(legends) == 2

    # Old legend no longer available
    assert legend not in legends

    # The legends are not empty
    assert not is_empty(legends[0])  # (2, 0)
    assert not is_empty(legends[1])  # (1, 1)

    assert get_texts(legends[0]) == ["three", "four", "five"]
    assert get_texts(legends[1]) == ["one", "two"]

    # Test gathering handles from the whole figure, here drawing them onto a
    # fully empty axis deliberately.
    hlpr.select_axis(0, 0)
    assert len(get_legends(hlpr.ax)) == 0

    hlpr.invoke_helper(
        "set_legend", skip_empty_axes=False, gather_from_fig=True
    )
    legend = get_legends(hlpr.ax)[0]
    assert get_texts(legend) == ["three", "four", "five", "one", "two"]

    # gather_from_fig always triggers handle search, but still retains a handle
    # that was not made part of a legend object yet. Duplicates are identified
    # by their label and are not included.
    hlpr.select_axis(0, 1)
    hlpr.ax.plot([6, 7, 8], label="six")
    assert not get_legends(hlpr.ax)

    hlpr.invoke_helper("set_legend", gather_from_fig=True)
    legend = get_legends(hlpr.ax)[0]
    assert get_texts(legend) == ["six", "three", "four", "five", "one", "two"]

    # Axes (0, 0) and (0, 1) should have large legends now
    print([get_len(lg) for lg in get_legends(hlpr.fig)])
    assert sum(get_len(lg) > 4 for lg in get_legends(hlpr.fig)) == 2

    # Can now conditionally hide them
    hlpr.invoke_helper("set_legend", axes="all", hiding_threshold=4)
    legends = get_legends(hlpr.fig)
    assert len(legends) == 4  # (0, 0) (0, 1) (1, 1) (2, 0)

    # ... but because we have legends on otherwise empty axes, the helper will
    # skip those axes and not hide the legend on axis (0, 0)
    assert [get_len(lg) for lg in legends] == [5, 3, 1, 2]

    # Apply it to all axes now, which will lead to (partly invisible) legends
    # on some axes, but the one at (0, 0) will now be hidden / empty
    hlpr.invoke_helper(
        "set_legend", axes="all", hiding_threshold=4, skip_empty_axes=False
    )
    legends = get_legends(hlpr.fig)
    assert len(legends) == 6  # all axes
    assert [get_len(lg) for lg in legends] == [0, 0, 3, 1, 2, 0]


def test_helper_set_figlegend(hlpr):
    """Tests the `set_figlegend` helper directly"""
    hlpr.setup_figure(ncols=3, nrows=2)

    h = hlpr.ax.plot([1, 2, 3], [1, 2, 3])
    hlpr.track_handles_labels(h, ["foo"])
    hlpr.invoke_helper("set_figlegend", gather_from_fig=True)

    # Can't do it twice
    with pytest.raises(ValueError, match="was already set"):
        hlpr.invoke_helper("set_figlegend")


# -----------------------------------------------------------------------------


def test_utils_set_tick_locators_or_formatters():
    """Tests dantro.plot.utils.mpl.set_tick_locators_or_formatters"""
    from dantro.plot.utils.mpl import set_tick_locators_or_formatters

    # Bad kind
    with pytest.raises(ValueError, match="Bad kind"):
        set_tick_locators_or_formatters(ax=None, kind="invalid_key")

    # ... rest is tested via other tests
