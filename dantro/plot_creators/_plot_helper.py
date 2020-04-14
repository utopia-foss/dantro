"""This module implements the PlotHelper class"""

import os
import copy
import logging
import inspect
from itertools import product
from typing import Union, Callable, Tuple, List, Dict, Generator

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dantro.tools import recursive_update

# Public constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tools

class temporarily_changed_axis:
    """Context manager to temporarily change an axis in the PlotHelper"""

    def __init__(self, hlpr, *, tmp_ax_coords: Tuple[int]=None):
        """Initialize the context manager.

        Args:
            hlpr: The plot helper of which to select a temporary axis
            tmp_ax_coords (Tuple[int]): The coordinates of the temporary axis.
                If not given, will not change the axis.
        """
        self._hlpr = hlpr
        self._tmp_ax_coords = tmp_ax_coords
        self._old_ax_coords = None

    def __enter__(self):
        """Enter the context, selecting a temporary axis"""
        # Store the current axis' coordinates
        self._old_ax_coords = self._hlpr.ax_coords

        # If it needs to be changed, do it.
        if (    self._tmp_ax_coords is not None
            and self._tmp_ax_coords != self._old_ax_coords):
            log.debug("Temporarily changing from axis %s to %s ...",
                      self._old_ax_coords, self._tmp_ax_coords)
            self._hlpr.select_axis(*self._tmp_ax_coords)

        else:
            log.debug("No need to change current axis.")

    def __exit__(self, *args):
        """Change back to the initial axis. Errors are not handled."""
        if self._old_ax_coords != self._hlpr.ax_coords:
            log.debug("Changing back to axis %s ...", self._old_ax_coords)
            self._hlpr.select_axis(*self._old_ax_coords)

def coords_match(coords: Tuple[int], *,
                 match: Union[tuple, str], full_shape: Tuple[int]) -> bool:
    """Whether a coordinate is matched by a coordinate match tuple.

    Allowed values in the coordinate match tuple are:
        * integers: regarded as coordinates. If negative or exceeding the full
            shape, these are wrapped around.
        * Ellipsis: matches all coordinates
        * None: alias for Ellipsis

    Args:
        coords (Tuple[int]): The coordinate to match
        match (Union[tuple, str]): The match tuple, where None is
            interpreted as an Ellipsis and negative values are wrapped around
            by `full_shape`. Can also be 'all', which is equivalent to a
            (None, None) tuple. Can also be a list, which is then converted to
            a tuple.
        full_shape (Tuple[int]): The full shape of the axes; needed to wrap
            around negative values in `match`.

    Returns:
        bool: Whether `coords` matches `match`

    Raises:
        TypeError: `match` not being a tuple or a list
        ValueError: Any of the arguments not being 2-tuples.
    """
    # Convert the 'all argument'
    match = match if match != 'all' else (Ellipsis, Ellipsis)

    # Make sure it is a tuple, allowing conversion from lists
    if isinstance(match, list):
        match = tuple(match)

    elif not isinstance(match, tuple):
        raise TypeError("Argument `match` needs to be a 2-tuple, list, or a "
                        "string, but was {} with value '{}'!"
                        "".format(type(match), match))

    # Convert any Nones to Ellipsis
    match = tuple([m if m is not None else Ellipsis for m in match])

    # Check length and values, not allowing values exceeding the shape
    if any([len(t) != 2 for t in (coords, match, full_shape)]):
        raise ValueError("Need 2-tuples for arguments, got {}, {}, and {}!"
                         "".format(coords, match, full_shape))

    elif not all([m is Ellipsis or m < s for m, s in zip(match, full_shape)]):
        raise ValueError("Got match values {} exceeding the shape {}! Take "
                         "care that all values are strictly smaller than the "
                         "maximum value. Negative values are allowed and will "
                         "be evaluated via a modulo operation."
                         "".format(match, full_shape))

    for c, m, s in zip(coords, match, full_shape):
        if m is Ellipsis:
            # Always matches
            continue
        if (m % s) != c:
            # No match
            return False

    # Went through -> have a match
    return True

class EnterAnimationMode(Exception):
    """An exception that is used to convey to any
    :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` or derived
    creator that animation mode is to be entered instead of a regular
    single-file plot.

    It can and should be invoked via
    :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.enable_animation`.

    This exception can be raised from within a plot function to dynamically
    decide whether animation should happen or not. Its counterpart is
    :py:exc:`~dantro.plot_creators._plot_helper.ExitAnimationMode`.
    """

class ExitAnimationMode(Exception):
    """An exception that is used to convey to any
    :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` or derived
    creator that animation mode is to be exited and a regular single-file plot
    should be carried out.

    It can and should be invoked via
    :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.disable_animation`.

    This exception can be raised from within a plot function to dynamically
    decide whether animation should happen or not. Its counterpart is
    :py:exc:`~dantro.plot_creators._plot_helper.ExitAnimationMode`.
    """


# -----------------------------------------------------------------------------

class PlotHelper:
    """The PlotHelper takes care of the figure setup and saving and allows
    accessing matplotlib utilities through the plot configuration.
    """

    # Configuration keys with special meaning
    _SPECIAL_CFG_KEYS = ('setup_figure', 'save_figure')

    def __init__(self, *, out_path: str,
                 helper_defaults: dict=None,
                 update_helper_cfg: dict=None,
                 raise_on_error: bool=True,
                 animation_enabled: bool=False):
        """Initialize a Plot Helper with a certain configuration.

        This configuration is the so-called "base" configuration and is not
        axis-specific. There is the possibility to specify axis-specific
        configuration entries.

        All entries in the helper configuration are deemed 'enabled' unless
        they explicitly specify `enabled: false` in their configuration.

        Args:
            out_path (str): path to store the created figure
            helper_defaults (dict, optional): The basic configuration of the
                helpers.
            update_helper_cfg (dict, optional): A configuration used to update
                the existing helper defaults
            raise_on_error (bool, optional): Whether to raise on an exception
                created on helper invocation or just log the error
            animation_enabled (bool, optional): Whether animation mode is
                enabled.
        """
        # Determine available helper methods, store it as tuple
        self._AVAILABLE_HELPERS = [attr_name[6:] for attr_name in dir(self)
                                   if attr_name.startswith('_hlpr_')
                                   and callable(getattr(self, attr_name))]
        self._AVAILABLE_HELPERS = tuple(self._AVAILABLE_HELPERS)

        # Store (a copy of) the base configuration
        self._base_cfg = (copy.deepcopy(helper_defaults) if helper_defaults
                          else {})

        # Update the defaults
        if update_helper_cfg:
            self._base_cfg = recursive_update(self._base_cfg,
                                              copy.deepcopy(update_helper_cfg))

        # Extract the axis-specific update list
        self._axis_specific_updates = self._base_cfg.pop('axis_specific', {})

        # Check that all remaining entries are valid keys
        self._raise_on_invalid_helper_name(*self._base_cfg.keys(),
                                           special_cfg_keys_allowed=True)

        # Initialize the actual axis-specific configuration empty; it can only
        # be compiled once the figure is created.
        self._cfg = None

        # Store the other attributes
        self._out_path = out_path
        self._raise_on_error = raise_on_error
        self._animation_enabled = animation_enabled

        # Initialize attributes that are set at a later point
        self._fig = None
        self._axes = None
        self._current_ax_coords = None
        self._additional_axes = None
        self._animation_update = None

    # .........................................................................
    # Properties

    @property
    def _axis_cfg(self) -> dict:
        """Return the configuration for the current axis; not a deep copy!"""
        try:
            return self._cfg[self.ax_coords]

        except RuntimeError as err:
            raise RuntimeError("The axis-specific configuration is only "
                               "available while a figure is associated with "
                               "the PlotHelper!") from err

    @property
    def axis_cfg(self) -> dict:
        """Returns a deepcopy of the current axis' configuration."""
        return copy.deepcopy(self._axis_cfg)

    @property
    def base_cfg(self) -> dict:
        """Returns a deepcopy of the base configuration, i.e. the configuration
        that is not axis-specific.
        """
        return copy.deepcopy(self._base_cfg)

    @property
    def fig(self):
        """Returns the current figure"""
        if self._fig is None:
            raise ValueError("No figure initialized or already closed! Use "
                             "the `setup_figure` method to create a figure "
                             "instance.")
        return self._fig

    @property
    def ax(self):
        """Returns the current axis of the associated figure"""
        return self.fig.gca()

    @property
    def ax_coords(self) -> Tuple[int]:
        """Returns the current axis coordinates within a subfigure in shape
        ``(col, row)``.

        For example, the ``(0, 0)`` coordinate refers to the top left subplot
        of the figure. ``(1, 2)`` is the axis object in the second column,
        third row.
        """
        if self._current_ax_coords is None:
            raise RuntimeError("The current axis coordinate is only defined "
                               "while a figure is associated with the "
                               "PlotHelper!")

        return self._current_ax_coords
        # NOTE There _would_ be the possiblity to use the matplotlib axis
        #      properties self.ax.numCol and .numRow, which store the column
        #      and row the axis was created in via plt.subplots. However, that
        #      that information gets lost as soon as an additional axis is
        #      added to the figure, e.g. when adding a color bar. Thus, the
        #      current axis coordinates need to be stored in an attribute.

    @property
    def axes(self) -> np.ndarray:
        """Returns the axes array, which is of shape (#cols, #rows).

        The (0, 0) axis refers to the top left subplot of the figure.
        """
        return self._axes

    @property
    def available_helpers(self) -> Tuple[str]:
        """List of available helper names"""
        return self._AVAILABLE_HELPERS

    @property
    def enabled_helpers(self) -> list:
        """Returns a list of enabled helpers for the current axis"""
        return [hn for hn in self._axis_cfg
                if self._axis_cfg[hn].get('enabled', True)
                and hn not in self._SPECIAL_CFG_KEYS]

    @property
    def out_path(self) -> str:
        """Returns the output path of the plot"""
        return self._out_path

    @property
    def animation_enabled(self) -> bool:
        """Whether animation mode is currently enabled or not"""
        return self._animation_enabled

    @property
    def animation_update(self) -> Callable:
        """Returns the animation update generator callable"""
        if self._animation_update is None:
            raise ValueError("No animation update generator was registered "
                             "with the PlotHelper! Cannot perform animation "
                             "update.")
        return self._animation_update


    # .........................................................................
    # Figure setup and axis control

    def attach_figure_and_axes(self, *, fig, axes):
        """Attaches the given figure and axes to the PlotHelper. This method
        replaces an existing figure and existing axes with the ones given.

        As the PlotHelper relies on axes being accessible via coordinate pairs,
        multiple axes must be passed as two-dimensional array-like. Since the
        axes are internally stored as numpy array, the axes-grid must be
        complete.

        Note that by closing the old figure the existing axis-specific config
        and all existing axes are destroyed. In other words: All information
        previously provided via the provide_defaults and the mark_* methods is
        lost. Therefore, if needed, it is recommended to call this method at
        the beginning of the plotting function.

        .. note::

            This function assumes multiple axes to be passed in (y,x) format
            (as e.g. returned by matplotlib.pyplot.subplots with squeeze set to
            False) and internally transposes the axes-grid such that afterwards
            it is accessible via (x,y) coordinates.

        Args:
            fig: The new figure which replaces the existing.
            axes: single axis or 2d array-like containing the axes

        Raises:
            ValueError: On multiple axes not being passed in 2d format.

        """
        log.debug("Closing existing figure and re-associating with a new "
                  "figure ...")
        self.close_figure()

        # Assign the new figure
        self._fig = fig

        # Prepare the new axis object
        try:
            # Assuming it's a scalar, np.reshape leads to np.array being called
            # on the object, thus allowing any scalar type. Only in cases where
            # an np.ndarray with size > 1 is given will this reshape operation
            # fail.
            axes = np.reshape(axes, (1, 1))

        except ValueError:
            # Else, assume array-like, containing the axes in (y,x) format.
            # Transpose the axes such that they are accessible in the (x,y)
            # format, which is used internally throughout the PlotHelper.
            axes = np.array(axes).T

        # Ensure correct shape
        if axes.ndim != 2:
            raise ValueError("When attaching a figure with multiple axes, the "
                             "axes must be passed as a 2d array-like object! "
                             "Got object of shape {}.".format(axes.shape))

        # Everything ok, attach axes
        self._axes = axes

        log.debug("Figure %d and axes attached.", fig.number)

        # Select the (0, 0) axis, for consistency
        self.select_axis(0, 0)

        # Can now evaluate the axis-specific configuration
        self._cfg = self._compile_axis_specific_cfg()

    def setup_figure(self, **update_fig_kwargs):
        """Sets up a matplotlib figure instance and axes with the given
        configuration (by calling matplotlib.pyplot.subplots) and attaches
        both to the PlotHelper.

        If the ``scale_figsize_with_subplots_shape`` option is enabled here,
        this method will also take care of scaling the figure accordingly.

        Args:
            **update_fig_kwargs: Parameters that are used to update the
                figure setup parameters stored in `setup_figure`.
        """
        # Prepare arguments
        fig_kwargs = self.base_cfg.get('setup_figure', {})

        if update_fig_kwargs:
            fig_kwargs = recursive_update(fig_kwargs, update_fig_kwargs)

        # Need to handle scaling argument separately
        scale_figsize = fig_kwargs.pop('scale_figsize_with_subplots_shape',
                                       False)

        # Now, create the figure and axes and attach them
        fig, axes = plt.subplots(squeeze=False, **fig_kwargs)
        log.debug("Figure %d created.", fig.number)

        self.attach_figure_and_axes(fig=fig, axes=axes)

        # Scale figure, if needed
        if scale_figsize and self.axes.size > 1:
            log.debug("Scaling current figure size with subplots shape %s ...",
                      self.axes.shape)

            old_figsize = self.fig.get_size_inches()
            self.fig.set_size_inches(old_figsize[0] * self.axes.shape[0],
                                     old_figsize[1] * self.axes.shape[1])

            log.debug("Scaled figure size from %s to %s.",
                      old_figsize, self.fig.get_size_inches())

    def save_figure(self, *, close: bool=True):
        """Saves and (optionally, but default) closes the current figure

        Args:
            close (bool, optional): Whether to close the figure after saving.
        """
        self.fig.savefig(self.out_path, **self.base_cfg.get('save_figure', {}))
        log.debug("Figure saved.")

        if close:
            self.close_figure()

    def close_figure(self):
        """Closes the figure and disassociates it from the helper. This method
        has no effect if no figure is currently associated.

        This also removes the axes objects and deletes the axis-specific
        configuration. All information provided via provide_defaults and the
        mark_* methods is lost.
        """
        if self._fig is None:
            log.debug("No figure currently associated; nothing to close.")
            return

        fignum = self.fig.number
        plt.close(self.fig)
        log.debug("Figure %d closed.", fignum)

        self._fig = None
        self._axes = None
        self._current_ax_coords = None
        self._cfg = None
        log.debug("Associated data removed.")

    def select_axis(self, col: int, row: int):
        """Selects the axes at the given coordinate as the current axis.

        This does not perform a check on whether the axis is valid or already
        set.

        Args:
            col (int): The column to select, i.e. the x-coordinate. Can be
                negative, in which case it indexes backwards from the last
                column.
            row (int): The row to select, i.e. the y-coordinate. Can be
                negative, in which case it indexes backwards from the last row.

        Raises:
            ValueError: On failing to set the current axis

        """
        log.debug("Selecting axis (%s, %s) ...", col, row)

        try:
            self.fig.sca(self.axes[col, row])

        except IndexError as exc:
            raise ValueError("Could not select axis ({}, {}) from figure with "
                             "subplots of shape {}!"
                             "".format(col, row, self.axes.shape)) from exc

        else:
            self._current_ax_coords = (col % self.axes.shape[0],
                                       row % self.axes.shape[1])

    def coords_iter(self, *, match: Union[tuple, str]=None,
                    ) -> Generator[Tuple[int], None, None]:
        """Returns a generator to iterate over all coordinates that match
        `match`.

        Args:
            match (Union[tuple, str]): The coordinates to match; those that do
                not match this pattern (evaluated by `coords_match` function)
                will not be yielded. If not given, will iterate only over the
                currently selected axis.

        Yields:
            Generator[Tuple[int], None, None]: The axis coordinates generator
        """
        # If no match argument is given, match only current coordinates
        match = match if match is not None else self.ax_coords

        # Go over the cartesian product ranges specified by the subplots shape
        for coords in product(range(self.axes.shape[0]),
                              range(self.axes.shape[1])):
            if coords_match(coords, match=match, full_shape=self.axes.shape):
                yield coords

    # .........................................................................
    # Helper invocation and configuration

    def _invoke_helper(self, helper_name: str, *,
                       axes: Union[tuple, str]=None,
                       mark_disabled_after_use: bool=True,
                       **update_kwargs) -> None:
        """Invokes a single helper on the specified axes, if it is enabled, and
        marks it disabled afterwards. The given update parameters are used to
        update the existing configuration.

        Unlike the public invoke_helper method, this method checks for whether
        the helper is enabled, while the public method automatically assumes
        it is meant to be enabled.

        Args:
            helper_name (str): helper which is invoked
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): If True, the helper is
                marked as disabled after invoking it
            **update_kwargs: Update parameters for this specific plot helper.
                Note that these do not persist, but are only used for this
                invocation.

        Raises:
            ValueError: No matching helper function defined

        Returns:
            None
        """
        # Get the helper function
        try:
            helper = getattr(self, "_hlpr_" + helper_name)

        except AttributeError as err:
            raise ValueError("No helper with name '{}' available! "
                             "Available helpers: {}"
                             "".format(helper_name,
                                       ", ".join(self.available_helpers))
                             ) from err

        # Go over all matching axis coordinates
        for ax_coords in self.coords_iter(match=axes):
            # Temporarily change to this axis
            with temporarily_changed_axis(self, tmp_ax_coords=ax_coords):
                # Prepare the helper params for this axis, working on a copy
                helper_params = self.axis_cfg.get(helper_name, {})

                if update_kwargs:
                    recursive_update(helper_params,
                                     copy.deepcopy(update_kwargs))

                # If it is disabled, go to the next iteration
                if not helper_params.pop('enabled', True):
                    log.debug("Helper '%s' is not enabled for axis %s.",
                              helper_name, self.ax_coords)
                    continue

                # Invoke helper
                log.debug("Invoking helper function '%s' on axis %s ...",
                          helper_name, self.ax_coords)

                try:
                    helper(**helper_params)

                except Exception as exc:
                    # Build an informative error message
                    hp_params = "\n".join(["   {}: {}".format(k, repr(v))
                                           for k, v in helper_params.items()])
                    hp_doc = inspect.getdoc(helper)

                    msg = ("A {} was raised during invocation of the '{}' "
                           "helper: {}.\n\nIt was invoked with the following "
                           "arguments:\n{}\n\nMake sure these arguments were "
                           "valid. You may want to consult the helper's "
                           "docstring:\n\n{}"
                           "".format(exc.__class__.__name__, helper_name, exc,
                                     hp_params, hp_doc))

                    # Either log or raise
                    if self._raise_on_error:
                        raise ValueError(msg) from exc
                    log.error(msg)

                if mark_disabled_after_use:
                    self.mark_disabled(helper_name)

            # Now back at previous axis, whatever happened above.

    def invoke_helper(self, helper_name: str, *,
                      axes: Union[tuple, str]=None,
                      mark_disabled_after_use: bool=True,
                      **update_kwargs):
        """Invokes a single helper on the specified axes.

        Args:
            helper_name (str): The name of the helper to invoke
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): If True, the helper is
                marked as disabled after the invocation.
            **update_kwargs: Update parameters for this specific plot helper.
                Note that these do not persist, but are only used for this
                invocation.
        """
        self._invoke_helper(helper_name, axes=axes, enabled=True,
                            mark_disabled_after_use=mark_disabled_after_use,
                            **update_kwargs)

    def invoke_helpers(self, *helper_names,
                       axes: Union[tuple, str]=None,
                       mark_disabled_after_use: bool=True,
                       **update_helpers):
        """Invoke all specified helpers on the specified axes.

        Args:
            *helper_names: The helper names to invoke
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): Whether to mark helpers
                disabled after they were used.
            **update_helpers: Update parameters for all plot helpers.
                These have to be grouped under the name of the helper in order
                to be correctly associated. Note that these do not persist,
                but are only used for this invocation.

        Deleted Parameters:
            helper_names (list): Helpers to be invoked
        """
        for helper_name in helper_names:
            self.invoke_helper(helper_name, axes=axes,
                               mark_disabled_after_use=mark_disabled_after_use,
                               **update_helpers.get(helper_name, {}))

    def invoke_enabled(self, *,
                       axes: Union[tuple, str]=None,
                       **update_helpers):
        """Invokes all enabled helpers with their current configuration on the
        matching axes.

        Args:
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            **update_helpers: Update parameters for all plot helpers.
                These have to be grouped under the name of the helper in order
                to be correctly associated. Note that these do not persist,
                but are only used for this invocation.
        """
        # Go over all axes matching the given argument
        for ax_coords in self.coords_iter(match=axes):
            # Temporarily change to this axis
            with temporarily_changed_axis(self, tmp_ax_coords=ax_coords):
                # Iterate over all enabled helpers on this axis
                for helper_name in self.enabled_helpers:
                    # See if there are update parameters for this helper
                    params = update_helpers.get(helper_name, {})

                    # Invoke the single helper on the current axis
                    self._invoke_helper(helper_name, **params)

    def provide_defaults(self, helper_name: str, *,
                         axes: Union[tuple, str]=None,
                         mark_enabled: bool=True, **update_kwargs):
        """Update or add a single entry to a helper's configuration.

        Args:
            helper_name (str): The name of the helper whose config is to change
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_enabled (bool, optional): Whether to mark the helper enabled
                by providing defaults
            **update_kwargs: dict containing the helper parameters with
                which the config is updated recursively
        """
        self._raise_on_invalid_helper_name(helper_name,
                                           special_cfg_keys_allowed=True)

        # Special configuration keys are updating the base configuration
        if helper_name in self._SPECIAL_CFG_KEYS:
            if helper_name not in self._base_cfg:
                self._base_cfg[helper_name] = dict()

            # Do an in-place recursive update; can return afterwards
            recursive_update(self._base_cfg[helper_name],
                             copy.deepcopy(update_kwargs))
            log.debug("Updated defaults for special configuration entry '%s'.",
                      helper_name)
            return

        # Go over all selected axes
        for ax_coords in self.coords_iter(match=axes):
            # Update or create the configuration entry, manually passing the
            # coordinates. Using the _axis_cfg property would require to enter
            # the temporary axis context, which is overkill at this point.
            if helper_name not in self._cfg[ax_coords]:
                # Create a new empty entry
                self._cfg[ax_coords][helper_name] = dict()

            # Can do an in-place recursive update now
            recursive_update(self._cfg[ax_coords][helper_name],
                             copy.deepcopy(update_kwargs))
            log.debug("Updated axis-specific defaults for helper '%s' and "
                      "axis %s.", helper_name, ax_coords)

            if mark_enabled and helper_name not in self._SPECIAL_CFG_KEYS:
                self.mark_enabled(helper_name, axes=ax_coords)

    def mark_enabled(self, *helper_names, axes: Union[tuple, str]=None):
        """Marks the specified helpers as enabled for the specified axes.

        Args:
            *helper_names: Helpers to be enabled
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
        """
        # Check the helper names
        self._raise_on_invalid_helper_name(*helper_names)

        # Go over all selected axes
        for ax_coords in self.coords_iter(match=axes):
            # Go over the specified helper names
            for helper_name in helper_names:
                # Create the empty dict, if necessary
                if helper_name not in self._cfg[ax_coords]:
                    self._cfg[ax_coords][helper_name] = dict()

                self._cfg[ax_coords][helper_name]['enabled'] = True
                log.debug("Marked helper '%s' enabled for axis %s.",
                          helper_name, ax_coords)

    def mark_disabled(self, *helper_names, axes: Union[tuple, str]=None):
        """Marks the specified helpers as disabled for the specified axes.

        Args:
            *helper_names: Helpers to be disabled
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
        """
        # Check the helper names
        self._raise_on_invalid_helper_name(*helper_names)

        # Go over all selected axes
        for ax_coords in self.coords_iter(match=axes):
            # Go over the specified helper names
            for helper_name in helper_names:
                # Create the empty dict, if necessary
                if helper_name not in self._cfg[ax_coords]:
                    self._cfg[ax_coords][helper_name] = dict()

                self._cfg[ax_coords][helper_name]['enabled'] = False
                log.debug("Marked helper '%s' disabled for axis %s.",
                          helper_name, ax_coords)

    # .........................................................................
    # Animation interface

    def register_animation_update(self, update_func: Callable):
        """Registers a generator used for animations.

        Args:
            update_func (Callable): Generator object over which is iterated
                over to create an animation. This needs
        """
        self._animation_update = update_func
        log.debug("Registered animation update function.")

    def enable_animation(self):
        """Can be invoked to enter animation mode. An action is only performed
        if the helper is **not** currently in animation mode.

        Raises:
            EnterAnimationMode: Conveys to the plot creator that animation
                mode is to be entered.
        """
        if not self.animation_enabled:
            log.debug("Enabling animation mode ...")
            self._animation_enabled = True
            raise EnterAnimationMode()
        log.debug("Animation mode is already enabled.")

    def disable_animation(self):
        """Can be invoked to exit animation mode. An action is only performed
        if the helper **is** currently in animation mode.

        Raises:
            ExitAnimationMode: Conveys to the plot creator that animation mode
                should be left.
        """
        if self.animation_enabled:
            log.debug("Disabling animation mode ...")
            self._animation_enabled = False
            raise ExitAnimationMode()
        log.debug("Animation mode is already disabled.")

    # .........................................................................
    # Private support methods

    def _compile_axis_specific_cfg(self) -> Dict[tuple, dict]:
        """With a figure set up, compiles the axis-specific helper config."""
        # The target dict of axis coordinates to fully resolved dicts
        cfg = dict()

        # Compile a base configuration dict that does not contain special keys
        base_cfg = {k: v for k, v in self._base_cfg.items()
                    if k not in self._SPECIAL_CFG_KEYS}
        # NOTE setup_figure and save_figure are handled separately

        # Go over all possible coordinates
        for ax_coords in self.coords_iter(match='all'):
            # Store a copy of the base configuration for these coordinates
            cfg[ax_coords] = copy.deepcopy(self._base_cfg)

            # Go over the list of updates and apply them
            for update_params in self._axis_specific_updates.values():
                # Work on a copy
                update_params = copy.deepcopy(update_params)

                # Extract the axis to update
                axis = update_params.pop('axis')

                # Check if there is a match
                if not coords_match(ax_coords, match=axis,
                                    full_shape=self.axes.shape):
                    # Nothing to update for this coordinate
                    continue

                # In-place update the parameter dict. Copies have been created
                # above already, so no need to do this here again.
                recursive_update(cfg[ax_coords], update_params)

        # Now created configurations for all axes and applied all updates
        return cfg

    def _raise_on_invalid_helper_name(self, *helper_names: str,
                                      special_cfg_keys_allowed: bool=False):
        """Makes sure the given helper names are valid.

        Args:
            *helper_names (str): Names of the helpers to check
            special_cfg_keys_allowed (bool, optional): Whether to regard the
                special configuration keys as valid or not.

        Raises:
            ValueError: On invalid helper name
        """
        valid_names = self.available_helpers

        if special_cfg_keys_allowed:
            valid_names += self._SPECIAL_CFG_KEYS

        for helper_name in helper_names:
            if helper_name in valid_names:
                # Is valid
                continue

            # Not valid!
            raise ValueError("No helper with name '{}' available! "
                             "Available helpers: {}. "
                             "Special configuration keys ({}) {} allowed for "
                             "this operation."
                             "".format(helper_name,
                                       ", ".join(self.available_helpers),
                                       ", ".join(self._SPECIAL_CFG_KEYS),
                                       ("are also" if special_cfg_keys_allowed
                                        else "are NOT")))

    # .........................................................................
    # Helper Methods -- acting on the figure

    def _hlpr_set_suptitle(self, *, title: str=None, **title_kwargs):
        """Set the figure title, i.e. matplotlib.Figure.suptitle

        Args:
            title (str, optional): The title to be set
            **title_kwargs: Passed on to plt.set_title
        """
        if title:
            self.fig.suptitle(title, **title_kwargs)


    # .........................................................................
    # Helper Methods -- acting on a single axis

    def _hlpr_set_title(self, *, title: str=None, **title_kwargs):
        """Set the title of the current axis

        Args:
            title (str, optional): The title to be set
            **title_kwargs: Passed on to plt.set_title
        """
        if title:
            self.ax.set_title(title, **title_kwargs)

    def _hlpr_set_labels(self, *,
                         x: Union[str, dict]=None,
                         y: Union[str, dict]=None,
                         only_label_outer: bool=False):
        """Set the x and y label of the current axis

        Args:
            x (Union[str, dict], optional): Either the label as a string or
                a dict with key `label`, where all further keys are passed on
                to plt.set_xlabel
            y (Union[str, dict], optional): Either the label as a string or
                a dict with key `label`, where all further keys are passed on
                to plt.set_ylabel
            only_label_outer (bool, optional): Whether to label only outer axes
        """

        def set_label(func: Callable, *, label: str=None, **label_kwargs):
            # NOTE Can be extended here in the future to do more clever things
            return func(label, label_kwargs)

        if x:
            x = x if not isinstance(x, str) else dict(label=x)
            set_label(self.ax.set_xlabel, **x)

        if y:
            y = y if not isinstance(y, str) else dict(label=y)
            set_label(self.ax.set_ylabel, **y)

        if only_label_outer:
            self.ax.label_outer()

    def _hlpr_set_limits(self, *,
                         x: Union[tuple, dict]=None,
                         y: Union[tuple, dict]=None):
        """Set the x and y limit for the current axis

        x and y can have the following shapes:
            None           Limits are not set
            tuple, list    Specify lower and upper values
            dict           expecting keys `lower` and/or `upper`

        Each entries of the tuple or dict values can be:
            None           Set automatically / do not set
            numeric        Set to this value explicitly
            min            Set to the data minimum value on that axis
            max            Set to the data maximum value on that axis

        Args:
            x (Union[tuple, dict], optional): Set the x-axis limits. For valid
                argument values, see above.
            y (Union[tuple, dict], optional): Set the y-axis limits. For valid
                argument values, see above.
        """
        def parse_args(args: Union[tuple, dict], *, ax):
            """Parses the limit arguments."""

            def parse_arg(arg: Union[float, str]) -> Union[float, None]:
                """Parses a single limit argument to either be float or None"""
                if not isinstance(arg, str):
                    # Nothing to parse
                    return arg

                if arg == 'min':
                    arg = ax.get_data_interval()[0]
                elif arg == 'max':
                    arg = ax.get_data_interval()[1]
                else:
                    raise ValueError("Got an invalid str-type argument '{}' "
                                     "to set_limits helper. Allowed: min, max."
                                     "".format(arg))

                # Check that it is finite
                if not np.isfinite(arg):
                    raise ValueError("Could not get a finite value from the "
                                     "axis data to use for setting axis "
                                     "limits to 'min' or 'max', presumably "
                                     "because the axis is still empty.")

                return arg

            # Special case: dict
            if isinstance(args, dict):
                # Make sure there are only allowed keys
                if [k for k in args.keys() if k not in ('lower', 'upper')]:
                    raise ValueError("There are invalid keys present in a "
                                     "dict-type argument to set_limits! Only "
                                     "accepting keys 'lower' and 'upper', but "
                                     "got: {}".format(args))

                # Unpack into tuple
                args = (args.get('lower', None), args.get('upper', None))


            # Make sure it is a list or tuple of size 2
            if not isinstance(args, (tuple, list)):
                raise TypeError("Argument for set_limits helper needs to be "
                                "a dict, list, or a tuple, but was of type {} "
                                "with value '{}'!"
                                "".format(type(args), args))

            if len(args) != 2:
                raise ValueError("Argument for set_limits helper needs to be "
                                 "a list or tuple of length 2 or a dict with "
                                 "keys 'upper' and/or 'lower', but was {}!"
                                 "".format(args))

            # Parse and return
            return (parse_arg(args[0]), parse_arg(args[1]))

        # Now set the limits, using the helper functions defined above
        if x is not None:
            self.ax.set_xlim(*parse_args(x, ax=self.ax.xaxis))

        if y is not None:
            self.ax.set_ylim(*parse_args(y, ax=self.ax.yaxis))

    def _hlpr_set_legend(self, *, use_legend: bool=True, **legend_kwargs):
        """Set a legend for the current axis"""
        if use_legend:
            handles, labels = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, labels, **legend_kwargs)

    def _hlpr_set_texts(self, *, texts: list):
        """Set a list of texts for the current axis

        Args:
            texts: The list of text dicts, that are passed to
            matplotlib.pyplot.text
        """
        for text_args in texts:
            self.ax.text(**text_args)

    def _hlpr_set_hv_lines(self, *, hlines: list=None, vlines: list=None):
        """Set one or multiple horizontal or vertical lines.

        Args:
            hlines (list, optional): list of numeric positions of the lines or
                or list of dicts with key `pos` determining the position, key
                `limits` determining the relative limits of the line, and all
                additional arguments being passed on to the matplotlib
                function.
            vlines (list, optional): list of numeric positions of the lines or
                or list of dicts with key `pos` determining the position, key
                `limits` determining the relative limits of the line, and all
                additional arguments being passed on to the matplotlib
                function.
        """

        def set_line(func: Callable, *, pos: float, limits: tuple=(0., 1.),
                     **line_kwargs):
            """Helper function to invoke the matplotlib function that sets
            a horizontal or vertical line."""
            try:
                pos = float(pos)

            except Exception as err:
                raise ValueError("Got non-numeric value '{}' for `pos` "
                                 "argument in set_hv_lines helper!"
                                 "".format(pos))

            func(pos, *limits, **line_kwargs)

        if hlines is not None:
            for line_spec in hlines:
                if isinstance(line_spec, dict):
                    set_line(self.ax.axhline, **line_spec)
                else:
                    set_line(self.ax.axhline, pos=line_spec)

        if vlines is not None:
            for line_spec in vlines:
                if isinstance(line_spec, dict):
                    set_line(self.ax.axvline, **line_spec)
                else:
                    set_line(self.ax.axvline, pos=line_spec)

    def _hlpr_set_scales(self, *,
                         x: Union[str, dict]=None,
                         y: Union[str, dict]=None):
        """Set a scale for the current axis"""

        def set_scale(func: Callable, *, scale: str=None, **scale_kwargs):
            func(scale, **scale_kwargs)

        if x:
            x = x if not isinstance(x, str) else dict(scale=x)
            set_scale(self.ax.set_xscale, **x)

        if y:
            y = y if not isinstance(y, str) else dict(scale=y)
            set_scale(self.ax.set_yscale, **y)
