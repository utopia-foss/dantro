"""This module implements the PlotHelper class"""

import copy
import logging
import os
from collections import defaultdict
from itertools import product
from typing import Any, Callable, Dict, Generator, List, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from paramspace.tools import recursive_replace

from .._import_tools import LazyLoader
from ..exceptions import *
from ..tools import recursive_update

# Lazy loading modules that take a long time to import
sns = LazyLoader("seaborn")

# Public constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tools


class temporarily_changed_axis:
    """Context manager to temporarily change an axis in the
    :py:class:`~dantro.plot_creators._plot_helper.PlotHelper`
    """

    def __init__(
        self, hlpr: "PlotHelper", tmp_ax_coords: Tuple[int, int] = None
    ):
        """Initialize the context manager.

        Args:
            hlpr (PlotHelper): The plot helper of which to select a temporary
                axis
            tmp_ax_coords (Tuple[int], optional): The coordinates of the
                temporary axis. If not given, will **not** change the axis.
        """
        self._hlpr = hlpr
        self._tmp_ax_coords = tmp_ax_coords
        self._old_ax_coords = None

    def __enter__(self):
        """Enter the context, selecting a temporary axis"""
        # Store the current axis' coordinates
        self._old_ax_coords = self._hlpr.ax_coords

        # If it needs to be changed, do it.
        if (
            self._tmp_ax_coords is not None
            and self._tmp_ax_coords != self._old_ax_coords
        ):
            log.debug(
                "Temporarily changing from axis %s to %s ...",
                self._old_ax_coords,
                self._tmp_ax_coords,
            )
            self._hlpr.select_axis(*self._tmp_ax_coords)

        else:
            log.trace("No need to change current axis.")

    def __exit__(self, *args):
        """Change back to the initial axis. Errors are not handled."""
        if self._old_ax_coords != self._hlpr.ax_coords:
            log.debug("Changing back to axis %s ...", self._old_ax_coords)
            self._hlpr.select_axis(*self._old_ax_coords)


def coords_match(
    coords: Tuple[int], *, match: Union[tuple, str], full_shape: Tuple[int]
) -> bool:
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
            by ``full_shape``. Can also be 'all', which is equivalent to a
            (None, None) tuple. Can also be a list, which is then converted to
            a tuple.
        full_shape (Tuple[int]): The full shape of the axes; needed to wrap
            around negative values in ``match``.

    Returns:
        bool: Whether ``coords`` matches ``match``

    Raises:
        TypeError: ``match`` not being a tuple or a list
        ValueError: Any of the arguments not being 2-tuples.
    """
    # Convert the 'all argument'
    match = match if match != "all" else (Ellipsis, Ellipsis)

    # Make sure it is a tuple, allowing conversion from lists
    if isinstance(match, list):
        match = tuple(match)

    elif not isinstance(match, tuple):
        raise TypeError(
            "Argument `match` needs to be a 2-tuple, list, or a "
            "string, but was {} with value '{}'!"
            "".format(type(match), match)
        )

    # Convert any Nones to Ellipsis
    match = tuple([m if m is not None else Ellipsis for m in match])

    # Check length and values, not allowing values exceeding the shape
    if any([len(t) != 2 for t in (coords, match, full_shape)]):
        raise ValueError(
            "Need 2-tuples for arguments, got {}, {}, and {}!".format(
                coords, match, full_shape
            )
        )

    elif not all([m is Ellipsis or m < s for m, s in zip(match, full_shape)]):
        raise ValueError(
            "Got match values {} exceeding the shape {}! Take "
            "care that all values are strictly smaller than the "
            "maximum value. Negative values are allowed and will "
            "be evaluated via a modulo operation."
            "".format(match, full_shape)
        )

    for c, m, s in zip(coords, match, full_shape):
        if m is Ellipsis:
            # Always matches
            continue
        if (m % s) != c:
            # No match
            return False

    # Went through -> have a match
    return True


def remove_duplicate_handles_labels(h: list, l: list) -> Tuple[list, list]:
    """Returns new aligned lists of handles and labels from which duplicates
    (identified by label) are removed.

    This maintains the order and association by keeping track of seen items;
    see https://stackoverflow.com/a/480227/1827608 for more information.

    Args:
        h (list): List of artist handles
        l (list): List of labels

    Returns:
        Tuple[list, list]: handles and labels
    """
    seen = set()
    hls = [
        (_h, _l) for _h, _l in zip(h, l) if not (_l in seen or seen.add(_l))
    ]
    h, l = [hl[0] for hl in hls], [hl[1] for hl in hls]
    return h, l


def gather_handles_labels(mpo) -> Tuple[list, list]:
    """Uses ``.findobj`` to search a figure or axis for legend objects and
    returns lists of handles and (string) labels.
    """
    h, l = [], []
    for lg in mpo.findobj(mpl.legend.Legend):
        h += [_h for _h in lg.legendHandles]
        l += [_t.get_text() for _t in lg.texts]

    # Remove duplicates and return
    return remove_duplicate_handles_labels(h, l)


def prepare_legend_args(
    h, l, *, custom_labels: list, hiding_threshold: int
) -> Tuple[list, list, bool]:

    # Might want to use custom labels
    if custom_labels:
        log.remark("Using custom labels:  " + ", ".join(custom_labels))
        l = custom_labels

    # Evaluate the hiding threshold
    past_thresh = (
        hiding_threshold is not None and min(len(h), len(l)) > hiding_threshold
    )
    if past_thresh:
        log.remark(
            "With %d handles and %d labels, passed hiding threshold of %d.",
            len(h),
            len(l),
            hiding_threshold,
        )
    else:
        log.remark(
            "Have %d handles and %d labels available for the legend.",
            len(h),
            len(l),
        )

    return h, l, past_thresh


def calculate_space_needed_hv(
    fig, obj, *, spacing_h: float = 0.02, spacing_v: float = 0.02
) -> Tuple[float, float]:
    """Calculates the horizontal and vertical space needed for an object in
    this figure.

    .. note:: This will invoke ``fig.draw`` two times and resize the figure!

    Args:
        fig: The figure
        obj: The object in that figure to fit
        spacing_h (float, optional): Added to space needed (in inches)
        spacing_v (float, optional): Added to space needed (in inches)

    Returns:
        Tuple[float, float]: the horizontal and vertical space needed to fit
            into the figure (in inches).
    """

    def get_obj_width_height(obj) -> tuple:
        extent = obj.get_window_extent()
        return (extent.width / fig.dpi, extent.height / fig.dpi)

    # Draw the figure to have an object window
    fig.draw(fig.canvas.get_renderer())

    # Calculate and set the new width and height of the figure so the obj fits
    fig_width, fig_height = fig.get_figwidth(), fig.get_figheight()
    obj_width, obj_height = get_obj_width_height(obj)

    fig.set_figwidth(fig_width + obj_width)
    fig.set_figheight(fig_height + obj_height)

    # Draw again to get new object transformations
    fig.draw(fig.canvas.get_renderer())
    obj_width, obj_height = get_obj_width_height(obj)

    # Now calculate the needed space horizontally and vertically
    space_needed_h = obj_width / (fig_width + obj_width) + spacing_h
    space_needed_v = obj_height / (fig_height + obj_height) + spacing_v

    return space_needed_h, space_needed_v


# -----------------------------------------------------------------------------


class PlotHelper:
    """The PlotHelper takes care of the figure setup and saving and allows
    accessing matplotlib utilities through the plot configuration.
    """

    # Configuration keys with special meaning
    _SPECIAL_CFG_KEYS = ("setup_figure", "save_figure")

    # Helpers that are applied on the figure level
    _FIGURE_HELPERS = ("set_suptitle", "set_figlegend")

    def __init__(
        self,
        *,
        out_path: str,
        helper_defaults: dict = None,
        update_helper_cfg: dict = None,
        raise_on_error: bool = True,
        animation_enabled: bool = False,
    ):
        """Initialize a Plot Helper with a certain configuration.

        This configuration is the so-called "base" configuration and is not
        axis-specific. There is the possibility to specify axis-specific
        configuration entries.

        All entries in the helper configuration are deemed 'enabled' unless
        they explicitly specify `enabled: false` in their configuration.

        Args:
            out_path (str): path to store the created figure. This may be an
                absolute path or a relative path; the latter is regarded as
                relative to the current working directory. The home directory
                indicator ``~`` is expanded.
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
        self._AVAILABLE_HELPERS = [
            attr_name[6:]
            for attr_name in dir(self)
            if attr_name.startswith("_hlpr_")
            and callable(getattr(self, attr_name))
        ]
        self._AVAILABLE_HELPERS = tuple(self._AVAILABLE_HELPERS)

        # Store (a copy of) the base configuration
        self._base_cfg = (
            copy.deepcopy(helper_defaults) if helper_defaults else {}
        )

        # Update the defaults
        if update_helper_cfg:
            self._base_cfg = recursive_update(
                self._base_cfg, copy.deepcopy(update_helper_cfg)
            )

        # Extract the axis-specific update list
        self._axis_specific_updates = self._base_cfg.pop("axis_specific", {})

        # Check that all remaining entries are valid keys
        self._raise_on_invalid_helper_name(
            *self._base_cfg.keys(), special_cfg_keys_allowed=True
        )

        # Initialize the actual axis-specific configuration empty; it can only
        # be compiled once the figure is created.
        self._cfg = None

        # Store the other attributes
        self._out_path = os.path.expanduser(out_path)
        self._raise_on_error = raise_on_error
        self._animation_enabled = animation_enabled

        # Initialize attributes that are set at a later point
        self._fig = None
        self._axes = None
        self._current_ax_coords = None

        self._additional_axes = None
        self._handles_labels = defaultdict(dict)
        self._figlegend = None

        self._animation_update = None
        self._invoke_before_grab = False

        log.debug("PlotHelper initialized.")

    # .........................................................................
    # Properties

    @property
    def _axis_cfg(self) -> dict:
        """Return the configuration for the current axis; not a deep copy!"""
        try:
            return self._cfg[self.ax_coords]

        except RuntimeError as err:
            raise RuntimeError(
                "The axis-specific configuration is only available while a "
                "figure is associated with the PlotHelper!"
            ) from err

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
            raise ValueError(
                "No figure initialized or already closed! Use "
                "the `setup_figure` method to create a figure "
                "instance."
            )
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
            raise RuntimeError(
                "The current axis coordinate is only defined "
                "while a figure is associated with the "
                "PlotHelper!"
            )

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
        """Returns a list of enabled helpers *for the current axis*"""
        return [
            helper_name
            for helper_name in self._axis_cfg
            if self._axis_cfg[helper_name].get("enabled", True)
            and helper_name
            not in (self._SPECIAL_CFG_KEYS + self._FIGURE_HELPERS)
        ]

    @property
    def enabled_figure_helpers(self) -> list:
        """Returns a list of enabled figure-level helpers"""
        return [
            helper_name
            for helper_name in self._base_cfg
            if self._base_cfg[helper_name].get("enabled", True)
            and helper_name in self._FIGURE_HELPERS
        ]

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
            raise ValueError(
                "No animation update generator was registered "
                "with the PlotHelper! Cannot perform animation "
                "update."
            )
        return self._animation_update

    @property
    def invoke_before_grab(self) -> bool:
        """Whether the helpers are to be invoked before grabbing each frame of
        an animation.
        """
        return self._invoke_before_grab

    @property
    def raise_on_error(self) -> bool:
        """Whether the PlotHelper was configured to raise exceptions"""
        return self._raise_on_error

    @property
    def axis_handles_labels(self) -> Tuple[list, list]:
        """Returns the tracked axis handles and labels for the current axis"""
        return (
            self._handles_labels[self.ax_coords].get("handles", []),
            self._handles_labels[self.ax_coords].get("labels", []),
        )

    @property
    def all_handles_labels(self) -> Tuple[list, list]:
        """Returns all associated handles and labels"""
        handles, labels = [], []
        for hl in self._handles_labels.values():
            handles += hl.get("handles", [])
            labels += hl.get("labels", [])

        return handles, labels

    @property
    def axis_is_empty(self) -> bool:
        """Returns true if the current axis is empty, i.e. has no artists added
        to it. This is basically the inverse of ``mpl.axes.Axes.has_data``.
        """
        return not self.ax.has_data()

    # .........................................................................
    # Figure setup and axis control

    def attach_figure_and_axes(
        self, *, fig, axes, skip_if_identical: bool = False
    ) -> None:
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
            skip_if_identical (bool, optional): If True, will check if the
                given ``fig`` is *identical* to the already associated figure;
                if so, will do nothing. This can be useful if one cannot be
                sure if the figure was already associated. In such a case, note
                that the ``axes`` argument is completely ignored.

        Raises:
            ValueError: On multiple axes not being passed in 2d format.

        Returns:
            None

        """
        if skip_if_identical and fig is self._fig:
            log.debug("Figure was already associated; not doing anything.")
            return

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
            raise ValueError(
                "When attaching a figure with multiple axes, the "
                "axes must be passed as a 2d array-like object! "
                f"Got object of shape {axes.shape}."
            )

        # Everything ok, close figure and attach new figure and axes
        log.debug(
            "Closing existing figure and re-associating with a new figure ..."
        )
        self.close_figure()

        self._fig = fig
        self._axes = axes

        log.debug("Figure %d and axes attached.", fig.number)

        # Reset some tracking attributes
        self._handles_labels = defaultdict(dict)
        self._additional_axes = None
        self._figlegend = None

        # Can now evaluate the axis-specific configuration
        self._cfg = self._compile_axis_specific_cfg()

        # Select the (0, 0) axis, for consistency
        self.select_axis(0, 0)

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
        fig_kwargs = self.base_cfg.get("setup_figure", {})

        if update_fig_kwargs:
            fig_kwargs = recursive_update(fig_kwargs, update_fig_kwargs)

        # Need to handle scaling argument separately
        scale_figsize = fig_kwargs.pop(
            "scale_figsize_with_subplots_shape", False
        )

        # Now, create the figure and axes and attach them
        fig, axes = plt.subplots(squeeze=False, **fig_kwargs)
        log.debug("Figure %d created.", fig.number)

        self.attach_figure_and_axes(fig=fig, axes=axes)

        # Scale figure, if needed
        if scale_figsize and self.axes.size > 1:
            log.debug(
                "Scaling current figure size with subplots shape %s ...",
                self.axes.shape,
            )

            old_figsize = self.fig.get_size_inches()
            self.fig.set_size_inches(
                old_figsize[0] * self.axes.shape[0],
                old_figsize[1] * self.axes.shape[1],
            )

            log.debug(
                "Scaled figure size from %s to %s.",
                old_figsize,
                self.fig.get_size_inches(),
            )

    def save_figure(self, *, close: bool = True):
        """Saves and (optionally, but default) closes the current figure

        Args:
            close (bool, optional): Whether to close the figure after saving.
        """
        self.fig.savefig(self.out_path, **self.base_cfg.get("save_figure", {}))
        log.debug("Figure saved at: %s", self.out_path)

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
        self._additional_axes = None
        self._handles_labels = defaultdict(list)
        self._figlegend = None
        log.debug("Associated data removed.")

    def select_axis(self, col: int = None, row: int = None, *, ax=None):
        """Sets the current axis.

        Setting the axis can happen in three ways, depending on the arguments:

            - The axis object at the given ``col`` and ``row`` coordinates.
            - An explicitly given axis object (if ``ax`` is given)
            - The current axis (if all arguments are None)

        This method can be used to change to a different associated axis to
        continue plotting on that axis.

        Calling this method may also become necessary if the current axis is
        changed in a part of the program where the plot helper is not
        involved; in such a case, the currently selected axis may have been
        changed directly via the matplotlib interface.
        This method can then be used to synchronize the two again.

        .. note::

            The ``col`` and ``row`` values are wrapped around according to the
            shape of the associated ``axes`` array, thereby allowing to specify
            them as negative values for indexing from the back.

        Args:
            col (int, optional): The column to select, i.e. the x-coordinate.
                Can be negative, in which case it indexes backwards from the
                last column.
            row (int, optional): The row to select, i.e. the y-coordinate. Can
                be negative, in which case it indexes backwards from the last
                row.
            ax (optional): If given this axis object, tries to look it up from
                the associated axes array.

        Raises:
            ValueError: On failing to set the current axis or if the given axis
                object or the result of ``plt.gca()`` was not part of the
                associated axes array. To associate the correct figure and
                axes, use
                :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.attach_figure_and_axes`.

        """
        # Check arguments
        # Column and row need either both be given or both NOT be given. (XOR)
        if (col is None) != (row is None):
            raise ValueError(
                "Need both `col` and `row` arguments to select the current "
                f"axis via its coordinates! Got  col: {col},  row: {row}"
            )

        # Without any arguments, use the current axis for the lookup
        if col is None and row is None and ax is None:
            ax = plt.gca()

        # With an axis argument now given, it means that we need to retrieve
        # the col and row values via the axis object
        if ax is not None:
            # ... and we require that `col` and `row` are not given.
            if col is not None or row is not None:
                raise ValueError(
                    "Cannot specify arguments `col` and/or `row` if also "
                    "setting the `ax` argument for `select_axis`!"
                )

            # Look up column and row
            col, row = self._find_axis_coords(ax)

        # Wrap around negative values
        col = col if col >= 0 else col % self.axes.shape[0]
        row = row if row >= 0 else row % self.axes.shape[1]

        # Now ready to select
        log.debug("Selecting axis (%d, %d) ...", col, row)

        try:
            self.fig.sca(self.axes[col, row])

        except IndexError as exc:
            raise ValueError(
                f"Could not select axis ({col}, {row}) from figure with "
                f"associated axes array of shape {self.axes.shape}!"
            ) from exc

        else:
            self._current_ax_coords = (
                col % self.axes.shape[0],
                row % self.axes.shape[1],
            )

            log.debug(
                "Axis (%d, %d) selected. Enabled helpers: %s",
                self.ax_coords[0],
                self.ax_coords[1],
                ", ".join(self.enabled_helpers),
            )

    def coords_iter(
        self,
        *,
        match: Union[tuple, str] = None,
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
        for coords in product(
            range(self.axes.shape[0]), range(self.axes.shape[1])
        ):
            if coords_match(coords, match=match, full_shape=self.axes.shape):
                yield coords

    # .........................................................................
    # Helper invocation and configuration

    def _invoke_helper(
        self,
        helper_name: str,
        *,
        axes: Union[tuple, str] = None,
        mark_disabled_after_use: bool = True,
        raise_on_error: bool = None,
        **update_kwargs,
    ) -> None:
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
            raise_on_error (bool, optional): If given, overwrites the default
                controlled via the ``raise_on_error`` attribute.
            **update_kwargs: Update parameters for this specific plot helper.
                Note that these do not persist, but are only used for this
                invocation.

        Raises:
            PlotHelperErrors: On failing plot helper invocations
            ValueError: No matching helper function defined
        """
        is_figure_helper = helper_name in self._FIGURE_HELPERS

        def invoke_now(
            helper_name: str, errors: list, *, ax_coords: tuple = None
        ):
            """Invokes a single helper and keeps track of errors by mutably
            appending to the given ``errors`` list.
            """
            # Prepare the helper params, always working on a copy.
            # Depending on the helper, these can be either axis-specific or
            # figure-specific configurations.
            if not is_figure_helper:
                helper_params = self.axis_cfg.get(helper_name, {})
            else:
                helper_params = self.base_cfg.get(helper_name, {})

            if update_kwargs:
                recursive_update(helper_params, copy.deepcopy(update_kwargs))

            # If it was disabled, have nothing else to do here
            if not helper_params.pop("enabled", True):
                log.debug(
                    "Helper '%s' is not enabled for axis %s.",
                    helper_name,
                    self.ax_coords,
                )
                return

            # Also skip it if it is an axis-level helper and the axis is empty
            if (
                not is_figure_helper
                and helper_params.pop("skip_empty_axes", True)
                and self.axis_is_empty
            ):
                log.debug(
                    "Helper '%s' skipped for empty axis %s.",
                    helper_name,
                    self.ax_coords,
                )
                return

            # Invoke helper
            log.debug(
                "Invoking helper function '%s' on axis %s ...",
                helper_name,
                self.ax_coords,
            )

            try:
                helper(**helper_params)

            except Exception as exc:
                errors.append(
                    PlotHelperError(
                        exc,
                        name=helper_name,
                        ax_coords=ax_coords,
                        params=helper_params,
                    )
                )

            if mark_disabled_after_use:
                self.mark_disabled(helper_name)

        # Initial checks
        try:
            helper = getattr(self, "_hlpr_" + helper_name)

        except AttributeError as err:
            _available = ", ".join(self.available_helpers)
            raise PlotConfigError(
                f"No helper with name '{helper_name}' available! "
                f"Available helpers: {_available}"
            ) from err

        # A list to collect error messages in; is updated mutably.
        errors = []

        # If this is a figure-level helper, handle separately:
        if is_figure_helper:
            invoke_now(helper_name, errors)
            self._handle_errors(*errors, raise_on_error=raise_on_error)
            return

        # Go over all matching axis coordinates, temporarily change to an axis
        # and invoke the helper there. Keeps track of
        for ax_coords in self.coords_iter(match=axes):
            with temporarily_changed_axis(self, tmp_ax_coords=ax_coords):
                invoke_now(helper_name, errors)

        # Finally, handle the gathered errors
        self._handle_errors(*errors, raise_on_error=raise_on_error)

    def invoke_helper(
        self,
        helper_name: str,
        *,
        axes: Union[tuple, str] = None,
        mark_disabled_after_use: bool = True,
        raise_on_error: bool = None,
        **update_kwargs,
    ):
        """Invokes a single helper on the specified axes.

        Args:
            helper_name (str): The name of the helper to invoke
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): If True, the helper is
                marked as disabled after the invocation.
            raise_on_error (bool, optional): If given, overwrites the default
                controlled via the ``raise_on_error`` attribute.
            **update_kwargs: Update parameters for this specific plot helper.
                Note that these do not persist, but are only used for this
                invocation.

        Raises:
            PlotHelperErrors: On failing plot helper invocation
        """
        self._invoke_helper(
            helper_name,
            axes=axes,
            enabled=True,
            mark_disabled_after_use=mark_disabled_after_use,
            raise_on_error=raise_on_error,
            **update_kwargs,
        )

    def invoke_helpers(
        self,
        *helper_names,
        axes: Union[tuple, str] = None,
        mark_disabled_after_use: bool = True,
        raise_on_error: bool = None,
        **update_helpers,
    ):
        """Invoke all specified helpers on the specified axes.

        Args:
            *helper_names: The helper names to invoke
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): Whether to mark helpers
                disabled after they were used.
            raise_on_error (bool, optional): If given, overwrites the default
                controlled via the ``raise_on_error`` attribute.
            **update_helpers: Update parameters for all plot helpers.
                These have to be grouped under the name of the helper in order
                to be correctly associated. Note that these do not persist,
                but are only used for this invocation.

        Raises:
            PlotHelperErrors: On failing plot helper invocations
        """
        errors = []
        for helper_name in helper_names:
            try:
                self.invoke_helper(
                    helper_name,
                    axes=axes,
                    mark_disabled_after_use=mark_disabled_after_use,
                    raise_on_error=True,
                    **update_helpers.get(helper_name, {}),
                )

            except PlotHelperErrors as pherrs:
                errors += pherrs.errors

        self._handle_errors(*errors, raise_on_error=raise_on_error)

    def invoke_enabled(
        self,
        *,
        axes: Union[tuple, str] = None,
        mark_disabled_after_use: bool = True,
        raise_on_error: bool = None,
        **update_helpers,
    ):
        """Invokes all enabled helpers with their current configuration on the
        matching axes and all enabled figure-level helpers on the figure.

        Internally, this first invokes all figure-level helpers and then calls
        :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.invoke_helpers`
        with all enabled helpers for all axes matching the ``axes`` argument.

        .. note::

            When setting ``mark_disabled_after_use = False``, this will lead to
            figure-level helpers being invoked multiple times. As some of these
            helpers do not allow multiple invocation, invoking this method a
            second time *might* fail if not disabling them as part of the first
            call to this method.

        Args:
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
            mark_disabled_after_use (bool, optional): If True, the helper is
                marked as disabled after the invocation.
            raise_on_error (bool, optional): If given, overwrites the default
                controlled via the ``raise_on_error`` attribute.
            **update_helpers: Update parameters for all plot helpers.
                These have to be grouped under the name of the helper in order
                to be correctly associated. Note that these do not persist,
                but are only used for this invocation.

        Raises:
            PlotHelperErrors: On failing plot helper invocations
        """

        def invoke(*helper_names, errors: list):
            """Small helper to invoke the given helpers and keep track of
            errors via the given ``errors`` list (mutably changed)
            """
            try:
                self.invoke_helpers(
                    *helper_names,
                    mark_disabled_after_use=mark_disabled_after_use,
                    raise_on_error=True,
                    **update_helpers,
                )

            except PlotHelperErrors as pherrs:
                errors += pherrs.errors

        # Mutably extended list of errors
        errors = []

        # Handle figure-level helpers first ...
        invoke(*self.enabled_figure_helpers, errors=errors)

        # ... and then axis-level helpers, each on their own axis
        for ax_coords in self.coords_iter(match=axes):
            with temporarily_changed_axis(self, tmp_ax_coords=ax_coords):
                invoke(*self.enabled_helpers, errors=errors)

        self._handle_errors(*errors, raise_on_error=raise_on_error)

    def provide_defaults(
        self,
        helper_name: str,
        *,
        axes: Union[tuple, str] = None,
        mark_enabled: bool = True,
        **update_kwargs,
    ):
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
        self._raise_on_invalid_helper_name(
            helper_name, special_cfg_keys_allowed=True
        )

        # Special configuration keys and figure-level helpers are updating the
        # base configuration
        if helper_name in (self._SPECIAL_CFG_KEYS + self._FIGURE_HELPERS):
            if helper_name not in self._base_cfg:
                self._base_cfg[helper_name] = dict()

            # Do an in-place recursive update; can return afterwards
            recursive_update(
                self._base_cfg[helper_name], copy.deepcopy(update_kwargs)
            )
            log.debug(
                "Updated defaults for special configuration entry '%s'.",
                helper_name,
            )
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
            recursive_update(
                self._cfg[ax_coords][helper_name], copy.deepcopy(update_kwargs)
            )
            log.debug(
                "Updated axis-specific defaults for helper '%s' and axis %s.",
                helper_name,
                ax_coords,
            )

            if mark_enabled and helper_name not in self._SPECIAL_CFG_KEYS:
                self.mark_enabled(helper_name, axes=ax_coords)

    def mark_enabled(self, *helper_names, axes: Union[tuple, str] = None):
        """Marks the specified helpers as enabled for the specified axes.

        Args:
            *helper_names: Helpers to be enabled
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
        """
        # Check the helper names
        self._raise_on_invalid_helper_name(*helper_names)

        # Handle the figure-level helpers separately
        fig_helpers = [hn for hn in helper_names if hn in self._FIGURE_HELPERS]
        helper_names = [hn for hn in helper_names if hn not in fig_helpers]

        for helper_name in fig_helpers:
            if helper_name not in self._base_cfg:
                self._base_cfg[helper_name] = dict()

            self._base_cfg[helper_name]["enabled"] = True
            log.debug(
                "Marked figure-level helper '%s' as enabled.", helper_name
            )

        # Go over all selected axes
        for ax_coords in self.coords_iter(match=axes):
            # Go over the specified helper names
            for helper_name in helper_names:
                # Create the empty dict, if necessary
                if helper_name not in self._cfg[ax_coords]:
                    self._cfg[ax_coords][helper_name] = dict()

                self._cfg[ax_coords][helper_name]["enabled"] = True
                log.debug(
                    "Marked helper '%s' enabled for axis %s.",
                    helper_name,
                    ax_coords,
                )

    def mark_disabled(self, *helper_names, axes: Union[tuple, str] = None):
        """Marks the specified helpers as disabled for the specified axes.

        Args:
            *helper_names: Helpers to be disabled
            axes (Union[tuple, str], optional): A coordinate match tuple of
                the axes to invoke this helper on. If not given, will invoke
                only on the current axes.
        """
        # Check the helper names
        self._raise_on_invalid_helper_name(*helper_names)

        # Handle the figure-level helpers separately
        fig_helpers = [hn for hn in helper_names if hn in self._FIGURE_HELPERS]
        helper_names = [hn for hn in helper_names if hn not in fig_helpers]

        for helper_name in fig_helpers:
            if helper_name not in self._base_cfg:
                self._base_cfg[helper_name] = dict()

            self._base_cfg[helper_name]["enabled"] = False
            log.debug(
                "Marked figure-level helper '%s' as disabled.", helper_name
            )

        # Go over all selected axes
        for ax_coords in self.coords_iter(match=axes):
            # Go over the specified helper names
            for helper_name in helper_names:
                # Create the empty dict, if necessary
                if helper_name not in self._cfg[ax_coords]:
                    self._cfg[ax_coords][helper_name] = dict()

                self._cfg[ax_coords][helper_name]["enabled"] = False
                log.debug(
                    "Marked helper '%s' disabled for axis %s.",
                    helper_name,
                    ax_coords,
                )

    # .........................................................................
    # Methods to track artists (e.g. legend handles and labels)

    def track_handles_labels(self, handles: list, labels):
        """Keep track of legend handles and/or labels for the current axis.

        Args:
            handles (list): The handles to keep track of
            labels (list): The associated labels
        """
        if len(handles) != len(labels):
            raise ValueError(
                "Lists of handles and labels need to be of the same size but "
                f"were {len(handles)} != {len(labels)}, respectively."
            )

        if self.ax_coords not in self._handles_labels:
            self._handles_labels[self.ax_coords] = defaultdict(list)

        self._handles_labels[self.ax_coords]["handles"] += handles
        self._handles_labels[self.ax_coords]["labels"] += labels

        log.debug(
            "Axis %s: now tracking %d handles and labels (%d new).",
            self.ax_coords,
            len(self._handles_labels[self.ax_coords]["handles"]),
            len(handles),
        )

    # .........................................................................
    # Animation interface

    def register_animation_update(
        self,
        update_func: Callable,
        *,
        invoke_helpers_before_grab: bool = False,
    ):
        """Registers a generator used for animations.

        Args:
            update_func (Callable): Generator object over which is iterated
                over to create an animation. This needs
            invoke_helpers_before_grab (bool, optional): Whether to invoke all
                enabled plot helpers before grabbing a frame. This should be
                set to ``True`` if the animation update function overwrites the
                effects of the previously applied helpers.
        """
        self._animation_update = update_func
        self._invoke_before_grab = invoke_helpers_before_grab
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
        base_cfg = {
            k: v
            for k, v in self._base_cfg.items()
            if k not in self._SPECIAL_CFG_KEYS
        }
        # NOTE setup_figure and save_figure are handled separately

        # Go over all possible coordinates
        for ax_coords in self.coords_iter(match="all"):
            # Store a copy of the base configuration for these coordinates
            cfg[ax_coords] = copy.deepcopy(self._base_cfg)

            # Go over the list of updates and apply them
            for update_params in self._axis_specific_updates.values():
                if any(k in self._FIGURE_HELPERS for k in update_params):
                    raise PlotConfigError(
                        "Cannot set axis-specific configuration for figure-"
                        "level helpers. Remove any keys that refer to figure-"
                        f"level helpers ({', '.join(self._FIGURE_HELPERS)})\n"
                        f"Given parameters were:\n  {update_params}."
                    )

                # Work on a copy
                update_params = copy.deepcopy(update_params)

                # Extract the axis to update
                axis = update_params.pop("axis")

                # Check if there is a match
                if not coords_match(
                    ax_coords, match=axis, full_shape=self.axes.shape
                ):
                    # Nothing to update for this coordinate
                    continue

                # In-place update the parameter dict. Copies have been created
                # above already, so no need to do this here again.
                recursive_update(cfg[ax_coords], update_params)

        # Now created configurations for all axes and applied all updates
        return cfg

    def _raise_on_invalid_helper_name(
        self, *helper_names: str, special_cfg_keys_allowed: bool = False
    ):
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
            raise ValueError(
                "No helper with name '{}' available! "
                "Available helpers: {}. "
                "Special configuration keys ({}) {} allowed for "
                "this operation."
                "".format(
                    helper_name,
                    ", ".join(self.available_helpers),
                    ", ".join(self._SPECIAL_CFG_KEYS),
                    ("are also" if special_cfg_keys_allowed else "are NOT"),
                )
            )

    def _handle_errors(self, *errors, raise_on_error: bool = None):
        """Helper method to handle errors"""
        if not errors:
            return

        phe = PlotHelperErrors(*errors)
        if self._raise_on_error or raise_on_error:
            raise phe
        log.warning(phe)

    def _find_axis_coords(self, ax) -> Tuple[int, int]:
        """Find the coordinates of the given axis object in the axes array"""
        try:
            col, row = np.where(self.axes == ax)
            col, row = col[0], row[0]

        except Exception as exc:
            raise ValueError(
                "Could not find the given axis within the associated axes "
                "array! Is the correct figure associated?"
            ) from exc

        return col, row

    # -------------------------------------------------------------------------
    # -- Helper methods -------------------------------------------------------
    # -------------------------------------------------------------------------
    # .........................................................................
    # ... acting on the figure

    def _hlpr_set_suptitle(
        self,
        *,
        title: str = None,
        margin: float = 0.025,
        **title_kwargs,
    ):
        """Set the figure title, i.e. ``matplotlib.Figure.suptitle``.

        This figure-level helper automatically vertically adjusts the subplot
        sizes to fit the suptitle into the figure without overlapping. This is
        *not* done if the ``title`` string is empty or if the y-position is
        specified via the ``y`` argument. When repetitively invoking this
        helper on the same figure, the subplot sizes are re-adjusted each time.

        Args:
            title (str, optional): The title to be set
            margin (float, optional): An additional vertical margin between
                the figure and the suptitle.
            **title_kwargs: Passed on to ``fig.suptitle``
        """
        st = self.fig.suptitle(title, **title_kwargs)

        if title and not "y" in title_kwargs:
            # Make some figure adjustments such that it does not overlap with
            # the already existing parts.
            _, space_needed = calculate_space_needed_hv(self.fig, st)
            self.fig.subplots_adjust(top=(1.0 - (space_needed + margin)))

    def _hlpr_set_figlegend(
        self,
        *,
        gather_from_fig: bool = False,
        custom_labels: Sequence[str] = (),
        hiding_threshold: int = None,
        loc="center right",
        margin: float = 0.015,
        **legend_kwargs,
    ):
        """Sets a figure legend.

        As a source of handles and labels, uses all those tracked via
        :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.track_handles_labels`.
        Furthermore, ``gather_from_fig`` controls whether to additionally
        retrieve already existing handles and labels from any legend used
        within the figure, e.g. on all of the axes.

        For legend locations on the **right** side of the figure, this will
        additionally adjust the subplot shape to accomodate for the figure
        legend without overlapping. This is not done for other ``loc`` values.

        If no handles could be retrieved by the above procedure, no figure
        legend will be added.

        Args:
            gather_from_fig (bool, optional): Whether to extract figure handles
                and labels from already existing legends used in the figure.
            custom_labels (Sequence[str], optional): Custom labels to assign
                for the given handles.
            hiding_threshold (int, optional): If there are more handles or
                labels available than the threshold, no figure legend will be
                added.
            loc (str, optional): The location of the figure legend. Note that
                figure adjustments are done for a figure legend on the right
                side of the figure; these are not done if the figure is located
                somewhere else, which might lead to it overlapping with other
                axes.
            margin (float, optional): An additional horizontal margin between
                the figure legend and the axes.
            **legend_kwargs: Passed on to ``fig.legend``.

        Raises:
            RuntimeError: If a figure legend was already set via the helper.
        """
        # Ensure that this is not called multiple times, because it's quite
        # costly to do all of the below (and can really mess up the figure
        # if applied multiple times).
        if self._figlegend:
            raise RuntimeError(
                "A figure legend was already set for this figure! "
                "Will not set another one."
            )

        # Now get the handles and labels . . . . . . . . . . . . . . . . . . .
        h, l = self.all_handles_labels

        if gather_from_fig:
            _h, _l = gather_handles_labels(self.fig)
            log.debug(
                "Gathered %d handles and labels from the whole figure.",
                len(_h),
            )
            h += _h
            l += _l

        h, l = remove_duplicate_handles_labels(h, l)

        # Evaluate custom labels and hiding threshold
        h, l, past_thresh = prepare_legend_args(
            h,
            l,
            custom_labels=custom_labels,
            hiding_threshold=hiding_threshold,
        )
        if not h or past_thresh:
            return

        # Construct the legend and keep track of it
        figlegend = self.fig.legend(h, l, loc=loc, **legend_kwargs)
        self._figlegend = figlegend

        # Figure adjustments . . . . . . . . . . . . . . . . . . . . . . . . .
        # TODO Extend to all locs
        if "right" in loc:
            # These are necessary to have enough space for the legend.
            space_needed, _ = calculate_space_needed_hv(self.fig, figlegend)
            self.fig.subplots_adjust(right=(1.0 - (space_needed + margin)))

    # .........................................................................
    # ... acting on a single axis

    def _hlpr_set_title(self, *, title: str = None, **title_kwargs):
        """Sets the title of the current axis.

        Args:
            title (str, optional): The title to be set
            **title_kwargs: Passed on to ``ax.set_title``
        """
        self.ax.set_title(title, **title_kwargs)

    def _hlpr_set_labels(
        self,
        *,
        x: Union[str, dict] = None,
        y: Union[str, dict] = None,
        only_label_outer: bool = False,
    ):
        """Sets the x and y label of the current axis.

        Args:
            x (Union[str, dict], optional): Either the label as a string or
                a dict with key ``label``, where all further keys are passed on
                to plt.set_xlabel
            y (Union[str, dict], optional): Either the label as a string or
                a dict with key ``label``, where all further keys are passed on
                to plt.set_ylabel
            only_label_outer (bool, optional): If true, call ``ax.label_outer``
                such that only tick labels on "outer" axes are visible:
                x-labels are only kept for subplots on the last row; y-labels
                only for subplots on the first column. Note that this applies
                to both axes and may lead to existing axes being hidden.
        """

        def set_label(func: Callable, *, label: str = None, **label_kwargs):
            return func(label, label_kwargs)

        if x:
            x = x if not isinstance(x, str) else dict(label=x)
            set_label(self.ax.set_xlabel, **x)

        if y:
            y = y if not isinstance(y, str) else dict(label=y)
            set_label(self.ax.set_ylabel, **y)

        if only_label_outer:
            self.ax.label_outer()

    def _hlpr_set_limits(
        self,
        *,
        x: Union[Sequence[Union[float, str]], dict] = None,
        y: Union[Sequence[Union[float, str]], dict] = None,
    ):
        """Sets the x and y limit for the current axis.

        The ``x`` and ``y`` arguments can have the following form:

            - None:         Limits are not set
            - sequence:     Specify lower and upper values
            - dict:         Expecting keys ``lower`` and/or ``upper``

        The sequence or dict values can be:

            - None          Set automatically / do not set
            - numeric       Set to this value explicitly
            - ``min``       Set to the data minimum value on that axis
            - ``max``       Set to the data maximum value on that axis

        Args:
            x (Union[Sequence[Union[float, str]], dict], optional): The limits
                to set on the x-axis
            y (Union[Sequence[Union[float, str]], dict], optional): The limits
                to set on the y-axis
        """

        def parse_args(
            args: Union[Sequence[Union[float, str]], dict], *, ax
        ) -> Tuple[float, float]:
            """Parses the limit arguments."""

            def parse_arg(arg: Union[float, str]) -> Union[float, None]:
                """Parses a single limit argument to either be float or None"""
                if not isinstance(arg, str):
                    # Nothing to parse
                    return arg

                if arg == "min":
                    arg = ax.get_data_interval()[0]
                elif arg == "max":
                    arg = ax.get_data_interval()[1]
                else:
                    raise PlotConfigError(
                        f"Got an invalid str-type argument '{arg}' to the "
                        "set_limits helper. Allowed: 'min', 'max', None, or a "
                        "numerical value specifying the lower or upper limit."
                    )

                # Check that it is finite
                if not np.isfinite(arg):
                    raise PlotConfigError(
                        "Could not get a finite value from the axis data to "
                        "use for setting axis limits to 'min' or 'max', "
                        "presumably because the axis is still empty."
                    )

                return arg

            # Special case: dict
            if isinstance(args, dict):
                # Make sure there are only allowed keys
                if [k for k in args.keys() if k not in ("lower", "upper")]:
                    raise PlotConfigError(
                        "There are invalid keys present in a dict-type "
                        "argument to set_limits! Only accepting keys 'lower' "
                        f"and 'upper', but got: {args}"
                    )

                lower = args.get("lower", None)
                upper = args.get("upper", None)

            else:
                lower, upper = args  # ... assuming sequence of length 2

            # Parse individually, then return
            return (parse_arg(lower), parse_arg(upper))

        # Now set the limits, using the helper functions defined above
        if x is not None:
            self.ax.set_xlim(*parse_args(x, ax=self.ax.xaxis))

        if y is not None:
            self.ax.set_ylim(*parse_args(y, ax=self.ax.yaxis))

    def _hlpr_set_legend(
        self,
        *,
        use_legend: bool = True,
        gather_from_fig: bool = False,
        custom_labels: Sequence[str] = (),
        hiding_threshold: int = None,
        **legend_kwargs,
    ):
        """Sets a legend for the current axis.

        As a first step, this helper tries to extract all relevant legend
        handles and labels. If a legend was set previously and *no* handles and
        labels could be extracted in the typical way (i.e., using the
        ``ax.get_legend_handles_labels`` method) it will be attempted to
        retrieve them from existing ``matplotlib.legend.Legend`` objects on the
        current axis.
        If ``gather_from_fig`` is given, the *whole* figure will be inspected,
        regardless of whether handles were found previously.

        Additionally, all axis-specific handles and labels tracked via
        :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.track_handles_labels`
        will always be added.

        .. note::

            - If no handles can be found, the legend is hidden, also meaning
              that the ``legend_kwargs`` will not be passed on.
            - During gathering of handles and labels from the current axis or
              the figure, duplicates will be removed; duplicates are detected
              via their label strings, *not* via their handle.

        .. hint::

            To set a figure-level legend, use the ``set_figlegend`` helper.

        Args:
            use_legend (bool, optional): Whether to set a legend or not. If
                False, the legend will be removed.
            gather_from_fig (bool, optional): If set, will gather legend
                handles and labels from the whole figure. This can be useful to
                set if the relevant information is found on another axis or in
                a figure legend.
            custom_labels (Sequence[str], optional): If given, use these labels
                and associate them with existing labels. Note that if fewer
                labels are given than handles are available, those without a
                label will not be drawn.
            hiding_threshold (int, optional): If given, will hide legends
                that have more than this number of handles registered.
            **legend_kwargs: Passed on to ``ax.legend``
        """

        def hide_legend():
            legend = self.ax.legend((), fancybox=False, frameon=False)
            legend.set_visible(False)

        # Warn about usage of the old interface.
        # TODO Remove in Version 1.0
        if legend_kwargs.pop("use_figlegend", None):
            log.warning(
                "The `use_figlegend` argument is no longer supported "
                "for the `set_legend` helper. Use `set_figlegend` instead."
            )

        # Do explicit hiding first, don't need to do all the rest in this case
        if not use_legend:
            log.remark("Hiding legend ...")
            hide_legend()
            return

        # Try to get (dangling) handles and labels from the axis
        h, l = self.ax.get_legend_handles_labels()
        # NOTE Will be empty if ax.legend() was called already

        # Add those that have been tracked explicitly
        _h, _l = self.axis_handles_labels
        h += _h
        l += _l

        # If there were no handles available in this way, try to gather the
        # information from the current axis or the whole figure
        if not h or gather_from_fig:
            _h, _l = gather_handles_labels(
                self.fig if gather_from_fig else self.ax
            )
            log.debug(
                "Gathered %d handles and labels from %s.",
                len(_h),
                "the whole figure"
                if (not h and gather_from_fig)
                else "current axis",
            )

            h += _h
            l += _l

        # Remove potential duplicate handles (identified by the labels)
        h, l = remove_duplicate_handles_labels(h, l)

        # ... and evaluate the custom labels and hiding threshold
        h, l, past_thresh = prepare_legend_args(
            h,
            l,
            custom_labels=custom_labels,
            hiding_threshold=hiding_threshold,
        )

        # Hide or draw the legend
        if past_thresh or not h:
            hide_legend()

        else:
            self.ax.legend(h, l, **legend_kwargs)

    def _hlpr_set_texts(self, *, texts: Sequence[dict]):
        """Sets multiple text elements for the current axis.

        Args:
            texts (Sequence[dict]): The sequence of text dicts, that are
                passed to matplotlib.pyplot.text
        """
        for text_args in texts:
            self.ax.text(**text_args)

    def _hlpr_set_hv_lines(self, *, hlines: list = None, vlines: list = None):
        """Sets one or multiple horizontal or vertical lines.

        Args:
            hlines (list, optional): list of numeric positions of the lines or
                or list of dicts with key ``pos`` determining the position, key
                ``limits`` determining the relative limits of the line, and all
                additional arguments being passed on to the matplotlib
                function.
            vlines (list, optional): list of numeric positions of the lines or
                or list of dicts with key ``pos`` determining the position, key
                ``limits`` determining the relative limits of the line, and all
                additional arguments being passed on to the matplotlib
                function.
        """

        def set_line(
            func: Callable,
            *,
            pos: float,
            limits: tuple = (0.0, 1.0),
            **line_kwargs,
        ):
            try:
                pos = float(pos)

            except Exception as err:
                raise PlotConfigError(
                    f"Got non-numeric value '{pos}' for `pos` argument in "
                    "set_hv_lines helper!"
                )

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

    def _hlpr_set_scales(
        self, *, x: Union[str, dict] = None, y: Union[str, dict] = None
    ):
        """Sets the scales for the current axis

        The arguments are used to call ``ax.set_xscale`` or ``ax.set_yscale``,
        respectively.
        For string-like arguments, the value is directly used to set the scale
        for that axis, e.g. ``linear``, ``log``, ``symlog``.
        Otherwise, dict-like arguments are expected where a ``scale`` key is
        present and defines which type of scale to use. All further arguments
        are passed on; these are relevant for the symmetrical logarithmic
        scale, for example.

        For more information, see
        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xscale.html

        Args:
            x (Union[str, dict], optional): The scales to use on the x-axis
            y (Union[str, dict], optional): The scales to use on the y-axis
        """

        def set_scale(func: Callable, *, scale: str = None, **scale_kwargs):
            func(scale, **scale_kwargs)

        if x:
            x = x if not isinstance(x, str) else dict(scale=x)
            set_scale(self.ax.set_xscale, **x)

        if y:
            y = y if not isinstance(y, str) else dict(scale=y)
            set_scale(self.ax.set_yscale, **y)

    def _hlpr_set_ticks(self, *, x: dict = None, y: dict = None):
        """Sets the ticks for the current axis

        The arguments are used to call ``ax.set_xticks`` or ``ax.set_yticks``,
        and ``ax.set_xticklabels`` or ``ax.set_yticklabels``, respectively.
        The dict-like arguments may contain the keys ``major`` and/or
        ``minor``, referring to major or minor tick locations and labels,
        respectively.
        They should either be list-like, directly specifying the ticks'
        locations, or dict-like requiring a ``locs`` key that contains the
        ticks' locations and is passed on to matplotlib.axes.Axes.set_xticks
        as ``ticks`` argument. Further kwargs such as ``labels`` can be given
        and are passed on to matplotlib.axes.Axes.set_xticklabels.

        Example:

        .. code-block:: yaml

            set_ticks:
              x:
                major: [2, 4, 6, 8]
                minor:
                  locs: [1, 3, 5, 7, 9]
              y:
                major:
                  locs: [0, 1000, 2000, 3000]
                  labels: [0, 1k, 2k, 3k]
                  # ... further kwargs here specify label aesthetics


        For more information, see:

            - https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
            - https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html

        Args:
            x (Union[list, dict], optional): The ticks and optionally their
                labels to set on the x-axis
            y (Union[list, dict], optional): The ticks and optionally their
                labels to set on the y-axis
        """

        def set_ticks(
            func: Callable,
            *,
            ticks: list,
            minor: bool,
        ):
            func(ticks, minor=minor)

        def set_ticklabels(
            func: Callable, *, labels: list, **ticklabels_kwargs
        ):
            func(labels, **ticklabels_kwargs)

        def set_ticks_and_labels(*, axis: str, minor: bool, axis_cfg: dict):
            axis_cfg = (
                axis_cfg
                if not isinstance(axis_cfg, (list, tuple))
                else dict(locs=axis_cfg)
            )

            if axis_cfg.get("labels") and not axis_cfg.get("locs"):
                raise PlotConfigError(
                    "Labels can only be set through the `labels` argument "
                    "if their location is also given through the `locs` "
                    "argument. However, `locs` is not given."
                )

            # Get the axis specific
            if axis == "x":
                ticks_func = self.ax.set_xticks
                tickslabel_func = self.ax.set_xticklabels
            elif axis == "y":
                ticks_func = self.ax.set_yticks
                tickslabel_func = self.ax.set_yticklabels

            # Set the ticks
            set_ticks(ticks_func, ticks=axis_cfg.pop("locs"), minor=minor)

            # Set the tick labels, passing on the additional kwargs
            if axis_cfg.get("labels"):
                set_ticklabels(tickslabel_func, minor=minor, **axis_cfg)

        if x:
            if x.get("major"):
                set_ticks_and_labels(
                    axis="x", minor=False, axis_cfg=x["major"]
                )

            if x.get("minor"):
                set_ticks_and_labels(axis="x", minor=True, axis_cfg=x["minor"])

        if y:
            if y.get("major"):
                set_ticks_and_labels(
                    axis="y", minor=False, axis_cfg=y["major"]
                )

            if y.get("minor"):
                set_ticks_and_labels(axis="y", minor=True, axis_cfg=y["minor"])

    # .........................................................................
    # ... using seaborn

    def _hlpr_despine(self, **kwargs):
        """Despines the current *axis* using ``seaborn.despine``.

        To despine the whole *figure*, apply this helper to all axes.
        Refer to the seaborn documentation for available arguments:
        https://seaborn.pydata.org/generated/seaborn.despine.html

        Args:
            **kwargs: Passed on to ``seaborn.despine``.
        """
        if "fig" in kwargs:
            raise ValueError(
                "Got unexpected `fig` argument! To apply the `despine` helper "
                "to all axes, invoke the helper on each axis separately."
            )
        sns.despine(ax=self.ax, **kwargs)
