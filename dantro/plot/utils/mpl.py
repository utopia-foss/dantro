"""Matplotlib utilities which are used in
:py:class:`~dantro.plot.plot_helper.PlotHelper` or other places.
"""

import logging
from itertools import chain
from typing import List, Tuple

from .color_mngr import ColorManager

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class figure_leak_prevention:
    """Context manager that aims to prevent superfluous matplotlib figures
    persisting beyond the context. Such figure objects can aggregate and start
    memory issues or even representation errors.

    Specifically, it does the following:

        * When entering, stores all current figure numbers
        * When exiting regularly, all figures that were opened within the
          context are closed, except the currently selected figure.
        * When exiting with an exception, the behaviour is the same, unless the
          ``close_current_fig_on_raise`` is set, in which case the currently
          selected figure is **not** excluded from closing.

    """

    def __init__(self, *, close_current_fig_on_raise: bool = False):
        """Initialize the context manager

        Args:
            close_current_fig_on_raise (bool, optional): If True, the
                currently selected figure will **not** be exempt from the
                figure closure in case an exception occurs. This flag has no
                effect when the context is exited without an exception.
        """
        self._fignums = None
        self._close_current = close_current_fig_on_raise

    def __enter__(self):
        """Upon entering, store all currently open figure numbers"""
        import matplotlib.pyplot as plt

        self._fignums = plt.get_fignums()
        log.trace(
            "Entering figure_leak_prevention context. Open figures: %s",
            self._fignums,
        )

    def __exit__(self, exc_type: type, *args) -> None:
        """Iterates over all currently open figures and closes all figures
        that were not previously open, except the currently selected figure.

        If an exception is detected, i.e. ``exc_type` is **not** None, the
        current figure is only closed if the context manager was entered with
        the ``close_current_fig_on_raise`` flag set.
        """
        import matplotlib.pyplot as plt

        log.trace(
            "Exiting figure_leak_prevention (exception: %s) ...", exc_type
        )

        # Determine whether to exclude the current figure or not
        exclude_current = not self._close_current or exc_type is None
        cfn = plt.gcf().number
        log.trace("  Current figure: %d", cfn)

        for n in plt.get_fignums():
            if n in self._fignums or (exclude_current and n == cfn):
                continue
            plt.close(n)
            log.trace("  Closed figure %d.", n)


def calculate_space_needed_hv(
    fig: "matplotlib.figure.Figure",
    obj,
    *,
    spacing_h: float = 0.02,
    spacing_v: float = 0.02,
) -> Tuple[float, float]:
    """Calculates the horizontal and vertical space needed for an object in
    this figure.

    .. note::

        This will invoke :py:meth:`matplotlib.figure.Figure.draw` two times
        and cause resizing of the figure!

    Args:
        fig (matplotlib.figure.Figure): The figure
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
    import matplotlib as mpl

    h, l = [], []
    for lg in mpo.findobj(mpl.legend.Legend):
        h += [_h for _h in lg.legendHandles]
        l += [_t.get_text() for _t in lg.texts]

    # Remove duplicates and return
    return remove_duplicate_handles_labels(h, l)


def prepare_legend_args(
    h, l, *, custom_labels: List[str], hiding_threshold: int
) -> Tuple[list, list, bool]:
    """A utility function that allows setting custom legend handles and
    implements some logic to hide all handles if there are too many."""
    import matplotlib as mpl

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


def set_tick_locators_or_formatters(
    *,
    ax: "matplotlib.axes.Axes",
    kind: str,
    x: dict = None,
    y: dict = None,
    z: dict = None,
):
    """Sets the tick locators or formatters.
    Look at the :py:class:`~dantro.plot.plot_helper.PlotHelper` methods
    ``_hlpr_set_tick_{locators/formatters}`` for more information.

    Names are looked up in the :py:mod:`matplotlib.ticker` and
    :py:mod:`matplotlib.dates` modules.

    Args:
        ax (matplotlib.axes.Axes): The axes object
        kind (str): Whether to set a ``locator`` or a ``formatter``.
        x (dict, optional): The config for the x-axis tick locator/formatter
        y (dict, optional): The config for the y-axis tick locator/formatter
        z (dict, optional): The config for the z-axis tick locator/formatter
    """
    # Safe guard against calling this with unexpected arguments from
    # within the actual helper methods; not part of public interface.
    if kind not in ("locator", "formatter"):
        raise ValueError(f"Bad kind: {kind}")

    def _set_locator_or_formatter(
        *,
        _ax: "matplotlib.axes.Axes",
        _axis: str,
        _major: bool,
        _kind: str,
        name: str,
        args: tuple = (),
        **kwargs,
    ):
        """Set the tick locator or formatter on a specific axis.

        Args:
            _ax (matplotlib.axes.Axes): The axes object to work on.
            _axis (str):    The axis name, ``x`` or ``y``
            _major (bool):  Whether to set the major or minor ticks
            _kind (str):    The kind of function to set: ``locator`` or
                ``formatter``
            name (str):     The name of the locator or formatter, looked up in
                :py:mod:`matplotlib.ticker` and :py:mod:`matplotlib.dates`.
            args (tuple):   Args passed on to the respective locator
                            or formatter setter function.
            **kwargs:       Kwargs passed on to the respective locator
                            or formatter setter function.
        """
        import matplotlib as mpl

        # Get the ticker from the name, looking it up in both the mpl.ticker
        # and the mpl.dates modules
        try:
            ticker = getattr(mpl.ticker, name)
        except AttributeError as err:
            try:
                ticker = getattr(mpl.dates, name)
            except AttributeError as err:
                # Customize the error message for (i) no name (ii) wrong name
                # for locators and formatters.
                _avail = ", ".join(
                    s
                    for s in chain(dir(mpl.ticker), dir(mpl.dates))
                    if _kind.capitalize() in s
                )
                raise AttributeError(
                    f"The given {_kind} name '{name}' is not valid! "
                    f"Choose from: {_avail}"
                ) from err

        # Get the locator or formatter function for the respective
        # major or minor axis.
        ax_obj = getattr(_ax, f"{_axis}axis")
        setter = getattr(
            ax_obj, f"set_{'major' if _major else 'minor'}_{_kind}"
        )

        try:
            setter(ticker(*args, **kwargs))
        except Exception as exc:
            raise ValueError(
                f"Failed setting {'major' if _major else 'minor'} {_kind} "
                f"'{name}' for {_axis}-axis! Check the matplotlib "
                "documentation for valid arguments. "
                f"Got:\n  args: {args}\n  kwargs: {kwargs}"
            ) from exc

    # Decide which tick locator or formatter to set, and set it
    if x:
        if x.get("major"):
            _set_locator_or_formatter(
                _ax=ax,
                _kind=kind,
                _axis="x",
                _major=True,
                **x["major"],
            )

        if x.get("minor"):
            _set_locator_or_formatter(
                _ax=ax,
                _kind=kind,
                _axis="x",
                _major=False,
                **x["minor"],
            )

    if y:
        if y.get("major"):
            _set_locator_or_formatter(
                _ax=ax,
                _kind=kind,
                _axis="y",
                _major=True,
                **y["major"],
            )

        if y.get("minor"):
            _set_locator_or_formatter(
                _ax=ax,
                _kind=kind,
                _axis="y",
                _major=False,
                **y["minor"],
            )

    if hasattr(ax, "zaxis"):
        if z:
            if z.get("major"):
                _set_locator_or_formatter(
                    _ax=ax,
                    _kind=kind,
                    _axis="z",
                    _major=True,
                    **z["major"],
                )

            if z.get("minor"):
                _set_locator_or_formatter(
                    _ax=ax,
                    _kind=kind,
                    _axis="z",
                    _major=False,
                    **z["minor"],
                )
