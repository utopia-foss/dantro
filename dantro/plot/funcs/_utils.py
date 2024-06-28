"""A module that implements a bunch of plot utilities used in the plotting
functions. These can be shared tools between the plotting functions.
"""

import copy
import math
from typing import Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------


def determine_ideal_col_wrap(
    N: int, *, fill_last_row: bool = True, fill_ratio_thrs: float = 3 / 4
) -> Optional[int]:
    """Given a number of subplots to place in a grid, determines the ideal
    number of columns to wrap after such that:

        1. The resulting grid is most "square"
        2. If ``fill_last_row`` is set, we compromise on squared-ness in order
           to have a last row that is more filled (avoiding lonely plots)

    To get to the square-like configuration, uses:

    .. code-block:: python

        col_wrap = math.ceil(math.sqrt(N))

    With ``fill_last_row``, will improve the fill ratio

    Args:
        N (int): Number of elements to place in the grid. If this is below 4,
            will return None.
        fill_last_row (bool, optional): Whether to not only optimize for a
            square-like grid, but to also reduce lonely plots in the last row
        fill_ratio_thrs (float, optional): If the fill ratio of the last row
            is greater or equal this number already without optimization, will
            not begin optimization.

    Returns:
        Optional[int]: The determined column wrapping number.
            Will be None for ``N < 4``.
    """

    def last_row(cw: int) -> Tuple[int, int]:
        """(filled, empty) in last row"""
        n_rows_filled = N // cw
        n_last_row = N - cw * n_rows_filled
        return n_last_row, cw - n_last_row

    def ratio_filled(cw: int) -> float:
        n_full, _ = last_row(cw)
        if n_full == 0:
            return 1.0
        return n_full / cw

    if N < 4:
        return None

    cw = math.ceil(math.sqrt(N))
    if not fill_last_row or ratio_filled(cw) >= fill_ratio_thrs:
        return cw

    # Try some fill ratios and return the best one.
    # Also include the deviation from the square-like setting to decide in
    # situations where the fill ratio is the same.
    # The -_cw is added to prefer larger col_wrap values if identical.
    dcw = max(2, cw // 3)
    ratios_and_cws = [
        (1.0 - ratio_filled(_cw), abs(_cw - cw), -_cw, _cw)
        for _cw in range(max(3, cw - dcw), min(cw + dcw, N // 2 + 1) + 1)
    ]
    return sorted(ratios_and_cws)[0][-1]


def plot_errorbar(
    *,
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    fill_between: bool = False,
    fill_between_kwargs: dict = None,
    **errorbar_kwargs,
):
    """Given the data and (optionally) the y-error data, plots a single
    errorbar line. With ``fill_between=True``, a shaded area is plotted instead
    of the errorbar markers.

    The following ``fill_between_kwargs`` defaults are assumed:

        - ``color = line_color``
        - ``alpha = 0.2 * line_alpha``
        - ``lw = 0.``

    Args:
        ax: The axis to plot on
        x (numpy.ndarray): The x data to use
        y (numpy.ndarray): The y-data to use for ``ax.errorbar``. Needs to be
            1D and have coordinates associated which will be used for the
            x-values.
        yerr (numpy.ndarray): The y-error data
        fill_between (bool, optional): Whether to use plt.fill_between or
            plt.errorbar to plot y-errors
        fill_between_kwargs (dict, optional): Passed on to plt.fill_between
        **errorbar_kwargs: Passed on to plt.errorbar

    Raises:
        ValueError: On non-1D data

    Returns:
        The matplotlib legend handle of the errorbar line or of the errorbands
    """
    if y.ndim != 1 or (yerr is not None and yerr.ndim != 1):
        raise ValueError(
            "Expected 1D `y` and `yerr` data for errorbar plot but selected "
            f"data was {y.ndim}- and {yerr.ndim}-dimensional, respecetively!\n"
            f"\ny: {y}\nyerr: {yerr}"
        )

    # Data is ok.
    # Plot the data against its coordinates, including the y-error data only if
    # no fill-between is to be performed
    ebar = ax.errorbar(
        x, y, yerr=yerr if not fill_between else None, **errorbar_kwargs
    )

    if not fill_between or yerr is None:
        # Return the regular errorbar artist
        return ebar
    # else: plot yerr as shaded area using fill_between

    # Find out the colour of the error bar line by inspecting line collection
    lc, _, _ = ebar
    line_color = lc.get_c()
    line_alpha = lc.get_alpha() if lc.get_alpha() else 1.0

    # Prepare fill between arguments, setting some default values
    fb_kwargs = (
        copy.deepcopy(fill_between_kwargs) if fill_between_kwargs else {}
    )

    fb_kwargs["color"] = fb_kwargs.get("color", line_color)
    fb_kwargs["alpha"] = fb_kwargs.get("alpha", line_alpha * 0.2)
    fb_kwargs["lw"] = fb_kwargs.get("lw", 0.0)

    # Fill.
    fb = ax.fill_between(x, y1=(y - yerr), y2=(y + yerr), **fb_kwargs)

    # Return the artist that is to be used in the legend
    return (ebar, fb)
