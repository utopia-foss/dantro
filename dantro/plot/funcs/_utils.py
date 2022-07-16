"""A module that implements a bunch of plot utilities used in the plotting
functions. These can be shared tools between the plotting functions.
"""

import copy

import numpy as np

# -----------------------------------------------------------------------------


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
