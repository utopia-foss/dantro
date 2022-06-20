"""Holds basic plot functions for use with PyPlotCreator"""

import logging

import matplotlib.pyplot as plt

from ...data_mngr import DataManager
from ..utils.plot_func import is_plot_func

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


@is_plot_func(creator="pyplot", use_helper=False)
def lineplot(
    dm: DataManager,
    *,
    out_path: str,
    y: str,
    x: str = None,
    fmt: str = None,
    save_kwargs: dict = None,
    **plot_kwargs,
):
    """Performs a simple lineplot using :py:func:`matplotlib.pyplot.plot`.

    Args:
        dm (DataManager): The data manager from which to retrieve the data
        out_path (str): Where to store the plot to
        y (str): The path to get to the y-data from the data tree
        x (str, optional): The path to get to the x-data from the data tree
        save_kwargs (dict, optional): Keyword arguments for
            :py:func:`matplotlib.pyplot.savefig`
        **plot_kwargs: Passed on to :py:func:`matplotlib.pyplot.plot`.
    """
    # Get the data
    x_data = dm[x] if x else None
    y_data = dm[y]

    # Assemble the arguments
    args = [x_data, y_data] if x_data is not None else [y_data]
    if fmt:
        args.append(fmt)

    # Call the plot function
    plt.plot(*args, **plot_kwargs)

    # Save and close figure
    plt.savefig(out_path, **(save_kwargs if save_kwargs else {}))
    plt.close()
