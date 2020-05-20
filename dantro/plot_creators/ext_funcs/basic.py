"""Holds basic plot functions for use with ExternalPlotCreator"""

from ...data_mngr import DataManager
from ..pcr_ext import is_plot_func
from .mpl_setup import *


# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

@is_plot_func(creator_name='external', use_helper=False)
def lineplot(dm: DataManager, *, out_path: str, y: str, x: str=None,
             fmt: str=None, save_kwargs: dict=None, **plot_kwargs):
    """Performs a simple lineplot.

    Args:
        dm (DataManager): The data manager from which to retrieve the data
        out_path (str): Where to store the plot to
        y (str): The path to get to the y-data in the data manager
        x (str, optional): The path to get to the x-data in the data manager
        save_kwargs (dict, optional): kwargs to the plt.savefig function
        **plt_kwargs: Passed on to plt.plot
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
