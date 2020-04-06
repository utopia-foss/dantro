"""Generic, DAG-supporting plots"""

import matplotlib.pyplot as plt
import xarray as xr

from ..pcr_ext import is_plot_func
from .._plot_helper import PlotHelper

# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True, required_dag_tags=('data',),
              supports_animation=True)
def facet_grid(*,
               data: dict,
               hlpr: PlotHelper,
               kind: str = None,
               frames: str = None,
               suptitle_kwargs: dict = None,
               **plot_kwargs):
    """This is a generic FacetGrid plot function with preprocessed DAG data.

    This function calls the data['data'].plot function if no plot kind is given 
    else the specified data['data'].plot.<kind> function. It is designed for 
    `xarray plotting <http://xarray.pydata.org/en/stable/plotting.html>`_ i.e.
    xarray.DataArray and xarray.Dataset plotting capabilities. 
    Specifying the kind of plot requires the data to either be an 
    xarray.DataArray or xarray.Dataset of specific dimensionality, see
    `the correponding docu <http://xarray.pydata.org/en/stable/api.html#plotting>`_. 
    The function creates a `FacetGrid <http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html`>_
    object that automatically layouts and chooses a visual representation 
    of the data through a declarative approach if not explicitely specified: 
    users can specify the data plotted as ``x``, ``y``, ``row``, ``col``, 
    and/or ``hue`` (available options are listed in the corresponding
    `plot function documentation <http://xarray.pydata.org/en/stable/api.html#plotting>`_
    ). 

    Args:
        data (dict): The data selected by the DAG framework
        hlpr (PlotHelper): The plot helper
        kind (str): The kind of plot to use. Options are ``contourf``, 
            ``contour``, ``imshow``, ``line``, ``pcolormesh``, ``step``, 
            ``hist``, ``scatter``. If None is given, xarray automatically 
            determines it using the dimensionality of the data.
        frames (str): The dimension from which to create frames which results
            in the creation of an animation. If frames=None a single plot
            is generated.
        suptitle_kwargs (dict): Kwargs passed on to the PlotHelper's
            `set_suptitle` function in case of enabled animation.
            If no title is given in the suptitle kwargs the frame dimension
            and its current value are used as title.
        **plot_kwargs: Passed on ot xarray.plot or xarray.plot.<kind>
    """
    # If `frames` argument is given, enter animation mode
    if frames is not None:
        # Enable the animation mode
        hlpr.enable_animation()

        # Check that frames is of type string
        if not isinstance(frames, str):
            raise TypeError("'frames' needs to be a string but was of type {}!"
                            "".format(type(frames)))

    # Get the Dataset or DataArray to plot
    d = data['data']

    def plot_frame(_d):
        """Plot a FacetGrid frame"""
        # Use the automatically deduced, default plot kind, e.g. line plot for
        # 1D
        if kind is None:
            # Directly call the plot function of the underlying data object
            # NOTE rv usually is a xarray.FaceGrid object but not always:
            #      `hist` returns what matplotlib.pyplot.hist returns
            rv = _d.plot(**plot_kwargs)

        # Use a specific kind of plot function
        else:
            # Gather all possible kinds of plots
            KINDS = ('contourf', 'contour', 'imshow', 'line', 'pcolormesh',
                     'step', 'hist', 'scatter')

            # Raise an error if the given kind is unknown
            if kind not in KINDS:
                raise ValueError("Got an unknown plot kind `{}`! Valid choices "
                                 "are: {}".format(kind, ", ".join(KINDS)))

            try:
                # Retrieve the specialized plot function
                # NOTE rv usually is a xarray.FaceGrid object but not always:
                #      `hist` returns what matplotlib.pyplot.hist returns.
                #      This leads to the question why `hist`s do not seem to be
                #      possible in `xarray.FacetGrid`s, although they would be
                #      useful? Gaining a deeper understanding of this issue and
                #      corresponding xarray functionality is something to
                #      investigate in the future. :)
                plot_func = getattr(_d.plot, kind)

            except AttributeError as err:
                raise AttributeError("The plot kind '{}' seems not to be "
                                     "available for data of type {}! Please "
                                     "check the documentation regarding the "
                                     "expected data types."
                                     "".format(kind, type(_d))) from err
            # Before invoking the plot function it is important to first
            # close the PlotHelper figure because it will be overwritten
            # and then generate a new figure because the spezialized plot 
            # functions do not generate a new figure automatically.
            hlpr.close_figure()
            figure = plt.figure()

            # Invoke the specialized plot function
            rv = plot_func(**plot_kwargs)
            
        # Attach the figure and the axes to the PlotHelper
        if isinstance(rv, xr.plot.FacetGrid):
            fig = rv.fig
            axes = rv.axes
        else:
            # Best guess: there's only one axis and figure, attach those to the
            # helper
            fig = plt.gcf()
            axes = plt.gca()

        hlpr.attach_figure_and_axes(fig=fig, axes=axes)

    def update():
        """The animation update function: a python generator"""
        # Go over all available frame data dimension
        for f_value, f_data in d.groupby(frames):
            # Plot a frame. It attaches the new figure and axes to the hlpr
            plot_frame(f_data)

            if suptitle_kwargs is not None:
                if 'title' not in suptitle_kwargs:
                    suptitle_kwargs['title'] = "{dim:} : {value:}".format(
                        dim=frames, value=f_value)

            # Set the title with current time step
            if suptitle_kwargs is not None:
                hlpr.invoke_helper('set_suptitle', suptitle_kwargs)
            else:
                hlpr.invoke_helper('set_suptitle',
                                   title="{dim:} : {value:}".format(
                                       dim=frames, value=f_value))

            yield

    # If `frames` argument is given select the data corresponding to the
    # first frames value.
    if frames is not None:
        # Plot the first frame
        plot_frame(d.isel({frames: 0}))
    else:
        # Just plot a figure which will not be updated.
        plot_frame(d)

    # Register the animation update with the helper
    hlpr.register_animation_update(update)
