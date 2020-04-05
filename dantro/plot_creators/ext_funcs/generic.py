"""Generic, DAG-supporting plots"""

import matplotlib.pyplot as plt

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
        """Plot a FaceGrid frame"""
        # Use the automatically deduced, default plot kind, e.g. line plot for 1D
        if kind is None:
            # Directly call the plot function of the underlying data object
            # NOTE rv usually is a xarray.FaceGrid object but not always:
            #      `hist` returns what matplotlib.pyplot.hist returns
            rv = _d.plot(**plot_kwargs)

        # Use a specific kind of plot function
        else:
            # Gather all possible kinds of plots
            KINDS = ('contourf', 'contour', 'imshow', 'line', 'pcolormesh', 'step',
                     'hist', 'scatter')

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
                raise AttributeError("The plot kind '{}' seems not to be available "
                                     "for data of type {}! Please check the "
                                     "documentation regarding the expected data "
                                     "types."
                                     .format(kind, type(d))) from err

            # Invoke the specialized plot function
            rv = plot_func(**plot_kwargs)

        # Attach the figure and the axes to the PlotHelper
        fig = plt.gcf()

        # get the axes of the FaceGrid plot.
        # NOTE 'hist': in case of a histogram, the interface is different because
        #      the plot.hist method does not return a FacetGrid object but
        #      a single matlotlib.pyplot.hist object
        # if kind == 'hist':
        axes = fig.gca()
        # else:
        #     axes = rv.axes

        hlpr.attach_figure_and_axes(fig=fig, axes=axes)

    def update():
        """The animation update function: a python generator"""
        # Go over all available frame data dimension
        for f_value, f_data in d.groupby(frames):
            # Clear the axis and plot the frame
            hlpr.ax.clear()
            plot_frame(f_data)

            # Set the title with current time step
            hlpr.invoke_helper('set_title',
                               title="Frames from dimension `{}` at value {}"
                               .format(frames, f_value))

            # Done with this frame. Yield control to the plot framework,
            # which will take care of grabbing the frame.

            # Attach the figure and the axes to the PlotHelper
            fig = plt.gcf()

            # get the axes of the FaceGrid plot.
            # NOTE 'hist': in case of a histogram, the interface is different because
            #      the plot.hist method does not return a FacetGrid object but
            #      a single matlotlib.pyplot.hist object
            # if kind == 'hist':
            axes = fig.gca()
            # else:
            #     axes = rv.axes

            hlpr.attach_figure_and_axes(fig=fig, axes=axes)
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
