"""Generic, DAG-supporting plots"""

import copy

import matplotlib.pyplot as plt
import xarray as xr

from ..pcr_ext import is_plot_func, PlotHelper

# Local variables
# The available plot kinds for the xarray plotting interface
_XR_PLOT_KINDS = ('contourf', 'contour', 'imshow', 'line', 'pcolormesh',
                  'step', 'hist', 'scatter')


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
    `the correponding documentation <http://xarray.pydata.org/en/stable/api.html#plotting>`_.

    In most cases, the function creates a so-called
    `FacetGrid <http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html`>_
    object that automatically layouts and chooses a visual representation that
    fits the dimensionality of the data.
    To specify which data dimension should be represented in which way, it
    supports a basic declarative syntax: via the optional keyword arguments
    ``x``, ``y``, ``row``, ``col``, and/or ``hue`` (available options are
    listed in the corresponding
    `plot function documentation <http://xarray.pydata.org/en/stable/api.html#plotting>`_),
    the data dimensions to represent in the corresponding way can be selected.

    dantro adds the ``frames`` argument, which behaves in a similar way but
    leads to an animation being generated, thus opening up one further
    dimension of representation.

    .. note::

        When specifying ``frames``, the ``animation`` arguments need to be
        specified. See :ref:`here <pcr_ext_animations>` for more information
        on the expected animation parameters.

        The value of the ``animation.enabled`` key is not relevant for this
        function; it will automatically enter or exit animation mode,
        depending on whether the ``frames`` argument is given or not. This uses
        the :ref:`animation mode switching <pcr_ext_animation_mode_switching>`
        feature.

    Args:
        data (dict): The data selected by the DAG framework
        hlpr (PlotHelper): The plot helper
        kind (str, optional): The kind of plot to use. Options are:
            ``contourf``, ``contour``, ``imshow``, ``line``, ``pcolormesh``,
            ``step``, ``hist``, ``scatter``.
            If None is given, xarray automatically determines it using the
            dimensionality of the data.
        frames (str, optional): The data dimension from which to create frames.
            If given, this results in the creation of an animation. If not
            given, a single plot is generated.
        suptitle_kwargs (dict, optional): Key passed on to the PlotHelper's
            ``set_suptitle`` helper function. Only used if animations are
            enabled. The ``title`` entry can be a format string with the
            following keys, which are updated for each frame of the animation:
            ``dim``, ``value``. Example: ``{dim:} : {value:.2g}``.
        **plot_kwargs: Passed on to ``<data>.plot`` or ``<data>.plot.<kind>``
    """
    def plot_frame(_d):
        """Plot a FacetGrid frame"""
        # Use the automatically deduced plot kind, e.g. line plot for 1D
        if kind is None:
            # Directly call the plot function of the underlying data object
            # NOTE rv usually is a xarray.FaceGrid object but not always:
            #      `hist` returns what matplotlib.pyplot.hist returns
            rv = _d.plot(**plot_kwargs)

        # Use a specific kind of plot function
        else:
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
                                     "expected data types. For xarray data "
                                     "structures, valid choices are: {}"
                                     "".format(kind, type(_d),
                                               ", ".join(_XR_PLOT_KINDS))
                                     ) from err
            # Before invoking the plot function it is important to first
            # close the PlotHelper figure because it will be overwritten
            # and then generate a new figure because the spezialized plot
            # functions do not generate a new figure automatically.
            hlpr.close_figure()
            fig = plt.figure()

            # Invoke the specialized plot function. If that fails for any
            # reason, the newly opened figure should be properly closed.
            try:
                rv = plot_func(**plot_kwargs)

            except Exception:
                plt.close(fig)
                raise

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
        # Done with this frame now.

    # Actual plotting routine starts here .....................................

    # Get the Dataset, DataArray, or other compatible data
    d = data['data']

    # If no animation is desired, the plotting routine is really simple
    if not frames:
        # Exit animation mode, if it was enabled. Then plot the figure. Done.
        hlpr.disable_animation()
        plot_frame(d)

        return

    # else: Animation is desired. Might have to enable it.
    # If not already in animation mode, the plot function will be exited here
    # and be invoked anew in animation mode. It will end up in this branch
    # again, and will then be able to proceed past this point...
    hlpr.enable_animation()

    # Prepare some parameters for the update routine
    suptitle_kwargs = suptitle_kwargs if suptitle_kwargs else {}
    if 'title' not in suptitle_kwargs:
        suptitle_kwargs['title'] = "{dim:} = {value:.3g}"

    # Define an animation update function. All frames are plotted therein.
    # There is no need to plot the first frame _outside_ the update function,
    # because it would be discarded anyway.
    def update():
        """The animation update function: a python generator"""
        # Go over all available frame data dimension
        for f_value, f_data in d.groupby(frames):
            # Plot a frame. It attaches the new figure and axes to the hlpr
            plot_frame(f_data)

            # Apply the suptitle format string, then invoke the helper
            st_kwargs = copy.deepcopy(suptitle_kwargs)
            st_kwargs['title'] = st_kwargs['title'].format(dim=frames,
                                                           value=f_value)
            hlpr.invoke_helper('set_suptitle', **st_kwargs)

            # Done with this frame. Let the writer grab it.
            yield

    # Register the animation update with the helper
    hlpr.register_animation_update(update)
