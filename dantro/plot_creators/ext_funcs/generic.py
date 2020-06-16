"""Generic, DAG-based plot functions for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import logging
import copy

import matplotlib.pyplot as plt
import xarray as xr

from ..pcr_ext import is_plot_func, PlotHelper, figure_leak_prevention
from ._utils import plot_errorbar as _plot_errorbar

# Local constants
log = logging.getLogger(__name__)

# The available plot kinds for the xarray plotting interface
_XR_PLOT_KINDS = ('contourf', 'contour', 'imshow', 'line', 'pcolormesh',
                  'step', 'hist', 'scatter')


# -----------------------------------------------------------------------------

@is_plot_func(supports_animation=True,
              use_dag=True, required_dag_tags=('y', 'yerr'))
def errorbar(*, data: dict, hlpr: PlotHelper,
             x: str=None, frames: str=None,
             hue: str=None, hue_fstr: str='{value:}',
             use_bands: bool=False,
             fill_between_kwargs: dict=None,
             suptitle_kwargs: dict=None,
             **errorbar_kwargs) -> None:
    """A DAG-based generic errorbar plot.

    This plot expects data to be provided via :ref:`plot_creator_dag`.
    Expected tags are ``y`` and ``yerr`` and both should be labelled
    xarray.DataArray objects.

    Uses :py:func:`~dantro.plot_creators.ext_funcs._utils.plot_errorbar` for
    plotting individual lines.

    Args:
        data (dict): The DAG results dict which should contain entries ``y``
            and ``yerr``, both labelled xr.DataArray objects.
        hlpr (PlotHelper): The PlotHelper
        x (str, optional): The dimension to represent on the x-axis. If not
            given, it will be inferred. If no coordinates are associated with
            the dimension, trivial coordinates are used.
        frames (str, optional): Name of the dimension to represent via the
            frames of an animation. If given, this will automatically enable
            animation mode and requires ``animation`` arguments being specified
            in the plot configuration. See :ref:`pcr_ext_animations`.
        hue (str, optional): Name of the dimension to respresent via the hue
            colors of the errorbar lines. For adjusting the property cycle of
            the lines, see :ref:`pcr_ext_style`.
        hue_fstr (str, optional): A format string for the labels used when
            creating a plot with ``hue`` specified. Available keys: ``dim``
            (which is equivalent to ``hue``), and ``value``, which is set to
            the value of the corresponding coordinate along that dimension.
        use_bands (bool, optional): Whether to use error bands instead of bars
        fill_between_kwargs (dict, optional): If using error bands, these
            arguments are passed on to ``hlpr.ax.fill_between``. This would be
            the place to adjust ``alpha``; if not given, the corresponding
            line's ``alpha`` value will be used, reduced to 20%.
        suptitle_kwargs (dict, optional): Key passed on to the PlotHelper's
            ``set_suptitle`` helper function. Only used if animations are
            enabled. The ``title`` entry can be a format string with the
            following keys, which are updated for each frame of the animation:
            ``dim``, ``value``. Default: ``{dim:} = {value:.3g}``.
        **errorbar_kwargs: Passed on to ``hlpr.ax.errorbar``

    Returns:
        None

    Raises:
        ValueError: Description
    """
    def plot_frame(*, y: xr.DataArray, yerr: xr.DataArray,
                   x: str, hue: str=None, **kwargs):
        """Invokes a single errorbar plot, potentially for multiple hues."""
        # Get the x-data from the coordinates
        _x = y.coords[x]

        if hue is None:
            _plot_errorbar(ax=hlpr.ax, x=_x, y=y, yerr=yerr,
                           fill_between=use_bands, **kwargs)
            return

        # else: will plot multiple lines
        # Keep track of legend handles and labels
        handles, labels = [], []

        # Group by the hue dimension and perform plots
        hue_iter = zip(y.groupby(hue), yerr.groupby(hue))
        for (_y_coord, _y), (_yerr_coord, _yerr) in hue_iter:
            label = hue_fstr.format(dim=hue, value=_y_coord)
            handle = _plot_errorbar(ax=hlpr.ax, x=_x, y=_y, yerr=_yerr,
                                    label=label, fill_between=use_bands,
                                    **kwargs)
            handles.append(handle)
            labels.append(label)

        # Register the custom legend handles
        hlpr.ax.legend(handles, labels, title=hue)
        # NOTE Other aesthetics are specified by the user via the PlotHelper

    # Retrieve and prepare data ...............................................
    y = data['y']
    yerr = data['yerr']

    # Check shape
    if y.sizes != yerr.sizes:
        _sizes_y = ", ".join([f"{k}: {v:d}" for k, v in y.sizes.items()])
        _sizes_yerr = ", ".join([f"{k}: {v:d}" for k, v in yerr.sizes.items()])
        raise ValueError("The 'y' and 'yerr' data need to be of the same size "
                         f"but had sizes ({_sizes_y}) and ({_sizes_yerr}), "
                         "respectively!")
    # TODO Check that coordinates match?
    # Only need to check vals from now on, knowing that the shape is the same.

    # Determine expected dimension number of data arrays and check
    expected_ndim = 1 + bool(hue) + bool(frames)
    if y.ndim != expected_ndim:
        _dims = ", ".join(y.dims)
        raise ValueError(f"Data has unexpected number of dimensions! With "
                         f"`hue: {hue}` and `frames: {frames}`, expected data "
                         f"to be {expected_ndim}-dimensional, but got "
                         f"{y.ndim}-dimensional data with dimensions: {_dims}")

    # If x is not given, can infer it now
    if x is None:
        x = [dim for dim in y.dims if dim not in (hue, frames)][0]
    hlpr.invoke_helper('set_labels', x=dict(label=x),
                       mark_disabled_after_use=False)


    # Actual plotting routine starts here .....................................
    # If no animation is desired, the plotting routine is really simple
    if not frames:
        # Exit animation mode, if it was enabled. Then plot the figure. Done.
        hlpr.disable_animation()
        plot_frame(y=y, yerr=yerr, x=x, hue=hue,
                   fill_between_kwargs=fill_between_kwargs, **errorbar_kwargs)
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
        frames_iter = zip(y.groupby(frames), yerr.groupby(frames))

        for (_y_coord, _y), (_yerr_coord, _yerr) in frames_iter:
            hlpr.ax.clear()

            plot_frame(y=_y, yerr=_yerr, x=x, hue=hue,
                       fill_between_kwargs=fill_between_kwargs,
                       **errorbar_kwargs)

            # Convey frame coordinate information via suptile
            st_kwargs = copy.deepcopy(suptitle_kwargs)
            st_kwargs['title'] = st_kwargs['title'].format(dim=frames,
                                                           value=_y_coord)
            hlpr.invoke_helper('set_suptitle', **st_kwargs,
                               mark_disabled_after_use=False)

            # Done with this frame. Let the writer grab it.
            yield

    # Register the animation update with the helper
    hlpr.register_animation_update(update, invoke_helpers_before_grab=True)


@is_plot_func(supports_animation=True,
              use_dag=True, required_dag_tags=('y', 'yerr'))
def errorbands(*, data: dict, hlpr: PlotHelper, **kwargs):
    """A DAG-based generic errorbands plot.

    Invokes :py:func:`~dantro.plot_creators.ext_funcs.generic.errorbar` with
    ``use_bands = True``. See there for available arguments.
    """
    return errorbar(data=data, hlpr=hlpr, use_bands=True, **kwargs)


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

    .. warning::

        Depending on ``kind`` and the dimensionality of the data, some plot
        functions might create their own figure, disregarding any previously
        set up figure. This includes the figure from the plot helper.

        To control figure aesthetics, you can either specify matplotlib RC
        :ref:`style parameters <pcr_ext_style>` (via the ``style`` argument),
        or you can use the ``plot_kwargs`` to pass arguments to the respective
        plot functions. For the latter, refer to the respective documentation
        to find out about available arguments.

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
            ``dim``, ``value``. Default: ``{dim:} = {value:.3g}``.
        **plot_kwargs: Passed on to ``<data>.plot`` or ``<data>.plot.<kind>``
    """
    def plot_frame(_d):
        """Plot a FacetGrid frame"""
        # Use the automatically deduced plot kind, e.g. line plot for 1D
        if kind is None:
            # If the helper figure is to be used, make sure there's absolutely
            # nothing on it; this prevents plotting artifacts.
            hlpr.fig.clear()

            # Directly call the plot function of the underlying data object,
            # making sure that any additionally created figure will not survive
            # in case of an exception being raised from that plot function.
            with figure_leak_prevention(close_current_fig_on_raise=True):
                rv = _d.plot(**plot_kwargs)
            # NOTE See extended NOTE below about the type of the return value.

        # Use a specific kind of plot function
        else:
            # Retrieve the specialized plot function
            try:
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

            # Make sure to work on a fully cleared figure. This is important
            # for *some* specialized plot functions and for certain
            # dimensionality of the data: in these specific cases, an existing
            # figure can be re-used, in some cases leading to plotting
            # artifacts.
            # In other cases, a new figure is opened by the plot function. The
            # currently attached helper figure is then discarded below.
            hlpr.fig.clear()

            # Invoke the specialized plot function, taking care that no figures
            # that are additionally created survive beyond that point, which
            # would lead to figure leakage, gobbling up memory.
            with figure_leak_prevention(close_current_fig_on_raise=True):
                rv = plot_func(**plot_kwargs)
            # NOTE rv usually is a xarray.FaceGrid object but not always:
            #      `hist` returns what matplotlib.pyplot.hist returns.
            #      This leads to the question why `hist`s do not seem to be
            #      possible in `xarray.FacetGrid`s, although they would be
            #      useful? Gaining a deeper understanding of this issue and
            #      corresponding xarray functionality is something to
            #      investigate in the future. :)

        # Determine which figure and axes to attach to the PlotHelper
        if isinstance(rv, xr.plot.FacetGrid):
            fig = rv.fig
            axes = rv.axes
        else:
            # Best guess: there's only one axis and figure, use those
            fig = plt.gcf()
            axes = plt.gca()

        # When now attaching the new figure and axes, the previously existing
        # figure (the one .clear()-ed above) is closed and discarded.
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
    hlpr.register_animation_update(update, invoke_helpers_before_grab=True)
