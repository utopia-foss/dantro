"""Generic, DAG-based plot functions for the
:py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived plot
creators.
"""

import copy
import logging
import math
from typing import Tuple, Union

import matplotlib.pyplot as plt
import xarray as xr

from ..pcr_ext import PlotHelper, figure_leak_prevention, is_plot_func
from ._utils import plot_errorbar as _plot_errorbar

# Local constants
log = logging.getLogger(__name__)

# fmt: off

# The available plot kinds for the *xarray* plotting interface, together with
# the supported layout specifier keywords.
_XR_PLOT_KINDS = {
    "scatter":      ("x", "y", "hue", "col", "row"),
    "line":         ("x", "hue", "col", "row"),
    "step":         ("x", "col", "row"),
    "contourf":     ("x", "y", "col", "row"),
    "contour":      ("x", "y", "col", "row"),
    "imshow":       ("x", "y", "col", "row"),
    "pcolormesh":   ("x", "y", "col", "row"),
    "hist":         (),
}

# The available plot kinds for the *dantro* plotting interface, together with
# the supported layout specifiers, which include the ``frames`` option.
_DANTRO_PLOT_KINDS = {  # --- start literalinclude
    # based on facet_grid
    "scatter":      ("x", "y", "hue", "col", "row", "frames"),
    "line":         ("x", "hue", "col", "row", "frames"),
    "step":         ("x", "col", "row", "frames"),
    "contourf":     ("x", "y", "col", "row", "frames"),
    "contour":      ("x", "y", "col", "row", "frames"),
    "imshow":       ("x", "y", "col", "row", "frames"),
    "pcolormesh":   ("x", "y", "col", "row", "frames"),
    "hist":         ("frames",),

    # based on other generic functions
    "errorbar":     ("x", "hue", "frames"),
}   # --- end literalinclude

# A mapping from data dimensionality to preferred plot kind, used in automatic
# plot kind selection. This assumes the specifiers of ``_DANTRO_PLOT_KINDS``.
_AUTO_PLOT_KINDS = {  # --- start literalinclude
    1:           "line",
    2:           "pcolormesh",
    3:           "pcolormesh",
    4:           "pcolormesh",
    5:           "pcolormesh",
    "fallback":  "hist",
    "dataset":   "scatter",
}   # --- end literalinclude


# All layout encoding specifiers used in the facet_grid or similar plots
_LAYOUT_SPECIFIERS = ("x", "y", "hue", "col", "row", "frames")

# fmt: on


# -- Helper functions ---------------------------------------------------------


def determine_plot_kind(
    d: Union[xr.DataArray, xr.Dataset],
    *,
    kind: Union[str, dict],
    default_kind_map: dict = _AUTO_PLOT_KINDS,
) -> str:
    """Determines the plot kind to use for the given data. If ``kind: auto``,
    this will determine the plot kind depending on the dimensionality of the
    data. Otherwise, it will simply return ``kind``.

    Args:
        d (Union[xr.DataArray, xr.Dataset]): The data for which to determine
            the plot kind.
        kind (Union[str, dict]): The given kind argument. If it is ``auto``,
            the ``kind_map`` is used to determine the ``kind`` from the
            dimensionality of ``d``.
            If it is a dict, ``auto`` is implied and the dict is assumed to be
            a (ndim -> kind) mapping, *updating* the ``default_kind_map``.
        default_kind_map (dict, optional): The default mapping to use for
            ``kind: auto``, with keys being ``d``'s dimensionality and values
            being the plot kind to use.
            There are two special keys, ``fallback`` and ``dataset``. The
            value belonging to ``dataset`` is used for data that is dataset-
            like, i.e. does not have an ``ndim`` attribute. The value of
            ``fallback`` specifies the plot kind for data dimensionalities
            that match no other key.

    Returns:
        str: The selected plot kind. This is equal to the *given* ``kind`` if
            it was None or a string unequal to ``auto``.
    """
    # Was the plot kind already specified?
    if kind is None or (isinstance(kind, str) and kind != "auto"):
        # Yes. Just return that value.
        return kind

    # else: Need to determine it by inspecting the data and the kind mapping.
    # First, determine the mapping.
    kind_map = copy.deepcopy(default_kind_map)
    if isinstance(kind, dict):
        kind_map.update(kind)

    # For datasets, always fall back to the specified default kind
    if not hasattr(d, "ndim"):
        return kind_map["dataset"]

    # Select the kind from the dimensionality. If this fails, use the default
    # value instead.
    try:
        kind = kind_map[d.ndim]
    except KeyError:
        kind = kind_map["fallback"]

    log.remark("Using plot kind '%s' for %d-dimensional data.", kind, d.ndim)
    return kind


def determine_layout_encoding(
    d: Union[xr.DataArray, xr.Dataset],
    *,
    kind: str,
    auto_encoding: Union[bool, dict],
    default_encodings: dict,
    **all_plot_kwargs,
) -> dict:
    """Determines the plot kind and layout encoding for the given data.

    If ``auto_encoding`` does not evaluate to true or ``kind is None``, this
    function does nothing and simply returns all given plotting arguments.

    Otherwise, it uses the chosen plot ``kind`` to associate layout specifiers
    with dimension names of ``d``.
    The available layout encoding specifiers (``x``, ``y``, ``col`` etc.) can
    be specified in two ways:

        - By default, ``default_encodings`` is used as a map from plot kind to
          a sequence of available layout specifiers.
        - If ``auto_encoding`` is a dictionary, the default map will be updated
          with that dictionary.

    The association is done in the following way:

        1. Inspecting ``all_plot_kwargs``, all layout encoding specifiers are
           extracted, dropping those that evaluate to False.
        2. The encodings mapping is determined (see above).
        3. The available dimension names are determined from ``d``.
        4. Depending on ``kind`` and the already fixed specifiers, the *free*
           encoding specifiers and dimension names are extracted.
        5. These free specifiers are associated with free dimension names,
           in order of descending dimension size.

    **Example:** Assume, the available specifiers are ``('x', 'y', 'col')`` and
    the data has dimensions ``dim0``, ``dim1`` and ``dim2``. Let's further say
    that ``y`` was already fixed to ``dim2``, leaving ``x`` and ``col`` as free
    specifiers and ``dim0`` and ``dim1`` as free dimensions.
    With ``x`` being specified before ``col`` in the list of available
    specifiers, ``x`` would be associated to the remaining dimension with the
    *larger* size and ``col`` to the remaining one.

    An encodings mapping may look like this:

    .. literalinclude:: ../../dantro/plot_creators/ext_funcs/generic.py
        :language: python
        :start-after: _DANTRO_PLOT_KINDS = {  # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    Args:
        d (Union[xr.DataArray, xr.Dataset]): The data for which to create the
            layout association.
        kind (str): The chosen plot kind. If this was None, will directly
            return, because auto-encoding information is missing.
        auto_encoding (Union[bool, dict]): Whether to perform auto-encoding.
            If a dict, will regard it as a mapping of available encodings and
            update ``default_encodings``.
        default_encodings (dict): A map from plot kinds to available layout
            specifiers, e.g. ``{"line": ("x", "hue", "col", "row")}``.
        **all_plot_kwargs: All remaining plot function arguments, including any
            layout encoding arguments that aim to *fix* a dimension. Everything
            else is ignored.
    """
    if not auto_encoding or kind is None:
        log.debug("Layout auto-encoding was disabled (kind: %s).", kind)
        return all_plot_kwargs

    log.note(
        "Automatically determining layout encoding for kind '%s' ...", kind
    )

    # Split plotting kwargs into a dict of layout specifiers and one that only
    # includes the remaining plotting kwargs
    plot_kwargs = copy.deepcopy(all_plot_kwargs)
    specs = {k: v for k, v in plot_kwargs.items() if k in _LAYOUT_SPECIFIERS}
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if k not in specs}

    # Drop those specifiers that are effectively unset.
    specs = {s: dim_name for s, dim_name in specs.items() if dim_name}

    # Evaluate supported encodings, then get the available encoding specifiers
    encs = copy.deepcopy(default_encodings)
    if isinstance(auto_encoding, dict):
        encs.update(auto_encoding)

    encoding_specs = encs[kind]

    # -- Determine specifiers, depending on kind and dimensionality
    # Get all available dimension names, sorted by size (descending)
    dim_names = [
        name
        for name, _ in sorted(
            d.sizes.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    # Some dimensions and specifiers might already have been associated;
    # determine those that have *not* yet been associated:
    free_specs = [s for s in encoding_specs if s not in specs.keys()]
    free_dim_names = [name for name in dim_names if name not in specs.values()]

    log.debug("Fixed layout specifiers:  %s", specs)
    log.debug("Free specifiers:          %s", ", ".join(free_specs))
    log.debug("Free dimension names:     %s", ", ".join(free_dim_names))

    # From these two lists, update the specifier dictionary
    specs.update(
        {s: dim_name for s, dim_name in zip(free_specs, free_dim_names)}
    )
    log.remark(
        "Chosen layout encoding:   %s",
        ", ".join([f"{s}: {d}" for s, d in specs.items()]),
    )

    # -- Automatic column wrapping
    if plot_kwargs.get("col_wrap") == "auto":
        if specs.get("col") and not specs.get("row"):
            num_cols = d.sizes[specs["col"]]
            plot_kwargs["col_wrap"] = math.ceil(math.sqrt(num_cols))
            log.debug(
                "With %d expected columns, set automatic col_wrap to %d.",
                num_cols,
                plot_kwargs["col_wrap"],
            )
        else:
            plot_kwargs["col_wrap"] = None  # ... to avoid a plot warning

    # Finally, return the merged layout specifiers and plot kwargs
    return dict(**plot_kwargs, **specs)


# -----------------------------------------------------------------------------
# -- The actual plotting functions --------------------------------------------
# -----------------------------------------------------------------------------


@is_plot_func(
    supports_animation=True, use_dag=True, required_dag_tags=("y", "yerr")
)
def errorbar(
    *,
    data: dict,
    hlpr: PlotHelper,
    x: str = None,
    hue: str = None,
    frames: str = None,
    auto_encoding: Union[bool, dict] = False,
    hue_fstr: str = "{value:}",
    use_bands: bool = False,
    fill_between_kwargs: dict = None,
    suptitle_kwargs: dict = None,
    **errorbar_kwargs,
) -> None:
    """A DAG-based generic errorbar plot.

    This plot expects data to be provided via :ref:`plot_creator_dag`.
    Expected tags are ``y`` and ``yerr`` and both should be labelled
    xarray.DataArray objects. Depending on the given layout encoding specifiers
    (``x``, ``hue``, and ``frames``), data may be 1D, 2D, or 3D.
    The :ref:`auto-encoding feature <dag_generic_auto_encoding>` is supported.

    Uses :py:func:`~dantro.plot_creators.ext_funcs._utils.plot_errorbar` for
    plotting individual lines.

    Args:
        data (dict): The DAG results dict which should contain entries ``y``
            and ``yerr``, both labelled xr.DataArray objects.
        hlpr (PlotHelper): The PlotHelper
        x (str, optional): The dimension to represent on the x-axis. If not
            given, it will be inferred. If no coordinates are associated with
            the dimension, trivial coordinates are used.
        hue (str, optional): Name of the dimension to respresent via the hue
            colors of the errorbar lines. For adjusting the property cycle of
            the lines, see :ref:`pcr_ext_style`.
        frames (str, optional): Name of the dimension to represent via the
            frames of an animation. If given, this will automatically enable
            animation mode and requires ``animation`` arguments being specified
            in the plot configuration. See :ref:`pcr_ext_animations`.
        auto_encoding (Union[bool, dict], optional): Whether to choose the
            layout encoding options automatically, i.e. select ``x``, ``hue``,
            and ``frames`` according to the given data's dimensionality.
            For further options, can pass a dict.
            See :ref:`dag_generic_auto_encoding` for more info.
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
        ValueError: Upon badly shaped data
    """

    def plot_frame(
        *,
        y: xr.DataArray,
        yerr: xr.DataArray,
        x: str,
        hue: str = None,
        **kwargs,
    ):
        """Invokes a single errorbar plot, potentially for multiple hues."""
        # Set some helper properties
        hlpr.invoke_helper(
            "set_labels", x=dict(label=x), mark_disabled_after_use=False
        )

        # Get the x-data from the coordinates
        _x = y.coords[x]

        if hue is None:
            _plot_errorbar(
                ax=hlpr.ax,
                x=_x,
                y=y,
                yerr=yerr,
                fill_between=use_bands,
                **kwargs,
            )
            return

        # else: will plot multiple lines
        # Keep track of legend handles and labels
        handles, labels = [], []

        # Group by the hue dimension and perform plots. Depending on the xarray
        # version, this might or might not drop size-1 dimensions. To make sure
        # that this does not mess up everything, squeeze explicitly.
        hue_iter = zip(y.groupby(hue), yerr.groupby(hue))
        for (_y_coord, _y), (_yerr_coord, _yerr) in hue_iter:
            _y = _y.squeeze(drop=True)
            _yerr = _yerr.squeeze(drop=True)

            label = hue_fstr.format(dim=hue, value=_y_coord)
            handle = _plot_errorbar(
                ax=hlpr.ax,
                x=_x,
                y=_y,
                yerr=_yerr,
                label=label,
                fill_between=use_bands,
                **kwargs,
            )
            handles.append(handle)
            labels.append(label)

        # Register the custom legend handles
        hlpr.ax.legend(handles, labels, title=hue)
        # NOTE Other aesthetics are specified by the user via the PlotHelper

    # Retrieve and prepare data ...............................................
    y = data["y"]
    yerr = data["yerr"]

    # Check shape
    if y.sizes != yerr.sizes:
        _sizes_y = ", ".join([f"{k}: {v:d}" for k, v in y.sizes.items()])
        _sizes_yerr = ", ".join([f"{k}: {v:d}" for k, v in yerr.sizes.items()])
        raise ValueError(
            "The 'y' and 'yerr' data need to be of the same size "
            f"but had sizes ({_sizes_y}) and ({_sizes_yerr}), "
            "respectively!"
        )
    # TODO Check that coordinates match?
    # Only need to check vals from now on, knowing that the shape is the same.

    # Allow auto-encoding of layout specifiers
    # NOTE Need to pop all explicitly given specifiers in order to not have
    #      them appear as part of plot_kwargs further downstream.
    layout_encoding = determine_layout_encoding(
        y,
        kind="errorbar",
        auto_encoding=auto_encoding,
        default_encodings=_DANTRO_PLOT_KINDS,
        x=x,
        hue=hue,
        frames=frames,
    )
    x = layout_encoding.pop("x")
    hue = layout_encoding.pop("hue", None)
    frames = layout_encoding.pop("frames", None)

    # Determine expected dimension number of data arrays and check
    expected_ndim = 1 + bool(hue) + bool(frames)
    if y.ndim != expected_ndim:
        _dims = ", ".join(y.dims)
        raise ValueError(
            "Data has unexpected number of dimensions! With "
            f"`hue: {hue}` and `frames: {frames}`, expected data "
            f"to be {expected_ndim}-dimensional, but got "
            f"{y.ndim}-dimensional data with dimensions: {_dims}"
        )

    # Even without auto-encoding, we need x. At this point, we may infer it
    if not x:
        x = [dim for dim in y.dims if dim not in (hue, frames)][0]

    # Actual plotting routine starts here .....................................
    # If no animation is desired, the plotting routine is really simple
    if not frames:
        # Exit animation mode, if it was enabled. Then plot the figure. Done.
        hlpr.disable_animation()
        plot_frame(
            y=y,
            yerr=yerr,
            x=x,
            hue=hue,
            fill_between_kwargs=fill_between_kwargs,
            **errorbar_kwargs,
        )
        return

    # else: Animation is desired. Might have to enable it.
    # If not already in animation mode, the plot function will be exited here
    # and be invoked anew in animation mode. It will end up in this branch
    # again, and will then be able to proceed past this point...
    hlpr.enable_animation()

    # Prepare some parameters for the update routine
    suptitle_kwargs = suptitle_kwargs if suptitle_kwargs else {}
    if "title" not in suptitle_kwargs:
        suptitle_kwargs["title"] = "{dim:} = {value:.3g}"

    # Define an animation update function. All frames are plotted therein.
    # There is no need to plot the first frame _outside_ the update function,
    # because it would be discarded anyway.
    def update():
        """The animation update function: a python generator"""
        frames_iter = zip(y.groupby(frames), yerr.groupby(frames))

        for (_y_coord, _y), (_yerr_coord, _yerr) in frames_iter:
            hlpr.ax.clear()

            plot_frame(
                y=_y,
                yerr=_yerr,
                x=x,
                hue=hue,
                fill_between_kwargs=fill_between_kwargs,
                **errorbar_kwargs,
            )

            # Convey frame coordinate information via suptile
            st_kwargs = copy.deepcopy(suptitle_kwargs)
            st_kwargs["title"] = st_kwargs["title"].format(
                dim=frames, value=_y_coord
            )
            hlpr.invoke_helper(
                "set_suptitle", **st_kwargs, mark_disabled_after_use=False
            )

            # Done with this frame. Let the writer grab it.
            yield

    # Register the animation update with the helper
    hlpr.register_animation_update(update, invoke_helpers_before_grab=True)


@is_plot_func(
    supports_animation=True, use_dag=True, required_dag_tags=("y", "yerr")
)
def errorbands(*, data: dict, hlpr: PlotHelper, **kwargs):
    """A DAG-based generic errorbands plot.

    Invokes :py:func:`~dantro.plot_creators.ext_funcs.generic.errorbar` with
    ``use_bands = True``. See there for available arguments.
    """
    return errorbar(data=data, hlpr=hlpr, use_bands=True, **kwargs)


# -----------------------------------------------------------------------------


@is_plot_func(
    use_dag=True, required_dag_tags=("data",), supports_animation=True
)
def facet_grid(
    *,
    data: dict,
    hlpr: PlotHelper,
    kind: Union[str, dict] = None,
    frames: str = None,
    auto_encoding: Union[bool, dict] = False,
    suptitle_kwargs: dict = None,
    **plot_kwargs,
):
    """A generic facet grid plot function for high dimensional data.

    This function calls the ``data['data'].plot`` function if no plot ``kind``
    is given, otherwise ``data['data'].plot.<kind>``. It is designed for
    `plotting with xarray objects <http://xarray.pydata.org/en/stable/plotting.html>`_, i.e.
    `xr.DataArray <http://xarray.pydata.org/en/stable/plotting.html#dataarrays>`_
    and
    `xr.Dataset <http://xarray.pydata.org/en/stable/plotting.html#datasets>`_.
    Specifying the kind of plot requires the data to be of one of those types
    and have a dimensionality that can be represented in these plots. See
    `the correponding API documentation <http://xarray.pydata.org/en/stable/api.html#plotting>`_ for more information.

    In most cases, this function creates a so-called
    `FacetGrid <http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html>`_
    object that automatically layouts and chooses a visual representation that
    fits the dimensionality of the data. To specify which data dimension
    should be represented in which way, it supports a declarative syntax: via
    the optional keyword arguments ``x``, ``y``, ``row``, ``col``, and/or
    ``hue`` (available options are listed in the corresponding
    `plot function documentation <http://xarray.pydata.org/en/stable/api.html#plotting>`_),
    the representation of the data dimensions can be selected. This is
    referred to as "layout encoding".

    dantro not only wraps this interface, but adds the following functionality:

        * the ``frames`` layout encoding argument, which behaves in the same
          way as the other encodings, but leads to an *animation* being
          generated, thus opening up one further dimension of representation,
        * the ``auto_encoding`` feature, which allows to select layout-
          encodings automatically,
        * and the ``kind: 'auto'`` option, which can be used in conjunction
          with ``auto_encoding`` to choose the plot kind automatically as well.
        * allows ``col_wrap: 'auto'``, which selects the value such that the
          figure becomes more square-like (requires ``auto_encoding: true``)

    For details about auto-encoding and how the plot ``kind`` is chosen, see
    :py:func:`~dantro.plot_creators.ext_funcs.generic.determine_layout_encoding`
    and :py:func:`~dantro.plot_creators.ext_funcs.generic.determine_plot_kind`.

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
        data (dict): The data selected by the data transformation framework,
            expecting the ``data`` key.
        hlpr (PlotHelper): The plot helper
        kind (str, optional): The kind of plot to use. Options are:
            ``contourf``, ``contour``, ``imshow``, ``line``, ``pcolormesh``,
            ``step``, ``hist``, ``scatter``.
            With ``auto``, dantro chooses an appropriate kind by itself; this
            setting is useful when also using the ``auto_encoding`` feature;
            see :ref:`dag_generic_facet_grid_auto_kind` for more information.
            If None is given, xarray automatically determines it using the
            dimensionality of the data, frequently falling back to ``hist``
            for higher-dimensional data or lacking specifiers.
        frames (str, optional): Data dimension from which to create animation
            frames. If given, this results in the creation of an animation. If
            not given, a single plot is generated. Note that this requires
            ``animation`` options as part of the plot configuration.
        auto_encoding (Union[bool, dict], optional): Whether to choose the
            layout encoding options automatically. For further options, can
            pass a dict. See :ref:`dag_generic_auto_encoding` for more info.
        suptitle_kwargs (dict, optional): Key passed on to the PlotHelper's
            ``set_suptitle`` helper function. Only used if animations are
            enabled. The ``title`` entry can be a format string with the
            following keys, which are updated for each frame of the animation:
            ``dim``, ``value``. Default: ``{dim:} = {value:.3g}``.
        **plot_kwargs: Passed on to ``<data>.plot`` or ``<data>.plot.<kind>``
            These should include the layout encoding specifiers (``x``, ``y``,
            ``hue``, ``col``, and/or ``row``).

    Raises:
        AttributeError: Upon unsupported ``kind`` value
        ValueError: Upon *any* upstream error in invocation of the xarray
            plotting capabilities. This wraps the given error message and
            provides additional information that helps to track down why the
            plotting failed.
    """

    def plot_frame(_d, *, kind: str, plot_kwargs: dict):
        """Plot a FacetGrid frame"""
        # Retrieve the generic or specialized plot function, depending on kind
        if kind is None:
            plot_func = _d.plot

        else:
            try:
                plot_func = getattr(_d.plot, kind)

            except AttributeError as err:
                _available = ", ".join(_XR_PLOT_KINDS)
                raise AttributeError(
                    f"The plot kind '{kind}' seems not to be available for "
                    f"data of type {type(_d)}! Please check the documentation "
                    "regarding the expected data types. For xarray data "
                    f"structures, valid choices are: {_available}"
                ) from err

        # Make sure to work on a fully cleared figure. This is important for
        # *some* specialized plot functions and for certain dimensionality of
        # the data: in these specific cases, an existing figure can be
        # re-used, in some cases leading to plotting artifacts.
        # In other cases, a new figure is opened by the plot function. The
        # currently attached helper figure is then discarded below.
        hlpr.fig.clear()

        # Invoke the specialized plot function, taking care that no figures
        # that are additionally created survive beyond that point, which would
        # lead to figure leakage, gobbling up memory.
        with figure_leak_prevention(close_current_fig_on_raise=True):
            try:
                rv = plot_func(**plot_kwargs)

            except Exception as exc:
                raise ValueError(
                    "facet_grid plotting failed, most probably because the "
                    "dimensionality of the data, the chosen plot kind "
                    f"({kind}) and the specified layout encoding were not "
                    "compatible.\n"
                    f"The upstream error was a {type(exc).__name__}: {exc}\n\n"
                    f"facet_grid arguments:\n  {plot_kwargs}\n\n"
                    f"Data:\n  {_d}\n"
                ) from exc
        # NOTE rv usually is a xarray.FaceGrid object but not always: `hist`
        #      returns what matplotlib.pyplot.hist returns.
        #      This leads to the question why `hist`s do not seem to be
        #      possible in `xarray.FacetGrid`s, although they would be useful?
        #      Gaining a deeper understanding of this issue and corresponding
        #      xarray functionality is something to investigate in the future.

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
    d = data["data"]

    # Determine kind and encoding, updating the plot kwargs accordingly.
    # NOTE Need to pop all explicitly given specifiers in order to not have
    #      them appear as part of plot_kwargs further downstream.
    kind = determine_plot_kind(d, kind=kind, default_kind_map=_AUTO_PLOT_KINDS)
    plot_kwargs = determine_layout_encoding(
        d,
        kind=kind,
        auto_encoding=auto_encoding,
        default_encodings=_DANTRO_PLOT_KINDS,
        frames=frames,
        **plot_kwargs,
    )
    frames = plot_kwargs.pop("frames", None)

    # If no animation is desired, the plotting routine is really simple
    if not frames:
        # Exit animation mode, if it was enabled. Then plot the figure. Done.
        hlpr.disable_animation()
        plot_frame(d, kind=kind, plot_kwargs=plot_kwargs)

        return

    # else: Animation is desired. Might have to enable it.
    # If not already in animation mode, the plot function will be exited here
    # and be invoked anew in animation mode. It will end up in this branch
    # again, and will then be able to proceed past this point...
    hlpr.enable_animation()

    # Prepare some parameters for the update routine
    suptitle_kwargs = suptitle_kwargs if suptitle_kwargs else {}
    if "title" not in suptitle_kwargs:
        suptitle_kwargs["title"] = "{dim:} = {value:.3g}"

    # Define an animation update function. All frames are plotted therein.
    # There is no need to plot the first frame _outside_ the update function,
    # because it would be discarded anyway.
    def update():
        """The animation update function: a python generator"""
        # Go over all available frame data dimension
        for f_value, f_data in d.groupby(frames):
            # Plot a frame. It attaches the new figure and axes to the hlpr
            plot_frame(f_data, kind=kind, plot_kwargs=plot_kwargs)

            # Apply the suptitle format string, then invoke the helper
            st_kwargs = copy.deepcopy(suptitle_kwargs)
            st_kwargs["title"] = st_kwargs["title"].format(
                dim=frames, value=f_value
            )
            hlpr.invoke_helper("set_suptitle", **st_kwargs)

            # Done with this frame. Let the writer grab it.
            yield

    # Register the animation update with the helper
    hlpr.register_animation_update(update, invoke_helpers_before_grab=True)
