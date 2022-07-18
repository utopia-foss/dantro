"""Generic, DAG-based plot functions for the
:py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` and derived plot
creators.
"""

import copy
import logging
import math
import numbers
import warnings
from functools import partial as _partial
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.colors as mcolors

from ..._import_tools import LazyLoader
from ...exceptions import PlottingError
from ...tools import recursive_update
from ..plot_helper import PlotHelper
from ..utils import figure_leak_prevention, is_plot_func
from ..utils.color_mngr import ColorManager, parse_cmap_and_norm_kwargs
from ._utils import plot_errorbar as _plot_errorbar

# Local constants and lazy module imports
log = logging.getLogger(__name__)

plt = LazyLoader("matplotlib.pyplot")
xr = LazyLoader("xarray")

# fmt: off

_XR_PLOT_KINDS = {  # --- start literalinclude
    "scatter":      ("hue", "col", "row"),
    "line":         ("x", "hue", "col", "row"),
    "step":         ("x", "col", "row"),
    "contourf":     ("x", "y", "col", "row"),
    "contour":      ("x", "y", "col", "row"),
    "imshow":       ("x", "y", "col", "row"),
    "pcolormesh":   ("x", "y", "col", "row"),
    "hist":         (),
}   # --- end literalinclude
"""The available plot kinds for the *xarray* plotting interface, together with
the supported layout specifier keywords."""

_FACET_GRID_KINDS = {
    # based on xarray plotting functions
    "scatter":      ("hue", "col", "row", "frames"),
    "line":         ("x", "hue", "col", "row", "frames"),
    "step":         ("x", "col", "row", "frames"),
    "contourf":     ("x", "y", "col", "row", "frames"),
    "contour":      ("x", "y", "col", "row", "frames"),
    "imshow":       ("x", "y", "col", "row", "frames"),
    "pcolormesh":   ("x", "y", "col", "row", "frames"),
    "hist":         ("frames",),

    # based on dantro plotting functions
    # NOTE These are dynamically added but generally look similar to the above:
    # "errorbars":    ("x", "hue", "col", "row", "frames"),
    # "scatter3d":    ("hue", "col", "row", "frames"),
}
"""The available plot kinds for the *dantro* plotting interface, together with
the supported layout specifiers, which include the ``frames`` option."""

_AUTO_PLOT_KINDS = {  # --- start literalinclude
    1:               "line",
    2:               "pcolormesh",
    3:               "pcolormesh",
    4:               "pcolormesh",
    5:               "pcolormesh",
    "with_hue":      "line",         # used when `hue` is explicitly set
    "with_x_and_y":  "pcolormesh",   # used when _both_ `x` and `y` were set
    "dataset":       "scatter",      # used for xr.Dataset-like data
    "fallback":      "hist",         # used when none of the above matches
}   # --- end literalinclude
"""A mapping from data dimensionality to preferred plot kind, used in automatic
plot kind selection. This assumes the specifiers of ``_FACET_GRID_KINDS``"""

# fmt: on

_FACET_GRID_FUNCS: Dict[str, Callable] = {}
"""A dict mapping additional facet grid kinds to callables.
This is populated by the ``make_facet_grid_plot`` decorator."""


# -----------------------------------------------------------------------------
# -- Helper functions ---------------------------------------------------------
# -----------------------------------------------------------------------------


def determine_plot_kind(
    d: Union["xarray.DataArray", "xarray.Dataset"],
    *,
    kind: Union[str, dict],
    default_kind_map: dict = _AUTO_PLOT_KINDS,
    **plot_kwargs,
) -> str:
    """Determines the plot kind to use for the given data. If ``kind: auto``,
    this will determine the plot kind depending on the dimensionality of the
    data and other (potentially fixed) encoding specifiers. Otherwise, it will
    simply return ``kind``.

    **What if layout encodings were partly fixed?** There are two special cases
    where this is of relevance, and both these cases are covered explicitly:

        - If *both* ``x`` and ``y`` are given, ``line``- or ``hist``-like plot
          kinds are no longer possible; hence, a ``pcolormesh``-like kind has
          to be chosen.
        - In turn, if ``hue`` was given, ``pcolormesh``-like plot kinds are no
          longer applicable, thus a ``line``-like argument needs to be chosen.

    These two special cases are specified via the extra keys ``with_x_and_y``
    and ``with_hue`` in the kind mapping.

    A kind mapping may look like this:

    .. literalinclude:: ../../dantro/plot/funcs/generic.py
        :language: python
        :start-after: _AUTO_PLOT_KINDS = {  # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    Args:
        d (Union[xarray.DataArray, xarray.Dataset]): The data for which to
            determine the plot kind.
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
        **plot_kwargs: All remaining plot function arguments, including any
            layout encoding arguments that aim to *fix* a dimension; these are
            used to determine the ``with_hue`` and ``with_x_and_y`` special
            cases. Everything else is ignored.

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

    # Handle special cases ...
    # ... for datasets: always fall back to the specified default kind
    if not hasattr(d, "ndim"):
        return kind_map["dataset"]

    # ... for given x *and* y layout specifiers
    elif plot_kwargs.get("x") and plot_kwargs.get("y"):
        return kind_map["with_x_and_y"]

    # ... for given hue layout specifier
    elif plot_kwargs.get("hue"):
        return kind_map["with_hue"]

    # Select the kind from the dimensionality. If this fails, use the default
    # value instead.
    try:
        kind = kind_map[d.ndim]
    except KeyError:
        kind = kind_map["fallback"]

    log.remark("Using plot kind '%s' for %d-dimensional data.", kind, d.ndim)
    return kind


def determine_encoding(
    dims: Union[List[str], Dict[str, int]],
    *,
    kind: str,
    auto_encoding: Union[bool, dict],
    default_encodings: dict,
    allow_y_for_x: List[str] = ("line",),
    plot_kwargs: dict,
) -> dict:
    """Determines the layout encoding for the given plot kind and the available
    data dimensions (as specified by the ``dims`` argument).

    If ``auto_encoding`` does not evaluate to true or ``kind is None``, this
    function does nothing and simply returns all given plotting arguments.

    Otherwise, it uses the chosen plot ``kind`` to associate layout specifiers
    with dimension names of ``d``.
    The available layout encoding specifiers (``x``, ``y``, ``col`` etc.) can
    be specified in two ways:

        - By default, ``default_encodings`` is used as a map from plot kind to
          a sequence of available layout specifiers.
        - If ``auto_encoding`` is a dictionary, the default map will be
          *updated* with that dictionary.

    The association is done in the following way:

        1. Inspecting ``plot_kwargs``, all layout encoding specifiers are
           extracted, dropping those that evaluate to False.
        2. The encodings mapping is determined (see above).
        3. The available dimension names are determined from ``dims``.
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

    .. literalinclude:: ../../dantro/plot/funcs/generic.py
        :language: python
        :start-after: _XR_PLOT_KINDS = {  # --- start literalinclude
        :end-before:  }   # --- end literalinclude
        :dedent: 4

    This function also implements **automatic column wrapping**, aiming to
    produce a more square-like figure with column wrapping. The prerequisites
    are the following:

        * The ``dims`` argument is a dict, containing size information
        * The ``col_wrap`` argument is given and set to ``"auto"``
        * The ``col`` specifier is in use
        * The ``row`` specifier is *not* used, i.e. wrapping is possible
        * There are more than three columns

    In such a case, ``col_wrap`` will be set to ``ceil(sqrt(num_cols))``.
    Otherwise, the entry will be removed from the plot arguments.

    Args:
        dims (Union[List[str], Dict[str, int]]): The dimension names (and, if
            given as dict: their sizes) that are to be encoded. If no sizes are
            provided, the assignment order will be the same as in the given
            sequence of dimension names. If sizes are given, these will be used
            to sort the dimension names in descending order of their sizes.
        kind (str): The chosen plot kind. If this was None, will directly
            return, because auto-encoding information is missing.
        auto_encoding (Union[bool, dict]): Whether to perform auto-encoding.
            If a dict, will regard it as a mapping of available encodings and
            update ``default_encodings``.
        default_encodings (dict): A map from plot kinds to available layout
            specifiers, e.g. ``{"line": ("x", "hue", "col", "row")}``.
        allow_y_for_x (List[str], optional): A list of plot kinds for which the
            following replacement will be allowed: if a ``y`` specifier is
            given but *no* ``x`` specifier, the ``"x"`` in the list of
            available encodings will be replaced by a ``"y"``. This is to
            support plots that allow *either* an ``x`` or a ``y`` specifier,
            like the ``line`` kind.
        plot_kwargs (dict): The actual plot function arguments, including any
            layout encoding arguments that aim to *fix* a dimension. Everything
            else is ignored.
    """
    if not auto_encoding or kind is None:
        log.debug("Layout auto-encoding was disabled (kind: %s).", kind)
        return plot_kwargs

    log.note(
        "Automatically determining layout encoding for kind '%s' ...", kind
    )

    # Evaluate supported encodings, then get the available encoding specifiers
    encs = copy.deepcopy(default_encodings)
    if isinstance(auto_encoding, dict):
        encs.update(auto_encoding)

    encoding_specs = encs[kind]

    # Special case for line-like kinds
    if allow_y_for_x and kind in allow_y_for_x:
        if plot_kwargs.get("y") and not plot_kwargs.get("x"):
            encoding_specs = tuple(
                s if s != "x" else "y" for s in encoding_specs
            )

    # Split plotting kwargs into a dict of layout specifiers and one that only
    # includes the remaining plotting kwargs
    plot_kwargs = copy.deepcopy(plot_kwargs)
    specs = {k: v for k, v in plot_kwargs.items() if k in encoding_specs}
    plot_kwargs = {k: v for k, v in plot_kwargs.items() if k not in specs}

    # -- Determine specifiers, depending on kind and dimensionality
    # Get all available dimension names. If size-information is available,
    # sort them by size (descending), otherwise just use them as they are.
    if isinstance(dims, dict):
        dim_names = [
            name
            for name, _ in sorted(
                dims.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
    else:
        dim_names = list(dims)

    # Some dimensions and specifiers might already have been associated;
    # determine those that have *not* yet been associated:
    free_specs = [s for s in encoding_specs if not specs.get(s)]
    free_dim_names = [name for name in dim_names if name not in specs.values()]

    log.debug("   given specifiers:  %s", specs)
    log.debug("   free specifiers:   %s", ", ".join(free_specs))
    log.debug("   free dimensions:   %s", ", ".join(free_dim_names))

    # From these two lists, update the specifier dictionary
    specs.update(
        {s: dim_name for s, dim_name in zip(free_specs, free_dim_names)}
    )

    # Drop those specifiers that are effectively unset.
    specs = {s: dim_name for s, dim_name in specs.items() if dim_name}

    # Provide information about the chosen encoding
    log.remark(
        "   encoding:  %s",
        ", ".join([f"{s}: {d}" for s, d in specs.items()]),
    )
    log.remark(
        "   free:      %s",
        ", ".join([k for k in encoding_specs if k not in specs]),
    )

    # -- Automatic column wrapping
    if plot_kwargs.get("col_wrap") == "auto":
        if (
            not specs.get("row")
            and specs.get("col")
            and hasattr(dims, "items")  # i.e.: dict-like
            and dims[specs["col"]] > 3
        ):
            num_cols = dims[specs["col"]]
            plot_kwargs["col_wrap"] = math.ceil(math.sqrt(num_cols))
            log.remark(
                "   col_wrap:  %d  (length of col dimension: %d)",
                plot_kwargs["col_wrap"],
                num_cols,
            )
        else:
            # Remove it to avoid a plot warning or "unexpected argument"
            del plot_kwargs["col_wrap"]

    # Finally, return the merged layout specifiers and plot kwargs
    return dict(**plot_kwargs, **specs)


class make_facet_grid_plot:
    """This is a decorator class that transforms a plot function that works on
    a single axis into one that supports faceting via
    :py:class:`xarray.plot.FacetGrid`.

    Additionally, it allows to register the plotting function with the generic
    :py:func:`~dantro.plot.funcs.generic.facet_grid` plot by adding the
    callable to ``_FACET_GRID_FUNCS``.
    """

    MAP_FUNCS = {
        "dataset": lambda fg, f, **kws: fg.map_dataset(f, **kws),
        "dataarray": lambda fg, f, **kws: fg.map_dataarray(f, **kws),
        "dataarray_line": lambda fg, f, **kws: fg.map_dataarray_line(f, **kws),
    }
    """The available mapping functions in :py:class:`xarray.plot.FacetGrid`"""

    DEFAULT_ENCODINGS = ("col", "row", "frames")
    """The default encodings the facet grid supplies; these are those supported
    by the generic facet grid function"""

    DEFAULT_DROP_KWARGS = ("_fg", "meta_data", "hue_style", "add_guide")
    """The default kwargs that are to be dropped rather than passed on to the
    wrapped plotting function.
    Can be customized via ``drop_kwargs`` argument."""

    def __init__(
        self,
        *,
        map_as: str,
        encodings: Tuple[str],
        supported_hue_styles: Tuple[str] = None,
        register_as_kind: Union[bool, str] = True,
        overwrite_existing: bool = False,
        drop_kwargs: Tuple[str] = DEFAULT_DROP_KWARGS,
        parse_cmap_and_norm_kwargs: bool = True,
        **default_kwargs,
    ):
        """Initialize the decorator, making the decorated function capable of
        performing a facet grid plot.

        Args:
            map_as (str): Which mapping to use. Available: ``dataset``,
                ``dataarray`` and ``dataarray_line``.
            encodings (Tuple[str]): The encodings supported by the wrapped
                plot function, e.g. ``("x", "hue")``.
                Note that these *need to be dimensionality-reducing encodings*
                that have a qualitatively similar effect as ``col`` & ``row``
                in that they consume a data *dimension*. This is in contrast to
                plots that may represent multiple data *variables*, e.g. if the
                data comes from a  :py:class:`xarray.Dataset`; those should not
                be specified here.
            supported_hue_styles (Tuple[str]): Which hue styles are
                supported by the wrapped plot function. It is suggested to set
                this value if mapping via ``dataset`` or ``dataarray_line`` in
                order to disallow configurations that will not work with the
                wrapped plot function. If set to None, no check will be done.
            register_as_kind (Union[bool, str], optional): If boolean, controls
                *whether* to register the wrapped function with the generic
                facet grid plot, using its own name. If a string, uses that
                name for registration.
            overwrite_existing (bool, optional): Whether to overwrite an
                existing registration in ``_FACET_GRID_FUNCS``. If False, an
                existing entry of the same ``register_as_kind`` value will
                lead to an error.
            drop_kwargs (Tuple[str], optional): Which keyword arguments to
                drop before invocation of the wrapped function; this can be
                useful to trim down the signature of the wrapped function.
            parse_cmap_and_norm_kwargs (bool, optional): Whether to parse
                colormap-related plot function arguments using the
                :py:func:`~dantro.plot.utils.color_mngr.parse_cmap_and_norm_kwargs`
                function. Should be set to false if the decorated plot function
                takes care of these arguments itself.
            **default_kwargs: Additional arguments that are passed to the
                single-axis plotting function. These are used both when calling
                it via the selected mapping function and when invoking it
                without a facet grid.
                These are recursively updated with those given upon plot
                function invocation.
        """
        try:
            self.map_func = self.MAP_FUNCS[map_as]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported value for `map_as` argument: '{map_as}'! Needs "
                f"to be one of:  {', '.join(self.MAP_FUNCS)}"
            )

        self.encodings = encodings
        self.supported_hue_styles = supported_hue_styles
        self.register_as_kind = register_as_kind
        self.overwrite_existing = overwrite_existing
        self.drop_kwargs = drop_kwargs if drop_kwargs else ()
        self.default_kwargs = default_kwargs
        self.parse_cmap_and_norm_kwargs = parse_cmap_and_norm_kwargs

    def parse_wpf_kwargs(self, data, **kwargs) -> dict:
        """Parses the keyword arguments in preparation for invoking the wrapped
        plot function. This can happen both in context of a facet grid mapping
        and a single invocation.
        """
        # Update from defaults
        kwargs = recursive_update(copy.deepcopy(self.default_kwargs), kwargs)

        # Some checks
        if (
            self.supported_hue_styles is not None
            and "hue_style" in kwargs
            and kwargs["hue_style"] not in self.supported_hue_styles
        ):
            raise ValueError(
                f"The selected `hue_style` '{kwargs['hue_style']}' is not "
                "supported for this plotting function! May only be:  "
                f"{', '.join(self.supported_hue_styles)}"
            )

        # Parse colormap-related arguments
        if self.parse_cmap_and_norm_kwargs:
            kwargs = parse_cmap_and_norm_kwargs(**copy.deepcopy(kwargs))

        # Can do more pre-processing here
        # ...

        return kwargs

    def __call__(self, plot_single_axis: Callable) -> Callable:
        """Generates a standalone DAG-based plotting function that supports
        faceting. Additionally, integrates it as ``kind`` for the
        general facet grid plotting function by adding it to the global
        ``_FACET_GRID_FUNCS`` dictionary.
        """
        # First, wrap the single-axis plot function to achieve helper support
        def wrapped_plot_func(
            *args,
            hlpr: PlotHelper,
            _is_facetgrid: bool,
            ax=None,
            _fg: "xr.plot.FacetGrid" = None,
            **kwargs,
        ):
            """Wraps the single-axis plotting function and performs the
            following additional operations before invoking it:

                1. Sync the plot helper to the given axis (if faceting)
                2. Evaluates ``drop_kwargs`` to reduce the passed arguments
            """
            # If this is called as part of a facet grid plot, we need to sync
            # the helper to the given axis, otherwise the helper cannot be used
            if _is_facetgrid:
                hlpr.select_axis(ax=ax)

            # Prepare kwargs, optionally dropping some keys that bloat the
            # function signature ...
            kwargs["_fg"] = _fg
            kwargs["_is_facetgrid"] = _is_facetgrid
            kwargs = {
                k: v for k, v in kwargs.items() if k not in self.drop_kwargs
            }

            # Now invoke the single-axis plotting function
            return plot_single_axis(*args, hlpr=hlpr, **kwargs)

        # Get the mapping function
        map_to_facet_grid = self.map_func

        # Now, generate the facet-grid supporting function
        def fgplot(
            data,
            *,
            hlpr=None,
            col: str = None,
            row: str = None,
            col_wrap: int = None,
            sharex: bool = True,
            sharey: bool = True,
            figsize: tuple = None,
            aspect: float = 1.0,
            size: float = 3.0,
            subplot_kws: dict = None,
            **kwargs,
        ):
            """A facet-grid capable version of the given plot function.

            Explicitly named arguments here are passed to the setup of the
            :py:class:`xarray.plot.FacetGrid`; all ``kwargs`` are passed on to
            the selected mapping function and subsequently: the wrapped
            single-axis plot function.
            """
            # Without columns or rows, cannot use facet grid. Make a primitive
            # plot instead, directly using the wrapped plot function.
            if not col and not row:
                log.debug("No `col` or `row` set. Not using a facet grid.")

                kwargs = self.parse_wpf_kwargs(data, **kwargs)
                log.debug(
                    "Invoking single-axis plot function with kwargs:  %s",
                    kwargs,
                )

                hlpr.setup_figure()  # TODO Find out why this is necessary ...
                return wrapped_plot_func(
                    data, hlpr=hlpr, _is_facetgrid=False, **kwargs
                )

            # Prepare facet grid and helper
            log.debug(
                "Setting up a facet grid (col: %s, row: %s) ...", col, row
            )
            fg = xr.plot.FacetGrid(
                data,
                col=col,
                row=row,
                col_wrap=col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                aspect=aspect,
                size=size,
                subplot_kws=subplot_kws if subplot_kws else {},
            )
            hlpr.attach_figure_and_axes(fig=fg.fig, axes=fg.axes)

            # Make the FacetGrid object available to the helper
            hlpr._attrs["facet_grid"] = fg

            # Parse arguments expected by wrapped plot function
            kwargs = self.parse_wpf_kwargs(data, **kwargs)

            # Prepare mapping keyword arguments and apply the mapping
            log.debug("Invoking mapping function with kwargs  %s  ...", kwargs)
            try:
                map_to_facet_grid(
                    fg, wrapped_plot_func, hlpr=hlpr, _fg=fg, **kwargs
                )

            except Exception as exc:
                raise PlottingError(
                    f"Failed mapping {type(data)} data to facet grid! Check "
                    "the given arguments, dimensionality, dimension names, "
                    "and whether the dimensions have coordinates associated.\n"
                    f"Got a {type(exc).__name__}: {exc}"
                ) from exc

            # Return the FacetGrid object for further handling
            return fg

        # facet grid plot function constructed now.
        # ... register it as a single-axis facet grid plot kind.
        if self.register_as_kind:
            if isinstance(self.register_as_kind, str):
                regname = self.register_as_kind
            else:
                regname = plot_single_axis.__name__

            if regname in _FACET_GRID_FUNCS or regname in _XR_PLOT_KINDS:
                if not self.overwrite_existing:
                    _in_use = ", ".join(
                        list(_FACET_GRID_FUNCS) + list(_XR_PLOT_KINDS)
                    )
                    raise ValueError(
                        f"The plot function name '{regname}' is already used! "
                        "Either set `register_as_kind` to a different value, "
                        "or set `overwrite_existing`. Registered functions: "
                        f"{_in_use}"
                    )

            # Register the callable for the non-standalone case
            _FACET_GRID_FUNCS[regname] = fgplot
            log.debug("Registered '%s' as special facet grid kind.", regname)

            _FACET_GRID_KINDS[regname] = (
                self.encodings + self.DEFAULT_ENCODINGS
            )
            log.debug(
                "Registered '%s' encodings:  %s",
                regname,
                ", ".join(_FACET_GRID_KINDS[regname]),
            )

        # Build the standalone plot function, which takes the place of the
        # decorated plot function
        @is_plot_func(use_dag=True, required_dag_tags=("data",))
        def standalone(*, data: dict, hlpr: PlotHelper, **kwargs):
            try:
                return fgplot(data["data"], hlpr=hlpr, **kwargs)

            except Exception as exc:
                raise PlottingError(
                    "Standalone facet grid plotting for plot function "
                    f"'{plot_single_axis.__name__}' failed!\n"
                    f"Got {type(exc).__name__}: {exc}\n\n"
                    f"Given arguments:\n  {kwargs}\n\n"
                    f"Selected data:\n  {data['data']}\n"
                ) from exc

        return standalone


# -----------------------------------------------------------------------------
# -- Facet Grid ---------------------------------------------------------------
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
    squeeze: bool = True,
    **plot_kwargs,
):
    """A generic facet grid plot function for high dimensional data.

    This function calls the ``data['data'].plot`` function if no plot ``kind``
    is given, otherwise ``data['data'].plot.<kind>``.
    It is designed for `plotting with xarray objects <http://xarray.pydata.org/en/stable/plotting.html>`_,
    i.e. :py:class:`xarray.DataArray` and :py:class:`xarray.Dataset`.
    Specifying the kind of plot requires the data to be of one of those types
    and have a dimensionality that can be represented in these plots. See
    `the correponding API documentation <https://xarray.pydata.org/en/stable/api.html#plotting>`_ for more information.

    In most cases, this function creates a so-called
    :py:class:`xarray.plot.FacetGrid` object that automatically layouts and
    chooses a visual representation that fits the dimensionality of the data.
    To specify which data dimension should be represented in which way, it
    supports a declarative syntax: via the optional keyword arguments ``x``,
    ``y``, ``row``, ``col``, and/or ``hue`` (available options are listed in
    the corresponding `plot function documentation <https://xarray.pydata.org/en/stable/api.html#plotting>`_),
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
        * allows to register additional plot ``kind`` values that create plots
          with a custom single-axis plotting function, using the
          :py:class:`~dantro.plot.funcs.generic.make_facet_grid_plot`
          decorator.

    For details about auto-encoding and how the plot ``kind`` is chosen, see
    :py:func:`~dantro.plot.funcs.generic.determine_encoding`
    and :py:func:`~dantro.plot.funcs.generic.determine_plot_kind`.

    .. note::

        When specifying ``frames``, the ``animation`` arguments need to be
        specified. See :ref:`here <pcr_pyplot_animations>` for more information
        on the expected animation parameters.

        The value of the ``animation.enabled`` key is not relevant for this
        function; it will automatically enter or exit animation mode,
        depending on whether the ``frames`` argument is given or not. This uses
        the :ref:`animation mode switching <pcr_pyplot_animation_mode_switching>`
        feature.

    .. note::

        Internally, this function calls ``.squeeze`` on the selected data, thus
        being more tolerant with data that has size-1 dimension coordinates.
        To suppress this behaviour, set the ``squeeze`` argument accordingly.

    .. warning::

        Depending on ``kind`` and the dimensionality of the data, some plot
        functions might create their own figure, disregarding any previously
        set up figure. This includes the figure from the plot helper.

        To control figure aesthetics, you can either specify matplotlib RC
        :ref:`style parameters <pcr_pyplot_style>` (via the ``style`` argument),
        or you can use the ``plot_kwargs`` to pass arguments to the respective
        plot functions. For the latter, refer to the respective documentation
        to find out about available arguments.

    Args:
        data (dict): The data selected by the data transformation framework,
            expecting the ``data`` key.
        hlpr (PlotHelper): The plot helper
        kind (str, optional): The kind of plot to use. Options are:
            ``contourf``, ``contour``, ``imshow``, ``line``, ``pcolormesh``,
            ``step``, ``hist``, ``scatter``, ``errorbars`` and any plot kinds
            that were additionally registered via the
            :py:class:`~dantro.plot.funcs.generic.make_facet_grid_plot`
            decorator.
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
        squeeze (bool, optional): whether to squeeze the data before plotting,
            such that size-1 dimensions do not take up encoding dimensions.
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
    # Make sure to have the latest module-level variables available here; this
    # is important to ensure that those `kind`s registered by the
    # make_facet_grid_plot decorator are available here.
    from .generic import _FACET_GRID_FUNCS, _FACET_GRID_KINDS

    # .........................................................................
    def plot_frame(_d, *, kind: str, plot_kwargs: dict):
        """Plot a FacetGrid frame"""
        # Squeeze size-1 dimension coordinates to non-dimension coordinates
        if squeeze:
            _d = _d.squeeze()

        # Retrieve the generic or specialized plot function, depending on kind
        if kind is None:
            plot_func = _d.plot

        elif kind in _FACET_GRID_FUNCS:
            _plot_func = _FACET_GRID_FUNCS[kind]

            # Bind the data and helper to the function
            plot_func = _partial(_plot_func, _d, hlpr=hlpr)

        else:
            try:
                plot_func = getattr(_d.plot, kind)

            except AttributeError as err:
                _available_xr = ", ".join(_XR_PLOT_KINDS)
                _available_dtr = ", ".join(_FACET_GRID_FUNCS)
                raise AttributeError(
                    f"The plot kind '{kind}' seems not to be available for "
                    f"data of type {type(_d)}! Please check the documentation "
                    "regarding the expected data types. For xarray data "
                    f"structures, valid choices are:  {_available_xr}.\n"
                    "Additionally, the following facet grid kinds were "
                    f"registered from within dantro:  {_available_dtr}"
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
                raise PlottingError(
                    "facet_grid plotting failed, most probably because the "
                    "dimensionality of the data, the chosen plot kind "
                    f"({kind}) and the specified layout encoding were not "
                    "compatible or because the selected data was missing "
                    "coordinates for one or more dimensions.\n"
                    "For debugging, inspect the chained traceback and the "
                    "information below.\n\n"
                    f"The upstream error was a {type(exc).__name__}: {exc}\n\n"
                    f"xr.plot.FacetGrid arguments:\n  {plot_kwargs}\n\n"
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
            # Use figure already attached to helper if they exist; these are
            # certain to be the currently relevant figure and axes.
            # If not available, let matplotlib decide.
            fig = plt.gcf() if hlpr._fig is None else hlpr.fig
            axes = plt.gca() if hlpr._axes is None else hlpr.axes

        # When now attaching the new figure and axes, the previously existing
        # figure (the one .clear()-ed above) is closed and discarded.
        # If the figure extracted here is identical to the already-associated
        # figure, nothing happens.
        hlpr.attach_figure_and_axes(fig=fig, axes=axes, skip_if_identical=True)

        # Store the FacetGrid instance for potential later manipulation
        hlpr._attrs["facet_grid"] = rv

        # Done with this frame now.

    # Actual plotting routine starts here .....................................
    # Get the Dataset, DataArray, or other compatible data
    d = data["data"]

    # Determine kind and encoding, updating the plot kwargs accordingly.
    # NOTE Need to pop all explicitly given specifiers in order to not have
    #      them appear as part of plot_kwargs further downstream.
    kind = determine_plot_kind(
        d, kind=kind, default_kind_map=_AUTO_PLOT_KINDS, **plot_kwargs
    )
    plot_kwargs = determine_encoding(
        d.sizes,
        kind=kind,
        auto_encoding=auto_encoding,
        default_encodings=_FACET_GRID_KINDS,
        plot_kwargs=dict(
            frames=frames,
            **plot_kwargs,
        ),
    )
    frames = plot_kwargs.pop("frames", None)

    # Parse colorbar-related arguments
    plot_kwargs = parse_cmap_and_norm_kwargs(**plot_kwargs)

    # Done parsing arguments
    log.note("Facet grid plot of kind '%s' now commencing ...", kind)

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


# -- Additional facet-grid supporting plots -----------------------------------

# TODO Should support errors along x as well!
@make_facet_grid_plot(
    map_as="dataset",
    encodings=("x", "hue"),
    supported_hue_styles=("discrete",),
    #
    # defaults
    hue_style="discrete",
    add_guide=False,
)
def errorbars(
    ds: "xarray.Dataset",
    *,
    _is_facetgrid: bool,
    hlpr: PlotHelper,
    y: str,
    yerr: str,
    x: str = None,
    hue: str = None,
    hue_fstr: str = "{value:}",
    use_bands: bool = False,
    add_legend: bool = True,
    **kwargs,
):
    """An errorbar plot supporting facet grid.

    This function makes use of a decorator to implement faceting support:
    :py:class:`~dantro.plot.funcs.generic.make_facet_grid_plot`.
    It additionally registers this plot as an available plot ``kind`` in
    :py:func:`~dantro.plot.funcs.generic.facet_grid`.

    .. note::

        This plot function is heavily wrapped by the decorator, which is why
        not all functionality is exposed here. Instead, the arguments seen here
        are those that apply to a *single* subplot of a facet grid.

    Uses :py:func:`~dantro.plot.funcs._utils.plot_errorbar` for
    plotting individual lines.

    Args:
        ds (xarray.Dataset): The dataset containing the errorbar data
        _is_facetgrid (bool): Indicates whether this plot is called as part of
            a facet grid or whether no faceting takes place (i.e. when neither
            columns nor rows are available for faceting). In such a case, this
            plot supplies metadata to the plot helper to draw axis labels etc.
            (For internal use only, no need to pass this parameter.)
        hlpr (PlotHelper): The plot helper, exposing the currently selected
            axis via ``hlpr.ax``.
        y (str): Which data variable to use for the y-axis values
        yerr (str): Which data variable to use for the errorbars or bands
        x (str, optional): Which data dimension to plot on the x-axis
        hue (str, optional): Which data dimension to represent via hues
        hue_fstr (str, optional): A format string that is used to build the
            label of discrete hue encoding.
        use_bands (bool, optional): Whether to use errorbands instead of bars.
        add_legend (bool, optional): Whether to add a legend to the individual
            plot or to the figure
        **kwargs: Passed on to ``hlpr.ax.errorbar`` via
            :py:func:`~dantro.plot.funcs._utils.plot_errorbar`.
    """
    # Prepare data
    _y = ds[y]
    _yerr = ds[yerr]

    # Try to infer x, if not given
    x = x if x else [dim for dim in _y.dims if dim not in (hue,)][0]
    _x = ds.coords[x]

    # If this is not a facet grid, still show some labels
    if not _is_facetgrid:
        # FIXME Should do this via helper, but not working (see #82)
        # hlpr.provide_defaults("set_labels", x=x, y=f"{y} & {yerr}")

        # Workaround:
        hlpr.ax.set_xlabel(x)
        hlpr.ax.set_ylabel(f"{y}, {yerr}")

    # Case: No hue dimension -> plot single errorbar line
    if hue is None:
        _plot_errorbar(
            ax=hlpr.ax,
            x=_x,
            y=_y,
            yerr=_yerr,
            fill_between=use_bands,
            **kwargs,
        )
        return

    # else: will plot multiple lines
    # Keep track of legend handles and labels
    _handles, _labels = [], []

    # Group by the hue dimension and perform plots. To be a bit more permissive
    # regarding data shape, squeeze out any additional dimensions that might
    # have been left over.
    hue_iter = zip(_y.groupby(hue), _yerr.groupby(hue))
    for (_y_coord, _y_vals), (_yerr_coord, _yerr_vals) in hue_iter:
        _y_vals = _y_vals.squeeze(drop=True)
        _yerr_vals = _yerr_vals.squeeze(drop=True)

        label = hue_fstr.format(dim=hue, value=_y_coord)
        handle = _plot_errorbar(
            ax=hlpr.ax,
            x=_x,
            y=_y_vals,
            yerr=_yerr_vals,
            label=label,
            fill_between=use_bands,
            **kwargs,
        )
        _handles.append(handle)
        _labels.append(label)

    # Either do a single-axis legend or prepare for figure-level legend
    if not _is_facetgrid:
        if add_legend:
            hlpr.ax.legend(_handles, _labels, title=hue)

    else:
        hlpr.track_handles_labels(_handles, _labels)
        if add_legend:
            hlpr.provide_defaults("set_figlegend", title=hue)


# .............................................................................


@make_facet_grid_plot(
    map_as="dataset",
    register_as_kind="scatter3d",
    encodings=("hue", "markersize"),  # TODO correct?!
    supported_hue_styles=("continuous",),
    parse_cmap_and_norm_kwargs=False,
    # defaults
    # hue_style="discrete",  # FIXME setting to 'discrete' fails, but shouldn't
)
def scatter3d(
    ds: "xarray.Dataset",
    *,
    _is_facetgrid: bool,
    hlpr: PlotHelper,
    x: str,
    y: str,
    z: str,
    hue: str = None,
    markersize: Union[float, str] = None,
    size_mapping: dict = None,
    cmap: Union[str, dict, mcolors.Colormap] = None,
    norm: Union[str, dict, mcolors.Normalize] = None,
    vmin: float = None,
    vmax: float = None,
    add_colorbar: bool = True,
    cbar_kwargs: dict = None,
    **kwargs,
):
    """A 3-dimensional scatter plot supporting facet grid.

    This function makes use of a decorator to implement faceting support:
    :py:class:`~dantro.plot.funcs.generic.make_facet_grid_plot`.
    It additionally registers this plot as an available plot ``kind`` in
    :py:func:`~dantro.plot.funcs.generic.facet_grid`.

    .. note::

        This plot relies on the figure projection having been set to 3D,
        which can be achieved via:

        .. code-block:: yaml

            my_3d_plot:
              # ...
              # for faceting:
              subplot_kws: &projection
                projection: 3d

              # for single plot:
              helpers:
                set_figure:
                  subplot_kw:  # sic
                    <<: *projection

        There *may* also be a base plot configuration that does this.

    .. warning::

        Support of :ref:`auto-encoding <dag_generic_auto_encoding>` and of the
        ``hue`` and ``markersize`` encodings is not as general as it could be.
        If you get dimensionality- or size-related errors, that's probably due
        to an incompatible combination of encodings.

    .. note::

        This plot function is heavily wrapped by the decorator, which is why
        not all functionality is exposed here. Instead, the arguments seen here
        are those that apply to a *single* subplot of a facet grid.

    Args:
        ds (xarray.Dataset): The dataset containing the data
        _is_facetgrid (bool): Indicates whether this plot is called as part of
            a facet grid or whether no faceting takes place (i.e. when neither
            columns nor rows are available for faceting). In such a case, this
            plot supplies metadata to the plot helper to draw axis labels etc.
            (For internal use only, no need to pass this parameter.)
        hlpr (PlotHelper): The plot helper, exposing the currently selected
            axis via ``hlpr.ax``.
        x (str): Which data variable to plot on the x-axis
        y (str): Which data variable to plot on the y-axis
        z (str): Which data variable to plot on the z-axis
        hue (str, optional): Which dimension or variable to represent via hues
        markersize: (str, optional): Which data *dimension* to plot using the
            markersize. Note that if ``hue`` is given this needs to match the
            size of that dimension.
            Whether using data *variables*  here depends on the dimensionality
            of the data; don't be surprised by a cryptic error message from
            deep within xarray.
        size_mapping: (dict, optional): A dictionary containing the facet grid
            ``size_mapping``. Is overwritten by ``markersize``, if passed.
        cmap (Union[str, dict, matplotlib.colors.Colormap], optional): The
            colormap, passed to the
            :py:class:`~dantro.plot.utils.color_mngr.ColorManager`.
        norm (Union[str, dict, matplotlib.colors.Normalize], optional):
            The norm that is applied for the color-mapping.
        vmin (float, optional): The lower bound of the color-mapping,
            passed to the
            :py:class:`~dantro.plot.utils.color_mngr.ColorManager`.
            Ignored if norm evaluates to ``BoundaryNorm``.
        vmax (float, optional): The upper bound of the color-mapping,
            passed to the
            :py:class:`~dantro.plot.utils.color_mngr.ColorManager`.
            Ignored if norm evaluates to ``BoundaryNorm``.
        add_colorbar (bool, optional): Whether to add a colorbar
        cbar_kwargs (dict, optional): Arguments for colorbar creation.
        **kwargs: Passed on to :py:func:`matplotlib.axes.Axes.scatter` or, if
            ``z`` is given, the equivalent 3D axes.

    Raises:
        AttributeError: If the active axes does not have a ``zaxis``.
            In that case, you probably forgot to set the figure's projection,
            see above.
    """

    def get_var(v: str) -> xr.DataArray:
        """Retrieves a data variable from the dataset, making some checks"""
        d = ds[v]
        if d.ndim != 1:
            raise ValueError(
                f"Unexpected data dimensionality for variable '{v}'! "
                "On the subplot-level, data variables should be 1D, but "
                f"ds['{v}'] was {d.ndim}-dimensional: {dict(d.sizes)}"
            )
        return d

    if not hasattr(hlpr.ax, "zaxis"):
        raise AttributeError(
            "Missing z-axis! Did you set the "
            "projection (via `subplot_kws` or `setup_figure` helper)?"
        )

    cm = ColorManager(
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    shared_kwargs = dict(
        c=get_var(hue) if hue is not None else None,
        cmap=cm.cmap if cmap is not None else None,
        norm=cm.norm if norm is not None else None,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
    )

    # Add the 's' key to the kwargs. If both size_mapping and markersize are
    # passed, 'markersize' will take precedent.
    if size_mapping is not None:
        shared_kwargs["s"] = size_mapping.values

    if not _is_facetgrid and markersize is not None:
        shared_kwargs["s"] = get_var(markersize).values

    im = hlpr.ax.scatter(
        get_var(x),
        get_var(y),
        get_var(z),
        **shared_kwargs,
        **kwargs,
    )

    # Postprocess
    if not _is_facetgrid and hue is not None and add_colorbar:
        # TODO This should read information from the FacetGrid's cbar_kwargs,
        #      which are also parsed there...
        cm.create_cbar(
            im,
            fig=hlpr.fig,
            ax=hlpr.ax,
            **(cbar_kwargs if cbar_kwargs else {}),
        )

    # FIXME Should do this via helper, but not working (see #82)
    # hlpr.provide_defaults("set_labels", x=x, y=y, z=z)
    hlpr.ax.set_xlabel(x)
    hlpr.ax.set_ylabel(y)
    hlpr.ax.set_zlabel(z)

    return im
