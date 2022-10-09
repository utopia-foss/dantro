"""Implements the :py:class:`.ColorManager` which simplifies working with
:py:class:`matplotlib.colors.Colormap` and related objects."""

import copy
import logging
from math import ceil, floor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.colors
import numpy as np
from matplotlib.colors import to_rgb

from ...tools import make_columns, parse_str_to_args_and_kwargs

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

NORMS = {
    "Normalize": mpl.colors.Normalize,
    "BoundaryNorm": mpl.colors.BoundaryNorm,
    "CenteredNorm": mpl.colors.CenteredNorm,
    "NoNorm": mpl.colors.NoNorm,
    "LogNorm": mpl.colors.LogNorm,
    "PowerNorm": mpl.colors.PowerNorm,
    "SymLogNorm": mpl.colors.SymLogNorm,
    "TwoSlopeNorm": mpl.colors.TwoSlopeNorm,
    "FuncNorm": mpl.colors.FuncNorm,
}
"""matplotlib color normalizations supported by the :py:class:`.ColorManager`.
See the :py:mod:`matplotlib.colors` module for more information.
"""


# -----------------------------------------------------------------------------


class ColorManager:
    """Custom color manager which provides an interface to the
    :py:mod:`matplotlib.colors` module and aims to simplify working with
    colormaps, colorbars, and different normalizations.
    """

    _NORMS_NOT_SUPPORTING_VMIN_VMAX: Tuple[str] = (
        "BoundaryNorm",
        "CenteredNorm",
    )
    """Names of norms that do *not* support getting passed the ``vmin`` and
    ``vmax`` arguments."""

    _POSSIBLE_CMAP_KWARGS: Tuple[str] = (
        "name",
        "colors",
        "segmentdata",
        "bad",
        "under",
        "over",
        "reversed",
        "N",
        "gamma",
        #
        # ColorManager-internal
        "placeholder_color",
        "continuous",
        "from_values",
    )
    """Keyword arguments that are used by matplotlib or the ColorManager to
    construct colormaps. If using the implicit syntax for defining labels and
    colormap values, these can *not* be used for labels."""

    _SNS_COLOR_PALETTE_PREFIX: str = "color_palette::"
    """If a colormap ``name`` starts with this string, will use
    :py:func:`seaborn.color_palette` to generate the colormap"""

    _SNS_DIVERGING_PALETTE_PREFIX: str = "diverging::"
    """If a colormap ``name`` starts with this string, will use
    :py:func:`seaborn.diverging_palette` to generate the colormap, parsing the
    remaining parts of the name into positional and keyword arguments."""

    # .........................................................................

    def __init__(
        self,
        *,
        cmap: Union[str, dict, list, mpl.colors.Colormap] = None,
        norm: Union[str, dict, mpl.colors.Normalize] = None,
        labels: Union[List[str], Dict[float, str]] = None,
        vmin: float = None,
        vmax: float = None,
        discretized: bool = None,
    ):
        """Initializes a :py:class:`.ColorManager` by building the colormap,
        the norm, and the colorbar labels.

        Refer to the :ref:`dedicated documentation page <color_mngr>` for
        examples and integration instructions.

        Args:
            cmap (Union[str, dict, list, matplotlib.colors.Colormap], optional):
                The colormap specification.
                If this is not already a :py:class:`matplotlib.colors.Colormap`
                instance, it will be parsed into a dict-like specification,
                which has the options as shown below.

                * If ``cmap`` is a string, it is turned into
                  ``dict(name=cmap)``.
                * If ``cmap`` is a list (or tuple), it will be converted to
                  ``dict(from_values=cmap)``, creating a segmented colormap.
                  See below for more information.

                In dict form, the following arguments are available:

                ``name`` (str, optional):
                    Name of a registered matplotlib colormap or None to use a
                    default. For available colormap names, see
                    `here <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.

                    Also **supports seaborn colormaps**. If the name starts
                    with the :py:attr:`._SNS_COLOR_PALETTE_PREFIX` string,
                    :py:func:`seaborn.color_palette` is used to generate the
                    colormap.
                    If starting with :py:attr:`._SNS_DIVERGING_PALETTE_PREFIX`,
                    :py:func:`seaborn.diverging_palette` is invoked, using
                    argument specified as part of the ``name``.

                    This opens many possibilities, as shown in the
                    `seaborn documentation <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.
                    For example:

                    .. code-block:: text

                        color_palette::YlOrBr
                        color_palette::icefire
                        color_palette::icefire_r          # reversed
                        color_palette::light:b            # white -> blue
                        color_palette::dark:b             # black -> blue
                        color_palette::light:#69d         # custom color
                        color_palette::light:#69d_r       # ... reversed
                        color_palette::dark:salmon_r      # named, reversed
                        color_palette::ch:s=-.2,r=.6      # cubehelix

                        diverging::220,20
                        diverging::145,300,s=60
                        diverging::250, 30, l=65, center=dark

                    Here, the ``ch:<key>=<val>,<key>=<val>`` syntax is used to
                    create a :py:func:`seaborn.cubehelix_palette`.
                    The same ``<arg>,<arg>,<key>=<val>,<key>=<val>`` syntax is
                    used for the diverging palette.

                    .. note::

                        When specifying these via YAML, make sure to put the
                        string into single or double quotes to avoid it being
                        interpreted as a YAML mapping.

                ``from_values`` (Union[dict, list], optional):
                    Dict of colors keyed by bin-specifier. If given, ``name``
                    is ignored and a discrete colormap is created from the list
                    of specified colors. The ``norm`` is then set to
                    :py:class:`matplotlib.colors.BoundaryNorm`.

                    The bins can be specified either by bin-centers (Scalar) or
                    by bin-intervals (2-tuples). For the former, the deduced
                    bin-edges are assumed halfway between the bin-centers. For
                    the latter, the given intervals must be pairwise connected.
                    In both cases, the bins must monotonically increase.

                    If a list of colors is passed they are automatically
                    assigned to the bin-centers ``[0, 1, 2, ...]``, potentially
                    shifted depending on ``vmin`` and ``vmax``. Inferring
                    these values is done in :py:meth:`_infer_pos_map`.

                    Alternatively, a continuous, linearly interpolated colormap
                    can be generated by setting the ``continuous`` flag, see
                    below. This will construct a
                    :py:class:`~matplotlib.colors.LinearSegmentedColormap`.
                    In such a case, keys in ``from_values`` can only be scalar,
                    bin *intervals* cannot be specified.
                ``continuous`` (bool, optional):
                    If True, will interpret the ``from_values`` data as
                    specifying points between which a linear interpolation is
                    carried out. Will create a
                    :py:class:`~matplotlib.colors.LinearSegmentedColormap`.
                ``under`` (Union[str, dict], optional):
                    Passed on to
                    :py:meth:`~matplotlib.colors.Colormap.set_under`
                ``over`` (Union[str, dict], optional):
                    Passed on to
                    :py:meth:`~matplotlib.colors.Colormap.set_over`
                ``bad`` (Union[str, dict], optional):
                    Passed on to
                    :py:meth:`~matplotlib.colors.Colormap.set_bad`
                ``placeholder_color`` (str, optional):
                    ``None`` values in ``from_values`` are replaced with this
                    color (default: white).
                ``reversed`` (bool, optional):
                    If True, will reverse the colormap.
                ``labels_and_colors`` (dict, optional):
                    This is a shorthand syntax for specifying colorbar labels
                    and colors at the same time.
                    Keys refer to labels, values to colors.
                    The label positions and bounds are inferred using
                    :py:meth:`_infer_pos_map` and are affected by ``vmin`` and
                    ``vmax``. These may also be given implicitly via
                    ``**kwargs`` (see below), but *not* at the same time!

                    Effectively, the mapping is unpacked into two parts:
                    The keys are used to specify the values of the ``labels``
                    dict (on the top-level); the values are used to specify
                    the values of the ``cmap.from_values`` dict (see above).
                    The keys are inferred from the length of the sequence and
                    ``vmin`` and ``vmax``, expecting to map to an integer
                    data positions.

                    **Example:**

                    .. code-block:: yaml

                        cmap:
                          empty: darkkhaki            # -> 0
                          susceptible: forestgreen    # -> 1
                          exposed: darkorange         # ...
                          infected: firebrick
                          recovered: slategray
                          deceased: black
                          source: maroon
                          inert: moccasin             # -> 7

                          # can still set extremes here (should not appear)
                          under: red
                          over: red

                ``**kwargs`` (optional):
                    Depending on the argument names, these are either passed
                    to colormap instantiation *or* are used to specify the
                    ``labels_and_colors`` mapping. For the latter, labels may
                    not be named after arguments that are relevant for
                    colormap initialization
                    (:py:attr:`._POSSIBLE_CMAP_KWARGS`).

            norm (Union[str, dict, matplotlib.colors.Normalize], optional):
                The norm that is applied for the color-mapping. If it is a
                string, the matching norm in :py:mod:`matplotlib.colors`
                is created with default values.
                If it is a dict, the ``name`` entry specifies the norm and all
                further entries are passed to its constructor.
                Overwritten if a discrete colormap is specified via
                ``cmap.from_values``.
            labels (Union[List[str], Dict[float, str]], optional): Colorbar
                tick-labels keyed by tick position. If a list of labels is
                passed they are automatically assigned to the positions
                ``[0, 1, 2, ...]`` (if no ``vmin`` and ``vmax`` are given) or
                ``[vmin, vmin + 1, ..., vmax]`` otherwise.
            vmin (float, optional): The lower bound of the color-mapping.
                Not passed to :py:class:`matplotlib.colors.BoundaryNorm`, which
                does not support it.
                If given, this argument in combination with ``vmax`` needs to
                define an integer range that has the same number of values
                as needed for a colormap constructed from ``from_values`` or
                via the ``label -> color`` mapping.
                If ``discretized`` is set, this value will be set to
                ``ceil(vmin) - 0.5``.
            vmax (float, optional): The upper bound of the color-mapping.
                Not passed to :py:class:`matplotlib.colors.BoundaryNorm`, which
                does not support it.
                If given, this argument in combination with ``vmin`` needs to
                define an integer range that has the same number of values
                as needed for a colormap constructed from ``from_values`` or
                via the ``label -> color`` mapping.
                If ``discretized`` is set, this value will be set to
                ``floor(vmax) + 0.5``.
            discretized (bool, optional): If True, assumes that the data this
                colormap is to represent only has integer values and makes a
                number of changes to improve the overall visualization.
                For instance, if ``True``, the ``vmin`` and ``vmax`` values
                will be set to the appropriate half-integer such that tick
                positions are centered within the corresponding range.
                If ``None`` (default), will do this automatically if a colormap
                is constructed via ``from_values`` or via ``label -> color``
                mapping.
        """
        self._cmap = None
        self._norm = None
        self._labels = None
        self._vmin = vmin
        self._vmax = vmax
        self.discretized = discretized
        self._cmap_kwargs = None
        self._norm_kwargs = None

        cmap_kwargs = None
        norm_kwargs = dict(name=None)
        labels_infd = None

        # .. Parse and set the colormap .......................................

        if isinstance(cmap, mpl.colors.Colormap):
            self._cmap = cmap

        else:
            if isinstance(cmap, str) or cmap is None:
                cmap_kwargs = dict(name=cmap)
            elif isinstance(cmap, (list, tuple)):
                cmap_kwargs = dict(from_values=copy.copy(cmap))
            else:
                cmap_kwargs = copy.deepcopy(cmap)

            cmap_kwargs, norm_kwargs, labels_infd = self._parse_cmap_kwargs(
                **cmap_kwargs,
                _labels=labels,
            )

            self._cmap_kwargs = cmap_kwargs
            self._cmap = self._create_cmap(**cmap_kwargs)

        # .. Parse and set the norm ...........................................
        # If Normalize instance is given, set it directly. Otherwise parse the
        # norm_kwargs below.

        if isinstance(norm, mpl.colors.Normalize):
            self._norm = norm

        else:
            if norm is not None:
                if isinstance(norm, str) or norm is None:
                    norm_kwargs["name"] = norm
                else:
                    norm_kwargs = copy.deepcopy(norm)

            norm_kwargs = self._parse_norm_kwargs(**norm_kwargs)
            self._norm_kwargs = norm_kwargs
            self._norm = self._create_norm(**norm_kwargs)

        # The norm should be regarded as the authority over vmin and vmax
        if self.norm.scaled():
            self._vmin = self.norm.vmin
            self._vmax = self.norm.vmax

        # .. Labels ...........................................................
        self._labels = self._parse_cbar_labels(
            labels if labels is not None else labels_infd
        )

    @property
    def cmap(self) -> mpl.colors.Colormap:
        """Returns the constructed colormap object"""
        return self._cmap

    @property
    def norm(self) -> mpl.colors.Normalize:
        """Returns the constructed normalization object"""
        return self._norm

    @property
    def labels(self) -> dict:
        """A dict or list of colorbar labels"""
        return self._labels

    @property
    def vmin(self) -> Optional[float]:
        """The ``vmin`` value of the colormap and norm"""
        return self._vmin

    @property
    def vmax(self) -> Optional[float]:
        """The ``vmax`` value of the colormap and norm"""
        return self._vmax

    # .........................................................................

    def map_to_color(self, X: Union[float, np.ndarray]):
        """Maps the input data to color(s) by applying both norm and colormap.

        Args:
            X (Union[float, numpy.ndarray]): Data value(s) to convert to RGBA.

        Returns:
            Tuple of RGBA values if X is scalar, otherwise an array of RGBA
            values with a shape of ``X.shape + (4, )``.
        """
        return self.cmap(self.norm(X))

    def create_cbar(
        self,
        mappable: "matplotlib.cm.ScalarMappable",
        *,
        fig: "matplotlib.figure.Figure" = None,
        ax: "matplotlib.axes.Axes" = None,
        label: str = None,
        label_kwargs: dict = None,
        tick_params: dict = None,
        extend: str = "auto",
        **cbar_kwargs,
    ) -> "matplotlib.colorbar.Colorbar":
        """Creates a colorbar of a given mappable

        Args:
            mappable (matplotlib.cm.ScalarMappable): The mappable that is to be
                described by the colorbar.
            fig (matplotlib.figure.Figure, optional): The figure; if not
                given, will use the current figure as determined by
                :py:func:`~matplotlib.pyplot.gcf`.
            ax (matplotlib.axes.Axes, optional): The axes; if not given, will
                use the one given by :py:meth:`matplotlib.figure.Figure.gca`.
            label (str, optional): A label for the colorbar
            label_kwargs (dict, optional): Additional parameters passed to
                :py:meth:`matplotlib.colorbar.Colorbar.set_label`
            tick_params (dict, optional): Set colorbar tick parameters via the
                :py:meth:`matplotlib.axes.Axes.tick_params` method of the
                :py:class:`matplotlib.colorbar.Colorbar` axes.
            extend (str, optional): Whether to extend the colorbar axis to show
                the ``under`` and ``over`` values. If ``auto`` (default), will
                inspect whether the colormap has these values set and decide
                accordingly. Can also be set manually, possible values being
                ``neither``, ``min``, ``max``, and ``both``.
            **cbar_kwargs: Passed on to
                :py:meth:`matplotlib.figure.Figure.colorbar`

        Returns:
            matplotlib.colorbar.Colorbar: The created colorbar object
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            ax = fig.gca()

        # Determine extend
        if extend == "auto":
            EXTEND_MAP = {
                (False, False): "neither",
                (False, True): "max",
                (True, False): "min",
                (True, True): "both",
            }
            extend = EXTEND_MAP[
                (
                    self.cmap._rgba_under is not None,
                    self.cmap._rgba_over is not None,
                )
            ]

        # Create the colorbar and set its label, ticks, tick labels
        cb = fig.colorbar(mappable, ax=ax, extend=extend, **cbar_kwargs)

        if label:
            cb.set_label(label=label, **(label_kwargs if label_kwargs else {}))

        if self.labels is not None:
            cb.set_ticks(list(self.labels.keys()))
            cb.set_ticklabels(list(self.labels.values()))

        if tick_params:
            cb.ax.tick_params(**tick_params)

        return cb

    # .........................................................................

    def _parse_cmap_kwargs(
        self,
        *,
        _labels: Union[list, dict],
        name: str = None,
        continuous: bool = None,
        from_values: Union[list, dict] = None,
        placeholder_color: str = "w",
        labels_and_colors: dict = None,
        **kwargs,
    ) -> Tuple[dict, dict, dict]:
        """
        Args:
            _labels (Union[list, dict]): The (top-level!) ``labels`` argument.
                While not being parsed here, it is needed for informative error
                messages.
            name (str, optional): Name of the colormap
            continuous (bool, optional): Whether to create a continuous or a
                discrete colormap.
            from_values (Union[dict, list], optional): The values from which
                to create the colormap. Keys are either given explicitly or
                inferred using :py:meth:`_infer_pos_map`.
            placeholder_color (str, optional): Color used when a value in
                ``from_values`` did not specify a value.
            **kwargs: combined keyword arguments for the colormap creation and
                shorthand entries for ``label -> color`` mapping.
        """

        def parse_from_values(
            mapping: Union[Sequence[str], Dict[float, str]],
            *,
            cmap_kwargs: dict,
            norm_kwargs: dict,
            continuous: bool,
        ):
            """Populates ``cmap_kwargs`` and ``norm_kwargs`` from the given
            mapping of values to colors.
            """
            if not isinstance(mapping, dict):
                mapping = self._infer_pos_map(
                    mapping, vmin=self._vmin, vmax=self._vmax
                )

            # Replace all None entries by the placeholder color.
            mapping = {
                k: (v if v is not None else placeholder_color)
                for k, v in mapping.items()
            }

            # Distinguish between continous and discrete case
            if continuous:
                # Get the colordict used to generate the continuous colormap,
                # complying to the interface of LinearSegmentedColormap:
                # For each of RGB, construct a sequence of (x, y0, y1) tuples
                # that define how that color changes. The colormap will then
                # interpolate between the color values.
                cdict = dict()
                for num, col in enumerate(("red", "green", "blue")):
                    cdict[col] = [
                        (
                            x,
                            to_rgb(_color)[num],
                            to_rgb(_color)[num],
                        )
                        for x, _color in mapping.items()
                    ]
                cmap_kwargs["segmentdata"] = cdict
                cmap_kwargs["name"] = "LinearSegmentedColormap"

                log.remark("Configuring a linear colormap 'from values'. ")

            else:
                # Discrete case, potentially with binning
                cmap_kwargs["name"] = "ListedColormap"
                cmap_kwargs["colors"] = list(mapping.values())

                norm_kwargs["name"] = "BoundaryNorm"
                norm_kwargs["ncolors"] = len(mapping)
                norm_kwargs["boundaries"] = self._parse_boundaries(
                    list(mapping.keys()),
                    set_vmin_vmax=True,
                    discretized=(self.discretized or self.discretized is None),
                )
                log.remark(
                    "Configuring a discrete colormap 'from values'. "
                    "Setting 'norm' to BoundaryNorm with %d colors.",
                    norm_kwargs["ncolors"],
                )

            return cmap_kwargs, norm_kwargs

        # .....................................................................

        cmap_kwargs = dict(name=name)
        norm_kwargs = dict()

        # Filter out arguments that specify the colormap and those that may be
        # used to denote labels
        _labels_and_colors = {
            l: c
            for l, c in kwargs.items()
            if l not in self._POSSIBLE_CMAP_KWARGS
        }
        cmap_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in _labels_and_colors}
        )

        if _labels_and_colors:
            if labels_and_colors:
                raise ValueError(
                    "The label -> color mapping needs to be given _either_ "
                    "via the explicit `labels_and_colors` argument _or_ "
                    "implicitly via the **kwargs, but got both!"
                )
            labels_and_colors = _labels_and_colors

        # Basic checks
        if name and (
            from_values or continuous is not None or labels_and_colors
        ):
            raise ValueError(
                "Cannot use argument `name` in combination with argument(s) "
                "`from_values`, `continuous` and/or the shorthand syntax for "
                "specifying labels and colors!\n"
                "Got:\n"
                f"  name:                 {name}\n"
                f"  from_values:          {from_values}\n"
                f"  continuous:           {continuous}\n"
                f"  **labels_and_colors:  {labels_and_colors}\n"
            )

        # May have used the implicit syntax with labels and colors specified
        # within the ``cmap`` argument. If so, translate these to the long-form
        # and explicit syntax.
        if labels_and_colors:
            if from_values or continuous is not None:
                raise ValueError(
                    "Cannot use the shorthand syntax for specifying labels "
                    "and colors in combination with the arguments "
                    "`continuous`, `from_values`!\n"
                    "Either remove those arguments or those that were "
                    "interpreted as belonging to the label -> color mapping:  "
                    f"{', '.join(labels_and_colors)}"
                )

            _labels = self._infer_pos_map(
                labels_and_colors.keys(),
                vmin=self._vmin,
                vmax=self._vmax,
            )
            from_values = {k: labels_and_colors[l] for k, l in _labels.items()}

        # Parse configuration for custom color mapping from values, which can
        # be either discrete or continuous (interpolated between colors)
        if from_values:
            cmap_kwargs, norm_kwargs = parse_from_values(
                from_values,
                cmap_kwargs=cmap_kwargs,
                norm_kwargs=norm_kwargs,
                continuous=continuous,
            )

        return cmap_kwargs, norm_kwargs, _labels

    def _parse_norm_kwargs(self, *, name: str = None, **kws) -> dict:
        """Parses the norm arguments into a uniform shape"""
        norm_kwargs = dict(name=name, **kws)

        # Some norms accept no vmin/vmax argument
        if name not in self._NORMS_NOT_SUPPORTING_VMIN_VMAX:
            norm_kwargs["vmin"] = norm_kwargs.get("vmin", self._vmin)
            norm_kwargs["vmax"] = norm_kwargs.get("vmax", self._vmax)

        return norm_kwargs

    def _parse_cbar_labels(
        self, labels: Union[None, Dict[float, str], Sequence[str]]
    ) -> Optional[Dict[float, str]]:
        """Parses the ``labels`` argument into a uniform shape"""

        def skip_label(l: str) -> bool:
            if isinstance(l, str) and not l.strip():
                return True
            return False

        def format_label(l, *, pos: float) -> str:
            # can do stuff here in the future, e.g. formatting
            return l

        if labels is None:
            return None

        if not isinstance(labels, dict):
            labels = self._infer_pos_map(
                labels, vmin=self._vmin, vmax=self._vmax
            )

        labels = {
            pos: format_label(l, pos=pos)
            for pos, l in labels.items()
            if not skip_label(l)
        }

        return copy.deepcopy(labels)

    def _infer_pos_map(
        self, seq: Sequence[Any], *, vmin: int = None, vmax: int = None
    ) -> Dict[float, Any]:
        """Given a sequence, infers a mapping ``position -> value``, where the
        positions are numeric values and the values of the resulting dict
        are the ones from the given sequence.

        If ``vmin`` and ``vmax`` are given, they are used to help with
        inferring the values.
        *Note* that these arguments need to be explicitly passed.
        """
        if vmin is None and vmax is not None:
            _vmax = floor(vmax)
            _vmin = _vmax - len(seq) + 1
        else:
            _vmin = ceil(vmin) if vmin is not None else 0
            _vmax = floor(vmax) if vmax is not None else _vmin + (len(seq) - 1)

        rg = range(_vmin, _vmax + 1)

        if len(rg) != len(seq):
            raise ValueError(
                "Failed to infer data positions for the given sequence! There "
                "was a mismatch between the length of the given data "
                f"sequence ({len(seq)}) and the inferred number of candidate "
                f"positions ({len(rg)}).\n"
                "To address this issue, check the `vmin` and `vmax` arguments "
                "and make sure that they allow an integer mapping (with both "
                "`vmin` and `vmax` included). Note that these arguments are "
                "ceiled and floored, respectively, to arrive at integers.\n"
                f"  sequence:  {list(seq)}\n"
                f"  vmin:      {vmin} \t-> {_vmin} (after ceil)\n"
                f"  vmax:      {vmax} \t-> {_vmax} (after floor)\n"
                f"  positions: {str(rg)} -> {list(rg)}\n"
            )

        return {i: v for i, v in zip(rg, seq)}

    def _parse_boundaries(
        self,
        bins: Sequence,
        *,
        set_vmin_vmax: bool = False,
        discretized: bool = False,
    ) -> Tuple[float]:
        """Parses the boundaries for the
        :py:class:`~matplotlib.colors.BoundaryNorm`.

        Args:
            bins (Sequence): Either monotonically increasing sequence of bin
                centers or sequence of connected intervals (2-tuples).
            set_vmin_vmax (bool, optional): Description
            discretized (bool, optional): Description

        Returns:
            Tuple[float]:
                Monotonically increasing boundaries.

        Raises:
            ValueError: On disconnected intervals or decreasing boundaries.
        """

        def from_intervals(intervals) -> list:
            """Extracts bin edges from sequence of connected intervals"""
            b = [intervals[0][0]]

            for low, up in intervals:
                if up < low:
                    raise ValueError(
                        "Received decreasing boundaries: "
                        f"{up} < {low}\n"
                        "Boundaries should be monotonically increasing. Got:\n"
                        f"  {intervals}"
                    )

                elif b[-1] != low:
                    raise ValueError(
                        "Received disconnected intervals: Upper "
                        f"bound {b[-1]} and lower bound {low} of "
                        "the proximate interval do not match.\n"
                        f"Specified intervals:\n  {intervals}"
                    )

                b.append(up)

            return b

        def from_centers(centers) -> list:
            """Calculates the bin edges as the halfway points between adjacent
            bin centers."""
            centers = np.array(list(centers))

            if len(centers) < 2:
                raise ValueError(
                    "At least 2 bin centers must be given to "
                    f"create a BoundaryNorm. Got: {centers}"
                )

            halves = 0.5 * np.diff(centers)
            left = (
                self.vmin
                if self.vmin is not None
                else (centers[0] - halves[0])
            )
            right = (
                self.vmax
                if self.vmax is not None
                else (centers[-1] + halves[-1])
            )

            b = [left] + [c + h for c, h in zip(centers, halves)] + [right]
            return b

        # .....................................................................

        if isinstance(bins[0], tuple):
            boundaries = from_intervals(bins)
        else:
            boundaries = from_centers(bins)

        # Correction for discretized values
        left = boundaries[0]
        right = boundaries[-1]
        if discretized:
            if (left % 1) != 0.5:
                boundaries[0] = ceil(left) - 0.5

            if (right % 1) != 0.5:
                boundaries[-1] = floor(right) + 0.5

        if set_vmin_vmax:
            self._vmin = left
            self._vmax = right

        return tuple(boundaries)

    def _create_cmap(
        self,
        *,
        name: str = None,
        colors: list = None,
        segmentdata: dict = None,
        bad: Union[str, dict] = None,
        under: Union[str, dict] = None,
        over: Union[str, dict] = None,
        reversed: bool = False,
        N: int = None,
        gamma: float = 1.0,
    ) -> mpl.colors.Colormap:
        """Creates a colormap.

        Args:
            name (str, optional): The colormap name. Can either be the name of
                a registered colormap or ``ListedColormap``. ``None`` means
                that the default value from the RC parameters (``image.cmap``)
                is used.
                If the name starts with the
                :py:attr:`._SNS_COLOR_PALETTE_PREFIX`, the colormap can be
                created by :py:func:`seaborn.color_palette`.
                See `the seaborn docs <https://seaborn.pydata.org/tutorial/color_palettes.html>`_
                for available options.
            colors (list, optional): Passed on to
                :py:class:`matplotlib.colors.ListedColormap`, ignored otherwise
            segmentdata (dict, optional): Description
            bad (Union[str, dict], optional): Set color to be used for masked
                values.
            under (Union[str, dict], optional): Set the color for low
                out-of-range values when ``norm.clip = False``.
            over (Union[str, dict], optional): Set the color for high
                out-of-range values when ``norm.clip = False``.
            reversed (bool, optional): Reverses the colormap
            N (int, optional): Passed on to
                :py:class:`matplotlib.colors.ListedColormap` or
                :py:class:`matplotlib.colors.LinearSegmentedColormap`,
                ignored otherwise.
            gamma (float, optional): Passed on to
                :py:class:`matplotlib.colors.LinearSegmentedColormap`

        Returns:
            matplotlib.colors.Colormap: The created colormap.

        Raises:
            ValueError: On invalid colormap name.
        """

        import seaborn as sns

        SNS_CP_PREFIX = self._SNS_COLOR_PALETTE_PREFIX
        SNS_DIV_PREFIX = self._SNS_DIVERGING_PALETTE_PREFIX

        # Depending on the name, use different constructors
        if name == "ListedColormap":
            cmap = mpl.colors.ListedColormap(colors, name=name, N=N)

        elif name == "LinearSegmentedColormap":
            cmap = mpl.colors.LinearSegmentedColormap(
                name,
                segmentdata,
                N=(N if N is not None else 256),
                gamma=gamma,
            )

        elif name is not None and name.startswith(SNS_CP_PREFIX):
            name = name[len(SNS_CP_PREFIX) :].strip()
            cmap = sns.color_palette(name, as_cmap=True)

        elif name is not None and name.startswith(SNS_DIV_PREFIX):
            # Parse strings like 'diverging::65,0,sep=12' into args and kwargs
            args, kwargs = parse_str_to_args_and_kwargs(
                name[len(SNS_DIV_PREFIX) :], sep=","
            )

            try:
                cmap = sns.diverging_palette(*args, **kwargs, as_cmap=True)
            except Exception as exc:
                raise ValueError(
                    "Failed constructing a seaborn diverging palette from the "
                    f"given string-specification '{name}'! "
                    f"Got a {type(exc).__name__}: {exc}\n\n"
                    "Check that no arguments are missing and all given "
                    "arguments are valid. The above string was parsed into "
                    "the following positional and keyword arguments:\n"
                    f"  args:    {args}\n"
                    f"  kwargs:  {kwargs}\n"
                ) from exc

        else:
            if name is None:
                name = mpl.rcParams["image.cmap"]

            # Get the colormap from the ColormapRegistry
            try:
                cmap = mpl.colormaps[name]
            except KeyError as err:
                _avail = make_columns(
                    sorted(
                        [cm for cm in mpl.colormaps if not cm.endswith("_r")]
                    )
                )
                raise ValueError(
                    f"'{name}' is not a known colormap name!\n"
                    f"Available named colormaps:\n{_avail}\n"
                    "Additional ways to specify colormaps by name:\n"
                    "  - Add '_r' suffix to the name to reverse it\n"
                    f"  - Add '{SNS_CP_PREFIX}' prefix to define a seaborn "
                    "color palette\n"
                    f"  - Add '{SNS_DIV_PREFIX}' prefix to specify a "
                    "diverging seaborn color map\n\n"
                    "See dantro ColorManager documentation for more."
                ) from err

        # Parse some parameters
        if isinstance(bad, str):
            bad = dict(color=bad)

        if isinstance(under, str):
            under = dict(color=under)

        if isinstance(over, str):
            over = dict(color=over)

        # Set bad, under, over
        if bad is not None:
            cmap.set_bad(**bad)

        if under is not None:
            cmap.set_under(**under)

        if over is not None:
            cmap.set_over(**over)

        # Optionally, reverse the colormap
        if reversed:
            cmap = cmap.reversed()

        return cmap

    def _create_norm(
        self, name: str = None, **norm_kwargs
    ) -> "matplotlib.colors.Normalize":
        r"""Creates a norm.

        Args:
            name (str, optional): The norm name. Must name a
                :py:class:`matplotlib.colors.Normalize` instance (see
                `matplotlib.colors <https://matplotlib.org/api/colors_api.html>`_).
                ``None`` means that the base class, ``Normalize``, is used.
            **norm_kwargs: Passed on to the constructor of the norm.

        Returns:
            matplotlib.colors.Normalize: The created norm.

        Raises:
            ValueError: On invalid norm specification.
        """
        if name is None:
            name = "Normalize"

        if name not in NORMS:
            available_norms = ", ".join(NORMS)
            raise ValueError(
                f"Received invalid norm specifier '{name}'! "
                f"Must be one of:  {available_norms}"
            )

        return NORMS[name](**norm_kwargs)


# -- Supporting functions -----------------------------------------------------


def parse_cmap_and_norm_kwargs(
    *, _key_map: dict = None, use_color_manager: bool = True, **kws
) -> dict:
    """A function that parses colormap-related keyword arguments and passes
    them through the :py:class:`.ColorManager`, making its functionality
    available in places that would otherwise not be able to use the expanded
    syntax of the color manager.

    .. note::

        The resulting dict will only have the ``cmap`` and ``cbar`` kwargs
        (or their mapped equivalents) set from the color manager, all other
        arguments are simply passed through.

        In particular, this means that the ``labels`` feature of the color
        manager is *not* supported, because this function has no ability to
        set the colorbar.

    Args:
        _key_map (dict, optional): If custom keyword argument keys are
            expected as output, e.g. ``hue_cmap`` instead of ``cmap``, set the
            values to these custom names: ``{"cmap": "hue_cmap"}``.
            Expected keys are ``cmap``, ``norm``, ``vmin``, ``vmax``. If not
            set or partially not set, will use defaults.
        use_color_manager (bool, optional): If false, will simply pass through
        **kws: Keyword arguments to parse

    Returns:
        dict:
            The updated keyword arguments with ``cmap`` and ``norm`` (or
            equivalent keys according to ``_key_map``).
    """
    if not use_color_manager:
        return kws

    _key = dict(cmap="cmap", norm="norm", vmin="vmin", vmax="vmax")
    if _key_map is not None:
        _key.update(_key_map)

    if _key["cmap"] not in kws and _key["norm"] not in kws:
        return kws

    # otherwise: Create a ColorManager
    cm = ColorManager(
        cmap=kws.get(_key["cmap"]),
        norm=kws.get(_key["norm"]),
        vmin=kws.get(_key["vmin"]),
        vmax=kws.get(_key["vmax"]),
    )

    # Evaluate it, only setting keys if they were there before
    if kws.get(_key["cmap"]) is not None:
        kws[_key["cmap"]] = cm.cmap
    if kws.get(_key["norm"]) is not None:
        kws[_key["norm"]] = cm.norm

    return kws
