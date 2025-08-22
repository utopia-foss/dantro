"""Implements seaborn-based plotting functions"""

import logging
import operator
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import paramspace as psp
import seaborn as sns
import xarray as xr

from dantro.exceptions import PlotConfigError, PlottingError
from dantro.plot import PlotHelper, is_plot_func
from dantro.plot.funcs.generic import (
    UpdatePlotConfig,
    determine_encoding,
    figure_leak_prevention,
    make_facet_grid_plot,
)

log = logging.getLogger(__name__)

# -- Definitions --------------------------------------------------------------
# .. Seaborn's figure-level plot functions ....................................
SNS_PLOT_FUNCS = {
    "relplot": sns.relplot,
    "displot": sns.displot,
    "catplot": sns.catplot,
    "lmplot": sns.lmplot,
    "clustermap": sns.clustermap,
    "pairplot": sns.pairplot,
    "jointplot": sns.jointplot,
}

SNS_FACETGRID_KINDS = (
    "relplot",
    "displot",
    "catplot",
    "lmplot",
)

# .. Encodings for seaborn's figure-level plot functions ......................
# TODO Check if all are correct
SNS_ENCODINGS = {
    # FacetGrid: Distributions
    "displot": ("x", "col", "row", "hue"),
    "catplot": ("y", "hue", "col", "row"),
    # FacetGrid: Relational
    "relplot": ("x", "y", "hue", "col", "row", "style", "size"),
    "lmplot": ("x", "y", "hue", "col", "row"),
    # Others
    "clustermap": ("hue", "col", "row"),
    "pairplot": ("hue",),
    "jointplot": ("x", "y", "hue"),
}


# -- Utilities ----------------------------------------------------------------


def normalize_df_names(df: pd.DataFrame) -> pd.DataFrame:
    """In-place normalizes index and column names by prefixing ``index_`` or
    ``col_`` if they are not named.
    """

    def _normalize(names: List[Union[str, None]], prefix: str) -> List[str]:
        return [
            str(n) if n is not None else f"{prefix}_{i}"
            for i, n in enumerate(names)
        ]

    # Index names (MultiIndex or simple)
    if isinstance(df.index, pd.MultiIndex):
        df.index.set_names(_normalize(df.index.names, "index"), inplace=True)
    else:
        df.index.name = (
            str(df.index.name) if df.index.name is not None else "index_0"
        )

    # Column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns.set_names(_normalize(df.columns.names, "col"), inplace=True)
    else:
        df.columns = _normalize(df.columns, "col")

    return df


def apply_selection(df: pd.DataFrame, **sel) -> pd.DataFrame:
    """Apply a selection, defined by key-value pairs, on a DataFrame."""
    for var, val in sel.items():
        if var in df.columns:
            df = df.loc[df[var] == val]
        elif var in df.index.names:
            index_order = df.index.names
            df = df.reset_index(var)
            df = df[df[var] == val]
            df = df.set_index(var, append=True)
            df = df.reorder_levels(index_order)
        else:
            raise ValueError(
                f"Selector variable '{var}' is neither a valid column "
                "name nor an index, cannot apply selection!\n\n"
                f"Given DataFrame:\n{df.head()}\n"
            )

    return df


def build_df_selector(
    df: pd.DataFrame, vars: List[str], **sel
) -> Dict[str, Union[psp.ParamDim, Any]]:
    """Builds a selector dict for DataFrame selections, using
    :py:class:`~paramspace.paramdim.ParamDim` as values.

    This method also combines the parameter space selector with an existing
    selector dict, ``sel``, and throws an error if there is an overlap between
    keys in ``vars`` and ``sel``.
    """

    def get_unique_vals(var: str) -> list:
        if var in df.columns:
            vals = df[var].unique()
        elif var in df.index.names:
            vals = np.unique(df.index.get_level_values(var))
        else:
            raise ValueError(
                f"Variable '{var}' is neither a column nor an index, cannot "
                f"build a selector for it!\n\n{df.head()}\n"
            )

        return vals.tolist()

    unique_vals = {var: get_unique_vals(var) for var in vars}
    psp_sel: Dict[str, psp.ParamDim] = {
        str(var): psp.ParamDim(
            default=unique_vals[var][0],
            values=unique_vals[var],
        )
        for var in vars
    }

    if any(_sel in psp_sel for _sel in sel):
        raise PlotConfigError(
            f"Cannot combine parameter sweep selector ({', '.join(vars)}) "
            f"with existing selector ({sel}) because there are "
            "overlapping dimension names! Remove them and retry."
        )

    return dict(**psp_sel, **sel)


def convert_to_df(
    df: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    to_dataframe_kwargs: dict = None,
) -> pd.DataFrame:
    """Converts a xarray DataArray or Dataset to a pandas DataFrame"""
    if isinstance(df, (xr.Dataset, xr.DataArray)):
        tdf_kwargs = to_dataframe_kwargs if to_dataframe_kwargs else {}
        log.note("Attempting conversion to pd.DataFrame ...")
        log.remark(
            "   arguments:  %s",
            ", ".join(f"{k}: {v}" for k, v in tdf_kwargs.items()),
        )
        df = df.to_dataframe(**tdf_kwargs)

    elif isinstance(df, pd.DataFrame):
        # Let's work on a copy, just to be sure ...
        df = df.copy()

    else:
        raise TypeError(
            "Expected pd.DataFrame, xr.Dataset, or xr.DataArray, but got "
            f"{type(df).__name__}!"
        )

    return df


def _log_df_summary(df: pd.DataFrame):
    """Logs a summary of the the data frame properties"""

    def cols_to_str(cols) -> str:
        if len(cols) == 0:
            return "(none)"
        if isinstance(cols, pd.MultiIndex):
            return ", ".join("/".join(map(str, tup)) for tup in cols)
        return ", ".join(map(str, cols))

    def index_names_to_str(index) -> str:
        names = list(index.names)
        if not any(n is not None for n in names):
            return "(no named indices)"
        return ", ".join(
            str(n) if n is not None else "(unnamed)" for n in names
        )

    log.note("Evaluating data frame properties ...")
    log.remark("   length:           %d", len(df))
    log.remark("   shape:            %s", df.shape)
    log.remark("   size:             %d", df.size)
    log.remark("   columns:          %s", cols_to_str(df.columns))
    log.remark("   indices:          %s", index_names_to_str(df.index))


def sample_df(
    df: pd.DataFrame, sample: int, sample_kwargs: dict
) -> pd.DataFrame:
    if not sample_kwargs:
        sample_kwargs = {}
        if isinstance(sample, int) and sample < len(df):
            sample_kwargs["n"] = sample
        elif isinstance(sample, float) and 0.0 <= sample <= 1.0:
            sample_kwargs["frac"] = sample

    if sample_kwargs:
        log.note("Sampling from data frame ...")
        log.remark(
            "   arguments:  %s",
            ", ".join(f"{k}: {v}" for k, v in sample_kwargs.items()),
        )
        len_before = len(df)
        try:
            df = df.sample(**sample_kwargs)
        except Exception as exc:
            log.error(
                "   sampling failed with %s: %s", type(exc).__name__, exc
            )
            log.remark("Continuing without sampling.")
        else:
            log.remark(
                "   sampling succeeded. New length: %d (%d)",
                len(df),
                len(df) - len_before,
            )
    else:
        log.note("Sampling skipped (no arguments applicable).")

    return df


# -----------------------------------------------------------------------------


@is_plot_func(use_dag=True, required_dag_tags=("data",))
def snsplot(
    *,
    data: dict,
    hlpr: PlotHelper,
    sns_kind: str,
    free_indices: Tuple[str, ...] = None,
    optional_free_indices: Tuple[str, ...] = (),
    auto_encoding: Union[bool, dict] = None,
    auto_encoding_options: dict = None,
    reset_index: Union[bool, List[str]] = False,
    to_dataframe_kwargs: dict = None,
    normalize_names: bool = False,
    dropna: bool = False,
    dropna_kwargs: dict = None,
    sample: Union[bool, int] = False,
    sample_kwargs: dict = None,
    _sel: dict = None,
    **plot_kwargs,
) -> None:
    """An *experimental* interface to seaborn's figure-level plot functions.

    Seaborn plot functions are selected via the ``sns_kind`` argument:

    - ``relplot``:      :py:func:`seaborn.relplot`
    - ``displot``:      :py:func:`seaborn.displot`
    - ``catplot``:      :py:func:`seaborn.catplot`
    - ``lmplot``:       :py:func:`seaborn.lmplot`
    - ``clustermap``:   :py:func:`seaborn.clustermap` *(not faceting)*
    - ``pairplot``:     :py:func:`seaborn.pairplot`   *(not faceting)*
    - ``jointplot``:    :py:func:`seaborn.jointplot`  *(not faceting)*

    This plot function also supports the ``files`` encoding, which triggers
    plot config updating and thereby allows to represent data of arbitrary
    dimensionality; this is achieved by performing a parameter sweep plot where
    each point corresponds to a single plot file of a subspace of the data.

    .. warning::

        This plot function is still being experimented with and surely will
        show some quirks. Please report any errors or unexpected behavior and
        note that the interface may still change in future versions.

    Args:
        data (dict): The data transformation framework results, expecting a
            single entry ``data`` which can be a :py:class:`pandas.DataFrame`
            or an :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`.
        hlpr (PlotHelper): The plot helper instance
        sns_kind (str): Which seaborn plot to use, see list above.
        free_indices (Tuple[str, ...], optional): Which index names *not* to
            associate with a layout encoding; seaborn uses these to calculate
            the distribution statistics.
        optional_free_indices (Tuple[str, ...], optional): These indices will
            be added to the free indices *if they are part of the data frame*.
            Otherwise, they are silently ignored.
        auto_encoding (Union[bool, dict], optional): Whether to use
            auto-encoding to map encodings to data variables or dimensions;
            see :py:func:`~dantro.plot.funcs.generic.determine_encoding`.
        auto_encoding_options (dict, optional): Additional arguments for
            :py:func:`~dantro.plot.funcs.generic.determine_encoding`.
        reset_index (Union[bool, List[str]], optional): If a boolean, controls
            whether to reset indices such that only the ``free_indices`` remain
            as indices and all others are converted into columns.
            Otherwise, assumes it's a sequence of index names to reset.
        to_dataframe_kwargs (dict, optional): For xarray data types, this is
            used to convert the given data into a pandas.DataFrame.
        normalize_names (bool, optional): If True (default), unnamed columns
            and indices will get names assigned. This makes handling of various
            data frames easier.
        dropna (bool, optional): If True, will invoke ``.dropna`` on the data.
        dropna_kwargs (dict, optional): Additional arguments to the ``.dropna``
            call on the data.
        sample (Union[bool, int], optional): If True, will sample a subset from
            the final dataframe, controlled by ``sample_kwargs``. If an
            integer, will use that as the absolute number of samples to draw.
            If a float in the unit interval, will use it as the fraction of
            samples to draw.
        sample_kwargs (dict, optional): Passed to
            :py:meth:`pandas.DataFrame.sample`. May contain ``n`` for absolute
            or ``frac`` for relative number of samples to keep.
        _sel (dict, optional): Select a subset of the dataframe. (For internal
            use only!)
        **plot_kwargs: Passed on to the selected plotting function, containing
            the respective encoding variables, e.g. ``x``, ``y``, ``hue``,
            ``col``, ``row``, ``files``, ...
    """
    # Get and pre-process data
    df = data["data"]
    df = convert_to_df(df, to_dataframe_kwargs)

    # Take care of free indices, and unnamed columns or indices
    free_indices = list(free_indices) if free_indices else []
    if normalize_names:
        df = normalize_df_names(df)

    # Provide some information
    _log_df_summary(df)
    log.remark("   free indices:     %s", ", ".join(free_indices))
    log.remark("   optionally free:  %s", ", ".join(optional_free_indices))

    # Apply optionally free indices
    # TODO Add an option to make all indices free, excluding some ...
    free_indices += [n for n in optional_free_indices if n in df.index.names]

    # For some kinds, it makes sense to re-index such that only the free
    # indices are used as columns. Most plots require long-form data.
    # See:  https://seaborn.pydata.org/tutorial/data_structure.html
    reset_for = []
    if reset_index:
        if isinstance(reset_index, bool):
            # Reset all. There can be unnamed indices, i.e. df.index.names can
            # contain None, and we need to handle that case explicitly.
            if df.index.names:
                reset_for = [
                    n
                    for n in df.index.names
                    if n is not None and n not in free_indices
                ]
            else:
                reset_for = []
        else:
            # Assume that index names have been specified, use those
            reset_for = list(reset_index)

        if reset_for:
            df = df.reset_index(level=reset_for)
            log.remark("   reset index for:  %s", ", ".join(reset_for))

    # Can apply a data subselection to allow files iteration
    if _sel:
        log.note("Applying data selection ...")
        log.remark(
            "   selector:         %s",
            ",  ".join(f"{var}: {val}" for var, val in _sel.items()),
        )
        df = apply_selection(df, **_sel)

        log.remark("   new length:       %d", len(df))

    # Might want to drop null values
    if dropna:
        dropna_kwargs = dropna_kwargs if dropna_kwargs else {}
        log.note("Dropping null values ...")
        log.remark(
            "  Arguments:  %s",
            ", ".join(f"{k}: {v}" for k, v in dropna_kwargs.items()),
        )
        df = df.dropna(**dropna_kwargs)
        log.remark("   new length:       %d", len(df))

    # Sampling
    if sample:
        df = sample_df(df, sample, sample_kwargs)

    # Interface with auto-encoding
    # Need to pop any given `kind` argument (valid input to sns.pairplot)
    kind = plot_kwargs.pop("kind", None)
    ae_dims = {
        n: s
        for n, s in zip(
            df.index.names, getattr(df.index, "levshape", [len(df.index)])
        )
        if n not in free_indices
    }
    plot_kwargs, (encoding, _, _) = determine_encoding(
        ae_dims,
        kind=sns_kind,
        auto_encoding=auto_encoding,
        default_encodings=SNS_ENCODINGS,
        data_vars=list(df.columns),
        plot_kwargs=plot_kwargs,
        return_encoding_info=True,
        **(auto_encoding_options if auto_encoding_options else {}),
    )

    files = plot_kwargs.pop("files", None)
    free = plot_kwargs.pop("free", None)  # TODO Should this be an argument?
    _special_specs = ("free", "files")

    if free:
        free = (free,) if isinstance(free, str) else tuple(free)
        log.remark("   deliberately free:  %s", ", ".join(free))

    if kind is not None:
        plot_kwargs["kind"] = kind

    # Potentially perform files iteration
    # This may leave the plotting function via a messaging exception; the next
    # time we arrive here, `files` should no longer be set.
    if files:
        log.info(
            "Initiating %d-dimensional files iteration:  %s",
            len(files),
            ", ".join(files),
        )

        # Build selector, including existing values
        _sel = build_df_selector(df, files, **(_sel if _sel else {}))

        raise UpdatePlotConfig(
            "snsplot files iteration",
            from_pspace=dict(_sel=_sel),
            #
            # Explicitly pass the encoding so it's not tinkered with again
            **{s: d for s, d in encoding.items() if s not in _special_specs},
            free=free,
            files=None,  # TODO Consider setting to some informative value
        )

    # Retrieve the plot function
    try:
        plot_func = SNS_PLOT_FUNCS[sns_kind]

    except KeyError:
        _avail = ", ".join(SNS_PLOT_FUNCS)
        raise ValueError(
            f"Invalid plot kind '{sns_kind}'! Available: {_avail}"
        )

    # Actual plotting . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Close the existing figure; the seaborn functions create their own
    hlpr.close_figure()

    # Let seaborn do the plotting
    log.note("Now invoking sns.%s ...", sns_kind)

    try:
        with figure_leak_prevention():
            fg = plot_func(data=df, **plot_kwargs)

    except Exception as exc:
        raise PlottingError(
            f"sns.{sns_kind} failed! Got {type(exc).__name__}: {exc}\n\n"
            f"Data was:\n{df}\n\n"
            f"sns.{sns_kind} arguments:\n  {plot_kwargs}\n\n"
            "Selected snsplot arguments:\n"
            f"  reset_index:    {reset_index} (reset levels: {reset_for})\n"
            f"  auto_encoding:  {auto_encoding} -> {encoding}\n"
            "\nCheck the log output, data, and specified arguments and make "
            "sure the resulting DataFrame is in the desired shape for the "
            "selected plot function. Make sure that all required encodings "
            "have been specified in the `plot_kwargs`."
        ) from exc

    # Store FacetGrid (or similar) and encoding as helper attributes
    hlpr._attrs["facet_grid"] = fg
    hlpr._attrs["encoding"] = encoding

    # Attach the created figure, including a workaround for `col_wrap`, in
    # which case `fg.axes` is one-dimensional (for whatever reason)
    if isinstance(fg, sns.JointGrid):
        fig = fg.fig
        axes = [[fg.ax_joint]]  # TODO consider registering all axes

    else:
        # Assume it's FacetGrid-like
        fig = fg.fig
        axes = fg.axes
        if axes.ndim != 2:
            axes = axes.reshape((fg._nrow, fg._ncol))

    hlpr.attach_figure_and_axes(fig=fig, axes=axes)

    # TODO Animation?!

    return fg
