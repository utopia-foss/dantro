"""Tests the seaborn plotting function"""

import numpy as np
import pandas as pd
import pytest

from dantro._import_tools import get_resource_path
from dantro.plot.funcs import snsplot
from dantro.tools import load_yml

PLOTS_CFG_SNS = load_yml(get_resource_path("tests", "cfg/plots_sns.yml"))

# Import fixtures and test helper functions
from ..._fixtures import *
from ...test_plot_mngr import dm as _dm
from .test_generic import dm, invoke_facet_grid, out_dir


def invoke_snsplot(*args, to_test: str, **kwargs):
    return invoke_facet_grid(
        *args, plot_func=snsplot, to_test=PLOTS_CFG_SNS[to_test], **kwargs
    )


# -----------------------------------------------------------------------------


def test_normalize_df_names():
    from dantro.plot.funcs.snsplot import normalize_df_names

    # --- Case 1: simple unnamed index ---
    # Simple columns with mixed labels (incl. None, int)
    df1 = pd.DataFrame(
        {"a": [1, 2], None: [3, 4], 3: [5, 6]},
        index=[10, 11],
    )

    # Sanity preconditions
    assert df1.index.name is None
    assert list(df1.columns) == ["a", None, 3]

    before_id = id(df1)
    out1 = normalize_df_names(df1)

    # In-place
    assert id(out1) == before_id

    # Index name normalised
    assert out1.index.name == "index_0"

    # Column *labels* normalised (strings, None -> "col_i")
    assert list(out1.columns) == ["a", "col_1", "3"]

    # --- Case 2: simple *named* index; simple columns already strings ---
    df2 = pd.DataFrame({"x": [0], "y": [1]}).set_index(
        pd.Index([0], name="id")
    )
    out2 = normalize_df_names(df2)
    assert out2.index.name == "id"
    assert list(out2.columns) == ["x", "y"]  # left unchanged (but ensured str)

    # --- Case 3: MultiIndex index with one missing name ---
    mi_idx = pd.MultiIndex.from_product(
        [["s1", "s2"], [0, 1]], names=[None, "tp"]
    )
    df3 = pd.DataFrame({"val": np.arange(len(mi_idx))}, index=mi_idx)
    out3 = normalize_df_names(df3)
    assert out3.index.names == [
        "index_0",
        "tp",
    ]  # names filled, order preserved

    # Columns are simple here -> labels should be strings
    assert list(out3.columns) == ["val"]

    # --- Case 4: MultiIndex columns with missing level names ---
    # Labels must be preserved
    mi_cols = pd.MultiIndex.from_tuples(
        [("A", 1), ("B", 2)], names=[None, None]
    )
    df4 = pd.DataFrame([[1, 2], [3, 4]], columns=mi_cols)
    out4 = normalize_df_names(df4)

    # Level names normalised
    assert out4.columns.names == ["col_0", "col_1"]

    # Labels preserved
    assert list(out4.columns) == [("A", 1), ("B", 2)]

    # --- Case 5: Mixed: MultiIndex index & MultiIndex columns ---
    # All names missing
    mi_idx2 = pd.MultiIndex.from_product(
        [["u", "v"], [1, 2]], names=[None, None]
    )
    mi_cols2 = pd.MultiIndex.from_product(
        [["X", "Y"], ["L", "R"]], names=[None, None]
    )
    df5 = pd.DataFrame(
        np.arange(16).reshape(4, 4), index=mi_idx2, columns=mi_cols2
    )
    out5 = normalize_df_names(df5)
    assert out5.index.names == ["index_0", "index_1"]
    assert out5.columns.names == ["col_0", "col_1"]

    # Column labels unchanged
    assert list(out5.columns) == list(mi_cols2)


def test_apply_selection():
    from dantro.plot.funcs.snsplot import apply_selection

    # Build DataFrame with both column vars and a MultiIndex
    # (Similar to the seaborn fmri dataset, but simpler)
    idx = pd.MultiIndex.from_product(
        [["stim", "cue"], [0, 1]],
        names=["event", "time"],
    )
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s1", "s2"],
            "region": ["parietal", "frontal", "parietal", "frontal"],
            "signal": np.arange(4, dtype=float),
        },
        index=idx,
    )

    # 1) Select on a COLUMN only
    out_col = apply_selection(df, region="parietal")
    assert set(out_col["region"]) == {"parietal"}

    # Index untouched for column-only selection
    assert out_col.index.names == ["event", "time"]

    # 2) Select on an INDEX level only
    out_idx = apply_selection(df, event="stim")
    assert set(out_idx.index.get_level_values("event")) == {"stim"}

    # Order of index levels must be preserved
    assert out_idx.index.names == ["event", "time"]

    # 3) Combined selection on INDEX and COLUMN
    out_both = apply_selection(df, event="cue", region="frontal")
    assert set(out_both.index.get_level_values("event")) == {"cue"}
    assert set(out_both["region"]) == {"frontal"}
    assert out_both.index.names == ["event", "time"]

    # 4) Selecting an INDEX level that yields no rows -> empty df but no error
    out_empty = apply_selection(df, event="nonexistent")
    assert out_empty.empty
    assert out_empty.index.names == ["event", "time"]

    # 5) Unknown selector key -> ValueError
    with pytest.raises(
        ValueError, match="neither a valid column name nor an index"
    ):
        apply_selection(df, not_a_var=123)


# -----------------------------------------------------------------------------


def test_sns_basics(dm, out_dir):
    """Tests snsplot basics"""
    invoke_snsplot(dm=dm, out_dir=out_dir, to_test="basics")


def test_sns_auto_encoding(dm, out_dir):
    """Tests the auto-encoding feature of snsplot"""
    invoke_snsplot(dm=dm, out_dir=out_dir, to_test="auto_encoding")
