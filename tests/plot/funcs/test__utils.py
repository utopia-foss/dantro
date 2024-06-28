"""Tests the dantro.plot.funcs._utils module"""

import math

import pytest

import dantro.plot.funcs._utils as pfu

# -----------------------------------------------------------------------------


def test_determine_ideal_col_wrap_simple():
    """Tests automatic col wrapping"""
    determine_ideal_col_wrap = pfu.determine_ideal_col_wrap

    for N in range(1, 31):
        cw = determine_ideal_col_wrap(N, fill_last_row=False)
        if N < 4:
            assert cw is None
        else:
            assert cw == math.ceil(math.sqrt(N))


def test_determine_ideal_col_wrap_optimized():
    determine_ideal_col_wrap = pfu.determine_ideal_col_wrap

    expected_col_wrap = {
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
        10: 5,
        11: 4,
        12: 4,
        13: 5,
        14: 5,
        15: 4,
        16: 4,
        17: 6,
        18: 6,
        19: 5,
        20: 5,
        21: 7,
        22: 6,
        23: 6,
        24: 5,
        25: 5,
        26: 7,
        27: 7,
        28: 7,
        29: 6,
        30: 6,
    }

    for N in range(1, 51):
        cw = determine_ideal_col_wrap(N)
        cw_simple = determine_ideal_col_wrap(N, fill_last_row=False)

        if N < 4:
            assert cw is None
            continue

        n_rows_filled = N // cw
        n_last_row = N - cw * n_rows_filled
        if n_last_row == 0:
            n_last_row = cw
        fill_ratio = n_last_row / cw
        print(
            f"N: {N:2d}  ->  {cw:2d}  (instead of {cw_simple:2d},  "
            f"n_last_row: {n_last_row:2d},  fill_ratio: {fill_ratio:.4g})"
        )

        if N in expected_col_wrap:
            assert cw == expected_col_wrap[N]

        assert n_last_row > 1
        assert fill_ratio >= 0.5
