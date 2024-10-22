"""Tests the plot configuration tools module"""

from pprint import pformat, pprint

import pytest

import dantro.plot._cfg as _cfg
from dantro._import_tools import get_resource_path
from dantro.exceptions import *
from dantro.tools import DoNothingContext, load_yml

TEST_CFG_PATH = get_resource_path("tests", "cfg/plot_cfg.yml")

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def test_cfg() -> dict:
    return load_yml(TEST_CFG_PATH)


# -- Tests --------------------------------------------------------------------


def test_resolve_based_on(test_cfg):
    """Tests plot config resolution"""
    for case_name, case_cfg in test_cfg["resolve_based_on"].items():
        print(f"\n\n----- Testing case '{case_name}' ...")

        if not case_cfg.get("raises"):
            context = DoNothingContext()
        else:
            context = pytest.raises(
                globals().get(case_cfg["raises"]),
                match=case_cfg.get("match", ""),
            )

        with context:
            actual = _cfg.resolve_based_on(
                case_cfg["plots_cfg"],
                label=case_name,
                base_pools=case_cfg["base_pools"],
            )
            expected = case_cfg["expected"]

            print(f"\n-- Resolved:\n{pformat(actual)}")
            print(f"\n-- Expected:\n{pformat(expected)}")
            assert actual == expected
