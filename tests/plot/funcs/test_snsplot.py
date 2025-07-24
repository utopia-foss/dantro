"""Tests the seaborn plotting function"""

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


def test_sns_basics(dm, out_dir):
    """Tests the facet_grid with auto-encoding of kind and specifiers for
    datasets"""
    invoke_snsplot(dm=dm, out_dir=out_dir, to_test="basics")
