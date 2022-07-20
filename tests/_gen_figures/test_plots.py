"""DAG-related examples that are *not* using the plotting framework"""

import os

from dantro import PlotManager
from dantro.tools import load_yml

from .._fixtures import *
from ..plot.test_dag_plotting import DAG_PLOTS_CONFIG, dm

# -----------------------------------------------------------------------------


def test_plots(dm, out_dir):
    """Creates output from the (DAG-based) plotting tests and examples"""

    # Get the configuration and set up the plot manager
    plots = load_yml(DAG_PLOTS_CONFIG)
    pm = PlotManager(
        dm=dm,
        out_dir=out_dir,
        raise_exc=True,
        shared_creator_init_kwargs=dict(exist_ok=True),
        cfg_exists_action="overwrite",
        **plots["_pm_init_kwargs"],
    )

    # Specify which plots to create
    to_plot = ("doc_examples_errorbars",)

    # Safety check: all plots defined above are valid names
    invalid_plot_names = [p for p in to_plot if p not in plots["config_based"]]
    if invalid_plot_names:
        _avail = "\n".join(f"  {p}" for p in sorted(plots["config_based"]))
        raise ValueError(
            f"Invalid plot names:  {', '.join(invalid_plot_names)}\n"
            f"Available:\n{_avail}"
        )

    # Here we go ...
    for name, cfg in plots["config_based"].items():
        if name not in to_plot:
            continue

        print(f"... Case: '{name}' ...")
        pm.plot(name=name, **cfg["plot_cfg"])
