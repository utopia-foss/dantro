"""DAG-related examples that are *not* using the plotting framework"""

import os

from dantro.dag import TransformationDAG
from dantro.tools import load_yml

from . import import_from_dantro_tests


def visualize_dag_examples(*, out_dir: str):
    """Creates output from DAG doc_examples"""
    test_dag = import_from_dantro_tests("test_dag.py")
    dm = test_dag.create_dm()

    to_plot = (
        "meta_ops_deeply_nested",
        "doc_examples_define",
        "doc_examples_select_with_transform",
        "doc_examples_op_hooks_expression_symbolic",
        "doc_examples_meta_ops_prime_multiples",
        "doc_examples_meta_ops_multiple_return_values",
    )

    vis_params = dict(
        generation=dict(include_results=True),
    )

    for cfg_name, cfg in load_yml(test_dag.TRANSFORMATIONS_PATH).items():
        if cfg_name not in to_plot:
            continue
        if cfg.get("_raises"):
            continue

        print(f"... Case: '{cfg_name}' ...")

        tdag = TransformationDAG(dm=dm, **cfg["params"])
        tdag.compute(compute_only=cfg.get("compute_only", "all"))
        tdag.visualize(
            out_path=os.path.join(out_dir, f"{cfg_name}.pdf"),
            **vis_params,
        )
