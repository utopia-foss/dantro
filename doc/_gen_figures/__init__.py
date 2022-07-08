"""A module that implements python functions that generate figures which
are embedded into the documentation."""

import os
from typing import Callable, Dict


def import_from_dantro_tests(mod_file_path: str):
    from dantro._import_tools import import_module_from_file

    modstr = os.path.splitext(mod_file_path)[0].replace("/", ".")

    return import_module_from_file(
        mod_file_path,
        base_dir=os.path.join(os.path.dirname(__file__), "../../tests"),
        mod_name_fstr=f"tests.{modstr}",
    )


# .. Assemble all figure generator functions ..................................

from .dag import visualize_dag_examples

GENERATOR_FUNCS: Dict[str, Callable] = {
    "dag_vis": visualize_dag_examples,
}
"""The dict containing all generator functions"""
