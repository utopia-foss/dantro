"""A module that implements python functions that generate figures which
are embedded into the documentation."""

from typing import Callable, Dict

from .dag import visualize_dag_examples

# .. Assemble all figure generator functions ..................................


GENERATOR_FUNCS: Dict[str, Callable] = {
    "dag_vis": visualize_dag_examples,
}
"""The dict containing all generator functions"""
