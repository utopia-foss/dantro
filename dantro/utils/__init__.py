"""This submodule contains some utility classes and functions"""

from .coords import (
    extract_coords,
    extract_coords_from_attrs,
    extract_coords_from_data,
    extract_coords_from_name,
    extract_dim_names,
)
from .data_ops import apply_operation, available_operations, register_operation
from .link import Link
from .ordereddict import IntOrderedDict, KeyOrderedDict
