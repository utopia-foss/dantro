"""This submodule contains some utility classes and functions"""

from .ordereddict import KeyOrderedDict, IntOrderedDict
from .link import Link
from .coords import (extract_dim_names, extract_coords,
                     extract_coords_from_attrs, extract_coords_from_name,
                     extract_coords_from_data)
from .data_ops import register_operation, apply_operation, available_operations
