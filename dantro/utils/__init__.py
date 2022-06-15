"""This submodule contains some utility classes and functions"""

from .coords import (
    extract_coords,
    extract_coords_from_attrs,
    extract_coords_from_data,
    extract_coords_from_name,
    extract_dim_names,
)
from .link import Link, StrongLink
from .ordereddict import IntOrderedDict, KeyOrderedDict
