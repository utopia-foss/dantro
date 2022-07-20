"""This module holds the data operations database"""

import logging
import math
import operator
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from .._import_tools import (
    LazyLoader,
    get_from_module,
    import_module_or_object,
)
from ..exceptions import *
from ..tools import recursive_getitem
from ..utils.coords import extract_coords_from_attrs, extract_dim_names
from ..utils.ordereddict import KeyOrderedDict
from .arr_ops import *
from .ctrl_ops import *
from .expr_ops import *

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")
nx = LazyLoader("networkx")
pd = LazyLoader("pandas")
scipy = LazyLoader("scipy")


# -----------------------------------------------------------------------------
# -- The Operations Database --------------------------------------------------
# -----------------------------------------------------------------------------

# fmt: off
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
_OPERATIONS = KeyOrderedDict({
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - -
    "define":               lambda d: d,
    "pass":                 lambda d: d,

    # Functions useful for debugging
    "print":                print_data,

    # Control flow functions, e.g. conditionally skipping a plot
    "raise_SkipPlot":       raise_SkipPlot,

    # Working on imported modules (useful if other operations don't match)
    "from_module":          get_from_module,
    "import":               import_module_or_object,
    "call":                 lambda c, *a, **k: c(*a, **k),
    "import_and_call":
        lambda m, n, *a, **k: import_module_or_object(m, n)(*a, **k),

    # Defining and calling lambdas
    "lambda":               generate_lambda,
    "call_lambda":          lambda e, *a, **k: generate_lambda(e)(*a, **k),

    # Import from packages and call directly
    "math.":
        lambda ms, *a, **k: get_from_module(math, name=ms)(*a, **k),
    "np.":
        lambda ms, *a, **k: get_from_module(np, name=ms)(*a, **k),
    "xr.":
        lambda ms, *a, **k: get_from_module(xr, name=ms)(*a, **k),
    "scipy.":
        lambda ms, *a, **k: get_from_module(scipy, name=ms)(*a, **k),
    "nx.":
        lambda ms, *a, **k: get_from_module(nx, name=ms)(*a, **k),
    "pd.":
        lambda ms, *a, **k: get_from_module(pd, name=ms)(*a, **k),

    # Some commonly used types
    "list":                 list,
    "dict":                 dict,
    "tuple":                tuple,
    "set":                  set,

    "int":                  int,
    "float":                float,
    "complex":              complex,
    "bool":                 bool,
    "str":                  str,

    # Item access and manipulation
    "[]":                   lambda d, k:    d[k],
    "getitem":              lambda d, k:    d[k],
    "setitem":              lambda d, k, v: d.__setitem__(k, v),
    "recursive_getitem":    recursive_getitem,

    # Attribute-related
    ".":                    getattr,
    "getattr":              getattr,
    "setattr":              setattr,
    ".()":                  lambda d, attr, *a, **k: getattr(d, attr)(*a, **k),
    "callattr":             lambda d, attr, *a, **k: getattr(d, attr)(*a, **k),

    # Other common Python builtins
    "all":                  all,
    "any":                  any,
    "len":                  len,
    "min":                  min,
    "max":                  max,
    "sum":                  sum,
    "map":                  map,
    "repr":                 repr,
    "sorted":               sorted,

    # Common operations on strings
    ".format":              lambda s, *a, **k: s.format(*a, **k),
    ".join":                lambda s, *a, **k: s.join(*a, **k),
    ".split":               lambda s, *a, **k: s.split(*a, **k),


    # Numerical operations - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Broadly categorized in how many arguments they accept ...
    # Unary ...................................................................
    "neg":                  operator.neg,
    "pos":                  operator.pos,
    "truth":                operator.truth,
    "invert":               operator.invert,

    "increment":            lambda d: d + 1,
    "decrement":            lambda d: d - 1,
    "count_unique":         count_unique,

    # numpy
    ".T":                   lambda d: d.T,
    ".any":                 lambda d: d.any(),
    ".all":                 lambda d: d.all(),
    ".dtype":               lambda d: d.dtype,
    ".shape":               lambda d: d.shape,
    ".ndim":                lambda d: d.ndim,
    ".size":                lambda d: d.size,
    ".itemsize":            lambda d: d.itemsize,
    ".nbytes":              lambda d: d.nbytes,
    ".base":                lambda d: d.base,
    ".imag":                lambda d: d.imag,
    ".real":                lambda d: d.real,
    ".nonzero":             lambda d: d.nonzero,
    ".flat":                lambda d: d.flat,
    ".item":                lambda d: d.item(),

    # xarray
    ".data":                lambda d: d.data,
    ".values":              lambda d: d.values,
    ".name":                lambda d: d.name,
    ".head":                lambda d: d.head(),
    ".tail":                lambda d: d.tail(),
    ".isnull":              lambda d: d.isnull(),

    # logarithms and powers, all numpy-based
    "log":                  np.log,
    "log10":                np.log10,
    "log2":                 np.log2,
    "log1p":                np.log1p,
    "squared":              np.square,
    "sqrt":                 np.sqrt,
    "cubed":                lambda d: np.power(d, 3),
    "sqrt3":                lambda d: np.power(d, 1./3.),

    # normalization and cumulation
    "normalize_to_sum":     lambda d: d / np.sum(d),
    "normalize_to_max":     lambda d: d / np.max(d),
    "cumulate":             lambda d, **k: np.cumsum(d, **k),
    "cumulate_complementary": lambda d, **k: np.cumsum(d[::-1], **k)[::-1],


    # Binary ..................................................................
    # Elementwise operations
    "add":                  operator.add,
    "concat":               operator.concat,
    "div":                  operator.truediv,
    "truediv":              operator.truediv,
    "floordiv":             operator.floordiv,
    "lshift":               operator.lshift,
    "mod":                  operator.mod,
    "mul":                  operator.mul,
    "matmul":               operator.matmul,
    "pow":                  operator.pow,
    "rshift":               operator.rshift,
    "sub":                  operator.sub,

    # numpy
    "np.power":             np.power,
    "np.dot":               np.dot,

    # xarray
    ".coords":
        lambda d, k=None: d.coords if k is None else d.coords[k],
    ".attrs":
        lambda d, k=None: d.attrs if k is None else d.attrs[k],
    ".variables":
        lambda d, k=None: d.variables if k is None else d.variables[k],
    ".data_vars":
        lambda d, k=None: d.data_vars if k is None else d.data_vars[k],


    # N-ary ...................................................................
    # Masking; these use the corresponding dantro data operations and work with
    # xarray.DataArray objects.
    "create_mask":                  create_mask,
    "where":                        where,
    # NOTE For applying a mask, use the xr.where operation.
    # xarray.pydata.org/en/stable/indexing.html#assigning-values-with-indexing

    # Array generation
    "populate_ndarray":             populate_ndarray,
    "build_object_array":           build_object_array,
    "expand_object_array":          expand_object_array,

    # extract labelling info, e.g. for creating higher-dimensional arrays
    "extract_dim_names":            extract_dim_names,
    "extract_coords_from_attrs":    extract_coords_from_attrs,

    # dantro-specific wrappers around other library's functionality
    "dantro.apply_along_axis":      apply_along_axis,
    "dantro.multi_concat":          multi_concat,
    "dantro.merge":                 merge,
    "dantro.expand_dims":           expand_dims,

    # coordinate transformations, working on shallow copies by default
    "transform_coords":             transform_coords,
    ".coords.transform":            transform_coords,

    # evaluating symbolic expressions using sympy
    "expression":                   expression,
    # NOTE: The `^` operator acts as XOR; use `**` for exponentiation!

    # numpy
    # CAUTION Some of these work in-place ...
    ".sum":                 lambda d, *a, **k: d.sum(*a, **k),
    ".prod":                lambda d, *a, **k: d.prod(*a, **k),
    ".cumsum":              lambda d, *a, **k: d.cumsum(*a, **k),
    ".cumprod":             lambda d, *a, **k: d.cumprod(*a, **k),

    ".mean":                lambda d, *a, **k: d.mean(*a, **k),
    ".std":                 lambda d, *a, **k: d.std(*a, **k),
    ".min":                 lambda d, *a, **k: d.min(*a, **k),
    ".max":                 lambda d, *a, **k: d.max(*a, **k),
    ".min.item":            lambda d, *a, **k: d.min(*a, **k).item(),
    ".max.item":            lambda d, *a, **k: d.max(*a, **k).item(),
    ".var":                 lambda d, *a, **k: d.var(*a, **k),
    ".argmin":              lambda d, *a, **k: d.argmin(*a, **k),
    ".argmax":              lambda d, *a, **k: d.argmax(*a, **k),
    ".argsort":             lambda d, *a, **k: d.argsort(*a, **k),
    ".argpartition":        lambda d, *a, **k: d.argpartition(*a, **k),

    ".transpose":           lambda d, *ax: d.transpose(*ax),
    ".squeeze":             lambda d, **k: d.squeeze(**k),
    ".flatten":             lambda d, **k: d.flatten(**k),
    ".diagonal":            lambda d, **k: d.diagonal(**k),
    ".trace":               lambda d, **k: d.trace(**k),
    ".sort":                lambda d, **k: d.sort(**k),
    ".fill":                lambda d, val: d.fill(val),
    ".round":               lambda d, **k: d.round(**k),
    ".take":                lambda d, i, **k: d.take(i, **k),
    ".swapaxes":            lambda d, a1, a2: d.swapaxes(a1, a2),
    ".reshape":             lambda d, s, **k: d.reshape(s, **k),
    ".astype":              lambda d, t, **k: d.astype(t, **k),

    "np.array":             np.array,
    "np.empty":             np.empty,
    "np.zeros":             np.zeros,
    "np.ones":              np.ones,
    "np.full":              np.full,
    "np.empty_like":        np.empty_like,
    "np.zeros_like":        np.zeros_like,
    "np.ones_like":         np.ones_like,
    "np.full_like":         np.full_like,

    "np.eye":               np.eye,
    "np.arange":            np.arange,
    "np.linspace":          np.linspace,
    "np.logspace":          np.logspace,

    "np.invert":            np.invert,
    "np.transpose":         np.transpose,
    "np.flip":              np.flip,
    "np.diff":              np.diff,
    "np.reshape":           np.reshape,
    "np.take":              np.take,
    "np.repeat":            np.repeat,
    "np.stack":             np.stack,
    "np.hstack":            np.hstack,
    "np.vstack":            np.vstack,
    "np.concatenate":       np.concatenate,

    "np.abs":               np.abs,
    "np.ceil":              np.ceil,
    "np.floor":             np.floor,
    "np.round":             np.round,
    "np.fmod":              np.fmod,

    "np.where":             np.where,
    "np.digitize":          np.digitize,
    "np.histogram":         np.histogram,
    "np.count_nonzero":     np.count_nonzero,

    "np.any":               np.any,
    "np.all":               np.all,
    "np.allclose":          np.allclose,
    "np.isnan":             np.isnan,
    "np.isclose":           np.isclose,
    "np.isinf":             np.isinf,
    "np.isfinite":          np.isfinite,
    "np.isnat":             np.isnat,
    "np.isneginf":          np.isneginf,
    "np.isposinf":          np.isposinf,
    "np.isreal":            np.isreal,
    "np.isscalar":          np.isscalar,

    "np.mean":              np.mean,
    "np.std":               np.std,
    "np.min":               np.min,
    "np.max":               np.max,
    "np.var":               np.var,
    "np.argmin":            np.argmin,
    "np.argmax":            np.argmax,

    # xarray
    ".sel":                 lambda d, *a, **k: d.sel(*a, **k),
    ".isel":                lambda d, *a, **k: d.isel(*a, **k),
    ".sel.item":            lambda d, *a, **k: d.sel(*a, **k).item(),
    ".isel.item":           lambda d, *a, **k: d.isel(*a, **k).item(),
    ".drop_sel":            lambda d, *a, **k: d.drop_sel(*a, **k),
    ".drop_isel":           lambda d, *a, **k: d.drop_isel(*a, **k),
    ".drop_dims":           lambda d, *a, **k: d.drop_dims(*a, **k),
    ".squeeze_with_drop":   lambda d, *a, **k: d.squeeze(*a, **k, drop=True),
    ".median":              lambda d, *a, **k: d.median(*a, **k),
    ".quantile":            lambda d, *a, **k: d.quantile(*a, **k),
    ".count":               lambda d, *a, **k: d.count(*a, **k),
    ".diff":                lambda d, *a, **k: d.diff(*a, **k),
    ".where":               lambda d, c, *a, **k: d.where(c, *a, **k),
    ".notnull":             lambda d, *a, **k: d.notnull(*a, **k),
    ".ffill":               lambda d, *a, **k: d.ffill(*a, **k),
    ".bfill":               lambda d, *a, **k: d.bfill(*a, **k),
    ".fillna":              lambda d, *a, **k: d.fillna(*a, **k),
    ".interpolate_na":      lambda d, *a, **k: d.interpolate_na(*a, **k),
    ".dropna":              lambda d, *a, **k: d.dropna(*a, **k),
    ".isin":                lambda d, *a, **k: d.isin(*a, **k),
    ".roll":                lambda d, *a, **k: d.roll(*a, **k),
    ".thin":                lambda d, *a, **k: d.thin(*a, **k),
    ".weighted":            lambda d, *a, **k: d.weighted(*a, **k),

    ".rolling":             lambda d, *a, **k: d.rolling(*a, **k),
    ".coarsen":             lambda d, *a, **k: d.coarsen(*a, **k),

    ".groupby":             lambda d, g, **k: d.groupby(g, **k),
    ".groupby_bins":        lambda d, g, **k: d.groupby_bins(g, **k),
    ".map":                 lambda ds, func, **k: ds.map(func, **k),
    ".reduce":              lambda d, func, **k: d.reduce(func, **k),

    ".rename":              lambda d, *a, **k: d.rename(*a, **k),
    ".expand_dims":         lambda d, *a, **k: d.expand_dims(*a, **k),
    ".swap_dims":           lambda d, *a, **k: d.swap_dims(*a, **k),
    ".assign_coords":       lambda d, *a, **k: d.assign_coords(*a, **k),
    ".assign_attrs":        lambda d, *a, **k: d.assign_attrs(*a, **k),
    ".assign":              lambda d, *a, **k: d.assign(*a, **k),

    ".to_dataframe":        lambda d, *a, **k: d.to_dataframe(*a, **k),

    ".to_array":            lambda ds, *a, **k: ds.to_array(*a, **k),
    ".rename_dims":         lambda ds, *a, **k: ds.rename_dims(*a, **k),
    ".rename_vars":         lambda ds, *a, **k: ds.rename_vars(*a, **k),
    ".drop_vars":           lambda ds, *a, **k: ds.drop_vars(*a, **k),
    ".assign_var":          lambda ds, name, var: ds.assign({name: var}),

    "xr.Dataset":           lambda *a, **k: xr.Dataset(*a, **k),
    "xr.DataArray":         lambda *a, **k: xr.DataArray(*a, **k),
    "xr.zeros_like":        lambda *a, **k: xr.zeros_like(*a, **k),
    "xr.ones_like":         lambda *a, **k: xr.ones_like(*a, **k),
    "xr.full_like":         lambda *a, **k: xr.full_like(*a, **k),

    "xr.merge":             lambda *a, **k: xr.merge(*a, **k),
    "xr.concat":            lambda *a, **k: xr.concat(*a, **k),
    "xr.align":             lambda *a, **k: xr.align(*a, **k),
    "xr.combine_nested":    lambda *a, **k: xr.combine_nested(*a, **k),
    "xr.combine_by_coords": lambda *a, **k: xr.combine_by_coords(*a, **k),

    # ... method calls that require additional Python packages
    ".rolling_exp":         lambda d, *a, **k: d.rolling_exp(*a, **k),
    ".rank":                lambda d, *a, **k: d.rank(*a, **k),

    # fitting with xr.DataArray.polyfit or scipy.optimize
    ".polyfit":             lambda d, *a, **k: d.polyfit(*a, **k),
    "curve_fit":
        lambda *a, **k: import_module_or_object("scipy.optimize",
                                                name="curve_fit")(*a, **k),
    # NOTE: Use the "lambda" operation to generate the callable
}) # End of default operation definitions
# fmt: on

# Add the boolean operators
_OPERATIONS.update(BOOLEAN_OPERATORS)
