"""This module implements data processing operations for dantro objects"""

import logging
import operator
from difflib import get_close_matches
from typing import Callable, Any

import numpy as np
import xarray as xr

from .ordereddict import KeyOrderedDict
from ..base import BaseDataContainer, BaseDataGroup

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Some general helper operations

# Define boolean operators separately (might be useful elsewhere) & register
BOOLEAN_OPERATORS = {
    '==': operator.eq,  'eq': operator.eq,
    '<':  operator.lt,  'lt': operator.lt,
    '<=': operator.le,  'le': operator.le,
    '>':  operator.gt,  'gt': operator.gt,
    '>=': operator.ge,  'ge': operator.ge,
    '!=': operator.ne,  'ne': operator.ne,
    '^':  operator.xor, 'xor': operator.xor,
    # Expecting an iterable as second argument
    'in':               (lambda x, y: x in y),
    'not in':           (lambda x, y: x not in y),
    # Performing bitwise boolean operations to support numpy logic
    'in interval':      (lambda x, y: x >= y[0] & x <= y[1]),
    'not in interval':  (lambda x, y: x < y[0] | x > y[1]),
}

# .............................................................................

def print_data(data: Any) -> Any:
    """Prints and passes on the data.

    The print operation distinguishes between dantro types (in which case some
    more information is shown) and  non-dantro types.
    """
    # Distinguish between dantro types and others
    if isinstance(data, BaseDataContainer):
        print("{}, with data:\n{}\n".format(data, data.data))
    
    elif isinstance(data, BaseDataGroup):
        print("{}\n".format(data.tree))

    else:
        print(data)

    return data

# .............................................................................
# numpy and xarray operations

def create_mask(data: xr.DataArray,
                operator_name: str,
                rhs_value: float) -> xr.DataArray:
    """Given the data, returns a binary mask by applying the following
    comparison: ``data <operator> rhs value``.
    
    Args:
        data (xr.DataArray): The data to apply the comparison to. This is the
            lhs of the comparison.
        operator_name (str): The name of the binary operator function as
            registered in utopya.tools.OPERATOR_MAP
        rhs_value (float): The right-hand-side value
    
    Raises:
        KeyError: On invalid operator name
    
    Returns:
        xr.DataArray: Boolean mask
    """
    # Get the operator function
    try:
        comp_func = BOOLEAN_OPERATORS[operator_name]

    except KeyError as err:
        raise KeyError("No boolean operator '{}' available! "
                       "Available operators: {}"
                       "".format(operator_name,
                                 ", ".join(BOOLEAN_OPERATORS.keys()))
                       ) from err

    # Apply the comparison
    data = comp_func(data, rhs_value)

    # Create a new name
    name = data.name + " (masked by '{} {}')".format(operator_name, rhs_value)

    # Build a new xr.DataArray from that data, retaining all information
    return xr.DataArray(data=data,
                        name=name,
                        dims=data.dims,
                        coords=data.coords)

def where(data: xr.DataArray,
          operator_name: str,
          rhs_value: float) -> xr.DataArray:
    """Filter elements from the given data according to a condition. Only
    those elemens where the condition is fulfilled are not masked.

    NOTE This leads to a dtype change to float.
    """
    # Get the mask and apply it
    return data.where(create_mask(data,
                                  operator_name=operator_name,
                                  rhs_value=rhs_value))

def count_unique(data) -> xr.DataArray:
    """Applies np.unique to the given data and constructs a xr.DataArray for
    the results.
    """
    unique, counts = np.unique(data, return_counts=True)

    # Construct a new data array and return
    return xr.DataArray(data=counts,
                        name=data.name + " (unique counts)",
                        dims=('unique',),
                        coords=dict(unique=unique))


# -----------------------------------------------------------------------------
# The Operations Database -----------------------------------------------------
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
_OPERATIONS = KeyOrderedDict({
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    'define':       lambda d: d,
    'pass':         lambda d: d,
    'print':        print_data,
    
    # Item manipulation
    'getitem':      lambda d, k:    d[k],
    'setitem':      lambda d, k, v: d.__setitem__(k, v),

    # Attribute-related
    'getattr':      getattr,
    'setattr':      setattr,
    'callattr':     lambda d, attr, *a, **k: getattr(d, attr)(*a, **k),
    

    # Numerical operations - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Unary ...................................................................
    'increment':    lambda d: d + 1,
    'decrement':    lambda d: d - 1,
    'count_unique': count_unique,
    
    # numpy
    '.T':           lambda d: d.T,
    '.any':         lambda d: d.any(),
    '.all':         lambda d: d.all(),
    '.dtype':       lambda d: d.dtype,
    '.shape':       lambda d: d.shape,
    '.ndim':        lambda d: d.ndim,
    '.size':        lambda d: d.size,
    '.itemsize':    lambda d: d.itemsize,
    '.nbytes':      lambda d: d.nbytes,
    '.base':        lambda d: d.base,
    '.imag':        lambda d: d.imag,
    '.real':        lambda d: d.real,

    # xarray
    '.head':        lambda d: d.head(),
    '.tail':        lambda d: d.tail(),

    # logarithms and squares
    'log':          lambda d: np.log(d),
    'log10':        lambda d: np.log10(d),
    'log2':         lambda d: np.log2(d),
    'log1p':        lambda d: np.log1p(d),
    'squared':      lambda d: np.square(d),
    'sqrt':         lambda d: np.sqrt(d),
    'cubed':        lambda d: np.power(d, 3),
    'sqrt3':        lambda d: np.power(d, 1./.3),

    # Normalization and cumulation
    'normalize_to_sum':         lambda d: d / np.sum(d),
    'normalize_to_max':         lambda d: d / np.max(d),
    'cumulate':                 lambda d: np.cumsum(d),
    'cumulate_complementary':   lambda d: np.cumsum(d[::-1])[::-1],


    # Binary ..................................................................
    # Elementwise operations
    'add':          lambda d, v: operator.add(d, v),
    'concat':       lambda d, v: operator.concat(d, v),
    'div':          lambda d, v: operator.truediv(d, v),
    'truediv':      lambda d, v: operator.truediv(d, v),
    'floordiv':     lambda d, v: operator.floordiv(d, v),
    'lshift':       lambda d, v: operator.lshift(d, v),
    'mod':          lambda d, v: operator.mod(d, v),
    'mul':          lambda d, v: operator.mul(d, v),
    'matmul':       lambda d, v: operator.matmul(d, v),
    'rshift':       lambda d, v: operator.rshift(d, v),
    'sub':          lambda d, v: operator.sub(d, v),

    # numpy
    'power':        lambda d, e: np.power(d, e),

    # xarray
    '.coords':      lambda d, key: d.coords[key],


    # N-ary ...................................................................
    # numpy
    '.sum':         lambda d, **k: d.sum(**k),
    '.mean':        lambda d, **k: d.mean(**k),
    '.std':         lambda d, **k: d.std(**k),
    '.min':         lambda d, **k: d.min(**k),
    '.max':         lambda d, **k: d.max(**k),
    '.var':         lambda d, **k: d.var(**k),
    '.prod':        lambda d, **k: d.prod(**k),
    '.take':        lambda d, **k: d.take(**k),
    '.squeeze':     lambda d, **k: d.squeeze(**k),
    '.reshape':     lambda d, **k: d.reshape(**k),
    '.diagonal':    lambda d, **k: d.diagonal(**k),
    '.trace':       lambda d, **k: d.trace(**k),
    '.transpose':   lambda d, *a: d.transpose(*a),
    '.swapaxes':    lambda d, a1, a2: d.swapaxes(a1, a2),

    'invert':       lambda d, **k: np.invert(d, **k),
    'transpose':    lambda d, **k: np.transpose(d, **k),
    'diff':         lambda d, **k: np.diff(d, **k),
    
    # xarray
    '.sel':         lambda d, **k: d.sel(**k),
    '.isel':        lambda d, **k: d.isel(**k),
    '.median':      lambda d, **k: d.median(**k),
    '.quantile':    lambda d, **k: d.quantile(**k),
    '.argmin':      lambda d, **k: d.argmin(**k),
    '.argmax':      lambda d, **k: d.argmax(**k),
    '.count':       lambda d, **k: d.count(**k),
    '.diff':        lambda d, **k: d.diff(**k),

    # advanced
    'create_mask':  create_mask,
    'where':        where,
})

# Add the boolean operators
_OPERATIONS.update(BOOLEAN_OPERATORS)

# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation(*, name: str, func: Callable,
                       skip_existing: bool=False,
                       overwrite_existing: bool=False) -> None:
    """Adds an entry to the shared OPERATIONS registry.
    
    Args:
        name (str): The name of the operation
        func (Callable): The callable
        skip_existing (bool, optional): Description
        overwrite_existing (bool, optional): Description
    
    Raises:
        TypeError: On invalid name or non-callable for the func argument
        ValueError: On already existing operation name and no skipping or
            overwriting enabled.
    """
    if name in _OPERATIONS and not overwrite_existing:
        if skip_existing:
            return
        raise ValueError("Operation name '{}' already exists! Refusing to "
                         "register a new one. Set the overwrite_existing flag "
                         "to force overwriting.".format(name))

    elif not callable(func):
        raise TypeError("The given {} for operation '{}' is not callable! "
                        "".format(func, name))
    
    elif not isinstance(name, str):
        raise TypeError("Operation name need be a string, was {} with value "
                        "{}!".format(type(name), name))

    _OPERATIONS[name] = func

def apply_operation(op_name: str, *op_args, **op_kwargs) -> Any:
    """Apply an operation with the given arguments and then return it.
    
    Args:
        op_name (str): The name of the operation to carry out; need to be part
            of the OPERATIONS database.
        *op_args: The positional arguments to the operation
        **op_kwargs: The keyword arguments to the operation
    
    Returns:
        Any: The result of the operation
    
    Raises:
        KeyError: On invalid operation name. This also suggests possible other
            names that might match.
        RuntimeError: On failure to apply the operation
    """
    try:
        op = _OPERATIONS[op_name]

    except KeyError as err:
        # Find some close matches to make operation discovery easier
        possible_matches = get_close_matches(op_name, _OPERATIONS.keys(), n=5)

        raise KeyError("No operation '{}' registered! Did you mean: {} ?\n"
                       "All available operations:\n\t{}\n"
                       "If you need to register a new operation, use "
                       "dantro.utils.register_operation to do so."
                       "".format(op_name, ", ".join(possible_matches),
                                 "\n\t".join(_OPERATIONS.keys()))
                       ) from err

    # Compute and return the results
    try:
        return op(*op_args, **op_kwargs)

    except Exception as exc:
        raise type(exc)("Failed applying operation '{}'! {}\n"
                        "  args:   {}\n"
                        "  kwargs: {}\n"
                        "".format(op_name, str(exc), op_args, op_kwargs)
                        ) from exc
