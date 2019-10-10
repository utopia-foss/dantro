"""This module implements data processing operations for dantro objects"""

import logging
import operator
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
    """Prints and passes on the data"""
    if not isinstance(data, (BaseDataContainer, BaseDataGroup)):
        print(data)
    else:
        print("{}, with content:\n{}\n".format(data, data.data))

    return data

# .............................................................................
# numpy and xarray operations

def create_mask(data: xr.DataArray, *,
                operator_name: str, rhs_value: float) -> xr.DataArray:
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
        raise KeyError("No boolean operator with name '{}' available! "
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
                        coords=data.coords,
                        attrs=dict(**data.attrs))

def where(data: xr.DataArray, *,
          operator_name: str, rhs_value: float) -> xr.DataArray:
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
                        coords=dict(unique=unique),
                        attrs=dict(**data.attrs))


# -----------------------------------------------------------------------------
# The Operations Database -----------------------------------------------------
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
OPERATIONS = KeyOrderedDict({
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Unary ...................................................................
    'print':        print_data,
    
    # Binary ..................................................................
    # Item access
    'getitem':      lambda d, key: d[key],

    # N-ary ...................................................................
    # Attribute-related
    'getattr':      getattr,
    'callattr':     lambda d, attr, *a, **k: getattr(d, attr)(*a, **k),
    

    # Numerical operations - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # These are mainly attribute-call based, to
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
    '.size':        lambda d: d.size,
    '.itemsize':    lambda d: d.itemsize,
    '.nbytes':      lambda d: d.nbytes,
    '.base':        lambda d: d.base,
    '.imag':        lambda d: d.imag,
    '.real':        lambda d: d.real,

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
    # ...

    # xarray
    '.coords':      lambda d, key: d.coords[key],

    # N-ary ...................................................................
    # numpy
    '.sum':         lambda d, **k: d.sum(**k),
    '.mean':        lambda d, **k: d.mean(**k),
    '.std':         lambda d, **k: d.std(**k),
    '.min':         lambda d, **k: d.min(**k),
    '.max':         lambda d, **k: d.max(**k),
    '.squeeze':     lambda d, **k: d.squeeze(**k),

    'power':        lambda d, e: np.power(d, e),
    'invert':       lambda d, **k: np.invert(d, **k),
    'transpose':    lambda d, **k: np.transpose(d, **k),
    
    # xarray
    '.sel':         lambda d, **k: d.sel(**k),
    '.isel':        lambda d, **k: d.isel(**k),
    '.median':      lambda d, **k: d.median(**k),

    # advanced
    'create_mask':  create_mask,
    'where':        where,
})

# Add the boolean operators
OPERATIONS.update(BOOLEAN_OPERATORS)

# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation(*, name: str, func: Callable,
                       overwrite_existing: bool=False) -> None:
    """Sets the shared OPERATIONS registry"""
    if name in OPERATIONS and not overwrite_existing:
        raise ValueError("Operation name '{}' already exists! Refusing to "
                         "register a new one. Set the overwrite_existing flag "
                         "to force overwriting.".format(name))

    elif not callable(func):
        raise TypeError("The given {} for operation '{}' is not callable! "
                        "".format(func, name))

    OPERATIONS[name] = func

def apply_operation(op_name: str, *args, _maintain_container_type: bool=False,
                    **kwargs) -> Any:
    """Apply an operation with the given arguments and then return it.

    This function is also capable to attempt to maintain the result type of
    the operation. For that, the type of the first positional argument is
    inspected before the application of the operation. If it is a dantro
    container type and the result is not, it is attempted to feed the result
    to a new instance of the type.
    """
    op = OPERATIONS[op_name]

    if _maintain_container_type:
        if args and isinstance(args[0], BaseDataContainer):
            _ContCls = type(args[0])
            _cont_name = args[0].name
            _cont_attrs = args[0].attrs
        else:
            # Not applicable; turn back the flag
            _maintain_container_type = False

    # Compute the results
    res = op(*args, **kwargs)

    # If container type is to be maintained but does not match, attempt to
    # change it, carrying over name and attributes ...
    if _maintain_container_type and not isinstance(res, _ContCls):
        try:
            res = _ContCls(name="{}_after_{}".format(_cont_name, op_name),
                           data=res, attrs=_cont_attrs)
        except:
            pass

    return res

# -----------------------------------------------------------------------------
