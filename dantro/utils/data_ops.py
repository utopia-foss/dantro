"""This module implements data processing operations for dantro objects"""

import logging
import operator
from importlib import import_module as _import_module
from difflib import get_close_matches
from typing import Callable, Any, Sequence, Union

import numpy as np
import xarray as xr

from .ordereddict import KeyOrderedDict
from ..base import BaseDataContainer, BaseDataGroup
from ..tools import apply_along_axis

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
} # End of boolean operator definitions

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

def import_module_or_object(module: str=None, name: str=None):
    """Imports a module or an object using the specified module string and the
    object name.
    
    Args:
        module (str, optional): A module string, e.g. numpy.random. If this is
            not given, it will import from the :py:mod`builtins` module. Also,
            relative module strings are resolved from :py:mod:`dantro`.
        name (str, optional): The name of the object to retrieve from the
            chosen module and return. This may also be a dot-separated sequence
            of attribute names which can be used to traverse along attributes.
    
    Returns:
        The chosen module or object, i.e. the object found at <module>.<name>
    
    Raises:
        AttributeError: In cases where part of the ``name`` argument could not
            be resolved due to a bad attribute name.
    """
    module = module if module else 'builtins'
    mod = _import_module(module, package='dantro') if module else __builtin__

    if not name:
        return mod

    # Get the object by traversing along the attributes of the module. By
    # allowing a name to be a sequence, object imports become more versatile.
    # The first object to get the name from is the module itself:
    obj = mod

    for attr_name in name.split("."):
        try:
            obj = getattr(obj, attr_name)
        
        except AttributeError as err:
            raise AttributeError("Failed to retrieve attribute or attribute "
                                 "sequence '{}' from module '{}'! "
                                 "Intermediate {} {} has no attribute '{}'!"
                                 "".format(name, mod.__name__,
                                           type(obj).__name__, obj, attr_name)
                                 ) from err

    return obj


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
            registered in the ``BOOLEAN_OPERATORS`` constant.
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


# .............................................................................
# Working with multidimensional data, mostly xarray-based

def populate_ndarray(*objs, shape: tuple, dtype: str='float', order: str='C'
                     ) -> np.ndarray:
    """Populates an empty np.ndarray of the given dtype with the objects.
    
    Args:
        *objs: The objects to add to the 
        shape (tuple): The shape of the new array
        dtype (str, optional): Data type of the new array
        order (str, optional): Order of the new array
    
    Returns:
        np.ndarray: The newly created and populated array
    
    Raises:
        ValueError: If the number of given objects did not match the array size
    """
    arr = np.empty(shape, dtype=dtype, order=order)

    if len(objs) != arr.size:
        raise ValueError("Mismatch between array size ({}, shape: {}) and "
                         "number of given objects ({})!"
                         "".format(arr.size, arr.shape, len(objs)))

    it = np.nditer(arr, flags=('multi_index', 'refs_ok'))
    for obj, _ in zip(objs, it):
        arr[it.multi_index] = obj

    return arr


def multi_concat(arrs: np.ndarray, *, dims: Sequence[str]) -> xr.DataArray:
    """Concatenates ``xr.Dataset`` or ``xr.DataArray`` objects using
    ``xr.concat``. This function expects the xarray objects to be pre-aligned
    inside the numpy *object* array ``arrs``, with the number of dimensions
    matching the number of concatenation operations desired.
    The position inside the array carries information on where the objects that
    are to be concatenated are placed inside the higher dimensional coordinate
    system.

    Through multiple concatenation, the dimensionality of the contained objects
    is increased by ``dims``, while their dtype can be maintained.

    For the sequential application of ``xr.concat`` along the outer dimensions,
    the custom :py:func:`dantro.tools.apply_along_axis` is used.
    
    Args:
        arrs (np.ndarray): The array containing xarray objects which are to be
            concatenated. Each array dimension should correspond to one of the
            given ``dims``. For each of the dimensions, the ``xr.concat``
            operation is applied along the axis, effectively reducing the
            dimensionality of ``arrs`` to a scalar and increasing the
            dimensionality of the contained xarray objects until they
            additionally contain the dimensions specified in ``dims``.
        dims (Sequence[str]): A sequence of dimension names that is assumed to
            match the dimension names of the array. During each concatenation
            operation, the name is passed along to ``xr.concat`` where it is
            used to select the dimension of the *content* of ``arrs`` along
            which concatenation should occur.
    
    Raises:
        ValueError: If number of dimension names does not match the number of
            data dimensions.
    """
    # Check dimensionality
    if len(dims) != arrs.ndim:
        raise ValueError("The given sequence of dimension names, {}, did not "
                         "match the number of dimensions of data of shape {}!"
                         "".format(dims, arrs.shape))

    # Reverse-iterate over dimensions and concatenate them
    for dim_idx, dim_name in reversed(list(enumerate(dims))):
        log.debug("Concatenating along axis '%s' (idx: %d) ...",
                  dim_name, dim_idx)

        arrs = apply_along_axis(xr.concat, axis=dim_idx, arr=arrs,
                                dim=dim_name)
        # NOTE ``np.apply_along_axis`` would be what is desired here, but that
        #      function unfortunately tries to cast objects to np.arrays which
        #      is not what we want here at all! Thus, this function uses the
        #      custom dantro function of the same name instead.

    # Should be scalar now, get the element.
    return arrs.item()

def merge(arrs: Union[Sequence[Union[xr.DataArray, xr.Dataset]], np.ndarray],
          *, reduce_to_array: bool=False, **merge_kwargs
          ) -> Union[xr.Dataset, xr.DataArray]:
    """Merges the given sequence of xarray objects into an xr.Dataset.

    As a convenience, this also allows passing a numpy object array containing
    the xarray objects. Furthermore, if the resulting Dataset contains only a
    single data variable, that variable can be extracted as a DataArray which
    is then the return value of this operation.
    """
    if isinstance(arrs, np.ndarray):
        arrs = arrs.flat

    dset = xr.merge(arrs, **merge_kwargs)

    if not reduce_to_array:
        return dset
    
    elif len(dset.data_vars) != 1:
        raise ValueError("Can only reduce the Dataset resulting from the "
                         "xr.merge operation to a DataArray if one and only "
                         "one data variable is present in the Dataset! "
                         "Got: {}. Full data:\n{}"
                         "".format(", ".join(dset.data_vars), dset))

    # Get the name of the single data variable and then get the DataArray
    darr = dset[list(dset.data_vars.keys())[0]]
    # NOTE This is something else than the Dataset.to_array() method, which
    #      includes the name of the data variable as another coordinate. This
    #      is not desired, because it is not relevant.
    return darr


def expand_dims(d: Any, *, dim: dict=None, **kwargs) -> xr.DataArray:
    """Expands the dimensions of the given object.

    If the object does not support the `expand_dims` method, it will be
    attempted to convert it to an xr.DataArray.
    """
    if not hasattr(d, 'expand_dims'):
        d = xr.DataArray(d)

    return d.expand_dims(dim, **kwargs)


# -----------------------------------------------------------------------------
# The Operations Database -----------------------------------------------------
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
_OPERATIONS = KeyOrderedDict({
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    'define':       lambda d: d,
    'pass':         lambda d: d,
    'print':        print_data,

    'import':       import_module_or_object,
    'call':         lambda c, *a, **k: c(*a, **k),
    'import_and_call':
        lambda m, n, *a, **k: import_module_or_object(m, n)(*a, **k),

    # Some commonly used types
    'list':         list,
    'dict':         dict,
    'tuple':        tuple,
    'set':          set,

    'int':          int,
    'float':        float,
    'str':          str,
    
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
    'create_mask':          create_mask,
    'where':                where,
    'populate_ndarray':     populate_ndarray,

    # dantro-specific wrappers around other library's functionality
    'dantro.multi_concat':  multi_concat,
    'dantro.merge':         merge,
    'dantro.expand_dims':   expand_dims,

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
    'reshape':      lambda d, s, **k: np.reshape(d, s, **k),

    'np.array':     np.array,
    'np.empty':     np.empty,
    'np.zeros':     np.zeros,
    'np.ones':      np.ones,
    'np.arange':    np.arange,
    'np.linspace':  np.linspace,
    'np.logspace':  np.logspace,
    
    # xarray
    '.sel':         lambda d, **k: d.sel(**k),
    '.isel':        lambda d, **k: d.isel(**k),
    '.median':      lambda d, **k: d.median(**k),
    '.quantile':    lambda d, **k: d.quantile(**k),
    '.argmin':      lambda d, **k: d.argmin(**k),
    '.argmax':      lambda d, **k: d.argmax(**k),
    '.count':       lambda d, **k: d.count(**k),
    '.diff':        lambda d, **k: d.diff(**k),

    '.expand_dims':     lambda d, **k: d.expand_dims(**k),
    '.assign_coords':   lambda d, **k: d.assign_coords(**k),

    'xr.Dataset':   xr.Dataset,
    'xr.DataArray': xr.DataArray,
    'xr.merge':     xr.merge,
    'xr.concat':    xr.concat,
}) # End of default operation definitions

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

def apply_operation(op_name: str, *op_args, _log_level: int=5,
                    **op_kwargs) -> Any:
    """Apply an operation with the given arguments and then return it.
    
    Args:
        op_name (str): The name of the operation to carry out; need to be part
            of the OPERATIONS database.
        *op_args: The positional arguments to the operation
        _log_level (int, optional): Log level of the log messages created by
            this function.
        **op_kwargs: The keyword arguments to the operation
    
    Returns:
        Any: The result of the operation
    
    Raises:
        KeyError: On invalid operation name. This also suggests possible other
            names that might match.
        Exception: On failure to apply the operation, preserving the original
            exception.
    """
    try:
        op = _OPERATIONS[op_name]

    except KeyError as err:
        # Find some close matches to make operation discovery easier
        possible_matches = available_operations(match=op_name)

        raise ValueError("No operation '{}' registered! Did you mean: {} ?\n"
                         "Available operations:\n  - {}\n"
                         "If you need to register a new operation, use "
                         "dantro.utils.register_operation to do so."
                         "".format(op_name, ", ".join(possible_matches),
                                   "\n  - ".join(available_operations()))
                         ) from err

    # Compute and return the results
    log.log(_log_level, "Performing operation '%s' ...", op_name)
    try:
        return op(*op_args, **op_kwargs)

    except Exception as exc:
        raise RuntimeError("Failed applying operation '{}'! Got a {}: {}\n"
                           "  args:   {}\n"
                           "  kwargs: {}\n"
                           "".format(op_name, exc.__class__.__name__, str(exc),
                                     op_args, op_kwargs)
                           ) from exc


def available_operations(*, match: str=None, n: int=5) -> Sequence[str]:
    """Returns all available operation names or a fuzzy-matched subset of them.
    
    Args:
        match (str, optional): If given, fuzzy-matches the names and only
            returns close matches to this name.
        n (int, optional): Number of close matches to return. Passed on to
            difflib.get_close_matches
    
    Returns:
        Sequence[str]: All available operation names or the matched subset.
            The sequence is sorted alphabetically.
    """
    if match is None:
        return _OPERATIONS.keys()

    # Use fuzzy matching to return close matches
    return get_close_matches(match, _OPERATIONS.keys(), n=n)
