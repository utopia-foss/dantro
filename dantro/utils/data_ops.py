"""This module implements data processing operations for dantro objects"""

import re
import math
import logging
import operator
from importlib import import_module as _import_module
from difflib import get_close_matches as _get_close_matches
from typing import Callable, Any, Sequence, Union, Tuple, Iterable, List

import numpy as np
import xarray as xr
import scipy
import scipy.optimize

from sympy.parsing.sympy_parser import (parse_expr as _parse_expr,
                                        standard_transformations as _std_trf)

from .coords import extract_dim_names, extract_coords_from_attrs
from .ordereddict import KeyOrderedDict
from ..base import BaseDataContainer, BaseDataGroup
from ..tools import apply_along_axis, recursive_getitem

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Operation Implementations
# NOTE Operations should use only logger levels <= REMARK

# Define boolean operators separately; registered into _OPERATIONS below
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

def get_from_module(mod, *, name: str):
    """Retrieves an attribute from a module, if necessary traversing along the
    module string.

    Args:
        mod: Module to start looking at
        name (str): The ``.``-separated module string leading to the desired
            object.
    """
    obj = mod

    for attr_name in name.split("."):
        try:
            obj = getattr(obj, attr_name)

        except AttributeError as err:
            raise AttributeError(
                f"Failed to retrieve attribute or attribute sequence '{name}' "
                f"from module '{mod.__name__}'! Intermediate "
                f"{type(obj).__name__} {obj} has no attribute '{attr_name}'!"
            ) from err

    return obj

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
    return get_from_module(mod, name=name)


# .............................................................................
# symbolic math and expression parsing

def expression(expr: str, *,
               symbols: dict=None,
               evaluate: bool=True,
               transformations: Tuple[Callable]=_std_trf,
               astype: Union[type, str]=float):
    """Parses and evaluates a symbolic math expression using SymPy.

    For parsing, uses sympy's ``parse_expr`` function (see documentation of the
    `parsing module <https://docs.sympy.org/latest/modules/parsing.html>`_).
    The ``symbols`` are provided as ``local_dict``; the ``global_dict`` is not
    explicitly set and subsequently uses the sympy default value, containing
    all basic sympy symbols and notations.

    .. note::

        The expression given here is not Python code, but symbolic math.
        You cannot call arbitrary functions, but only those that are imported
        by ``from sympy import *``.

    .. hint::

        When using this expression as part of the :ref:`dag_framework`, it is
        attached to a so-called :ref:`syntax hook <dag_op_hooks_integration>`
        that makes it easier to specify the ``symbols`` parameter.
        See :ref:`here <dag_op_hook_expression>` for more information.

    .. warning::

        While the expression is symbolic math, be aware that smypy by default
        interprets the ``^`` operator as XOR.
        For exponentiation, use the``**`` operator or adjust the
        ``transformations`` argument as specified in the sympy documentation.

    .. warning::

        While the expression is symbolic math, it uses the ``**`` operator for
        exponentiation, unless a custom ``transformations`` argument is given.

        Thus, the ``^`` operator will lead to an XOR operation being performed!

    .. warning::

        The return object of this operation will *only* contain symbolic sympy
        objects if ``astype is None``. Otherwise, the type cast will evaluate
        all symbolic objects to the numerical equivalent specified by the given
        ``astype``.

    Args:
        expr (str): The expression to evaluate
        symbols (dict, optional): The symbols to use
        evaluate (bool, optional): Controls whether sympy evaluates ``expr``.
            This *may* lead to a fully evaluated result, but does not guarantee
            that no sympy objects are contained in the result. For ensuring
            a fully numerical result, see the ``astype`` argument.
        transformations (Tuple[Callable], optional): The ``transformations``
            argument for sympy's ``parse_expr``. By default, the sympy
            standard transformations are performed.
        astype (Union[type, str], optional): If given, performs a cast to this
            data type, fully evaluating all symbolic expressions.
            Default: Python ``float``.

    Raises:
        TypeError: Upon failing ``astype`` cast, e.g. due to free symbols
            remaining in the evaluated expression.
        ValueError: When parsing of ``expr`` failed.

    Returns:
        The result of the evaluated expression.
    """
    log.remark("Evaluating symbolic expression:  %s", expr)

    symbols = symbols if symbols else {}
    parse_kwargs = dict(evaluate=evaluate, transformations=transformations)

    # Now, parse the expression
    try:
        res = _parse_expr(expr, local_dict=symbols, **parse_kwargs)

    except Exception as exc:
        raise ValueError(
            f"Failed parsing expression '{expr}'! Got a "
            f"{exc.__class__.__name__}: {exc}. Check that the expression can "
            f"be evaluated with the available symbols "
            f"({', '.join(symbols) if symbols else 'none specified'}) "
            f"and inspect the chained exceptions for more information. Parse "
            f"arguments were: {parse_kwargs}"
        ) from exc

    # Finished here if no type cast is desired
    if astype is None:
        return res

    # If full evaluation is desired, do so via a numpy type cast. This works on
    # all sympy objects, but _importantly_ also works on numpy arrays
    # containing sympy objects.
    dtype = np.dtype(astype)
    log.debug("Applying %s ...", dtype)

    try:
        return dtype.type(res)

    except Exception as exc:
        raise TypeError(
            f"Failed casting the result of expression '{expr}' from "
            f"{type(res)} to {dtype}! This can also be due to free symbols "
            f"remaining in the evaluated expression. Either specify the free "
            f"symbols (got: {', '.join(symbols) if symbols else 'none'}) or "
            f"deactivate casting by specifying None as ``dtype`` argument. "
            f"The expression evaluated to:\n\n    {res}\n"
        ) from exc

def generate_lambda(expr: str) -> Callable:
    """Generates a lambda from a string. This is useful when working with
    callables in other operations.

    The ``expr`` argument needs to be a valid Python ``lambda`` expression, see
    `here <docs.python.org/3/tutorial/controlflow.html#lambda-expressions>`_.

    Inside the lambda body, the following names are available for use:

        * A large part of the ``builtins`` module
        * Every name from the Python ``math`` module, e.g. ``sin``, ``cos``, …
        * These modules (and their long form): ``np``, ``xr``, ``scipy``

    Internally, this uses ``eval`` but imposes the following restrictions:

        * The following strings may *not* appear in ``expr``: ``;``, ``__``.
        * There can be no nested ``lambda``, i.e. the only allowed lambda
          string is that in the beginning of ``expr``.
        * The dangerous parts from the ``builtins`` module are *not* available.

    Args:
        expr (str): The expression string to evaluate into a lambda.

    Returns:
        Callable: The generated Callable.

    Raises:
        SyntaxError: Upon failed evaluation of the given expression, invalid
            expression pattern, or disallowed strings in the lambda body.
    """
    ALLOWED_BUILTINS = (
        "abs", "all", "any", "callable", "chr", "divmod", "format", "hash",
        "hex", "id", "isinstance", "issubclass", "iter", "len", "max", "min",
        "next", "oct", "ord", "print", "repr", "round", "sorted", "sum",
        "None", "Ellipsis", "False", "True", "bool", "bytes", "bytearray",
        "complex", "dict", "enumerate", "filter", "float", "frozenset", "int",
        "list", "map", "range", "reversed", "set", "slice", "str", "tuple",
        "type", "zip", "open"
    )
    DISALLOWED_STRINGS = ("lambda", ";", "__")
    LAMBDA_PATTERN = r"^\s*lambda\s[\w\s\,\*]+\:(.+)$"
    #                   arguments definitions  : lambda body (capture group)
    # See also:  https://regex101.com/r/OmI8NY/2/

    # Check if the given expression matches this pattern
    pattern = re.compile(LAMBDA_PATTERN)
    match = pattern.match(expr)
    if match is None:
        raise SyntaxError(f"The given expression '{expr}' was not a valid "
                          "lambda expression!")

    # Check if the lambda body contains disallowed strings
    lambda_body = match[1]
    if any([bad_str in lambda_body for bad_str in DISALLOWED_STRINGS]):
        raise SyntaxError("Encountered one or more disallowed strings in the "
                          f"body ('{lambda_body}') of the given lambda "
                          f"expression ('{expr}'). Make sure none of the "
                          f"following strings appears there: "
                          f"{DISALLOWED_STRINGS}")

    # Ok, sanitized enough now. Prepare the globals dict, restricting access to
    # a subset of builtins and only allowing commonly used math functionality
    _g = dict(
        __builtins__={n: f for n, f in __builtins__.items()
                      if n in ALLOWED_BUILTINS},
        math=math, scipy=scipy, numpy=np, xarray=xr, np=np, xr=xr,
        **{n: f for n, f in math.__dict__.items() if not n.startswith("_")},
    )

    # Try evaluation now (with empty locals)
    try:
        f = eval(expr, _g, {})

    except Exception as exc:
        raise SyntaxError(f"Failed generating lambda object from expression "
                          f"'{expr}'! Got a {exc.__class__.__name__}: {exc}"
                          ) from exc

    return f

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
        raise KeyError(
            f"No boolean operator '{operator_name}' available! Available "
            f"operators: {', '.join(BOOLEAN_OPERATORS.keys())}"
        ) from err

    # Apply the comparison
    data = comp_func(data, rhs_value)

    # Create a new name
    name = data.name + f" (masked by '{operator_name} {rhs_value}')"

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

def count_unique(data, dims: List[str]=None) -> xr.DataArray:
    """Applies np.unique to the given data and constructs a xr.DataArray for
    the results.

    NaN values are filtered out.

    Args:
        data: The data
        dims (List[str], optional): The dimensions along which to apply
            np.unique. The other dimensions will be available after the
            operation. If not provided it is applied along all dims.

    """
    def _count_unique(data) -> xr.DataArray:
        unique, counts = np.unique(data, return_counts=True)
        
        # remove np.nan values
        # NOTE np.nan != np.nan, hence np.nan will count 1 for every occurrence,
        #      but duplicate values not allowed in coords.
        counts = counts[~np.isnan(unique)]
        unique = unique[~np.isnan(unique)]

        if isinstance(data, xr.DataArray):
            name = data.name + " (unique counts)"
        else:
            name = "data (unique counts)"

        # Construct a new data array and return
        return xr.DataArray(data=counts,
                            name=name,
                            dims=('unique',),
                            coords=dict(unique=unique))
    
    if not dims:
        return _count_unique(data)

    if not isinstance(data, xr.DataArray):
        raise TypeError("Data needs to be of type xr.DataArray, but was "
                        f"{type(data)}!")
    
    # use split-apply-combine along those dimensions not in dims
    split_dims = [dim for dim in data.dims if dim not in dims]
    
    if len(split_dims) == 0:
        return _count_unique(data)

    data = data.stack(_stack_cu=split_dims).groupby('_stack_cu')
    return data.map(_count_unique).unstack('_stack_cu')




# .............................................................................
# Working with multidimensional data, mostly xarray-based

def populate_ndarray(objs: Iterable,
                     shape: Tuple[int]=None,
                     dtype: Union[str, type, np.dtype]=float,
                     order: str='C',
                     out: np.ndarray=None,
                     ufunc: Callable=None
                     ) -> np.ndarray:
    """Populates an empty np.ndarray of the given dtype with the given objects
    by zipping over a new array of the given ``shape`` and the sequence of
    objects.

    Args:
        objs (Iterable): The objects to add to the np.ndarray. These objects
            are added in the order they are given here. Note that their final
            position inside the resulting array is furthermore determined by
            the ``order`` argument.
        shape (Tuple[int], optional): The shape of the new array. **Required**
            if no ``out`` array is given.
        dtype (Union[str, type, np.dtype], optional): dtype of the new array.
            Ignored if ``out`` is given.
        order (str, optional): Order of the new array, determines iteration
            order. Ignored if ``out`` is given.
        out (np.ndarray, optional): If given, populates this array rather than
            an empty array.
        ufunc (Callable, optional): If given, applies this unary function to
            each element before storing it in the to-be-returned ndarray.

    Returns:
        np.ndarray: The populated ``out`` array or the newly created one (if
            ``out`` was not given)

    Raises:
        TypeError: On missing
        ValueError: If the number of given objects did not match the array size
    """
    if out is None and shape is None:
        raise TypeError("Without an output array given, the `shape` argument "
                        "needs to be specified!")

    ufunc = ufunc if ufunc is not None else lambda e: e
    out = out if out is not None else np.empty(shape, dtype=dtype, order=order)

    if len(objs) != out.size:
        raise ValueError(
            f"Mismatch between array size ({out.size}, shape: {out.shape}) "
            f"and number of given objects ({len(objs)})!"
        )

    it = np.nditer(out, flags=('multi_index', 'refs_ok'))
    for obj, _ in zip(objs, it):
        out[it.multi_index] = ufunc(obj)

    return out

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
        raise ValueError(
            f"The given sequence of dimension names, {dims}, did not match "
            f"the number of dimensions of data of shape {arrs.shape}!"
        )

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
        raise ValueError(
            "Can only reduce the Dataset resulting from the xr.merge "
            "operation to a DataArray if one and only one data variable is "
            "present in the Dataset! "
            f"Got: {', '.join(dset.data_vars)}. Full data:\n{dset}"
        )

    # Get the name of the single data variable and then get the DataArray
    darr = dset[list(dset.data_vars.keys())[0]]
    # NOTE This is something else than the Dataset.to_array() method, which
    #      includes the name of the data variable as another coordinate. This
    #      is not desired, because it is not relevant.
    return darr

def expand_dims(d: Union[np.ndarray, xr.DataArray],
                *, dim: dict=None, **kwargs) -> xr.DataArray:
    """Expands the dimensions of the given object.

    If the object does not support the ``expand_dims`` method, it will be
    attempted to convert it to an xr.DataArray.

    Args:
        d (Union[np.ndarray, xr.DataArray]): The object to expand the
            dimensions of
        dim (dict, optional): Keys specify the dimensions to expand, values can
            either be an integer specifying the length of the dimension, or a
            sequence of coordinates.
        **kwargs: Passed on to ``expand_dims`` method

    Returns:
        xr.DataArray: The input data with expanded dimensions.
    """
    if not hasattr(d, 'expand_dims'):
        d = xr.DataArray(d)
    return d.expand_dims(dim, **kwargs)

def expand_object_array(d: xr.DataArray, *,
                        shape: Sequence[int]=None,
                        astype: Union[str, type, np.dtype]=None,
                        dims: Sequence[str]=None,
                        coords: Union[dict, str]='trivial',
                        combination_method: str='concat',
                        allow_reshaping_failure: bool=False,
                        **combination_kwargs) -> xr.DataArray:
    """Expands a labelled object-array that contains array-like objects into a
    higher-dimensional labelled array.

    ``d`` is expected to be an array *of arrays*, i.e. each element of the
    outer array is an object that itself is an ``np.ndarray``-like object.
    The ``shape`` is the expected shape of each of these *inner* arrays.
    Importantly, all these arrays need to have the exact same shape.

    Typically, e.g. when loading data from HDF5 files, the inner array will
    not be labelled but will consist of simple np.ndarrays.
    The arguments ``dims`` and ``coords`` are used to label the *inner* arrays.

    This uses :py:func:`~dantro.utils.data_ops.multi_concat` for concatenating
    or :py:func:`~dantro.utils.data_ops.merge` for merging the object arrays
    into a higher-dimensional array, where the latter option allows for missing
    values.

    .. TODO::

        Make reshaping and labelling optional if the inner array already is a
        labelled array. In such cases, the coordinate assignment is already
        done and all information for combination is already available.

    Args:
        d (xr.DataArray): The labelled object-array containing further arrays
            as elements (which are assumed to be unlabelled).
        shape (Sequence[int], optional): Shape of the inner arrays. If not
            given, the first element is used to determine the shape.
        astype (Union[str, type, np.dtype], optional): All inner arrays need to
            have the same dtype. If this argument is given, the arrays will be
            coerced to this dtype. For numeric data, ``float`` is typically a
            good fallback. Note that with ``combination_method == "merge"``,
            the choice here might not be respected.
        dims (Sequence[str], optional): Dimension names for labelling the
            inner arrays. This is necessary for proper alignment. The number of
            dimensions need to match the ``shape``. If not given, will use
            ``inner_dim_0`` and so on.
        coords (Union[dict, str], optional): Coordinates of the inner arrays.
            These are necessary to align the inner arrays with each other. With
            ``coords = "trivial"``, trivial coordinates will be assigned to all
            dimensions. If specifying a dict and giving ``"trivial"`` as value,
            that dimension will be assigned trivial coordinates.
        combination_method (str, optional): The combination method to use to
            combine the object array. For ``concat``, will use dantro's
            :py:func:`~dantro.utils.data_ops.multi_concat`, which preserves
            dtype but does not allow missing values. For ``merge``, will use
            :py:func:`~dantro.utils.data_ops.merge`, which allows missing
            values (masked using ``np.nan``) but leads to the dtype decaying
            to float.
        allow_reshaping_failure (bool, optional): If true, the expansion is not
            stopped if reshaping to ``shape`` fails for an element. This will
            lead to missing values at the respective coordinates and the
            ``combination_method`` will automatically be changed to ``merge``.
        **combination_kwargs: Passed on to the selected combination function,
            :py:func:`~dantro.utils.data_ops.multi_concat` or
            :py:func:`~dantro.utils.data_ops.merge`.

    Returns:
        xr.DataArray: A new, higher-dimensional labelled array.

    Raises:
        TypeError: If no ``shape`` can be extracted from the first element in
            the input data ``d``
        ValueError: On bad argument values for ``dims``, ``shape``, ``coords``
            or ``combination_method``.
    """
    def prepare_item(d: xr.DataArray, *,
                     midx: Sequence[int], shape: Sequence[int],
                     astype: Union[str, type, np.dtype, None],
                     name: str, dims: Sequence[str],
                     generate_coords: Callable
                     ) -> Union[xr.DataArray, None]:
        """Extracts the desired element and reshapes and labels it accordingly.
        If any of this fails, returns ``None``.
        """
        elem = d[midx]

        try:
            item = elem.item().reshape(shape)

        except Exception as exc:
            if allow_reshaping_failure:
                return None
            raise ValueError(
                f"Failed reshaping item at {midx} to {shape}! Make sure the "
                f"element\n\n{elem}\n\nallows reshaping. To discard values "
                "where reshaping fails, enable `allow_reshaping_failure`."
            ) from exc

        if astype is not None:
            item = item.astype(astype)

        return xr.DataArray(item, name=name, dims=dims,
                            coords=generate_coords(elem))

    # Make sure we are operating on labelled data
    d = xr.DataArray(d)

    # Try to deduce missing arguments and make sure arguments are ok
    if shape is None:
        try:
            shape = d.data.flat[0].shape
        except Exception as exc:
            raise TypeError(
                "Failed extracting a shape from the first element of the "
                f"given array:\n{d}\nCheck that the given array contains "
                "further np.ndarray-like objects. Alternatively, explicitly "
                "provide the `shape` argument."
            ) from exc

    if dims is None:
        dims = tuple([f"inner_dim_{n:d}" for n, _ in enumerate(shape)])

    if len(dims) != len(shape):
        raise ValueError(
            "Number of dimension names and number of dimensions of the inner "
            f"arrays needs to match! Got dimension names {dims} for array "
            f"elements of expected shape {shape}."
        )

    if coords == "trivial":
        coords = {n: "trivial" for n in dims}

    elif not isinstance(coords, dict):
        raise TypeError(
            f"Argument `coords` needs to be a dict or str, but was {coords}!"
        )

    if set(coords.keys()) != set(dims):
        raise ValueError(
            "Mismatch between dimension names and coordinate keys! Make sure "
            "there are coordinates specified for each dimension of the inner "
            f"arrays, {dims}! Got:\n{coords}"
        )

    # Handle trivial coordinates for each coordinate dimension separately
    coords = {n: (range(l) if isinstance(c, str) and c == "trivial" else c)
              for (n, c), l in zip(coords.items(), shape)}

    # Assemble info needed to bring individual array items into proper form
    item_name = d.name if d.name else "data"
    item_shape = tuple([1 for _ in d.shape]) + tuple(shape)
    item_dims = d.dims + tuple(dims)
    item_coords = lambda e: dict(**{n: [c.item()] for n,c in e.coords.items()},
                                 **coords)

    # The array that gathers all to-be-combined object arrays
    arrs = np.empty_like(d, dtype=object)
    arrs.fill(dict())  # are ignored in xr.merge

    # Transform each element to a labelled xr.DataArray that includes the outer
    # dimensions and coordinates; the latter is crucial for alignment.
    # Alongside, type coercion can be performed. For failed reshaping, the
    # element may be skipped.
    it = np.nditer(arrs.data, flags=('multi_index', 'refs_ok'))
    for _ in it:
        item = prepare_item(d, midx=it.multi_index, shape=item_shape,
                            astype=astype, name=item_name, dims=item_dims,
                            generate_coords=item_coords)

        if item is None:
            # Missing value; need to fall back to combination via merge
            combination_method = 'merge'
            continue
        arrs[it.multi_index] = item

    # Now, combine
    if combination_method == 'concat':
        return multi_concat(arrs, dims=d.dims, **combination_kwargs)

    elif combination_method == 'merge':
        return merge(arrs, reduce_to_array=True, **combination_kwargs)

    raise ValueError(f"Invalid combination method '{combination_method}'! "
                     "Choose from: 'concat', 'merge'.")


# -----------------------------------------------------------------------------
# The Operations Database -----------------------------------------------------
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
_OPERATIONS = KeyOrderedDict({
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - -
    'define':       lambda d: d,
    'pass':         lambda d: d,
    'print':        print_data,

    # Working on imported modules (useful if other operations don't match)
    'from_module':  get_from_module,
    'import':       import_module_or_object,
    'call':         lambda c, *a, **k: c(*a, **k),
    'import_and_call':
        lambda m, n, *a, **k: import_module_or_object(m, n)(*a, **k),

    'np.':          lambda ms, *a, **k: get_from_module(np, ms)(*a, **k),
    'xr.':          lambda ms, *a, **k: get_from_module(xr, ms)(*a, **k),
    'scipy.':       lambda ms, *a, **k: get_from_module(scipy, ms)(*a, **k),

    # Defining lambdas
    'lambda':       generate_lambda,
    'call_lambda':  lambda e, *a, **k: generate_lambda(e)(*a, **k),

    # Some commonly used types
    'list':         list,
    'dict':         dict,
    'tuple':        tuple,
    'set':          set,

    'int':          int,
    'float':        float,
    'complex':      complex,
    'bool':         bool,
    'str':          str,

    # Item access and manipulation
    'getitem':              lambda d, k:    d[k],
    'setitem':              lambda d, k, v: d.__setitem__(k, v),
    'recursive_getitem':    recursive_getitem,

    # Attribute-related
    'getattr':      getattr,
    'setattr':      setattr,
    'callattr':     lambda d, attr, *a, **k: getattr(d, attr)(*a, **k),

    # Other common Python builtins
    'all':          all,
    'any':          any,
    'len':          len,
    'min':          min,
    'max':          max,
    'sum':          sum,
    'map':          map,
    'repr':         repr,

    # Common operations on strings
    '.format':      lambda s, *a, **k: s.format(*a, **k),
    '.join':        lambda s, *a, **k: s.join(*a, **k),
    '.split':       lambda s, *a, **k: s.split(*a, **k),


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
    '.nonzero':     lambda d: d.nonzero,
    '.flat':        lambda d: d.flat,

    # xarray
    '.head':        lambda d: d.head(),
    '.tail':        lambda d: d.tail(),

    # logarithms and powers
    'log':          lambda d: np.log(d),
    'log10':        lambda d: np.log10(d),
    'log2':         lambda d: np.log2(d),
    'log1p':        lambda d: np.log1p(d),
    'squared':      lambda d: np.square(d),
    'sqrt':         lambda d: np.sqrt(d),
    'cubed':        lambda d: np.power(d, 3),
    'sqrt3':        lambda d: np.power(d, 1./.3),

    # normalization and cumulation
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
    'np.dot':       np.dot,

    # xarray
    '.coords':      lambda d, key: d.coords[key],


    # N-ary ...................................................................
    # Masking, array generation
    'create_mask':                  create_mask,
    'where':                        where,
    'populate_ndarray':             populate_ndarray,
    'expand_object_array':          expand_object_array,

    # extract labelling info, e.g. for creating higher-dimensional arrays
    'extract_dim_names':            extract_dim_names,
    'extract_coords_from_attrs':    extract_coords_from_attrs,

    # dantro-specific wrappers around other library's functionality
    'dantro.apply_along_axis':      apply_along_axis,
    'dantro.multi_concat':          multi_concat,
    'dantro.merge':                 merge,
    'dantro.expand_dims':           expand_dims,

    # evaluating symbolic expressions using sympy
    'expression':                   expression,
    # NOTE: The `^` operator acts as XOR; use `**` for exponentiation!

    # numpy
    '.sum':             lambda d, *a, **k: d.sum(*a, **k),
    '.prod':            lambda d, *a, **k: d.prod(*a, **k),
    '.cumsum':          lambda d, *a, **k: d.cumsum(*a, **k),
    '.cumprod':         lambda d, *a, **k: d.cumprod(*a, **k),

    '.mean':            lambda d, *a, **k: d.mean(*a, **k),
    '.std':             lambda d, *a, **k: d.std(*a, **k),
    '.min':             lambda d, *a, **k: d.min(*a, **k),
    '.max':             lambda d, *a, **k: d.max(*a, **k),
    '.var':             lambda d, *a, **k: d.var(*a, **k),
    '.argmin':          lambda d, *a, **k: d.argmin(*a, **k),
    '.argmax':          lambda d, *a, **k: d.argmax(*a, **k),
    '.argsort':         lambda d, *a, **k: d.argsort(*a, **k),
    '.argpartition':    lambda d, *a, **k: d.argpartition(*a, **k),

    '.transpose':       lambda d, *ax: d.transpose(*ax),
    '.squeeze':         lambda d, **k: d.squeeze(**k),
    '.flatten':         lambda d, **k: d.flatten(**k),
    '.diagonal':        lambda d, **k: d.diagonal(**k),
    '.trace':           lambda d, **k: d.trace(**k),
    '.sort':            lambda d, **k: d.sort(**k),
    '.fill':            lambda d, val: d.fill(val),
    '.round':           lambda d, **k: d.round(**k),
    '.take':            lambda d, i, **k: d.take(i, **k),
    '.swapaxes':        lambda d, a1, a2: d.swapaxes(a1, a2),
    '.reshape':         lambda d, s, **k: d.reshape(s, **k),
    '.astype':          lambda d, t, **k: d.astype(t, **k),

    'np.array':         np.array,
    'np.empty':         np.empty,
    'np.zeros':         np.zeros,
    'np.ones':          np.ones,
    'np.empty_like':    np.empty_like,
    'np.zeros_like':    np.zeros_like,
    'np.ones_like':     np.ones_like,

    'np.eye':           np.eye,
    'np.arange':        np.arange,
    'np.linspace':      np.linspace,
    'np.logspace':      np.logspace,

    'np.invert':        np.invert,
    'np.transpose':     np.transpose,
    'np.diff':          np.diff,
    'np.reshape':       np.reshape,
    'np.take':          np.take,
    'np.repeat':        np.repeat,
    'np.stack':         np.stack,
    'np.hstack':        np.hstack,
    'np.vstack':        np.vstack,
    'np.concatenate':   np.concatenate,

    'np.abs':           np.abs,
    'np.ceil':          np.ceil,
    'np.floor':         np.floor,
    'np.round':         np.round,

    'np.where':         np.where,
    'np.digitize':      np.digitize,
    'np.histogram':     np.histogram,
    'np.count_nonzero': np.count_nonzero,

    # xarray
    '.sel':             lambda d, *a, **k: d.sel(*a, **k),
    '.isel':            lambda d, *a, **k: d.isel(*a, **k),
    '.median':          lambda d, *a, **k: d.median(*a, **k),
    '.quantile':        lambda d, *a, **k: d.quantile(*a, **k),
    '.argmin':          lambda d, *a, **k: d.argmin(*a, **k),
    '.argmax':          lambda d, *a, **k: d.argmax(*a, **k),
    '.count':           lambda d, *a, **k: d.count(*a, **k),
    '.diff':            lambda d, *a, **k: d.diff(*a, **k),
    '.where':           lambda d, c, *a, **k: d.where(c, *a, **k),

    '.groupby':         lambda d, g, **k: d.groupby(g, **k),
    '.groupby_bins':    lambda d, g, **k: d.groupby_bins(g, **k),
    '.map':             lambda ds, func, **k: ds.map(func, **k),
    '.reduce':          lambda ds, func, **k: ds.reduce(func, **k),

    '.rename':          lambda d, *a, **k: d.rename(*a, **k),
    '.expand_dims':     lambda d, *a, **k: d.expand_dims(*a, **k),
    '.assign_coords':   lambda d, *a, **k: d.assign_coords(*a, **k),

    '.to_array':        lambda ds, *a, **k: ds.to_array(*a, **k),

    'xr.Dataset':       xr.Dataset,
    'xr.DataArray':     xr.DataArray,
    'xr.zeros_like':    xr.zeros_like,
    'xr.ones_like':     xr.ones_like,

    'xr.merge':             xr.merge,
    'xr.concat':            xr.concat,
    'xr.align':             xr.align,
    'xr.combine_nested':    xr.combine_nested,
    'xr.combine_by_coords': xr.combine_by_coords,

    # scipy
    'curve_fit':        scipy.optimize.curve_fit,
    # NOTE: Use the 'lambda' operation to generate the callable
}) # End of default operation definitions

# Add the boolean operators
_OPERATIONS.update(BOOLEAN_OPERATORS)


# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation(*, name: str, func: Callable,
                       skip_existing: bool=False,
                       overwrite_existing: bool=False) -> None:
    """Adds an entry to the shared operations registry.

    Args:
        name (str): The name of the operation
        func (Callable): The callable
        skip_existing (bool, optional): Whether to skip registration if the
            operation name is already registered. This suppresses the
            ValueError raised on existing operation name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing operation of the same name. If given, this takes
            precedence over ``skip_existing``.

    Raises:
        TypeError: On invalid name or non-callable for the func argument
        ValueError: On already existing operation name and no skipping or
            overwriting enabled.
    """
    if name in _OPERATIONS and not overwrite_existing:
        if skip_existing:
            log.debug("Operation '%s' is already registered and will not be "
                      "registered again.", name)
            return
        raise ValueError(
            f"Operation name '{name}' already exists! Refusing to register a "
            "new one. Set the overwrite_existing flag to force overwriting."
        )

    elif not callable(func):
        raise TypeError(
            f"The given {func} for operation '{name}' is not callable! "
        )

    elif not isinstance(name, str):
        raise TypeError(
            f"Operation name need be a string, was {type(name)} with "
            f"value {name}!"
        )

    _OPERATIONS[name] = func
    log.debug("Registered operation '%s'.", name)

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
        close_matches = available_operations(match=op_name)
        join_str = "\n  - "

        raise ValueError(
            f"No operation '{op_name}' registered! Did you mean: "
            f"{', '.join(close_matches) if close_matches else '(no match)'} ?"
            f"\nAvailable operations:{join_str}"
            f"{join_str.join(available_operations())}\nIf you need to "
            "register a new operation, use dantro.utils.register_operation."
        ) from err

    # Compute and return the results
    log.log(_log_level, "Performing operation '%s' ...", op_name)
    try:
        return op(*op_args, **op_kwargs)

    except Exception as exc:
        raise RuntimeError(
            f"Failed applying operation '{op_name}'! "
            f"Got a {exc.__class__.__name__}: {exc}\n"
            f"  args:   {op_args}\n"
            f"  kwargs: {op_kwargs}\n"
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
    return _get_close_matches(match, _OPERATIONS.keys(), n=n)
