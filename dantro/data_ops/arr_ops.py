"""Implements data operations that work on array-like data, e.g. from numpy
or xarray."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from .._import_tools import LazyLoader
from ._base_ops import BOOLEAN_OPERATORS

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")

# -----------------------------------------------------------------------------


def apply_along_axis(
    func: Callable, axis: int, arr: np.ndarray, *args, **kwargs
) -> np.ndarray:
    """This is like numpy's function of the same name, but does not try to
    cast the results of func to an :py:class:`numpy.ndarray` but tries to keep
    them as dtype object. Thus, the return value of this function will always
    have one fewer dimension then the input array.

    This goes along the equivalent formulation of
    :py:func:`numpy.apply_along_axis`, outlined in their documentation of the
    function.

    Args:
        func (Callable): The function to apply along the axis
        axis (int): Which axis to apply it to
        arr (numpy.ndarray): The array-like data to apply the function to
        *args: Passed to ``func``
        **kwargs: Passed to ``func``

    Returns:
        numpy.ndarray: with ``func`` applied along ``axis``, reducing the array
            dimensions by one.
    """
    # Get the shapes of the outer and inner iteration; both are tuples!
    shape_outer, shape_inner = arr.shape[:axis], arr.shape[axis + 1 :]
    num_outer = len(shape_outer)

    # These together give the shape of the output array
    out = np.zeros(shape_outer + shape_inner, dtype="object")
    out.fill(None)

    log.debug("Applying function '%s' along axis ...", func.__name__)
    log.debug("  input array:     %s, %s", arr.shape, arr.dtype)
    log.debug("  axis to reduce:  %d", axis)
    log.debug("  output will be:  %s, %s", out.shape, out.dtype)

    # Now loop over the output array and at each position fill it with the
    # result of the function call.
    it = np.nditer(out, flags=("refs_ok", "multi_index"))
    for _ in it:
        midx = it.multi_index

        # Build selector, which has the ellipsis at position `axis`, thus one
        # dimension higher than the out array and matching the input `arr`.
        sel = tuple(midx[:num_outer]) + (Ellipsis,) + tuple(midx[num_outer:])
        log.debug("  midx: %s  -->  selector: %s", midx, sel)

        # Apply function to selected parts of array, then write to the current
        # point in the iteration over the output array.
        out[midx] = func(arr[sel], *args, **kwargs)

    return out


def create_mask(
    data: "xarray.DataArray", operator_name: str, rhs_value: float
) -> "xarray.DataArray":
    """Given the data, returns a binary mask by applying the following
    comparison: ``data <operator> rhs value``.

    Args:
        data (xarray.DataArray): The data to apply the comparison to. This is
            the left-hand-side of the comparison.
        operator_name (str): The name of the binary operator function as
            registered in the ``BOOLEAN_OPERATORS`` database.
        rhs_value (float): The right-hand-side value

    Raises:
        KeyError: On invalid operator name

    Returns:
        xarray.DataArray: Boolean mask
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

    # If a name already exists, add some information about the masking to it
    if data.name:
        name = data.name + f" (masked by '{operator_name} {rhs_value}')"
    else:
        name = None

    # Build a new xr.DataArray from that data, retaining all information
    return xr.DataArray(
        data=data, name=name, dims=data.dims, coords=data.coords
    )


def where(
    data: "xarray.DataArray", operator_name: str, rhs_value: float, **kwargs
) -> "xarray.DataArray":
    """Filter elements from the given data according to a condition. Only
    those elemens where the condition is fulfilled are not masked.

    .. note::

        This typically leads to a dtype change to :py:attr:`numpy.float64`.

    Args:
        data (xarray.DataArray): The data to mask
        operator_name (str): The ``operator`` argument used in
            :py:func:`.arr_ops.create_mask`
        rhs_value (float): The ``rhs_value`` argument used in
            :py:func:`.arr_ops.create_mask`
        **kwargs: Passed on to ``.where()`` method call
    """
    return data.where(
        create_mask(data, operator_name=operator_name, rhs_value=rhs_value),
        **kwargs,
    )


def count_unique(data, dims: List[str] = None) -> "xarray.DataArray":
    """Applies :py:func:`numpy.unique` to the given data and constructs a
    :py:class:`xarray.DataArray` for the results.

    NaN values are filtered out.

    Args:
        data: The data
        dims (List[str], optional): The dimensions along which to apply
            np.unique. The other dimensions will be available after the
            operation. If not provided it is applied along all dims.

    """

    def _count_unique(data) -> "xarray.DataArray":
        unique, counts = np.unique(data, return_counts=True)

        # remove np.nan values
        # NOTE np.nan != np.nan, hence np.nan will count 1 for every
        #      occurrence, but duplicate values are not allowed in coords...
        counts = counts[~np.isnan(unique)]
        unique = unique[~np.isnan(unique)]

        if isinstance(data, xr.DataArray) and data.name:
            name = data.name + " (unique counts)"
        else:
            name = "unique counts"

        # Construct a new data array and return
        return xr.DataArray(
            data=counts,
            name=name,
            dims=("unique",),
            coords=dict(unique=unique),
        )

    if not dims:
        return _count_unique(data)

    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Data needs to be of type xr.DataArray, but was {type(data)}!"
        )

    # use split-apply-combine along those dimensions not in dims
    split_dims = [dim for dim in data.dims if dim not in dims]

    if len(split_dims) == 0:
        return _count_unique(data)

    data = data.stack(_stack_cu=split_dims).groupby("_stack_cu")
    return data.map(_count_unique).unstack("_stack_cu")


# .............................................................................
# Working with multidimensional data, mostly xarray-based


def populate_ndarray(
    objs: Iterable,
    shape: Tuple[int] = None,
    dtype: Union[str, type, np.dtype] = float,
    order: str = "C",
    out: np.ndarray = None,
    ufunc: Callable = None,
) -> np.ndarray:
    """Populates an empty :py:class:`numpy.ndarray` of the given ``dtype`` with
    the given objects by zipping over a new array of the given ``shape`` and
    the sequence of objects.

    Args:
        objs (Iterable): The objects to add to the :py:class:`numpy.ndarray`.
            These objects are added in the order they are given here. Note
            that their final position inside the resulting array is
            furthermore determined by the ``order`` argument.
        shape (Tuple[int], optional): The shape of the new array. **Required**
            if no ``out`` array is given.
        dtype (Union[str, type, numpy.dtype], optional): dtype of the new
            array. Ignored if ``out`` is given.
        order (str, optional): Order of the new array, determines iteration
            order. Ignored if ``out`` is given.
        out (numpy.ndarray, optional): If given, populates this array rather
            than an empty array.
        ufunc (Callable, optional): If given, applies this unary function to
            each element before storing it in the to-be-returned ndarray.

    Returns:
        numpy.ndarray: The populated ``out`` array or the newly created one (if
            ``out`` was not given)

    Raises:
        TypeError: On missing ``shape`` argument if ``out`` is not given
        ValueError: If the number of given objects did not match the array size
    """
    if out is None and shape is None:
        raise TypeError(
            "Without an output array given, the `shape` argument "
            "needs to be specified!"
        )

    ufunc = ufunc if ufunc is not None else lambda e: e
    out = out if out is not None else np.empty(shape, dtype=dtype, order=order)

    if len(objs) != out.size:
        raise ValueError(
            f"Mismatch between array size ({out.size}, shape: {out.shape}) "
            f"and number of given objects ({len(objs)})!"
        )

    it = np.nditer(out, flags=("multi_index", "refs_ok"))
    for obj, _ in zip(objs, it):
        out[it.multi_index] = ufunc(obj)

    return out


def build_object_array(
    objs: Union[Dict, Sequence],
    *,
    dims: Tuple[str] = ("label",),
    fillna: Any = None,
) -> "xarray.DataArray":
    """Creates a *simple* labelled multidimensional object array.

    It accepts simple iterable types like dictionaries or lists and unpacks
    them into the array, using the key or index (respectively) as coordinate
    for the entry. For dict-like entries, multi-dimensional coordinates can be
    specified by using tuples for keys.
    Subsequently, list-like iterable types (list, tuple etc.) will result in
    one-dimensional output array.

    .. warning::

        This data operation is built for *flexibility*, not for speed. It will
        call the :py:func:`.merge` operation for *every*
        element in the ``objs`` iterable, thus being slow and potentially
        creating an array with many empty elements.
        To efficiently populate an n-dimensional object array, use the
        :py:func:`.populate_ndarray` operation instead
        and build a labelled array from that output.

    Args:
        objs (Union[Dict, Sequence]): The objects to populate the object array
            with. If dict-like, keys are assumed to encode coordinates, which
            can be of the form ``coord0`` or ``(coord0, coord1, …)``, where the
            tuple-form requires as many coordinates as there are entries in the
            ``dims`` argument.
            If list- or tuple-like (more exactly: if missing the ``items``
            attribute) trivial indexing is used and ``dims`` needs to be 1D.
        dims (Tuple[str], optional): The names of the dimensions of the
            labelled array.
        fillna (Any, optional): The fill value for entries that are not
            covered by the dimensions specified by ``objs``. Note that this
            will replace all *null* values, which includes `NaN` but also
            ``None``. This operation is only called if ``fillna is not None``.

    Raises:
        ValueError: If coordinates and/or ``dims`` argument for individual
            entries did not match.
    """

    def get_coords(k, dims: Tuple[str]) -> dict:
        """Turn the iteration key into a valid coordinate dict"""
        if isinstance(k, tuple):
            if len(k) != len(dims):
                raise ValueError(
                    f"Given coordinates {k} could not be matched to the "
                    f"specified dimensions, {dims}! Make sure their sizes "
                    "agree."
                )
            return {_k: [_v] for _k, _v in zip(dims, k)}

        elif len(dims) != 1:
            raise ValueError(
                f"Got scalar coordinate '{k}' but have {len(dims)} dimensions "
                f"({dims}) specified. Either provide an appropriately sized "
                "coordinate tuple or reduce the number of `dims` to one."
            )
        return {dims[0]: [k]}

    # Determine the object iterator
    if not hasattr(objs, "items"):
        if len(dims) != 1:
            raise ValueError(
                "Can only create one-dimensional output data from the given "
                f"list-like object container, but got `dims`: {dims}. "
                f"Instead of {type(objs).__name__}, use a dict to specify "
                "multiple coordinates or adjust the `dims` argument to a "
                "single dimension name."
            )

        it = enumerate(objs)

    else:
        it = objs.items()

    # The (zero-sized) target array
    ndim = len(dims)
    out = xr.DataArray(
        np.zeros((0,) * ndim, dtype="object"),
        dims=dims,
        name="tmp",
        coords=dict(zip(dims, [[]] * ndim)),
    )

    # Populate it entry by entry, merging every entry into the existing array
    for k, v in it:
        coords = get_coords(k, dims)
        new_item = xr.DataArray(
            populate_ndarray([v], shape=(1,) * ndim, dtype="object"),
            dims=dims,
            coords=coords,
            name="tmp",
        )
        out = merge([out, new_item], reduce_to_array=True)

    out.name = None
    if fillna is not None:
        out = out.fillna(fillna)
    return out


def multi_concat(
    arrs: np.ndarray, *, dims: Sequence[str]
) -> "xarray.DataArray":
    """Concatenates :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
    objects using :py:func:`xarray.concat`. This function expects the xarray
    objects to be pre-aligned inside the numpy *object* array ``arrs``, with
    the number of dimensions matching the number of concatenation operations
    desired.
    The position inside the array carries information on where the objects that
    are to be concatenated are placed inside the higher dimensional coordinate
    system.

    Through multiple concatenation, the dimensionality of the contained objects
    is increased by ``dims``, while their dtype can be maintained.

    For the sequential application of :py:func:`xarray.concat` along the outer
    dimensions, the custom :py:func:`.apply_along_axis` function is used.

    Args:
        arrs (numpy.ndarray): The array containing xarray objects which are to
            be concatenated. Each array dimension should correspond to one of
            the given ``dims``. For each of the dimensions, the
            :py:func:`xarray.concat` operation is applied along the axis,
            effectively reducing the dimensionality of ``arrs`` to a scalar
            and increasing the dimensionality of the contained xarray objects
            until they additionally contain the dimensions specified in
            the ``dims`` argument.
        dims (Sequence[str]): A sequence of dimension names that is assumed to
            match the dimension names of the array. During each concatenation
            operation, the name is passed along to :py:func:`xarray.concat`
            where it is used to select the dimension of the *content* of
            ``arrs`` along which concatenation should occur.

    Raises:
        ValueError: If number of dimension names does not match the number of
            data dimensions.
    """
    if len(dims) != arrs.ndim:
        raise ValueError(
            f"The given sequence of dimension names, {dims}, did not match "
            f"the number of dimensions of data of shape {arrs.shape}!"
        )

    # Reverse-iterate over dimensions and concatenate them
    for dim_idx, dim_name in reversed(list(enumerate(dims))):
        log.debug(
            "Concatenating along axis '%s' (idx: %d) ...", dim_name, dim_idx
        )

        arrs = apply_along_axis(
            xr.concat, axis=dim_idx, arr=arrs, dim=dim_name
        )
        # NOTE ``np.apply_along_axis`` would be what is desired here, but that
        #      function unfortunately tries to cast objects to np.arrays which
        #      is not what we want here at all! Thus, this function uses the
        #      custom dantro function of the same name instead.

    # Should be scalar now, get the element.
    return arrs.item()


def merge(
    arrs: Union[
        Sequence[Union["xarray.DataArray", "xarray.Dataset"]], np.ndarray
    ],
    *,
    reduce_to_array: bool = False,
    **merge_kwargs,
) -> Union["xarray.Dataset", "xarray.DataArray"]:
    """Merges the given sequence of xarray objects into an
    :py:class:`xarray.Dataset`.

    As a convenience, this also allows passing a :py:class:`numpy.ndarray` of
    dtype ``object`` containing the xarray objects.
    Furthermore, if the resulting :py:class:`xarray.Dataset` contains only a
    single data variable, that variable can be extracted as a
    :py:class:`xarray.DataArray` by setting the ``reduce_to_array`` flag,
    making that array the return value of this operation.
    """
    if isinstance(arrs, np.ndarray):
        arrs = arrs.flat

    dset = xr.merge(arrs, **merge_kwargs)

    if not reduce_to_array:
        return dset

    if len(dset.data_vars) != 1:
        raise ValueError(
            "The Dataset resulting from the xr.merge operation can only be "
            "reduced to a DataArray, if one and only one data variable is "
            "present in the Dataset! "
            f"However, the merged Dataset contains {len(dset.data_vars)} data "
            f"variables:  {', '.join(dset.data_vars)}\n"
            f"Full dataset before attempting to reduce to an array:\n{dset}\n"
            "A typical reason for this is missing data; check that there were "
            "sufficiently populated xarray objects available for merging.\n"
        )

    # Get the name of the single data variable and then get the DataArray
    darr = dset[list(dset.data_vars.keys())[0]]
    # NOTE This is something else than the Dataset.to_array() method, which
    #      includes the name of the data variable as another coordinate. This
    #      is not desired, because it is not relevant.
    return darr


def expand_dims(
    d: Union[np.ndarray, "xarray.DataArray"], *, dim: dict = None, **kwargs
) -> "xarray.DataArray":
    """Expands the dimensions of the given object.

    If the object does not support a ``expand_dims`` method call, it will be
    attempted to convert it to an :py:class:`xarray.DataArray` first.

    Args:
        d (Union[numpy.ndarray, xarray.DataArray]): The object to expand the
            dimensions of
        dim (dict, optional): Keys specify the dimensions to expand, values can
            either be an integer specifying the length of the dimension, or a
            sequence of coordinates.
        **kwargs: Passed on to the ``expand_dims`` method call. For an xarray
            objects that would be :py:meth:`xarray.DataArray.expand_dims`.

    Returns:
        xarray.DataArray: The input data with expanded dimensions.
    """
    if not hasattr(d, "expand_dims"):
        d = xr.DataArray(d)
    return d.expand_dims(dim, **kwargs)


def expand_object_array(
    d: "xarray.DataArray",
    *,
    shape: Sequence[int] = None,
    astype: Union[str, type, np.dtype] = None,
    dims: Sequence[str] = None,
    coords: Union[dict, str] = "trivial",
    combination_method: str = "concat",
    allow_reshaping_failure: bool = False,
    **combination_kwargs,
) -> "xarray.DataArray":
    """Expands a labelled object-array that contains array-like objects into a
    higher-dimensional labelled array.

    ``d`` is expected to be an array *of arrays*, i.e. each element of the
    outer array is an object that itself is an :py:class:`numpy.ndarray`-like
    object. The ``shape`` is the expected shape of each of these *inner*
    arrays. *Importantly*, all these arrays need to have the exact same shape!

    Typically, e.g. when loading data from HDF5 files, the inner array will
    not be labelled but will consist of simple :py:class:`numpy.ndarray`
    objects.
    The arguments ``dims`` and ``coords`` are used to label the *inner* arrays.

    This uses :py:func:`.multi_concat` for concatenating
    or :py:func:`.merge` for merging the object arrays
    into a higher-dimensional array, where the latter option allows for missing
    values.

    .. TODO::

        Make reshaping and labelling optional if the inner array already is a
        labelled array. In such cases, the coordinate assignment is already
        done and all information for combination is already available.

    Args:
        d (xarray.DataArray): The labelled object-array containing further
            arrays as elements (which are assumed to be unlabelled).
        shape (Sequence[int], optional): Shape of the inner arrays. If not
            given, the first element is used to determine the shape.
        astype (Union[str, type, numpy.dtype], optional): All inner arrays
            need to have the same dtype. If this argument is given, the arrays
            will be coerced to this dtype. For numeric data, ``float`` is
            typically a good fallback.
            Note that with ``combination_method == "merge"``, the choice here
            might not be respected.
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
            :py:func:`.multi_concat`, which preserves
            dtype but does not allow missing values. For ``merge``, will use
            :py:func:`.merge`, which allows missing
            values (masked using ``np.nan``) but leads to the dtype decaying
            to float.
        allow_reshaping_failure (bool, optional): If true, the expansion is not
            stopped if reshaping to ``shape`` fails for an element. This will
            lead to missing values at the respective coordinates and the
            ``combination_method`` will automatically be changed to ``merge``.
        **combination_kwargs: Passed on to the selected combination function,
            :py:func:`.multi_concat` or
            :py:func:`.merge`.

    Returns:
        xarray.DataArray: A new, higher-dimensional labelled array.

    Raises:
        TypeError: If no ``shape`` can be extracted from the first element in
            the input data ``d``
        ValueError: On bad argument values for ``dims``, ``shape``, ``coords``
            or ``combination_method``.
    """

    def prepare_item(
        d: "xarray.DataArray",
        *,
        midx: Sequence[int],
        shape: Sequence[int],
        astype: Union[str, type, np.dtype, None],
        name: str,
        dims: Sequence[str],
        generate_coords: Callable,
    ) -> Union["xarray.DataArray", None]:
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

        return xr.DataArray(
            item, name=name, dims=dims, coords=generate_coords(elem)
        )

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
        dims = tuple(f"inner_dim_{n:d}" for n, _ in enumerate(shape))

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
    coords = {
        n: (range(l) if isinstance(c, str) and c == "trivial" else c)
        for (n, c), l in zip(coords.items(), shape)
    }

    # Assemble info needed to bring individual array items into proper form
    item_name = d.name if d.name else "data"
    item_shape = tuple(1 for _ in d.shape) + tuple(shape)
    item_dims = d.dims + tuple(dims)
    item_coords = lambda e: dict(
        **{n: [c.item()] for n, c in e.coords.items()}, **coords
    )

    # The array that gathers all to-be-combined object arrays
    arrs = np.empty_like(d, dtype=object)
    arrs.fill(dict())  # are ignored in xr.merge

    # Transform each element to a labelled xr.DataArray that includes the outer
    # dimensions and coordinates; the latter is crucial for alignment.
    # Alongside, type coercion can be performed. For failed reshaping, the
    # element may be skipped.
    it = np.nditer(arrs.data, flags=("multi_index", "refs_ok"))
    for _ in it:
        item = prepare_item(
            d,
            midx=it.multi_index,
            shape=item_shape,
            astype=astype,
            name=item_name,
            dims=item_dims,
            generate_coords=item_coords,
        )

        if item is None:
            # Missing value; need to fall back to combination via merge
            combination_method = "merge"
            continue
        arrs[it.multi_index] = item

    # Now, combine
    if combination_method == "concat":
        return multi_concat(arrs, dims=d.dims, **combination_kwargs)

    elif combination_method == "merge":
        return merge(arrs, reduce_to_array=True, **combination_kwargs)

    raise ValueError(
        f"Invalid combination method '{combination_method}'! "
        "Choose from: 'concat', 'merge'."
    )


# .. Coordinate transformations ...............................................


def transform_coords(
    d: "xarray.DataArray",
    dim: Union[str, Sequence[str]],
    func: Callable,
    *,
    func_kwargs: dict = None,
) -> "xarray.DataArray":
    """Assigns new, transformed coordinates to a data array by applying a
    function on the existing coordinates.

    Uses :py:meth:`xarray.DataArray.assign_coords` to set the new coordinates,
    which returns a shallow copy of the given object.

    Args:
        d (xarray.DataArray): The array to transform the ``dim`` coordinates of
        dim (Union[str, Sequence[str]]): The name or names of the coordinate
            dimension(s) to apply ``func`` to.
        func (Callable): The callable to apply to ``d.coords[dim]``
        func_kwargs (dict, optional): Passed to the function invocation like
            ``func(d.coords[dim], **func_kwargs)``
    """
    if isinstance(dim, str):
        dim = [dim]

    kws = func_kwargs if func_kwargs else {}
    return d.assign_coords({_dim: func(d.coords[_dim], **kws) for _dim in dim})
