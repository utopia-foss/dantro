"""This module provides coordinate parsing capabilities."""

import logging
from itertools import product
from typing import (
    Dict,
    Hashable,
    Iterable,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from ..abc import AbstractDataContainer
from ..tools import is_iterable, try_conversion
from .link import Link, StrongLink

log = logging.getLogger(__name__)

# Some type definitions .......................................................
TDims = Tuple[str]
"""A dimension sequence type"""

TCoord = TypeVar("TCoord", int, float, str, Hashable)
"""A single coordinate value type"""

TCoords = Sequence[TCoord]
"""A sequence of coordinate values"""

TCoordsDict = Dict[str, TCoords]
"""Several coordinates, bundled into a map of dimension name to coordinates"""


# Dimension name extraction ---------------------------------------------------


def extract_dim_names(
    attrs: dict, *, ndim: int, attr_name: str, attr_prefix: str
) -> TDims:
    """Extract dimension names from the given attributes.

    This can be done in two ways:
        1. A list of dimension names was specified in an attribute with the
           name specified by the ``attr_name`` argument
        2. One by one via attributes that start with the string prefix defined
           in ``attr_prefix``. This can be used if not all dimension names are
           available. Note that this will also *not* be used if option 1 is
           used!

    Args:
        attrs (dict): The dict-like object to read attributes from
        obj_logstr (str): A string that is given as context in log and error
            messages, ideally describing the object these attributes belong to
        ndim (int): The expected rank of the dimension names
        attr_name (str): The key to look for in ``attrs`` that would give a
            sequence of the dimension names.
        attr_prefix (str): The prefix to look for in the keys of the ``attrs``
            that would specify the name of a single dimension.

    Returns:
        Tuple[Union[str, None]]: The dimension names or None as placeholder

    Raises:
        TypeError: Attribute found at ``attr_name`` was a string, was not
            iterable or was not a sequence of strings
        ValueError: Length mismatch of attribute found at ``attr_name``
            and the data.

    """
    # The dimension names sequence to populate. None's are placeholders which
    # denote that no information was found for this dimension.
    dim_names = [None] * ndim

    # Distinguish two cases: All dimension names given directly as a list
    # or they are given separately, one by one
    if attr_name in attrs:
        dims_attr = attrs[attr_name]

        # Make sure it is an iterable of strings and of the right length
        if isinstance(dims_attr, str):
            raise TypeError(
                f"Attribute '{attr_name}' needs to be a sequence of "
                f"strings, but not directly a string! Got: {repr(dims_attr)}"
            )

        elif not is_iterable(dims_attr):
            raise TypeError(
                f"Attribute '{attr_name}' needs to be an iterable, but was "
                f"{type(dims_attr)} with value '{dims_attr}'!"
            )

        elif len(dims_attr) != ndim:
            raise ValueError(
                "Number of given dimension names does not match "
                f"the given rank of {ndim}! Names given: {dims_attr}"
            )

        # Data seems ok.
        # Create the sequence of dimension name, potentially needing to do
        # some further processing ...
        for dim_num, dim_name in enumerate(dims_attr):
            # Might be a numpy scalar or 1-sized array; resolve that
            if isinstance(dim_name, np.ndarray):
                dim_name = dim_name.item()

            if not isinstance(dim_name, str):
                raise TypeError(
                    "Dimension names need to be strings, got "
                    f"{type(dim_name)} with value '{dim_name}' for "
                    f"dimension {dim_num}!"
                )

            dim_names[dim_num] = dim_name

    # Have a populated list of dimension names now, all strings.
    # There might be additionally specified dimension names that overwrite
    # the names given here...

    # Go over all attributes and check for the prefix
    for attr_name, attr_val in attrs.items():
        if attr_name.startswith(attr_prefix):
            # Extract the integer dimension number
            try:
                dim_num = int(attr_name[len(attr_prefix) :])

            except ValueError as err:
                raise ValueError(
                    "Could not extract the dimension number from "
                    f"the container/group attribute named '{attr_name}'! "
                    "Take care that the part after the prefix "
                    f"('{attr_prefix}') can be converted to an integer."
                ) from err

            # Make sure its valid
            if dim_num < 0 or dim_num >= ndim:
                raise ValueError(
                    f"The dimension number {dim_num:d} extracted from "
                    f"attribute '{attr_name}' exceeds the given rank {ndim}!"
                )

            # Make sure the attribute value is a string
            if isinstance(attr_val, np.ndarray):
                attr_val = attr_val.item()  # ... assuming already decoded.

            if not isinstance(attr_val, str):
                raise TypeError(
                    "Dimension names need be strings, but the "
                    f"attribute '{attr_name}' provided {type(attr_val)} "
                    f"with value '{attr_val}'!"
                )

            # All good now. Write it to the dim name list
            dim_names[dim_num] = attr_val

    # Done.
    return tuple(dim_names)


# Coordinate extraction -------------------------------------------------------
# Low level extractor functions ...............................................
# These should all take the coordinate arguments as the first argument and then
# accept arbitrary keyword-only arguments, of which some may be used to
# determine the coordinate values


def _coords_start_and_step(
    cargs, *, data_shape: tuple, dim_num: int, **__
) -> Iterable[int]:
    """Interpret as integer start and step of range expression and use the
    length of the data dimension as number of steps
    """
    start, step = cargs
    stop = start + (step * data_shape[dim_num])
    return range(int(start), int(stop), int(step))


def _coords_trivial(
    _, *, data_shape: tuple, dim_num: int, **__
) -> Iterable[int]:
    """Returns trivial coordinates for the given dimension by creating a
    range iterator from the selected data shape.
    """
    return range(data_shape[dim_num])


def _coords_scalar(coord, **__) -> List[TCoord]:
    """Returns a single, scalar coordinate, i.e.: list of length 1"""
    if isinstance(coord, np.ndarray):
        coord = coord.item()

    # If it is not iterable, it is a scalar now. Expected return value is a
    # list, though, so convert it.
    if not is_iterable(coord):
        coord = [coord]

    if len(coord) != 1:
        raise ValueError(f"Expected scalar coordinate, but got: {coord}!")

    # Make sure it is a list, not a tuple
    return list(coord)


def _coords_linked(cargs, *, link_anchor_obj, **__) -> Link:
    """Creates a Link object which is to be used for coordinates"""
    # Need to parse potential numpy array arguments to string
    if isinstance(cargs, np.ndarray):
        cargs = cargs.item()

    # Problem: at this point, the container to be linked to might not
    # know its full path within the data tree. Thus, coordinate
    # resolution has to be postponed until it is clear.
    # For that reason, create a link object, which can forward to an
    # actual container once the coordinates are applied...
    return StrongLink(anchor=link_anchor_obj, rel_path=cargs)


# Map of extractors ...........................................................
# fmt: off
COORD_EXTRACTORS = {
    "values":           lambda cargs, **__: cargs,
    "range":            lambda cargs, **__: range(*cargs),
    "arange":           lambda cargs, **__: np.arange(*cargs),
    "linspace":         lambda cargs, **__: np.linspace(*cargs),
    "logspace":         lambda cargs, **__: np.logspace(*cargs),
    "start_and_step":   _coords_start_and_step,
    "trivial":          _coords_trivial,
    "scalar":           _coords_scalar,
    "linked":           _coords_linked,
}
# fmt: on

# Actual (high-level) extraction functions ....................................


def extract_coords_from_attrs(
    obj: Union[AbstractDataContainer, np.ndarray],
    *,
    dims: Tuple[Union[str, None]],
    strict: bool,
    coords_attr_prefix: str,
    default_mode: str,
    mode_attr_prefix: str = None,
    attrs: dict = None,
) -> TCoordsDict:
    """Extract coordinates from the given object's attributes.

    This is done by iterating over the given ``dims`` and then looking
    for attributes that are prefixed with ``coords_attr_prefix`` and ending in
    the name of the dimension, e.g. attributes like ``coords__time``.

    The value of that attribute is then evaluated according to a so-called
    attribute ``mode``. By default, the mode set by ``default_mode`` is used,
    but it can be set explicitly for each dimension by the ``mode_attr_prefix``
    parameter.

    The resulting number of coordinates for a dimension always need to match
    the length of that dimension. However, the corresponding error can only be
    raised once this information is applied.

    Args:
        obj (Union[AbstractDataContainer, numpy.ndarray]): The object to
            retrieve the attributes from (via the ``attrs`` attribute). If the
            ``attrs`` *argument* is given, will use those instead.
            It is furthermore expected that this object specifies the shape of
            the numerical data the coordinates are to be generated for by
            providing a ``shape`` property. This is possible with
            :py:class:`~dantro.containers.numeric.NumpyDataContainer` and
            derived classes.
        dims (Tuple[Union[str, None]]): Sequence of dimension names; this
            may also contain None's, which are ignored for coordinates.
        strict (bool): Whether to use strict checking, where no additional
            coordinate-specifying attributes are allowed.
        coords_attr_prefix (str): The attribute name prefix for coordinate
            specifications
        default_mode (str): The default coordinate extraction mode. Available
            modes:

            * ``values``: the explicit values (iterable) to use for coordinates
            * ``range``: range arguments
            * ``arange``: np.arange arguments
            * ``linspace``: np.linspace arguments
            * ``logspace``: np.logspace arguments
            * ``trivial``: The trivial indices. This does not require a value
              for the coordinate argument.
            * ``scalar``: makes sure only a single coordinate is provided
            * ``start_and_step``: the start and step values of an integer range
              expression; the stop value is deduced by looking at the length of
              the corresponding dimension. This is then passed to the python
              range function as (start, stop, step)
            * ``linked``: Load the coordinates from a linked object within the
              tree; this works only if ``link_anchor_obj`` is part of a data
              tree at the point of coordinate resolution!

        mode_attr_prefix (str, optional): The attribute name prefix that can
            be used to specify a non-default extraction mode. If not given, the
            default mode will be used.
        attrs (dict, optional): If given, these attributes will be used instead
            of attempting to extract attributes from ``obj``.

    Returns:
        TCoordsDict: The ``(dim_name -> coords)`` mapping

    Raises:
        ValueError: On invalid coordinates mode or (with strict attribute
            checking) on superfluous coordinate-setting attributes.

    """

    def get_coord(attrs: dict, dim_name: str, dim_num: int):
        """Determines coordinate values for a single dimension."""
        # The argument the coordinate will be determined from; may be None.
        cargs = attrs.get(coords_attr_prefix + dim_name)

        # Determine the mode to interpret the attribute values
        mode = default_mode
        if mode_attr_prefix:
            mode = attrs.get(mode_attr_prefix + dim_name, default_mode)

        # Might have to process the mode, e.g. because it is an array-type
        if isinstance(mode, np.ndarray):
            mode = mode.item()

        # Check if the mode is available
        if mode not in COORD_EXTRACTORS:
            _mode_attr_name = mode_attr_prefix + dim_name
            _extraction_modes = ", ".join(COORD_EXTRACTORS.keys())
            raise ValueError(
                f"Invalid mode '{mode}' to interpret coordinate "
                "attribute values! Check whether a mode "
                f"attribute '{_mode_attr_name}' is set. "
                f"Available modes: {_extraction_modes}"
            )

        # Invoke the method, passing on available arguments
        try:
            return COORD_EXTRACTORS[mode](
                cargs,
                dim_num=dim_num,
                data_shape=obj.shape,
                link_anchor_obj=obj,
            )

        except Exception as exc:
            raise type(exc)(
                "Failed extracting coordinates for dimension "
                f"'{dim_name}' via extraction mode '{mode}'! {exc}"
            ) from exc

    # If necessary, get the attributes from the given object
    if not attrs:
        attrs = obj.attrs

    # The to-be-populated mapping from dimension names to coordinates.
    coords_map = dict()

    # Get the coordinates for all labelled (!) dimensions; if not labelled, no
    # coordinate is expected.
    for dim_num, dim_name in enumerate(dims):
        if dim_name is None:
            continue

        # Is labelled; try to extract a coordinate and store it if found
        coords = get_coord(attrs, dim_name, dim_num)
        if coords is not None:
            coords_map[dim_name] = coords

    # Done.
    # Without strict attribute checking, can already return here
    if not strict:
        return coords_map

    # Need to do strict attribute checking ...
    # Determine the prefixes that are to be checked
    prefixes = [coords_attr_prefix]
    if mode_attr_prefix:
        prefixes.append(mode_attr_prefix)

    # Check all attribute name and prefix combinations
    for attr_name, prefix in product(attrs.keys(), prefixes):
        # See whether there are matching attributes that were not already
        # extracted above, i.e. appear as keys in the map
        if (
            attr_name.startswith(prefix)
            and attr_name[len(prefix) :] not in coords_map
        ):

            _dims_avail = ", ".join([d for d in dims if d is not None])
            _prefixes = ", ".join(prefixes)

            raise ValueError(
                f"Got superfluous attribute '{attr_name}' that "
                "does not match any of the available names of "
                f"labelled dimensions: {_dims_avail}. Valid "
                f"attribute prefixes: {_prefixes}. Either remove "
                "the attribute or turn strict attribute checking "
                "off."
            )

    # All good. Return the coordinate map.
    return coords_map


def extract_coords_from_name(
    obj: AbstractDataContainer,
    *,
    dims: TDims,
    separator: str,
    attempt_conversion: bool = True,
) -> TCoordsDict:
    """Given a container or group, extract the coordinates from its name.

    The name of the object may be a ``separator``-separated string, where each
    segment contains the coordinate value for one dimension.

    This function assumes that the coordinates for each dimension are scalar.
    Thus, the values of the returned dict are sequences of length 1.

    Args:
        obj (AbstractDataContainer): The object to get the coordinates of by
            inspecting its name.
        dims (TDims): The dimension names corresponding to the coordinates that
            are expected to be found in the object's name.
        separator (str): The separtor to apply on the name.
        attempt_conversion (bool, optional): Whether to attempt conversion of
            the string value to a numerical type.

    Returns:
        TCoordsDict: The coordinate dict, i.e. a mapping from the external
            dimension names to the coordinate values. In this case, there can
            only a single value for each dimension!

    Raises:
        ValueError: Raised upon failure to extract external coordinates:
            On ``ext_dims`` evaluating to False, f coordinates were missing for
            any of the external dimensions, if the number of coordinates
            extracted from the name did not match the number of external
            dimensions, if any of the strings extracted from the object's name
            were empty.
    """
    # Split the string and make some basic checks.
    coords = obj.name.split(separator)

    if len(coords) != len(dims):
        raise ValueError(
            "Number of coordinates extracted from the name of "
            f"{obj.logstr} does not match the number of expected "
            f"dimensions! Parsed coordinates: {coords}. "
            f"Dimensions: {dims}"
        )

    if not all(coords):
        raise ValueError(
            "One or more of the coordinates extracted from the "
            f"name of {obj.logstr} were empty! Got: {coords}"
        )

    # Build the dict, attempting conversion of the objects.
    coords = {
        dim_name: [try_conversion(c_val) if attempt_conversion else c_val]
        for dim_name, c_val in zip(dims, coords)
    }

    return coords


def extract_coords_from_data(
    obj: AbstractDataContainer, *, dims: TDims
) -> TCoordsDict:
    """Tries to extract the coordinates from the data of the given container
    or group. For that purpose, the ``obj`` needs to support the ``coords``
    property.

    Args:
        obj (AbstractDataContainer): The object that holds the data from which
            the coordinates are to be extracted.
        dims (TDims): The sequence of dimension names for which the coordinates
            are to be extracted.
    """
    return {dim_name: list(obj.coords[dim_name].values) for dim_name in dims}


# .............................................................................


def extract_coords(
    obj: AbstractDataContainer,
    *,
    mode: str,
    dims: TDims,
    use_cache: bool = False,
    cache_prefix: str = "__coords_cache_",
    **kwargs,
) -> TCoordsDict:
    """Wrapper around the more specific coordinate extraction functions.

    .. note::

        This function does not support the extraction of non-dimension
        coordinates.

    Args:
        obj (AbstractDataContainer): The object from which to extract the
            coordinates.
        mode (str): Which mode to use for extraction. Can be:

            * ``name``:   Use the name of the object
            * ``attrs``:  Use the attributes of the object
            * ``data``:   Use the data of the object

        dims (TDims): The dimensions for which the attributes are to be
            extracted. All dimension names given here are expected to be found.
        use_cache (bool, optional): Whether to use the object's attributes to
            write an extracted value to the cache and read it, if available.
        cache_prefix (str, optional): The prefix to use for writing the cache
            entries to the object attributes. Will suffix this with ``dims``
            and ``coords`` and store the respective data there.
        **kwargs: Passed on to the actual coordinates extraction method.

    Raises:
        NotImplementedError: If ``use_cache`` is set
    """
    EXTRACTORS = dict(
        name=extract_coords_from_name,
        attrs=extract_coords_from_attrs,
        data=extract_coords_from_data,
    )

    if use_cache:
        # TODO Read from cache
        raise NotImplementedError("use_cache")

    # Get the extraction function
    if mode not in EXTRACTORS:
        _extraction_modes = ", ".join(EXTRACTORS.keys())
        raise ValueError(
            f"Invalid extraction mode '{mode}'! Valid modes: "
            f"{_extraction_modes}"
        )

    extraction_func = EXTRACTORS[mode]
    try:
        coords = extraction_func(obj, dims=dims, **kwargs)

    except Exception as exc:
        raise type(exc)(
            f"Failed extracting coordinates of {obj} using "
            f"mode '{mode}': {exc}"
        ) from exc

    # TODO Write to cache

    return coords
