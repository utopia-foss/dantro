"""This module implements specialisations of the BaseDataContainer class."""

import logging
from itertools import product
from typing import Union, List, Dict, Tuple, Sequence

import numpy as np
import xarray as xr
import copy

from ..base import BaseDataContainer, ItemAccessMixin, CheckDataMixin
from ..mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin
from ..abc import AbstractDataProxy
from ..utils import Link

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class XrDataContainer(ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin,
                      CheckDataMixin, ItemAccessMixin, BaseDataContainer):
    """The XrDataContainer stores numerical xarray.DataArray data associated
    with dimensions, coordinates, and attributes.
    """
    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (xr.DataArray, np.ndarray,)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'raise'

    # Custom class variables for customizing XrDataContainer ..................
    # TODO Extend documentation
    # Define as class variable the name of the attribute that determines the
    # dimensions of the xarray.DataArray
    _XRC_DIMS_ATTR = 'dims'
    
    # Attributes prefixed with this string can be used to set names for
    # specific dimensions. The prefix should be followed by an integer-parsable
    # string, e.g. `dim_name__0` would be the dimension name for the 0th dim.
    _XRC_DIM_NAME_PREFIX = 'dim_name__'

    # Attributes prefixed with this string determine the coordinate values for
    # a specific dimension. The prefix should be followed by the _name_ of the
    # dimension, e.g. `coord__time`. The values are interpreted according to
    # the default coordinate mode or, if given, the coord_mode__* attribute
    _XRC_COORDS_ATTR_PREFIX = 'coords__'

    # The default mode by which coordinates are interpreted
    _XRC_COORDS_MODE_DEFAULT = 'list'

    # Prefix for the coordinate mode if a custom mode is to be used
    _XRC_COORDS_MODE_ATTR_PREFIX = 'coords_mode__'

    # Whether to inherit the other container attributes
    _XRC_INHERIT_CONTAINER_ATTRIBUTES = True

    # Whether to use strict attribute checking; throws errors if there are
    # container attributes available that match the prefix but don't match a
    # valid dimension name. Can be disabled for speed improvements
    _XRC_STRICT_ATTR_CHECKING = True

    # .........................................................................

    def __init__(self, *, name: str, data: Union[np.ndarray, xr.DataArray],
                 dims: Sequence[str]=None, coords: dict=None,
                 extract_metadata: bool=True, apply_metadata: bool=True,
                 **dc_kwargs):
        """Initialize a XrDataContainer and extract dimension and coordinate
        labels.
        
        Args:
            name (str): which name to give to the XrDataContainer
            data (Union[np.ndarray, xr.DataArray]): The data to store; anything
                that an xr.DataArray can take
            dims (Sequence[str], optional): The dimension names.
            coords (dict, optional): The coordinates. The keys of this dict
                have to correspond to the dimension names.
            extract_metadata (bool, optional): If True, missing ``dims`` or
                ``coords`` arguments are tried to be populated from the
                container attributes.
            apply_metadata (bool, optional): Whether to apply the extracted
                or passed ``dims`` and ``coords`` to the underlying data.
                This might not be desired in cases where the given ``data``
                already is a labelled ``xr.DataArray`` or where the data is a
                proxy and the labelling should be postponed.
            **dc_kwargs: passed to parent
        """
        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug("Received a list as `data` argument to %s '%s'. "
                      "Calling np.array on it ...", self.classname, name)
            data = np.array(data)

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)
        # NOTE The _data attribute is now set, but will be changed again below!

        # Set up cache attributes with given arguments
        self._dim_names = dims
        self._dim_to_coords_map = coords

        # Keep track of whether metadata was applied or not
        self._metadata_was_applied = False

        # If metadata is to be extracted from container attributes, do so now
        if extract_metadata:
            self._extract_metadata()

        # Apply the metadata, if set to do so (and not a proxy, which would not
        # allow it) ...
        if apply_metadata and not isinstance(self._data, AbstractDataProxy):
            self._apply_metadata()

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item.

        In this case, the dtype and sizes of the stored data is returned.
        Depending on whether metadata is available, the shape information is
        shown or the dimension names and the length of the dimensions are used.
        """
        return ("{dtype:}, {shape:}, {sup:}"
                "".format(dtype=self.dtype, shape=self._format_shape(),
                          sup=super()._format_info()))

    def _format_shape(self) -> str:
        """A __format__ helper for parsing shape information"""
        sizes = None
        if self._metadata_was_applied:
            # Can directly use the xarray information
            sizes = self.sizes.items()

        elif self._dim_names is not None:
            # Parse the metadata accordingly ...
            sizes = self._parse_sizes_from_metadata()

        if sizes is not None:
            return "({})".format(", ".join(["{}: {}".format(*kv)
                                            for kv in sizes]))
        return "shape {}".format(self.shape)

    def copy(self):
        """Return a new object with a deep copy of the data.
        
        Returns:
            XrDataContainer: A deep copy of this object.
        """
        log.debug("Creating copy of %s ...", self.logstr)

        return self.__class__(name=self.name + "_copy",
                              data=copy.deepcopy(self._data),
                              attrs=copy.deepcopy(self.attrs),
                              # Carry over cache attributes, needed for proxy
                              dims=self._dim_names,
                              coords=self._dim_to_coords_map,
                              # No need to extract or apply; if it is a proxy,
                              # the metadata is passed; if it is an xarray, it
                              # is already labelled and passed via `data`
                              extract_metadata=False,
                              apply_metadata=False)

    # Methods to extract and apply metadata ...................................

    def _extract_metadata(self):
        """Extracts metadata from the container attributes and stores them
        in the ``_dim_names`` and ``_dim_to_coords_map`` cache attributes.
        """
        log.debug("Extracting metadata for labelling %s ...", self.logstr)
        
        if self._dim_names is None:
            self._dim_names = self._extract_dim_names()

        if self._dim_to_coords_map is None:
            self._dim_to_coords_map = self._extract_coords()

    def _extract_dim_names(self) -> Tuple[Union[str, None]]:
        """Extract the dimension names from the container attributes.
        
        This can be done in two ways:
            1. A list of dimension names was specified in an attribute with the
               name specified in the _XRC_DIMS_ATTR class attribute
            2. One by one via container attributes that start with the string
               prefix defined in _XRC_DIM_NAME_PREFIX. This can be used if
               not all dimension names are available. Note that this will also
               _not_ be used if option 1 is used!
        
        Returns:
            Tuple[Union[str, None]]: The dimension names or None as placeholder
        
        Raises:
            TypeError: Attribute found at _XRC_DIMS_ATTR was a string, was not
                iterable or was not a sequence of strings
            ValueError: Length mismatch of attribute found at _XRC_DIMS_ATTR
                and the data.
        """
        def is_iterable(obj) -> bool:
            """Tries whether the given object is iterable."""
            try:
                (e for e in obj)
            except:
                return False
            return True
        
        # The dimension names sequence to populate. None's are placeholders and
        # are only resolved when applying the dimension names.
        dim_names = [None] * self.ndim

        # Distinguish two cases: All dimension names given directly as a list
        # or they are given separately, one by one
        if self._XRC_DIMS_ATTR in self.attrs:
            dims_attr = self.attrs[self._XRC_DIMS_ATTR]

            # Make sure it is an iterable of strings and of the right length
            if isinstance(dims_attr, str):
                raise TypeError("Attribute '{}' of {} needs to be a sequence "
                                "of strings, but not directly a string! "
                                "Got: {}"
                                "".format(self._XRC_DIMS_ATTR, self.logstr,
                                          repr(dims_attr)))
            
            elif not is_iterable(dims_attr):
                raise TypeError("Attribute '{}' of {} needs to be an "
                                "iterable, but was {} with value '{}'!"
                                "".format(self._XRC_DIMS_ATTR, self.logstr,
                                          type(dims_attr), dims_attr))

            elif len(dims_attr) != self.ndim:
                raise ValueError("Number of given dimension names does not "
                                 "match the rank of '{}'! Names given: {}. "
                                 "Rank: {}"
                                 "".format(self.logstr, dims_attr, self.ndim))

            # Data seems ok.
            # Create the sequence of dimension name, potentially needing to do
            # some further processing ...
            for dim_num, dim_name in enumerate(dims_attr):
                # Might be a numpy scalar or 1-sized array; resolve that
                if isinstance(dim_name, np.ndarray):
                    dim_name = dim_name.item()

                if not isinstance(dim_name, str):
                    raise TypeError("Dimension names for {} need to be "
                                    "strings, got {} with value '{}' for "
                                    "dimension {}!"
                                    "".format(self.logstr, type(dim_name),
                                              dim_name, dim_num))

                dim_names[dim_num] = dim_name

        # Have a populated list of dimension names now, all strings
        # There might be additionally specified dimension names that overwrite
        # the names given here...
        log.debug("Checking container attributes for the dimension name "
                  "prefix '%s' ...", self._XRC_DIM_NAME_PREFIX)

        # Go over all attributes and check for the prefix
        for attr_name, attr_val in self.attrs.items():
            if attr_name.startswith(self._XRC_DIM_NAME_PREFIX):
                # Extract the integer dimension number
                try:
                    dim_num = int(attr_name[len(self._XRC_DIM_NAME_PREFIX):])
                
                except ValueError as err:
                    raise ValueError("Could not extract the dimension "
                                     "number from the container attribute "
                                     "named '{}'! Take care that the part "
                                     "after the prefix ('{}') can be "
                                     "converted to an integer."
                                     "".format(attr_name,
                                               self._XRC_DIM_NAME_PREFIX)
                                     ) from err

                # Make sure its valid
                if dim_num < 0 or dim_num >= self.ndim:
                    raise ValueError("The dimension number {:d} extracted "
                                     "from attribute '{}' of {} exceeds "
                                     "the rank ({}) of the data!"
                                     "".format(dim_num, attr_name,
                                               self.logstr, self.ndim))

                # Make sure the attribute value is a string
                if isinstance(attr_val, np.ndarray):
                    # Need be single item and already decoded
                    attr_val = attr_val.item()

                if not isinstance(attr_val, str):
                    raise TypeError("Dimension names need be strings, but the "
                                    "attribute '{}' provided {} with value "
                                    "'{}'!"
                                    "".format(attr_name, type(attr_val),
                                              attr_val))

                # All good now. Write it to the dim name list
                dim_names[dim_num] = attr_val

        # Store the dimension names in the cache attribute
        return tuple(dim_names)
        
    
    def _extract_coords(self) -> dict:
        """Extract the coordinates to the dimensions using container attributes
        
        This is done by iterating over the dimension names of the DataArray
        and then looking for container attributes that are prefixed with
        _XRC_COORDS_ATTR_PREFIX and ending in the name of the dimension, e.g.
        attributes like 'coords__time'.

        The value of that attribute is then evaluated according to a so-called
        attribute 'mode'. By default, the mode set in _XRC_COORDS_MODE_DEFAULT
        is used, but it can be set explicitly for each dimension by setting
        a container attribute prefixed with _XRC_COORDS_MODE_ATTR_PREFIX.

        Available modes:
            * ``list``, ``values``: the explicit values to use for coordinates
            * ``arange``, ``range``, ``rangeexpr``, ``range_expr``: a range
                expression, i.e.: arguments passed to np.arange
            * ``start_and_step``: the start and step values of an integer range
                expression; the stop value is deduced by looking at the length
                of the corresponding dimension. This is then passed to the
                python range function as (start, stop, step)
            * ``linked``, ``from_path``: Load the coordinates from a linked
                data container within the tree (currently not implemented)

        The resulting number of coordinates for a dimension always need to
        match the length of that dimension.

        NOTE This expects that self._dim_names is already set.

        Returns:
            dict: The (dim_name -> coords) mapping
        
        Raises:
            ValueError: On invalid coordinates mode or (with strict attribute
                checking) on superfluous coordinate-setting attributes.
        """
        def extract_coords_from_attr(dim_name: str, dim_num: int):
            """Sets coordinates for a single dimension with the given name"""
            # Get the attribute
            cargs = self.attrs.get(self._XRC_COORDS_ATTR_PREFIX + dim_name)
            
            # Determine the mode to interpret the attribute values
            mode = self.attrs.get(self._XRC_COORDS_MODE_ATTR_PREFIX + dim_name,
                                  self._XRC_COORDS_MODE_DEFAULT)
            log.debug("Interpreting coordinate attribute for dimension '%s' "
                      "according to mode '%s' ...", dim_name, mode)

            # Distinguish by mode
            if mode in ['list', 'values']:
                # The attribute value are the coordinates
                return cargs
            
            elif mode in ['trivial', 'indices']:
                # Trivial coordinates
                return list(range(self.shape[dim_num]))

            elif mode in ['arange', 'range', 'rangeexpr', 'range_expr']:
                # Interpret values as a range expression
                return np.arange(*cargs)

            elif mode in ['start_and_step']:
                # Interpret as integer start and step of range expression
                # and use the length of the data dimension as number of steps
                start, step = cargs

                stop = start + (step * self.shape[dim_num])
                return list(range(int(start), int(stop), int(step)))

            elif mode in ['linked', 'from_path']:
                # Parse potential numpy array arguments to string
                if isinstance(cargs, np.ndarray):
                    cargs = cargs.item()

                # Problem: at this point, this container does not know its
                # full path within the data tree. Thus, coordinate resolution
                # has to be postponed until it is clear.
                # Instead, create a link object, which can forward to an actual
                # container once the coordinates are applied...
                return Link(anchor=self, rel_path=cargs)

            else:
                modes = ['list', 'values', 'trivial', 'indices',
                         'arange', 'range', 'rangeexpr', 'range_expr',
                         'start_and_step', 'linked', 'from_path']
                mode_attr_name = self._XRC_COORDS_MODE_ATTR_PREFIX + dim_name

                raise ValueError("Invalid mode '{}' to interpret coordinate "
                                 "attribute values! Check whether a mode "
                                 "attribute '{}' is set or the class variable "
                                 "for the default is not set correctly. "
                                 "Possible modes: {}"
                                 "".format(mode, mode_attr_name,
                                           ", ".join(modes)))
           
        # Dict to save the mapping from dim_names to coordinates
        coords_map = dict()

        # Get the coordinates for all labelled (!) dimensions
        for dim_num, dim_name in enumerate(self._dim_names):
            # If not labelled, not expecting a coordinate here
            if dim_name is None:
                continue

            # Otherwise, try to extract a coordinate and store it if found
            coords = extract_coords_from_attr(dim_name, dim_num)
            if coords is not None:
                coords_map[dim_name] = coords

        # Optionally, perform strict attribute checking
        if self._XRC_STRICT_ATTR_CHECKING:
            # Determine the prefixes that are to be checked
            prefixes = [self._XRC_COORDS_ATTR_PREFIX,
                        self._XRC_COORDS_MODE_ATTR_PREFIX]
            
            for attr_name, prefix in product(self.attrs.keys(), prefixes):
                # See whether there are matching attributes that were not
                # already extracted above
                if (    attr_name.startswith(prefix)
                    and attr_name[len(prefix):] not in coords_map):
                    print(coords_map)
                    dim_names_avail = [d for d in self._dim_names
                                       if d is not None]
                    raise ValueError("Got superfluous container attribute "
                                     "'{}' that does not match a labelled "
                                     "dimension in {}! "
                                     "Valid attribute prefixes: {}. "
                                     "Available dimension names: {}. "
                                     "Either remove the attribute or turn "
                                     "strict attribute checking off."
                                     "".format(attr_name, self.logstr,
                                               ", ".join(prefixes),
                                               ", ".join(dim_names_avail)))
            
        # All good. Return it.
        return coords_map

    def _inherit_attrs(self):
        """Carry over container attributes to the xr.DataArray attributes

        This does not include container attributes that are used for extracting
        metadata; it makes no sense to have them in the attributes of the
        already labelled xr.DataArray
        """
        def skip(attr_name: str) -> bool:
            return (   attr_name == self._XRC_DIMS_ATTR
                    or attr_name.startswith(self._XRC_DIM_NAME_PREFIX)
                    or attr_name.startswith(self._XRC_COORDS_ATTR_PREFIX)
                    or attr_name.startswith(self._XRC_COORDS_MODE_ATTR_PREFIX))

        for attr_name, attr_val in self.attrs.items():
            if not skip(attr_name):
                self.data.attrs[attr_name] = attr_val
    
    def _apply_metadata(self):
        """Applies the cached metadata to the underlying xr.DataArray"""
        # Make sure that data is an xarray
        if not isinstance(self.data, xr.DataArray):
            self._data = xr.DataArray(self.data)

        # Carry over the name
        self.data.name = self.name

        # Set the dimension names
        if self._dim_names:
            # Create a mapping from old to new names, then apply it
            new_names = {old: new
                         for old, new in zip(self.data.dims, self._dim_names)
                         if new is not None}
            
            log.debug("Renaming dimensions:  %s", new_names)
            self._data = self.data.rename(new_names)

        # Set the coordinates
        if self._dim_to_coords_map:
            log.debug("Associating coordinates:  %s", self._dim_to_coords_map)

            for dim_name, coords in self._dim_to_coords_map.items():
                # Need to handle links differently
                if isinstance(coords, Link):
                    # The target object is another DataContainer, which can not
                    # be used for association. Thus, just pass the raw data...
                    coords = coords.target_object.values  # np.ndarray now

                # Can associate now.
                try:
                    self.data.coords[dim_name] = coords
                
                except Exception as err:
                    raise ValueError("Could not associate coordinates {} for "
                                     "dimension '{}' due to a {}: {}."
                                     "".format(coords, dim_name,
                                               err.__class__.__name__, err)
                                     ) from err
        
        # Now write the rest of the attributes of the dataset to the xarray
        if self._XRC_INHERIT_CONTAINER_ATTRIBUTES:
            self._inherit_attrs()

        # Now set the flag that metadata was applied
        self._metadata_was_applied = True

    def _postprocess_proxy_resolution(self):
        """Only invoked from ``ProxySupportMixin``s, which have to be added to
        the class specifically. This function takes care to apply the
        potentially existing metadata after the proxy was resolved.
        """
        self._apply_metadata()

    def _parse_sizes_from_metadata(self) -> Sequence[Tuple[str, int]]:
        """Invoked from _format_shape when no metadata was applied but the
        dimension names are available. Should return data in the same form as
        xr.DataArray.sizes.items() does.
        """
        # Iterate over dimension names and shapes ...
        it = enumerate(zip(self._dim_names, self.shape))

        # ... and use the name from the metadata unless the name was None
        # which is a placeholder for "don't rename this dimension", in which
        # case it should be named via default names
        return tuple([(n if n else "dim_{}".format(i), l) for i, (n, l) in it])
