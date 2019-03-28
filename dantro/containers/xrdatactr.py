"""This module implements specialisations of the BaseDataContainer class."""

import logging
from typing import Union, List, Dict

import numpy as np
import xarray as xr

from ..base import BaseDataContainer, ItemAccessMixin, CheckDataMixin
from ..mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin
from ..abc import AbstractDataProxy

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
                 **dc_kwargs):
        """Initialize a XrDataContainer and extract dimension and coordinate
        labels.
        
        Args:
            name (str): which name to give to the XrDataContainer
            data (Union[np.ndarray, xr.DataArray]): The data to store; anything
                that an xr.DataArray can take
            **dc_kwargs: passed to parent parent (BaseDataContainer.__init__)
        """
        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug("Received a list as `data` argument to %s '%s'. "
                      "Calling np.array on it ...", self.classname, name)
            data = np.array(data)

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)
        # NOTE The _data attribute is now set, but will be changed again below!

        # Now, extract dimension names, coordinates, and attributes from the
        # container attributes and a mapping from coordinate number (as int)
        # to dimenasion name
        log.debug("Extracting metadata for labelling %s ...", self.logstr)
        self._dim_name_map, int_to_dim_map = self._extract_dim_names()
        self._dim_to_coords_map = self._extract_coords(int_to_dim_map)

        # Check if the passed data is a proxy
        if not isinstance(self._data, AbstractDataProxy):
            # Make sure that the _data member is an xarray
            self._data = xr.DataArray(self._data)
            # Apply the metadata to the xarray
            self._apply_metadata(self._dim_name_map, self._dim_to_coords_map)
            


    # Helper Methods ..........................................................

    def _extract_dim_names(self) -> None:
        """Extract the dimension names from the container attributes. This can
        be done in two ways:

            1. A list of dimension names specified in an attribute with the
               name specified in _XRC_DIMS_ATTR class attribute
            2. One by one via container attributes that start with the string
               prefix defined in _XRC_DIM_NAME_PREFIX. This can be used if
               not all dimension names are available. Note that this will also
               _not_ be used if option 1 is used!
        """
        def is_iterable(obj) -> bool:
            try:
                _ = (e for e in obj)
            except:
                return False
            return True

        # The dimensions name mapping (old_name -> new_name)
        dim_name_map = dict()
        int_to_name_map = dict()
        
        # Distinguish two cases: All dimension names given directly as a list
        # or they are given separately, one by one
        if self._XRC_DIMS_ATTR in self.attrs:
            dims = self.attrs[self._XRC_DIMS_ATTR]

            # Make sure it is an iterable of strings and of the right length
            if not is_iterable(dims):
                raise TypeError("Attribute '{}' of {} needs to be an "
                                "iterable, but was {} with value '{}'!"
                                "".format(self._XRC_DIMS_ATTR, self.logstr,
                                          type(dims), dims))

            elif not all([isinstance(d, str) for d in dims]):
                raise ValueError("Attribute '{}' of {} needs to be an "
                                 "iterable of strings, but was of types {}!"
                                 "".format(self._XRC_DIMS_ATTR, self.logstr,
                                           [type(d) for d in dims]))

            elif len(dims) != self.ndim:
                raise ValueError("Number of given dimension names does not "
                                 "match the rank of '{}'! Names given: {}. "
                                 "Rank: {}"
                                 "".format(self.logstr, dims, self.ndim))

            # ... and populate dimension name mapping
            for dim_num, dim_name in enumerate(dims):
                dim_name_map["dim_{}".format(dim_num)] = dim_name
                int_to_name_map[dim_num]= dim_name
        
        else:
            log.debug("Checking container attributes for the dimension name "
                      "prefix '%s' ...", self._XRC_DIM_NAME_PREFIX)

            # Need a mapping from the dimension number to the specified name
            int_to_name_map = dict()

            for attr_name, attr_val in self.attrs.items():
                if attr_name.startswith(self._XRC_DIM_NAME_PREFIX):
                    # Extract the integer dimension number
                    dim_num_str = attr_name.split(self._XRC_DIM_NAME_PREFIX)[1]
                    try:
                        dim_num = int(dim_num_str)
                    
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
                                         "from attribute '{}' exceeds the "
                                         "rank ({})!"
                                         "".format(dim_num, attr_name,
                                                   self.ndim))

                    # All good now. Add it to the dimension number to
                    # name mapping
                    int_to_name_map[dim_num] = attr_val

            # make sure that every dimension of the dataset is covered in the 
            # dim_name_map not only the ones specified in self.attrs (for use
            # by _extract_coords)
            for d_num in range(self.ndim):
                if d_num in int_to_name_map.keys():
                    dim_name_map["dim_{:d}".format(d_num)] = int_to_name_map[d_num]
                else:
                    dim_name_map["dim_{:d}".format(d_num)] = "dim_{:d}".format(d_num)
                    int_to_name_map[d_num] = "dim_{:d}".format(d_num)
        
        # Return the map
        return dim_name_map, int_to_name_map
        
    
    def _extract_coords(self, int_to_dim_names: dict) -> None:
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

        Arguments:
            dim_names (dict): A dictionary of the dimension names

        """
        def extract_coords_from_attr(dim_name: str, dim_num: int):
            """Sets coordinates for a single dimension with the given name"""
            # Check if there is an attribute available to use
            coords = self.attrs.get(self._XRC_COORDS_ATTR_PREFIX + dim_name,
                                    None)
            if not coords:
                return
            
            # Determine the mode to interpret the attribute values
            mode = self.attrs.get(self._XRC_COORDS_MODE_ATTR_PREFIX + dim_name,
                                  self._XRC_COORDS_MODE_DEFAULT)
            log.debug("Interpreting coordinate attribute for dimension '%s' "
                      "according to mode '%s' ...", dim_name, mode)

            # Distinguish by mode
            if mode in ['list', 'values']:
                # Nothing to do
                pass

            elif mode in ['arange', 'range', 'rangeexpr', 'range_expr']:
                # Interpret values as a range expression
                coords = np.arange(*coords)

            elif mode in ['start_and_step']:
                # Interpret as integer start and step of range expression
                # and use the length of the data dimension as number of steps
                start, step = coords
                stop = start + (step * self.shape[dim_num])
                coords = list(range(start, stop, step))

            elif mode in ['linked', 'from_path']:
                raise NotImplementedError("Linked datasets for coordinates "
                                          "are not yet supported!")

            else:
                modes = ['list', 'values', 'arange', 'range', 'rangeexpr',
                         'range_expr', 'start_and_step', 'linked', 'from_path']
                mode_attr_name = self._XRC_COORDS_MODE_ATTR_PREFIX + dim_name

                raise ValueError("Invalid mode '{}' to interpret coordinate "
                                 "attribute values! Check whether a mode "
                                 "attribute '{}' is set or the class variable "
                                 "for the default is not set correctly. "
                                 "Possible modes: {}"
                                 "".format(mode, mode_attr_name,
                                           ", ".join(modes)))
            return coords
           
        # Dict to save the mapping from dim_names to coordinates
        dim_to_coord = {}
        # Get the  coordinates for all available dimensions
        for dim_num, dim_name in int_to_dim_names.items():
            coords = extract_coords_from_attr(dim_name, dim_num)
            if coords is not None:
                dim_to_coord[dim_name] = coords

        # Can return now, if no strict attribute checking should take place
        if not self._XRC_STRICT_ATTR_CHECKING:
            return dim_to_coord

        # Determine the prefixes that are to be checked
        prefixes = [self._XRC_COORDS_ATTR_PREFIX,
                    self._XRC_COORDS_MODE_ATTR_PREFIX]
        
        for attr_name in self.attrs:
            for prefix in prefixes:
                if (    attr_name.startswith(prefix)
                    and attr_name.split(prefix)[1] 
                        not in int_to_dim_names.values()):
                    raise ValueError("Got superfluous container attribute "
                                     "'{}' that does not match a dimension "
                                     "name! Available dimensions: {}"
                                     "".format(attr_name, 
                                               int_to_dim_names.values()))
        return dim_to_coord

    def _inherit_attrs(self) -> None:
        """Carry over all remainig container attributes to the xr.DataArray"""
        def skip(attr_name: str) -> bool:
            return (   attr_name == self._XRC_DIMS_ATTR
                    or attr_name.startswith(self._XRC_DIM_NAME_PREFIX)
                    or attr_name.startswith(self._XRC_COORDS_ATTR_PREFIX)
                    or attr_name.startswith(self._XRC_COORDS_MODE_ATTR_PREFIX))

        for attr_name, attr_val in self.attrs.items():
            if skip(attr_name):
                continue

            self.data.attrs[attr_name] = attr_val
    
    def _apply_metadata(self, dim_to_name_map: dict, 
                        dim_to_coord_map: dict) -> None:
        """writes the metadata to the xarray
        
        Arguments:
            dim_to_name_map {dict} -- A map from the default dimension names to 
                                      the ones specified in the attrs
            dim_to_coord_map {dict} -- A map from dimension names to the
                                       coordinates belonging to this dimension
        
        Returns:
            None -- 
        """
        # Add some information to the data array
        self._data.name = self.path
        # Set the dimension names
        log.debug("Renaming dimensions using map: %s", dim_to_name_map)
        self._data = self._data.rename(dim_to_name_map)

        log.debug("Renaming dimensions using map: %s", dim_to_coord_map)
        # Set the coordinates for every dimension
        if dim_to_coord_map:
            for dim_name, coords in dim_to_coord_map.items():
                try:
                    self._data.coords[dim_name] = coords
                
                except Exception as err:
                    raise ValueError("Could not associate coordinates for "
                                        "dimension '{}' due to {}: {}"
                                        "".format(dim_name,
                                                err.__class__.__name__, err)
                                        ) from err
            # Now write the rest of the attributes of the dataset to the xarray
            if self._XRC_INHERIT_CONTAINER_ATTRIBUTES:
                self._inherit_attrs()

    def _postprocess_proxy_resolution(self):
        """
            Only used with proxies
        """
        # make sure that the data is cast into an xrDataArray
        self._data = xr.DataArray(self._data)
        # apply the metadata
        self._apply_metadata(self._dim_name_map, self._dim_to_coords_map)

