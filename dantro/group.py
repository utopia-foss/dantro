"""In this module, BaseDataContainer specialisations are implemented."""

import copy
import logging
import collections
from typing import Union, List, Dict

import numpy as np
import numpy.ma
import xarray as xr

from paramspace import ParamSpace

from dantro.base import BaseDataGroup, PATH_JOIN_CHAR
from dantro.container import NumpyDataContainer

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class OrderedDataGroup(BaseDataGroup, collections.abc.MutableMapping):
    """The OrderedDataGroup class manages groups of data containers, preserving
    the order in which they were added to this group.

    It uses an OrderedDict to associate containers with this group.
    """
    
    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("OrderedDataGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers,
                         StorageCls=collections.OrderedDict, **kwargs)

        # Done.
        log.debug("OrderedDataGroup.__init__ finished.")

# -----------------------------------------------------------------------------
# ParamSpaceGroup and associated classes

class ParamSpaceStateGroup(OrderedDataGroup):
    """A ParamSpaceStateGroup is meant as the member of the ParamSpaceGroup."""

    # The child class should not be of the same type as this class.
    _NEW_GROUP_CLS = OrderedDataGroup
    
    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a ParamSpaceStateGroup from the list of given containers.
        
        Args:
            name (str): The name of this group, which needs to be convertible
                to an integer.
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("ParamSpaceStateGroup.__init__ called.")

        # Assert that the name is valid, i.e. convertible to an integer
        try:
            int(name)
        except ValueError as err:
            raise ValueError("Only names that are representible as integers "
                             "are possible for {}!".format(self.classname)
                             ) from err

        # ... and not negative
        if int(name) < 0:
            raise ValueError("Name for {} needs to be positive when converted "
                             "to integer, was: {}".format(self.classname,name))

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # Done.
        log.debug("ParamSpaceStateGroup.__init__ finished.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ParamSpaceGroup(OrderedDataGroup):
    """The ParamSpaceGroup is associated with a ParamSpace object and the
    loaded results of an iteration over this parameter space.

    Thus, the groups that are stored in the ParamSpaceGroup need all relate to
    a state of the parameter space, identified by a zero-padded string name.
    In fact, this group allows no other kinds of groups stored inside.

    To make access to a specific state easier, it allows accessing a state by
    its state number as integer.
    """

    # Class variables that define some of the behaviour
    # Define which .attrs entry to return from the `pspace` property
    _PSPGRP_PSPACE_ATTR_NAME = 'pspace'

    # Define the class to use for the direct members of this group
    _NEW_GROUP_CLS = ParamSpaceStateGroup

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (ParamSpaceStateGroup,)

    # .........................................................................

    def __init__(self, *, name: str, pspace: ParamSpace=None, containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group.
            pspace (ParamSpace, optional): Can already pass a ParamSpace object
                here.
            containers (list, optional): A list of containers to add, which
                need to be ParamSpaceStateGroup objects.
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("ParamSpaceGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # If given, associate the parameter space object
        if pspace is not None:
            self.pspace = pspace

        # Private attribute needed for state access via strings
        self._num_digs = 0

        # Done.
        log.debug("ParamSpaceGroup.__init__ finished.")


    # Properties ..............................................................

    @property
    def pspace(self) -> Union[ParamSpace, None]:
        """Reads the entry named _PSPGRP_PSPACE_ATTR_NAME in .attrs and
        returns a ParamSpace object, if available there.
        
        Returns:
            Union[ParamSpace, None]: The associated parameter space, or None,
                if there is none associated yet.
        """
        return self.attrs.get(self._PSPGRP_PSPACE_ATTR_NAME, None)

    @pspace.setter
    def pspace(self, val: ParamSpace):
        """If not already set, sets the entry in the attributes that is
        accessed by the .pspace property
        """
        if self.pspace is not None:
            raise RuntimeError("The attribute for the parameter space of this "
                               "{} was already set, cannot set it again!"
                               "".format(self.logstr))

        
        elif not isinstance(val, ParamSpace):
            raise TypeError("The attribute for the parameter space of {} "
                            "needs to be a ParamSpace-derived object, was {}!"
                            "".format(self.logstr, type(val)))

        # Checked it, now set it
        self.attrs[self._PSPGRP_PSPACE_ATTR_NAME] = val
        log.debug("Associated %s with %s", val, self.logstr)


    # Item access .............................................................

    def _check_cont(self, cont: ParamSpaceStateGroup) -> None:
        """Asserts that only containers with valid names are added.
        
        Args:
            cont (ParamSpaceStateGroup): The state group to add
        
        Returns:
            None
        
        Raises:
            ValueError: For a state name that has an invalid length
        """
        # Check if this is the first container to be added. This also
        # determines the number of possible digits the state number can have
        if not len(self) or self._num_digs == 0:
            self._num_digs = len(cont.name)
            log.debug("Set _num_digs to %d.", self._num_digs)

        # Check the name against the already set number of digits
        elif len(cont.name) != self._num_digs:
            raise ValueError("Containers added to {} need names that have a "
                             "string representation of same length: for this "
                             "instance, a zero-padded integer of width {}. "
                             "Got: {}".format(self.logstr, self._num_digs,
                                              cont.name))

        # TODO could also check against .pspace.max_state_no ...
        # Everything ok. No return value needed.

    def __getitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow integer item access"""
        if isinstance(key, int):
            # Generate a padded string to access the state
            key = self._padded_id_from_int(key)

        # Use the parent method to return the value
        return super().__getitem__(key)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Adjusts the parent method to allow checking for integers"""
        if isinstance(key, int):
            # Generate a string from the given integer
            key = self._padded_id_from_int(key)

        return super().__contains__(key)

    def _padded_id_from_int(self, state_no: int) -> str:
        """This generates a zero-padded state number string from the given int.

        Note that the ParamSpaceGroup only allows its members to have keys of
        the same length.
        """
        # Check the requested state number to improve error messages
        if state_no < 0:
            raise KeyError("State numbers cannot be negative! {} is negative."
                           "".format(state_no))

        elif state_no > 10**self._num_digs - 1:
            raise KeyError("State numbers for {} cannot be larger than {}! "
                           "Requested state number: {}"
                           "".format(self.logstr,
                                     10**self._num_digs - 1, state_no))

        # Everything ok. Generate the zero-padded string.
        return "{sno:0{digs:d}d}".format(sno=state_no, digs=self._num_digs)


    # Data access .............................................................
    # TODO a select_multiple function that allows selecting multiple data
    #      fields in one iteration over parameter space ...

    def select(self, *, field: Union[str, List[str]]=None, fields: Dict[str, List[str]]=None, subspace: dict=None, rename_dims: dict=None) -> xr.Dataset:
        """Selects a multi-dimensional slab of this ParamSpaceGroup and returns
        it as an xarray.Dataset with labelled dimensions.
        
        Args:
            field (Union[str, List[str]]): The field of data to select. Should
                be path or a list of strings that points to an entry in the
                data tree.
            subspace (dict, optional): Selector for a subspace of the
                parameter space.
        
        Raises:
            ValueError: If no ParamSpace was associated with this group
        
        Returns:
            xr.Dataset: The selected data.
        """

        def parse_fields(*, field, fields) -> dict:
            """Parses the field and fields arguments into a uniform dict"""

            if field is not None and fields is not None:
                raise ValueError("Can only specify either of the arguments "
                                 "`field` or `fields`, got both!")

            elif field is None and fields is None:
                raise ValueError("Need to specify one of the arguments "
                                 "`field` or `fields`, got neither of them!")

            elif field is not None:
                # Generate a dict from the single field argument and put it
                # into a fields dict such that it can be processed like the
                # rest ...

                # Need to find a name from the path
                if isinstance(field, str):
                    path = field.split(PATH_JOIN_CHAR)
                    kwargs = {}

                elif isinstance(field, dict):
                    path = field.pop('path')
                    kwargs = field

                    if isinstance(path, str):
                        path = path.split(PATH_JOIN_CHAR)
                    
                else:
                    path = list(field)
                    kwargs = {}

                # Create the fields dict
                fields = dict()
                fields[path[-1]] = dict(path=path, **kwargs)

            # The fields variable is now available
            # Make sure it is of right type
            if not isinstance(fields, dict):
                raise TypeError("Argument `fields` needs to be a dict, "
                                "but was {}!".format(type(fields)))

            
            # Ensure entries are dicts of the proper structre
            for name, field in fields.items():
                if isinstance(field, str):
                    fields[name] = dict(path=field.split(PATH_JOIN_CHAR))

                elif isinstance(field, dict):
                    # Assert there are no invalid entries
                    if any([k not in ('path', 'dtype') for k in field.keys()]):
                        raise ValueError("There was an invalid key in the "
                                         "'{}' entry of the fields dict. "
                                         "Allowed keys: 'path', 'dtype'. "
                                         "Given dict: {}"
                                         "".format(name, field))

                else:
                    # Assume this is a sequence, make sure path is a list
                    fields[name] = dict(path=list(field))

            return fields

        def get_expaned_arr(*, grp, var_name: str, path: List[str], coords: dict, dtype: str=None) -> xr.DataArray:
            """Helper function to get a field from a group and create an
            expanded xr.DataArray from it.
            
            Args:
                grp: The group to search the field in
                var_name (str): The name of this field
                path (List[str]): The path to the field to get the data from
                coords (dict): The coordinates of the current point
                dtype (str, optional): The dtype to set.
            
            Returns:
                xr.DataArray: An expanded array
            """
            # Within that group, select a container using getitem
            cont = grp[path]

            # Now, an xr.DataArray has to be created from that container.
            if hasattr(cont, "to_xarray"):
                arr = cont.to_xarray(name=var_name)
                # TODO These do not actually exist, but should be implemented.
                #      They could also read in other meta data and already
                #      label the axes appropriately

            else:
                # For data without a specialized export function, access the
                # `data` attribute and extract the attributes manually ...
                
                # Still need to distinguish numeric containers and others:
                if isinstance(cont, NumpyDataContainer): # TODO generalise?
                    # Can take the data directly
                    data = cont.data
                    dims = None

                else:
                    # Need to wrap it in a list such that xarray does not try
                    # to resolve it somehow ... and later drop that dimension
                    data = [cont.data]
                    dims = ['__tmp']  # FIXME ugly
                
                # Get the attributes
                attrs = {k:v for k,v in cont.attrs.items()}

                # Use those to construct an xr.DataArray from it
                arr = xr.DataArray(data, name=var_name, dims=dims, attrs=attrs)

            # Now add the additional named dimensions with coordinates in front
            # and set their coordinates ...
            arr = arr.expand_dims(coords.keys())
            # NOTE While this creates a non-shallow copy of the data, there is
            #      no other way of doing this: a copy can only be avoided if
            #      the DataArray can re-use the existing array data – for the
            #      changes it needs to do to expand the dims, however, it will
            #      necessarily need to create a copy of the original array.
            #      Thus, we might as well let xarray take care of that instead
            #      of bothering with that ourselves ...

            arr = arr.assign_coords(**{k: [v] for k, v in coords.items()})
            # NOTE This creates a shallow (!) copy of the DataArray object.

            # Now, for object arrays, the temporary dimension can be dropped
            if '__tmp' in arr.dims:
                arr = arr.isel(dict(__tmp=0))

            # Finally, set the dtype
            if dtype is not None:
                arr = arr.astype(dtype)
                # NOTE for non-float dtypes, this will get lost in the merge!
                # TODO Find a better strategy for doing this!

            return arr

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Some initial checks

        if self.pspace is None:
            raise ValueError("Cannot get data from {} without having a "
                             "parameter space associated!".format(self.logstr))

        # Pre-process arguments . . . . . . . . . . . . . . . . . . . . . . . .
        # From field and fields arguments, generate a fields dict, such that
        # it can be handled uniformly below.
        fields = parse_fields(field=field, fields=fields)

        # TODO add log messages here


        # Work on a copy of the parameter space and apply the subspace masks
        psp = copy.deepcopy(self.pspace)

        if subspace is not None:
            psp.activate_subspace(**subspace)

        # Now, the data needs to be collected from each point in this subspace.
        data_pts = {field_name: [] for field_name in fields.keys()}

        # The most convenient and reliable merging can happen if the arrays
        # are merged directly using the xarray.merge feature.
        # To that end, the data needs to be collected and stored in separate
        # DataArrays which all have the first N dimensions matching to those of
        # the parameter subspace, but only with a single entry, namely that for
        # the specific parameter space coordinate that data is associated with.

        log.info("Collecting data from %d points in parameter space ...",
                 psp.volume)

        # Go over the parameter space...
        # TODO Could use np.nditer here instead?
        for state_no, coords in psp.iterator(with_info=('state_no',
                                                        'current_coords'),
                                             omit_pt=True):
            # Select the corresponding state group
            try:
                grp = self[state_no]

            except KeyError as err:
                # TODO consider not to bother ... missing data might be ok?!
                # TODO use custom exception class, e.g. from DataManager?
                raise ValueError("No state {} available in {}! Make sure the "
                                 "data was fully loaded."
                                 "".format(state_no, self.logstr)) from err

            # For each desired field ...
            for var_name, field in fields.items():
                arr = get_expaned_arr(grp=grp, var_name=var_name, **field,
                                      coords=coords)
                data_pts[var_name].append(arr)

        # All data points collected.
        # TODO consider warning if there are a high number of points. Also,
        #      it would be great if this could be parallelized ... via dask?!
        # TODO With multiple fields, there should be warnings if there would
        #      be a large amount of broadcasting needed ...

        # Merge now...
        log.info("Merging data arrays from %d points and %d field(s) ...",
                 psp.volume, len(fields))
        merged_dsets = []

        for var_name, _data_pts in data_pts.items():
            log.debug("Merging arrays of field '%s' ...", var_name)
            merged_dset = xr.merge(_data_pts)

            # Get the new variable's array out of the newly created dataset
            # and check if it was configured to change its dtype
            var_arr = merged_dset[var_name]
            dtype = fields[var_name].get('dtype')

            if dtype is not None and dtype != var_arr.dtype:
                merged_dset[var_name] = var_arr.astype(dtype)

            merged_dsets.append(merged_dset)

        # Finally, merge all the datasets together into a dataset with
        # potentially non-homogeneous data type. This will have at least the
        # dimensions given by the parameter space aligned, but there could
        # be potentially more dimensions!
        return xr.merge(merged_dsets)


    # Helper methods for data access ..........................................
