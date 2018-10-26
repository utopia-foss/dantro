"""In this module, BaseDataContainer specialisations are implemented."""

import copy
import logging
from typing import Union, List, Dict

import numpy as np
import numpy.ma
import xarray as xr

from paramspace import ParamSpace

from .ordered import OrderedDataGroup
from ..tools import apply_along_axis
from ..base import PATH_JOIN_CHAR
from ..container import NumpyDataContainer, XrContainer

# Local constants
log = logging.getLogger(__name__)

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

    def __delitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow item deletion by int key"""
        if isinstance(key, int):
            # Generate a padded string to access the state
            key = self._padded_id_from_int(key)

        # Use the parent method to return the value
        return super().__delitem__(key)

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

    def select(self, *, field: Union[str, List[str]]=None, fields: Dict[str, List[str]]=None, subspace: dict=None, method: str='concat', idx_as_label: bool=False, **kwargs) -> xr.Dataset:
        """Selects a multi-dimensional slab of this ParamSpaceGroup and the
        specified fields and returns them bundled into an xarray.Dataset with
        labelled dimensions and coordinates.
        
        Args:
            field (Union[str, List[str]], optional): The field of data to
                select. Should be path or a list of strings that points to an
                entry in the data tree. To select multiple fields, do not pass
                this argument but use the `fields` argument.
            fields (Dict[str, List[str]], optional): A dict specifying the
                fields that are to be loaded into the dataset. Keys will be
                the names of the resulting variables, while values should
                specify the path to the field in the data tree. Thus, they can
                be strings, lists of strings or dicts with the `path` key
                present. In the latter case, a dtype can be specified via the
                `dtype` key in the dict.
            subspace (dict, optional): Selector for a subspace of the
                parameter space. Adheres to the ParamSpace.activate_subspace
                signature.
            method (str, optional): How to combine the selected datasets.
                    * 'concat': concatenate sequentially along all parameter
                      space dimensions. This can preserve the data type but
                      it does not work if one data point is missing.
                    * 'merge': merge always works, even if data points are
                      missing, but will convert all dtypes to float.
            idx_as_label (bool, optional): If true, adds the trivial indices
                as labels for those dimensions where coordinate labels were not
                extractable from the loaded field. This allows merging for data
                with different extends in an unlabelled dimension.
            **kwargs: Passed along either to xr.concat or xr.merge, depending
                on the method argument.
        
        Raises:
            ValueError: Raised in multiple scenarios:
                * If no ParamSpace was associated with this group
                * For wrong argument values
                * If the data to select cannot be extracted with the given
                  argument values
                * Exceptions passed on from xarray
        
        Returns:
            xr.Dataset: The selected hyperslab of the parameter space, holding
                the desired fields.
        """

        def parse_fields(*, field, fields) -> dict:
            """Parses the field and fields arguments  into a uniform dict

            Return value is a dict of the following structure:
            <var_name_1>:
              path: <list of strings>
              dtype: <str, optional>
              dims: <list of strings, optional>
            <var_name_2>:
              ...
            """

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
                    path = field['path']
                    kwargs = {k:v for k, v in field.items() if k != "path"}
                    # Not using .pop here in order to not change the dict.

                    if isinstance(path, str):
                        path = path.split(PATH_JOIN_CHAR)
                    
                else:
                    path = list(field)
                    kwargs = {}

                # Create the fields dict, carrying over all other arguments
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
                    # Already is a dict.
                    # Need only assert there are no invalid entries
                    if any([k not in ('path', 'dtype', 'dims')
                            for k in field.keys()]):
                        raise ValueError("There was an invalid key in the "
                                         "'{}' entry of the fields dict. "
                                         "Allowed keys: 'path', 'dtype', "
                                         "'dims'. Given dict: {}"
                                         "".format(name, field))

                else:
                    # Assume this is a sequence, but better make sure ...
                    fields[name] = dict(path=list(field))

            return fields

        def get_state_grp(state_no: int) -> ParamSpaceStateGroup:
            """Returns the group corresponding to the given state"""
            try:
                return self[state_no]

            except KeyError as err:
                # TODO use custom exception class, e.g. from DataManager?
                raise ValueError("No state {} available in {}! Make sure the "
                                 "data was fully loaded."
                                 "".format(state_no, self.logstr)) from err

        def get_var(state_grp: ParamSpaceStateGroup, *, path: List[str], dtype: str=None, dims: List[str]=None) -> Union[xr.Variable, xr.DataArray]:
            """Extracts the field specified by the given path and returns it as
            either an xr.Variable or (for supported containers) directly as an
            xr.DataArray.
            
            We are using xr.Variables as defaults here, as they provide higher
            performance than xr.DataArrays; the latter have to be frequently
            unpacked and restructured in the merge operations.
            
            Args:
                state_grp (ParamSpaceStateGroup): The group to search `path` in
                path (List[str]): The path to a data container.
                dtype (str, optional): The desired dtype for the data.
                dims (List[str], optional): A list of dimension names for the
                    extracted data. If not given, will name them manually as
                    dim_0, dim_1, ...
            
            Returns:
                Union[xr.Variable, xr.DataArray]: The extracted data, which
                    can be either a data array (if the path led to an
                    xarray-interface supporting container) or an xr.Variable
                    (if not).
            """
            # First, get the desired container
            cont = state_grp[path]
            
            # Shortcut: specialised containers might already supply all the
            # information, including coordinates. In that case, return it as
            # a data array.
            if isinstance(cont, XrContainer):
                return cont.to_array()
                # TODO should the dims and dtype argument be handled here?!

            # If this was not the case, xr.Variable will have to be constructed
            # manually.
            # The only pre-requisite for the data is that it is np.array-like,
            # which is always possible; worst case: scalar of dtype "object".
            data = np.array(cont.data)
            # Can now assume data to comply to np.array interface

            # Generate dimension names, if not explicitly given.
            if dims is None:
                dims = ["dim_{}".format(i) for i in range(len(data.shape))]

            # TODO might need to add trivial indices here?!

            # Check the dtype and convert, if needed
            if dtype and data.dtype != dtype:
                log.debug("Converting data from '%s' with dtype %s to %s ...",
                          "/".join(path), data.dtype, dtype)
                data = data.astype(dtype)

            # Get the attributes
            attrs = cont.attrs.as_dict()

            # Check whether indices are to be added (var from outer scope!)
            if not idx_as_label:
                # Can use these to construct an xr.Variable
                return xr.Variable(dims, data, attrs=attrs)

            # else: will need to be a DataArray; Variable does not hold coords
            # For each dimension, add trivial coordinates
            coords = {d: range(data.shape[i]) for i, d in enumerate(dims)}

            return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)

        def expand_dset_dims(dset: xr.Dataset, *, coords: dict) -> xr.Dataset:
            """Expands the given dataset such that the new coordinates can be
            accomodated. Old coordinates stay intact.
            
            Note that this needs to create a copy of the data.
            
            Args:
                dset (xr.Dataset): The dataset to expand
                coords (dict): The coordinates of the current point, used to
                    expand the dimensions of the dataset.
            
            Returns:
                xr.Dataset: The expanded dataset
            """
            # Now add the additional named dimensions with coordinates in front
            dset = dset.expand_dims(coords.keys())
            # NOTE While this creates a non-shallow copy of the data, there is
            #      no other way of doing this: a copy can only be avoided if
            #      the DataArray can re-use the existing variables – for the
            #      changes it needs to do to expand the dims, however, it will
            #      necessarily need to create a copy of the original data.
            #      Thus, we might as well let xarray take care of that instead
            #      of bothering with that ourselves ...

            # ...and assign coordinates to them.
            dset = dset.assign_coords(**{k: [v] for k, v in coords.items()})
            # NOTE This creates only a shallow copy of the dataset. Thus, all
            #      already existing coordinates are carried over.

            return dset

        def combine(*, method: str, dsets: np.ndarray, psp: ParamSpace) -> xr.Dataset:
            """Tries to combine the given datasets either by concatenation or
            by merging and returns a combined xr.Dataset
            """
            # NOTE change for valid `method` value is carried out before this
            #      function is called.

            # Merging . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if method in ['merge']:
                log.info("Combining datasets by merging ...")
                # TODO consider warning about dtype changes?!

                dset = xr.merge(dsets.flat)

                log.info("Merge successful.")
                return dset

            # else: Concatenation . . . . . . . . . . . . . . . . . . . . . . .
            log.info("Combining datasets by concatenation along %d "
                     "dimensions ...", len(dsets.shape))

            # Go over all dimensions and concatenate
            # This effectively reduces the dsets array by one dimension in each
            # iteration by applying the xr.concat function along the axis
            # NOTE np.apply_along_axis would be what is desired here, but that
            #      function unfortunately tries to cast objects to np.arrays
            #      which is not what we want here at all! Thus, there is one
            #      implemented in dantro.tools ...
            idcs_and_names = list(enumerate(psp.dims.keys()))
            for dim_idx, dim_name in reversed(idcs_and_names):
                log.debug("Concatenating along axis '%s' (idx: %d) ...",
                          dim_name, dim_idx)

                dsets = apply_along_axis(xr.concat, axis=dim_idx, arr=dsets,
                                         dim=dim_name)

            log.info("Concatenation successful.")

            # Need to extract the single item from the now scalar dsets array
            return dsets.item()


        # End of definition of helper functions.
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Some initial checks

        if self.pspace is None:
            raise ValueError("Cannot get data from {} without having a "
                             "parameter space associated!".format(self.logstr))

        elif method not in ('concat', 'merge'):
            raise ValueError("Invalid value for argument `method`: '{}'. Can "
                             "be: 'concat' (default), 'merge'".format(method))

        # Pre-process arguments . . . . . . . . . . . . . . . . . . . . . . . .
        # From field and fields arguments, generate a fields dict, such that
        # it can be handled uniformly below.
        fields = parse_fields(field=field, fields=fields)

        # Work on a copy of the parameter space and apply the subspace masks
        psp = copy.deepcopy(self.pspace)

        if subspace:
            # Need the parameter space to be of non-zero volume
            if psp.volume == 0:
                raise ValueError("Cannot select a subspace because the "
                                 "associated parameter space has no "
                                 "dimensions defined! Remove the `subspace` "
                                 "argument in the call to this method.")

            try:
                psp.activate_subspace(**subspace)

            except KeyError as err:
                raise KeyError("Could not select a subspace because no "
                               "parameter dimension with name '{}' was found! "
                               "Make sure your subspace selector contains "
                               "only valid dimension names: {}"
                               "".format(str(err), ", ".join(psp.dims.keys()))
                               ) from err

        # Now, the data needs to be collected from each point in this subspace
        # and associated with the correct coordinate, such that the datasets
        # can later be merged and aligned by that coordinate.
        if psp.volume > 0:
            log.info("Collecting data for %d fields from %d points in "
                     "parameter space ...", len(fields), psp.volume)
        else:
            log.info("Collecting data for %d fields from a dimensionless "
                     "parameter space ...", len(fields))

        # Gather them in a multi-dimensional array
        dsets = np.zeros(psp.shape, dtype="object")
        dsets.fill(dict())  # these are ignored in xr.merge

        # Prepare the iterators
        psp_it = psp.iterator(with_info=('state_no', 'current_coords'),
                              omit_pt=True)
        arr_it = np.nditer(dsets, flags=('multi_index', 'refs_ok'))

        for (_state_no, _coords), _ in zip(psp_it, arr_it):
            # Select the corresponding state group
            try:
                _state_grp = get_state_grp(_state_no)

            except ValueError:
                if method == 'merge':
                    # In merge, this will mereley lead to a NaN ...
                    log.warning("Missing state group:  %d", _state_no)
                    continue
                # ...but for concatenation, it will result in an error.
                raise

            # Get the variables for all fields
            _vars = {k: get_var(_state_grp, **f) for k, f in fields.items()}

            # Construct a dataset from that ...
            _dset = xr.Dataset(_vars)

            # ... and expand its dimensions to accomodate the point in pspace
            _dset = expand_dset_dims(_dset, coords=_coords)

            # Store it in the array of datasets
            dsets[arr_it.multi_index] = _dset

        
        # All data points collected now.
        # TODO consider warning if there are a high number of points. Also,
        #      it would be great if this could be parallelized ... via dask?!

        # Finally, combine all the datasets together into a dataset with
        # potentially non-homogeneous data type. This will have at least the
        # dimensions given by the parameter space aligned, but there could
        # be potentially more dimensions!

        try:
            combined_dset = combine(method=method, dsets=dsets, psp=psp)

        except ValueError as err:
            raise ValueError("Combination of datasets failed; see below. This "
                             "is probably due to a failure of alignment, "
                             "which can be resolved by adding trivial "
                             "coordinates (i.e.: the indices) to unlabelled "
                             "dimensions by setting the `idx_as_label` "
                             "argument to True.") from err

        return combined_dset


    # Helper methods for data access ..........................................
