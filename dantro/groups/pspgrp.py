"""In this module, BaseDataContainer specializations that make use of features
from the paramspace package are implemented.
"""

import copy
import logging
from typing import Union, List, Dict, Callable, Sequence

import numpy as np
import numpy.ma
import xarray as xr

from paramspace import ParamSpace

from .ordered import OrderedDataGroup, IndexedDataGroup
from ..tools import apply_along_axis
from ..base import PATH_JOIN_CHAR
from ..containers import NumpyDataContainer, XrDataContainer
from ..mixins import PaddedIntegerItemAccessMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ParamSpaceStateGroup(OrderedDataGroup):
    """A ParamSpaceStateGroup is a member of the ParamSpaceGroup.

    While its own name need be interpretable as a positive integer (enforced
    in parent :py:class:`~dantro.groups.ParamSpaceGroup` but also here), it
    can hold members with any name.
    """
    _NEW_GROUP_CLS = OrderedDataGroup

    def _check_name(self, name: str) -> None:
        """Called by __init__ and overwritten here to check the name."""
        # Assert that the name is valid, i.e. convertible to an integer
        try:
            int(name)

        except ValueError as err:
            raise ValueError("Only names that are representible as integers "
                             "are possible for the name of {}, got '{}'!"
                             "".format(self.classname, name)
                             ) from err

        # ... and not negative
        if int(name) < 0:
            raise ValueError("Name for {} needs to be positive when converted "
                             "to integer, was: {}"
                             "".format(self.classname, name))

        # Still ask the parent method for its opinion on this matter
        super()._check_name(name)


class ParamSpaceGroup(PaddedIntegerItemAccessMixin, IndexedDataGroup):
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

    # A transformation callable that can be used during data selection
    _PSPGRP_TRANSFORMATOR = None

    # Define the class to use for the direct members of this group
    _NEW_GROUP_CLS = ParamSpaceStateGroup

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (ParamSpaceStateGroup,)

    # .........................................................................

    def __init__(self, *, name: str, pspace: ParamSpace=None,
                 containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group.
            pspace (ParamSpace, optional): Can already pass a ParamSpace object
                here.
            containers (list, optional): A list of containers to add, which
                need to be ParamSpaceStateGroup objects.
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """
        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # If given, associate the parameter space object
        if pspace is not None:
            self.pspace = pspace

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

    @property
    def only_default_data_present(self) -> bool:
        """Returns true if only data for the default point in parameter space
        is available in this group.
        """
        return (len(self) == 1) and (0 in self)


    # Data access .............................................................

    def select(self, *,
               field: Union[str, List[str]]=None,
               fields: Dict[str, List[str]]=None,
               subspace: dict=None,
               method: str='concat',
               idx_as_label: bool=False,
               **kwargs) -> xr.Dataset:
        """Selects a multi-dimensional slab of this ParamSpaceGroup and the
        specified fields and returns them bundled into an ``xarray.Dataset``
        with labelled dimensions and coordinates.
        
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
        
                    - ``concat``: concatenate sequentially along all parameter
                      space dimensions. This can preserve the data type but
                      it does not work if one data point is missing.
                    - ``merge``: merge always works, even if data points are
                      missing, but will convert all dtypes to float.
        
            idx_as_label (bool, optional): If true, adds the trivial indices
                as labels for those dimensions where coordinate labels were not
                extractable from the loaded field. This allows merging for data
                with different extends in an unlabelled dimension.
            **kwargs: Passed along either to xr.concat or xr.merge, depending
                on the ``method`` argument.
        
        Raises:
            KeyError: Description
            ValueError: Raised in multiple scenarios:
        
                - If no ParamSpace was associated with this group
                - For wrong argument values
                - If the data to select cannot be extracted with the given
                  argument values
                - Exceptions passed on from xarray
        
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
              ... further
            <var_name_2>:
              ...

            TODO Change such that its using strings for paths, not sequences.
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

            
            # Ensure values of the dict are dicts of the proper structre
            for name, field in fields.items():
                if isinstance(field, str):
                    fields[name] = dict(path=field.split(PATH_JOIN_CHAR))

                elif not isinstance(field, dict):
                    # Assume this is a sequence, but better make sure ...
                    fields[name] = dict(path=list(field))

                # else: Already a dict; nothing to do. Parameters carried over.

            return fields

        def get_state_grp(state_no: int) -> ParamSpaceStateGroup:
            """Returns the group corresponding to the given state"""
            try:
                return self[state_no]

            except (KeyError, ValueError) as err:
                # TODO use custom exception class, e.g. from DataManager?
                raise ValueError("No state {} available in {}! Make sure the "
                                 "data was fully loaded."
                                 "".format(state_no, self.logstr)) from err

        def get_var(state_grp: ParamSpaceStateGroup, *,
                    path: str,
                    dtype: str=None,
                    dims: List[str]=None,
                    transform: Sequence[dict]=None,
                    **transform_kwargs) -> Union[xr.Variable, xr.DataArray]:
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
                transform (Sequence[dict], optional): Optional transform
                    arguments; passed on to transformator as *args.
                **transform_kwargs: Passed on to the transformator as **kwargs.
            
            Returns:
                Union[xr.Variable, xr.DataArray]: The extracted data, which
                    can be either a data array (if the path led to an
                    xarray-interface supporting container) or an xr.Variable
                    (if not).
            """
            def convert_dtype(data, dtype, *, path):
                """Change the dtype of the data, if it does not match the
                specified one.
                """
                if data.dtype == dtype:
                    return data
                log.debug("Converting data from '%s' with dtype %s to %s ...",
                          path, data.dtype, dtype)
                return data.astype(dtype)

            # First, get the desired container
            cont = state_grp[path]

            # Apply the transformator on the container, if arguments given
            if transform or transform_kwargs:
                if self._PSPGRP_TRANSFORMATOR is None:
                    raise ValueError("Got transform arguments or kwargs, but "
                                     "no transformator callable was defined "
                                     "as class variable!")

                # Invoke the transformator on the container
                cont = self._PSPGRP_TRANSFORMATOR(cont, *transform,
                                                  **transform_kwargs)
            
            # Shortcut: specialised containers might already supply all the
            # information, including coordinates. In that case, return it as
            # a data array.
            if isinstance(cont, XrDataContainer):
                # Will return the underlying data. See if some dtype change
                # or dimension name relabelling was specified
                darr = cont.data

                if dtype is not None:
                    darr = convert_dtype(darr, dtype, path=path)

                if dims is not None:
                    darr = darr.rename({old:new for old, new
                                        in zip(darr.dims, dims)})

                return darr

            elif isinstance(cont, (xr.DataArray, xr.Dataset)):
                # Actually was not a container but already the data; skip below
                return cont

            # If this was not the case, xr.Variable will have to be constructed
            # manually from the container.
            # The only pre-requisite for the data is that it is np.array-like,
            # which is always possible; worst case: scalar of dtype "object".
            data = np.array(cont.data)
            # Can now assume data to comply to np.array interface

            # Check the dtype and convert, if needed
            if dtype is not None:
                data = convert_dtype(data, dtype, path=path)

            # Get the attributes
            attrs = {k: v for k, v in cont.attrs.items()}

            # Generate dimension names, if not explicitly given.
            if dims is None:
                dims = ["dim_{}".format(i) for i in range(len(data.shape))]

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
            dset = dset.expand_dims(dim=list(coords.keys()))
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

        def combine(*, method: str,
                    dsets: np.ndarray, psp: ParamSpace) -> xr.Dataset:
            """Tries to combine the given datasets either by concatenation or
            by merging and returns a combined xr.Dataset
            """
            # NOTE change for valid `method` value is carried out before this
            #      function is called.

            # Merging . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if method in ['merge']:
                log.remark("Combining datasets by merging ...")
                # TODO consider warning about dtype changes?!

                dset = xr.merge(dsets.flat)

                log.remark("Merge successful.")
                return dset

            # else: Concatenation . . . . . . . . . . . . . . . . . . . . . . .
            log.remark("Combining %d datasets by concatenation along %d "
                       "dimensions ...", dsets.size, len(dsets.shape))

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

            log.remark("Concatenation successful.")

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
        log.info("Data collected.")

        # Finally, combine all the datasets together into a dataset with
        # potentially non-homogeneous data type. This will have at least the
        # dimensions given by the parameter space aligned, but there could
        # be potentially more dimensions.

        try:
            dset = combine(method=method, dsets=dsets, psp=psp)

        except ValueError as err:
            raise ValueError("Combination of datasets failed; see below. This "
                             "is probably due to a failure of alignment, "
                             "which can be resolved by adding trivial "
                             "coordinates (i.e.: the indices) to unlabelled "
                             "dimensions by setting the `idx_as_label` "
                             "argument to True.") from err

        log.info("Data selected.")
        log.note("Available data variables:      %s",
                 ", ".join(dset.data_vars))
        log.note("Dataset dimensions and sizes:  %s",
                 ", ".join("{}: {}".format(*kv) for kv in dset.sizes.items()))
        return dset
