"""This module implements :py:class:`~dantro.base.BaseDataContainer`
specializations that make use of features from the
`paramspace <https://gitlab.com/blsqr/paramspace>`_ package, in particular the
:py:class:`~paramspace.paramspace.ParamSpace` class.
"""

import copy
import logging
from typing import Dict, List, Sequence, Union

import numpy as np
import numpy.ma
from paramspace import ParamSpace

from .._import_tools import LazyLoader
from ..base import PATH_JOIN_CHAR
from ..containers import XrDataContainer
from ..data_ops.arr_ops import multi_concat as _multi_concat
from ..mixins import PaddedIntegerItemAccessMixin
from .ordered import IndexedDataGroup, OrderedDataGroup

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")

# -----------------------------------------------------------------------------


class ParamSpaceStateGroup(OrderedDataGroup):
    """A ParamSpaceStateGroup is meant to be used as a member group of the
    :py:class:`~dantro.groups.psp.ParamSpaceGroup`.

    While its *own* name need be interpretable as a positive integer (enforced
    in the enclosing :py:class:`~dantro.groups.psp.ParamSpaceGroup` but also
    here), it can *hold* members with any name.
    """

    _NEW_GROUP_CLS = OrderedDataGroup

    def _check_name(self, name: str) -> None:
        """Called by __init__ and overwritten here to check the name."""
        # Assert that the name is valid, i.e. convertible to an integer
        try:
            int(name)

        except ValueError as err:
            raise ValueError(
                "Only names that are representible as integers are possible "
                f"for the name of {self.classname}, got '{name}'!"
            ) from err

        # ... and not negative
        if int(name) < 0:
            raise ValueError(
                f"Name for {self.classname} needs to be positive when "
                f"converted to integer, was: {name}"
            )

        # Still ask the parent method for its opinion on this matter
        super()._check_name(name)

    @property
    def coords(self) -> dict:
        """Retrieves the coordinates of this group within the parameter space
        described by the :py:class:`~dantro.groups.psp.ParamSpaceGroup`
        this group is enclosed in.

        Returns:
            dict: The coordinates of this group, keys being dimension names and
                values being the coordinate values for this group.
        """
        state_map = self.parent.pspace.state_map
        coords = state_map.where(state_map == int(self.name), drop=True).coords
        return {d: c.item() for d, c in coords.items()}


class ParamSpaceGroup(PaddedIntegerItemAccessMixin, IndexedDataGroup):
    """The ParamSpaceGroup is associated with a
    :py:class:`paramspace.paramspace.ParamSpace` object and the
    loaded results of an iteration over this parameter space.

    Thus, the groups that are stored in the ParamSpaceGroup need all relate to
    a state of the parameter space, identified by a zero-padded string name.
    In fact, this group allows no other kinds of groups stored inside.

    To make access to a specific state easier, it allows accessing a state by
    its state number as integer.
    """

    # Configure the class variables that define some of the behaviour
    # Define which .attrs entry to return from the `pspace` property
    _PSPGRP_PSPACE_ATTR_NAME = "pspace"

    # A transformation callable that can be used during data selection
    _PSPGRP_TRANSFORMATOR = None

    # Define the class to use for the direct members of this group
    _NEW_GROUP_CLS = ParamSpaceStateGroup

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (ParamSpaceStateGroup,)

    # .........................................................................

    def __init__(
        self,
        *,
        name: str,
        pspace: ParamSpace = None,
        containers: list = None,
        **kwargs,
    ):
        """Initialize a OrderedDataGroup from the list of given containers.

        Args:
            name (str): The name of this group.
            pspace (paramspace.paramspace.ParamSpace, optional): Can already
                pass a ParamSpace object here.
            containers (list, optional): A list of containers to add, which
                need to be
                :py:class:`~dantro.groups.psp.ParamSpaceStateGroup` objects.
            **kwargs: Further initialisation kwargs, e.g. ``attrs`` ...
        """
        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # If given, associate the parameter space object
        if pspace is not None:
            self.pspace = pspace

    # Properties ..............................................................

    @property
    def pspace(self) -> Union[ParamSpace, None]:
        """Reads the entry named ``_PSPGRP_PSPACE_ATTR_NAME`` in ``.attrs`` and
        returns a :py:class:`~paramspace.paramspace.ParamSpace` object, if
        available there.

        Returns:
            Union[paramspace.paramspace.ParamSpace, None]: The associated
                parameter space, or None, if there is none associated yet.
        """
        return self.attrs.get(self._PSPGRP_PSPACE_ATTR_NAME, None)

    @pspace.setter
    def pspace(self, val: ParamSpace):
        """If not already set, sets the entry in the attributes that is
        accessed by the ``.pspace`` property
        """
        if self.pspace is not None:
            raise RuntimeError(
                "The attribute for the parameter space of this "
                f"{self.logstr} was already set, cannot set it again!"
            )

        elif not isinstance(val, ParamSpace):
            raise TypeError(
                f"The attribute for the parameter space of {self.logstr} "
                f"needs to be a ParamSpace-derived object, was {type(val)}!"
            )

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

    def select(
        self,
        *,
        field: Union[str, List[str]] = None,
        fields: Dict[str, List[str]] = None,
        subspace: dict = None,
        method: str = "concat",
        idx_as_label: bool = False,
        base_path: str = None,
        **kwargs,
    ) -> "xarray.Dataset":
        """Selects a multi-dimensional slab of this ParamSpaceGroup and the
        specified fields and returns them bundled into an
        :py:class:`xarray.Dataset` with labelled dimensions and coordinates.

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
                parameter space. Adheres to the parameter space's
                :py:meth:`~paramspace.paramspace.ParamSpace.activate_subspace`
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
            base_path (str, optional): If given, ``path`` specifications for
                each field can be seen as relative to this path
            **kwargs: Passed along either to xr.concat or xr.merge, depending
                on the ``method`` argument.

        Raises:
            KeyError: On invalid state key.
            ValueError: Raised in multiple scenarios: If no
                :py:class:`~paramspace.paramspace.ParamSpace` was
                associated with this group, for wrong argument values, if the
                data to select cannot be extracted with the given argument
                values, exceptions passed on from xarray.

        Returns:
            xarray.Dataset: The selected hyperslab of the parameter space,
                holding the desired fields.
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
                raise ValueError(
                    "Can only specify either of the arguments "
                    "`field` or `fields`, got both!"
                )

            elif field is None and fields is None:
                raise ValueError(
                    "Need to specify one of the arguments "
                    "`field` or `fields`, got neither of them!"
                )

            elif field is not None:
                # Generate a dict from the single field argument and put it
                # into a fields dict such that it can be processed like the
                # rest ...

                # Need to find a name from the path
                if isinstance(field, str):
                    path = field.split(PATH_JOIN_CHAR)
                    kwargs = {}

                elif isinstance(field, dict):
                    path = field["path"]
                    kwargs = {k: v for k, v in field.items() if k != "path"}
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
                raise TypeError(
                    "Argument `fields` needs to be a dict, "
                    f"but was {type(fields)}!"
                )

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
                raise ValueError(
                    f"No state {state_no} available in {self.logstr}! Make "
                    "sure the data was fully loaded."
                ) from err

        def get_var(
            state_grp: ParamSpaceStateGroup,
            *,
            path: List[str],
            base_path: List[str] = None,
            dtype: str = None,
            dims: List[str] = None,
            transform: Sequence[dict] = None,
            **transform_kwargs,
        ) -> Union["xr.Variable", "xr.DataArray"]:
            """Extracts the field specified by the given path and returns it as
            either an xr.Variable or (for supported containers) directly as an
            xr.DataArray.

            We are using xr.Variables as defaults here, as they provide higher
            performance than xr.DataArrays; the latter have to be frequently
            unpacked and restructured in the merge operations.

            Args:
                state_grp (ParamSpaceStateGroup): The group to search `path` in
                path (List[str]): The path to a data container.
                base_path (List[str], optional): Will be prepended to the given
                    path, if given.
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

            Raises:
                ValueError: Missing transformator
            """

            def convert_dtype(data, dtype, *, path):
                """Change the dtype of the data, if it does not match the
                specified one.
                """
                if data.dtype == dtype:
                    return data
                log.debug(
                    "Converting data from '%s' with dtype %s to %s ...",
                    PATH_JOIN_CHAR.join(path),
                    data.dtype,
                    dtype,
                )
                return data.astype(dtype)

            # Prepare the path, ensuring to work on the list representation
            if not isinstance(path, list):
                path = path.split(PATH_JOIN_CHAR)

            if base_path:
                path = base_path + path

            # Now, get the desired container
            cont = state_grp[path]

            # Apply the transformator on the container, if arguments given
            if transform or transform_kwargs:
                if self._PSPGRP_TRANSFORMATOR is None:
                    raise ValueError(
                        "Got transform arguments or kwargs, but "
                        "no transformator callable was defined "
                        "as class variable!"
                    )

                # Invoke the transformator on the container
                cont = self._PSPGRP_TRANSFORMATOR(
                    cont, *transform, **transform_kwargs
                )

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
                    darr = darr.rename(
                        {old: new for old, new in zip(darr.dims, dims)}
                    )

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
                dims = [f"dim_{i}" for i in range(len(data.shape))]

            # Check whether indices are to be added (var from outer scope!)
            if not idx_as_label:
                # Can use these to construct an xr.Variable
                return xr.Variable(dims, data, attrs=attrs)

            # else: will need to be a DataArray; Variable does not hold coords
            # For each dimension, add trivial coordinates
            coords = {d: range(data.shape[i]) for i, d in enumerate(dims)}

            return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)

        def combine(
            *, method: str, dsets: np.ndarray, psp: ParamSpace
        ) -> "xr.Dataset":
            """Tries to combine the given datasets either by concatenation or
            by merging and returns a combined xr.Dataset
            """
            # NOTE change for valid `method` value is carried out before this
            #      function is called.

            # Merging . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if method in ["merge"]:
                log.remark("Combining datasets by merging ...")
                # TODO consider warning about dtype changes?!

                dset = xr.merge(dsets.flat)

                log.remark("Merge successful.")
                return dset

            # else: Concatenation . . . . . . . . . . . . . . . . . . . . . . .
            log.remark(
                "Combining %d datasets by concatenation along %d "
                "dimensions ...",
                dsets.size,
                len(dsets.shape),
            )

            # Reduce the dsets array to one dimension by applying xr.concat
            # along each axis. The returned object contains the combined data.
            reduced = _multi_concat(dsets, dims=psp.dims.keys())
            log.remark("Concatenation successful.")

            return reduced

        # End of definition of helper functions.
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Some initial checks

        if self.pspace is None:
            raise ValueError(
                f"Cannot get data from {self.logstr} without having a "
                "parameter space associated!"
            )

        elif method not in ("concat", "merge"):
            raise ValueError(
                f"Invalid value for argument `method`: '{method}'. Can "
                "be: 'concat' (default), 'merge'"
            )

        # Pre-process arguments . . . . . . . . . . . . . . . . . . . . . . . .
        # From field and fields arguments, generate a fields dict, such that
        # it can be handled uniformly below.
        fields = parse_fields(field=field, fields=fields)

        # Prepare the base path
        if base_path and not isinstance(base_path, list):
            base_path = base_path.split(PATH_JOIN_CHAR)

        # Work on a copy of the parameter space and apply the subspace masks
        psp = copy.deepcopy(self.pspace)

        if subspace:
            # Need the parameter space to be of non-zero volume
            if psp.volume == 0:
                raise ValueError(
                    "Cannot select a subspace because the "
                    "associated parameter space has no "
                    "dimensions defined! Remove the `subspace` "
                    "argument in the call to this method."
                )

            try:
                psp.activate_subspace(**subspace)

            except KeyError as err:
                _dim_names = ", ".join(psp.dims.keys())
                raise KeyError(
                    "Could not select a subspace! "
                    f"{type(err).__name__}: {err}\n"
                    "Make sure your subspace selector contains "
                    "only valid dimension names and coordinates. "
                    f"Available dimension names: {_dim_names}"
                ) from err

        # Now, the data needs to be collected from each point in this subspace
        # and associated with the correct coordinate, such that the datasets
        # can later be merged and aligned by that coordinate.
        if psp.volume > 0:
            log.info(
                "Collecting data for %d fields from %d points in "
                "parameter space ...",
                len(fields),
                psp.volume,
            )
        else:
            log.info(
                "Collecting data for %d fields from a dimensionless "
                "parameter space ...",
                len(fields),
            )

        # Gather them in a multi-dimensional array
        dsets = np.zeros(psp.shape, dtype="object")
        dsets.fill(dict())  # these are ignored in xr.merge

        # Prepare the iterators
        psp_it = psp.iterator(
            with_info=("state_no", "current_coords"), omit_pt=True
        )
        arr_it = np.nditer(dsets, flags=("multi_index", "refs_ok"))

        for (_state_no, _coords), _ in zip(psp_it, arr_it):
            # Select the corresponding state group
            try:
                _state_grp = get_state_grp(_state_no)

            except ValueError:
                if method == "merge":
                    # In merge, this will mereley lead to a NaN ...
                    log.warning("Missing state group:  %d", _state_no)
                    continue
                # ...but for concatenation, it will result in an error.
                raise

            # Get the variables for all fields
            _vars = {
                k: get_var(_state_grp, **f, base_path=base_path)
                for k, f in fields.items()
            }

            # Construct a dataset from that ...
            _dset = xr.Dataset(_vars)

            # ... and expand its dimensions to accomodate the point in pspace
            _dset = _dset.expand_dims({k: [v] for k, v in _coords.items()})

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
            raise ValueError(
                "Combination of datasets failed; see below. This "
                "is probably due to a failure of alignment, "
                "which can be resolved by adding trivial "
                "coordinates (i.e.: the indices) to unlabelled "
                "dimensions by setting the `idx_as_label` "
                "argument to True."
            ) from err

        log.info("Data selected.")
        log.note(
            "Available data variables:      %s", ", ".join(dset.data_vars)
        )
        log.note(
            "Dataset dimensions and sizes:  %s",
            ", ".join("{}: {}".format(*kv) for kv in dset.sizes.items()),
        )
        return dset
