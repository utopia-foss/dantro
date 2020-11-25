"""Implements the LabelledDataGroup, which allows to handle groups and
containers that can be associated with further coordinates.

This imitates the xarray selection interface and provides a uniform interface
to select data from these groups. Most importantly, it allows to combine all
the data of one group, allowing to conveniently work with heterogeneously
stored data.
"""

import logging
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from ..abc import AbstractDataContainer
from ..containers import XrDataContainer
from ..exceptions import *
from ..tools import apply_along_axis
from ..utils import extract_coords
from ..utils.coords import TCoord, TCoordsDict, TDims
from . import OrderedDataGroup

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class LabelledDataGroup(OrderedDataGroup):
    """A group that assumes that the members it contains can be labelled
    with dimension names and coordinates.

    Such a group has the great benefit to provide a selection interface that
    works fully on the dimension labels and coordinates and can cooperate with
    the xarray selection interface, i.e. the ``sel`` and ``isel`` methods.
    """

    # Let new containers be xarray-based
    _NEW_CONTAINER_CLS = XrDataContainer

    # Configuration options for this group ....................................
    # Whether to use deep selection by default
    LDG_ALLOW_DEEP_SELECTION = True

    # The dimensions of this group, i.e. the dimensions of the space covered by
    # the members of this group.
    LDG_DIMS = tuple()

    # How to extract coordinates of the members; for available modes, see the
    # `dantro.utils.coords.extract_coords` function.
    LDG_EXTRACT_COORDS_FROM = "data"

    # Configuration for mode 'attrs' . . . . . . . . . . . . . . . . . . . . .
    LDG_COORDS_ATTR_PREFIX = "ext_coords__"
    LDG_COORDS_MODE_ATTR_PREFIX = "ext_coords_mode__"
    LDG_COORDS_MODE_DEFAULT = "scalar"
    LDG_STRICT_ATTR_CHECKING = False

    # Configuration for mode 'name' . . . . . . . . . . . . . . . . . . . . . .
    LDG_COORDS_SEPARATOR_IN_NAME = ";"

    # .........................................................................

    def __init__(
        self,
        *args,
        dims: TDims = None,
        mode: str = None,
        allow_deep_selection: bool = None,
        **kwargs,
    ):
        """Initialize a LabelledDataGroup

        Args:
            *args: Passed on to
                :py:class:`~dantro.groups.ordered.OrderedDataGroup`
            dims (TDims, optional): The dimensions associated with this group.
                If not given, will use those defined in the ``LDG_DIMS`` class
                variable. These can *not* be changed afterwards!
            mode (str, optional): By which coordiante extraction mode to get
                the coordinates from the group members. Can be ``attrs``,
                ``name``, ``data`` or anything else specified in
                :py:func:`~dantro.utils.coords.extract_coords`.
            allow_deep_selection (bool, optional): Whether to allow deep
                selection. If not given, will use the
                ``LDG_ALLOW_DEEP_SELECTION`` class variable's value. Behaviour
                can be changed via the property of the same name.
            **kwargs: Passed on to
                :py:class:`~dantro.groups.ordered.OrderedDataGroup`
        """
        # Initialize the member map, which is needed if containers are added
        # during initialization (thus invoking _add_container_callback)
        self.__member_map = None

        super().__init__(*args, **kwargs)

        self._dims = dims if dims is not None else tuple(self.LDG_DIMS)

        self._mode = self.LDG_EXTRACT_COORDS_FROM
        if mode is not None:
            self._mode = mode

        self._allow_deep_selection = self.LDG_ALLOW_DEEP_SELECTION
        if allow_deep_selection is not None:
            self._allow_deep_selection = allow_deep_selection

    # Dimension and coordinates ...............................................

    @property
    def dims(self) -> Tuple[str]:
        """The names of the group-level dimensions this group manages.

        It _may_ contain dimensions that overlap with dimension names from the
        members; this is intentional.
        """
        return self.member_map.dims

    @property
    def ndim(self) -> int:
        """The rank of the space covered by the group-level dimensions."""
        return self.member_map.ndim

    @property
    def coords(self) -> Dict[str, List[TCoord]]:
        """Returns a dict-like container of group-level coordinate values keyed
        by dimension.
        """
        return self.member_map.coords

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the space covered by the group-level dimensions."""
        return self.member_map.shape

    # Additional properties ...................................................

    @property
    def allow_deep_selection(self) -> bool:
        """Whether deep selection is allowed."""
        return self._allow_deep_selection

    @allow_deep_selection.setter
    def allow_deep_selection(self, val: bool):
        """Change whether deep selection is allowed."""
        self._allow_deep_selection = val

    @property
    def member_map(self) -> xr.DataArray:
        """Returns an array that represents the space that the members of this
        group span, where each value (i.e. a specific coordinate combination)
        is the name of the corresponding member of this group.

        Upon first call, this is computed here. If members are added, it is
        tried to accomodate them in there; if not possible, the cache will be
        invalidated.

        The member map _may_ include empty strings, i.e. coordinate
        combinations that are not covered by any member. Also, they can contain
        duplicate names, as one member can cover multiple coordinates.

        .. note::

            The member map is invalidated when new members are added that can
            not be accomodated in it. It will be recalculated when needed.
        """
        if self.member_map_available:
            return self.__member_map

        # Member map is not available and has to be created anew.
        # First, extract the coordinates by iterating over all members and
        # aggregating their individual coordinates; sets guarantee uniqueness
        coords = {dim_name: set() for dim_name in self._dims}

        for cont_name, cont in self.items():
            cont_coords = self._get_coords_of(cont)

            for dim_name, coord_vals in cont_coords.items():
                coords[dim_name].update(coord_vals)

        # Convert to dict of lists
        coords = {dim_name: sorted(list(s)) for dim_name, s in coords.items()}

        # Now, derive the shape from the coordinates
        shape = tuple([len(coords[dim_name]) for dim_name in self._dims])

        # Create a DataArray containing empty strings, using the existing
        # dimension names (as set during initialization) and coordinates to
        # label it.
        mm_data = np.zeros(shape, dtype="object")
        mm_data.fill("")

        mm = xr.DataArray(
            data=mm_data,
            dims=self._dims,
            coords=coords,
        )

        # Iterate over members and populate the array with member names
        for cont_name, cont in self.items():
            cont_coords = self._get_coords_of(cont)
            # These coordinates describe a hypercube in coordinate space that
            # is to be associated with this container. Thus, the member map
            # should contain the name of the member for all these coordinates:
            mm.loc[cont_coords] = cont_name

        # Cache the map and return it
        self.__member_map = mm
        return mm

    @property
    def member_map_available(self) -> bool:
        """Whether the member map is available yet."""
        return self.__member_map is not None

    # Selection interface .....................................................

    def isel(
        self,
        indexers: dict = None,
        *,
        drop: bool = False,
        combination_method: str = "try_concat",
        deep: bool = None,
        **indexers_kwargs,
    ) -> xr.DataArray:
        """Return a new labelled `xr.DataArray` with an index-selected subset
        of members of this group.

        If deep selection is activated, those indexers that are not available
        in the group-managed dimensions are looked up in the members of this
        group.

        .. note::

            For data combination (via *any* ``combination_method``)
            dimensions that differ in size across group members have to be
            labelled, such that arrays can be
            `aligned <http://xarray.pydata.org/en/stable/generated/xarray.align.html>`_
            using the respective coordinates.
            See `the xarray documentation <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_
            for more information about coordinates.

        Args:
            indexers (dict, optional): A dict with keys matching dimensions and
                values given by scalars, slices or arrays of tick indices.
                As `xr.DataArray.isel`, uses pandas-like indexing, i.e.: slices
                do not include the terminal value.
            drop (bool, optional): Drop coordinate variables instead of making
                them scalar.
            combination_method (str, optional): How to combine group-level data
                with member-level data. Ignored if data from a single group
                member is selected, i.e. no data has to be combined. Can be:

                    * ``concat``: Concatenate. This can preserve the dtype, but
                      requires that no data is missing.
                    * ``merge``: Merge, using `xarray.merge`. This leads to a
                      type conversion to ``float64``, but allows members being
                      missing or coordinates not fully filling the available
                      space.
                    * ``try_concat``: Try concatenation, fall back to merging
                      if that was unsuccessful.

            deep (bool, optional): Whether to allow deep indexing, i.e.: that
                ``indexers`` may contain dimensions that don't refer to group-
                level dimensions but to dimensions that are only availble among
                the member data. If ``None``, will use the value returned by
                the ``allow_deep_selection`` property.
            **indexers_kwargs: Additional indexers

        Returns:
            xr.DataArray: The selected data, potentially a combination of data
                on group level and member-level data.
        """
        idxrs, deep_idxrs = self._parse_indexers(
            indexers, allow_deep=deep, **indexers_kwargs
        )

        # Use the (shallow) indexers to select (by index) those members that
        # are to be combined ...
        tbc = self.member_map.isel(idxrs, drop=drop)

        # Now, combine them, potentially also applying deep indexing
        return self._combine(
            tbc,
            combination_method=combination_method,
            shallow_indexers=idxrs,
            deep_indexers=deep_idxrs,
            by_index=True,
            drop=drop,
        )

    def sel(
        self,
        indexers: dict = None,
        *,
        method: str = None,
        tolerance: float = None,
        drop: bool = False,
        combination_method: str = "try_concat",
        deep: bool = None,
        **indexers_kwargs,
    ) -> xr.DataArray:
        """Return a new labelled `xr.DataArray` with a coordinate-selected
        subset of members of this group.

        If deep selection is activated, those indexers that are not available
        in the group-managed dimensions are looked up in the members of this
        group.

        .. note::

            For data combination (via *any* ``combination_method``)
            dimensions that differ in size across group members have to be
            labelled, such that arrays can be
            `aligned <http://xarray.pydata.org/en/stable/generated/xarray.align.html>`_
            using the respective coordinates.
            See `the xarray documentation <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_
            for more information about coordinates.


        Args:
            indexers (dict, optional): A dict with keys matching dimensions and
                values given by scalars, slices or arrays of tick labels.
                As `xr.DataArray.sel`, uses pandas-like indexing, i.e.: slices
                include the terminal value.
            method (str, optional): Method to use for inexact matches
            tolerance (float, optional): Maximum (absolute) distance between
                original and given label for inexact matches.
            drop (bool, optional): Drop coordinate variables instead of making
                them scalar.
            combination_method (str, optional): How to combine group-level data
                with member-level data. Ignored if data from a single group
                member is selected, i.e. no data has to be combined. Can be:

                    * ``concat``: Concatenate. This can preserve the dtype, but
                      requires that no data is missing.
                    * ``merge``: Merge, using `xarray.merge`. This leads to a
                      type conversion to ``float64``, but allows members being
                      missing or coordinates not fully filling the available
                      space.
                    * ``try_concat``: Try concatenation, fall back to merging
                      if that was unsuccessful.

            deep (bool, optional): Whether to allow deep indexing, i.e.: that
                ``indexers`` may contain dimensions that don't refer to group-
                level dimensions but to dimensions that are only availble among
                the member data. If ``None``, will use the value returned by
                the ``allow_deep_selection`` property.
            **indexers_kwargs: Additional indexers

        Returns:
            xr.DataArray: The selected data, potentially a combination of data
                on group level and member-level data.
        """
        idxrs, deep_idxrs = self._parse_indexers(
            indexers, allow_deep=deep, **indexers_kwargs
        )

        # Use the (shallow) indexers to select (by label) those members that
        # are to be combined ...
        tbc = self.member_map.sel(
            idxrs, method=method, tolerance=tolerance, drop=drop
        )

        # Now, combine them, potentially also applying deep indexing
        return self._combine(
            tbc,
            combination_method=combination_method,
            shallow_indexers=idxrs,
            deep_indexers=deep_idxrs,
            by_index=False,
            method=method,
            tolerance=tolerance,
            drop=drop,
        )

    # Helpers .................................................................
    # General . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _get_coords_of(self, obj: AbstractDataContainer) -> TCoordsDict:
        """Extract the coordinates for the given object using the
        :py:func:`~dantro.utils.coords.extract_coords` function.

        Args:
            obj (AbstractDataContainer): The object to get the coordinates of.

        Returns:
            TCoordsDict: The extracted coordinates
        """
        # Depending on the mode, compile the dict of additional parameters
        kwargs = dict()

        if self._mode == "attrs":
            kwargs["coords_attr_prefix"] = self.LDG_COORDS_ATTR_PREFIX
            kwargs["mode_attr_prefix"] = self.LDG_COORDS_MODE_ATTR_PREFIX
            kwargs["default_mode"] = self.LDG_COORDS_MODE_DEFAULT
            kwargs["strict"] = self.LDG_STRICT_ATTR_CHECKING

        elif self._mode == "name":
            kwargs["separator"] = self.LDG_COORDS_SEPARATOR_IN_NAME

        return extract_coords(obj, dims=self._dims, mode=self._mode, **kwargs)

    def _add_container_callback(self, cont: AbstractDataContainer) -> None:
        """Called by the base class after adding a container, this method
        checks whether the member map needs to be invalidated or whether the
        new container can be accomodated in it.

        If it can be accomodated, the member map will be adjusted such that for
        all coordinates associated with the given ``cont``, the member map
        points to the newly added container.

        Args:
            cont (AbstractDataContainer): The newly added container
        """
        # First, let the parent class do its thinkg
        super()._add_container_callback(cont)

        # Don't have to do anything if there is no member map yet
        if not self.member_map_available:
            return

        # There is a map. Check if it can accomodate the new container
        coords = self._get_coords_of(cont)

        try:
            self.__member_map.loc[coords] = cont.name

        except Exception:
            # Cannot accomodate it -> invalidate it
            self.__member_map = None

    # For selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _parse_indexers(
        self, indexers: dict, *, allow_deep: bool, **indexers_kwargs
    ) -> Tuple[dict, dict]:
        """Parses the given indexer arguments and split them into indexers for
        the selection of group members and deep selection.

        Args:
            indexers (dict): The indexers dict, may be empty
            allow_deep (bool): Whether to allow deep selection
            **indexers_kwargs: Additional indexers

        Returns:
            Tuple[dict, dict]: (shallow indexers, deep indexers)

        Raises:
            ValueError: If deep indexers were given but deep selection was not
                enabled
        """
        allow_deep = (
            allow_deep if allow_deep is not None else self.allow_deep_selection
        )
        idxrs = dict(**(indexers if indexers else {}), **indexers_kwargs)

        # Split by those for deep selection and those for this group
        deep_idxrs = {k: v for k, v in idxrs.items() if k not in self._dims}
        idxrs = {k: v for k, v in idxrs.items() if k in self._dims}

        if deep_idxrs and not allow_deep:
            _dim_names = ", ".join(self._dims)
            raise ValueError(
                f"Deep indexing is not allowed for {self.logstr}, but got "
                "indexers that don't match any of its dimension "
                f"names: {_dim_names}.  You can change this behavior using "
                "the allow_deep_selection property or the class "
                "variable LDG_ALLOW_DEEP_SELECTION."
            )

        return idxrs, deep_idxrs

    def _combine(
        self,
        cont_names: xr.DataArray,
        *,
        combination_method: str,
        shallow_indexers: dict,
        deep_indexers: dict,
        by_index: bool,
        **sel_kwargs,
    ) -> xr.Dataset:
        """Combine the given objects by the specified method. If deep indexers
        are given, apply the deep indexing on each of the members.

        This method receives a labelled array of container names, on which the
        selection already took place. The aim is now to align the objects these
        names refer to, including their coordinates, and thereby construct an
        array that contains both the dimensions given by the ``cont_names``
        array and each members' data dimensions.

        Available combination methods are based either on `xarray.merge`
        operations or `xarray.concat` along each dimension. For both these
        combination methods, the members of this group need to be prepared such
        that the operation can be applied, i.e.: they need to already be in an
        array capable of that operation and they need to directly or indirectly
        preserve coordinate information.

        For that purpose, an object-array is constructed that has the same
        shape as the given ``cont_names``. As the `xarray.Dataset` and
        `xarray.DataArray` types have issues with handling array-like objects
        in object arrays, this is done via a `numpy.ndarray`.

        Args:
            cont_names (xr.DataArray): The pre-selected member map object, i.e.
                a labelled array containing names of the desired members that
                are to be combined.
            combination_method (str): How to combine them: concat, try_concat,
                or merge. Concatenation will allow preserving the dtype of the
                underlying data.
            shallow_indexers (dict): Indexer arguments that were used for the
                group member selection.
            deep_indexers (dict): Indexer arguments for deep selection to be
                done before combination.
            by_index (bool): Whether the deep indexing should take place by
                index; if False, will use label-based selection.
            **sel_kwargs: Passed on to ``.sel`` or ``.isel``.

        Returns:
            xr.Dataset: The data of the members from ``cont_names``, combined
                using the given combination method.

        Raises:
            ValueError: On invalid combination method; on conflicting
                coordinate information on group-level and member-level.
            KeyError: In ``concat`` mode, upon missing members.
        """

        def get_cont(
            name: str, combination_method: str
        ) -> Tuple[Union[XrDataContainer, None], str]:
            """Retrieve the container from the group, potentially changing the
            combination method. If no container could be found, returns None,
            which denotes that further processing should be skipped
            """
            try:
                cont = self[name]

            except KeyError as err:
                if combination_method != "concat":
                    # Failing is ok. But cannot do anything else here
                    return None, combination_method

                # Otherwise, should raise!
                raise ItemAccessError(
                    self,
                    key=name,
                    suffix=(
                        "Make sure that the member can be found under this "
                        "name or change the combination method to 'merge' or "
                        "'try_concat'."
                    ),
                ) from err

            else:
                return cont, combination_method

        def process_cont(cont, coords) -> Tuple[xr.DataArray, dict]:
            """Process the given container and coordinates into a data array;
            this applies selection along container dimensions that overlap with
            the group dimensions as well as deep selection.
            """

            def all_equal(x, y) -> bool:
                """Check element-wise equality of two arrays."""
                try:
                    if x.dtype.kind == "f" or y.dtype.kind == "f":
                        return np.allclose(x, y)

                    else:
                        return np.array_equal(x, y)

                except Exception:
                    warnings.warn(
                        "Element-wise array comparison failed. Received: "
                        f"\nx: {x}\ny: {y}\nTreating them as not equal.",
                        DantroWarning,
                    )
                    return False

            # Apply the coordinates of the overlapping dimensions
            # (If there are none, select everything to get a DataArray)
            # Note that all coordinates in `coords` are scalar coordinates
            # since they describe a single entry within the member map.
            overlapping_indexers = {
                dim: coord for dim, coord in coords.items() if dim in cont.dims
            }

            darr = cont.sel(overlapping_indexers, drop=False)
            # The selection above is done with `drop=False`, because `coords`
            # might contain coordinates that were not passed as indexers to
            # `.sel` or `.isel` and therefore should not be dropped.

            # This is to ensure that the array that is used matches only a
            # single coordinate combination, i.e. one _point_ in the space
            # spanned by self.member_map.coords.
            # This operation _might_ increase memory usage temporarily, because
            # data from the same member is accessed multiple times and only one
            # coordinate combination is extracted from it (instead of selecting
            # all at once; which would however require a different architecture
            # and not allow using the convenient xarray interface).

            # Apply the deep indexers
            if deep_indexers:
                if by_index:
                    darr = darr.isel(deep_indexers, **sel_kwargs)
                else:
                    darr = darr.sel(deep_indexers, **sel_kwargs)

            # If the selection is done with `drop=True`, non-dimension
            # coordinates were already removed from `coords`. If there are
            # overlapping dimensions, these have to be applied again.
            dropped_shallow_indexers = {
                dim: coord
                for dim, coord in shallow_indexers.items()
                if dim not in coords and dim in cont.dims
            }

            if dropped_shallow_indexers:
                if by_index:
                    darr = darr.isel(dropped_shallow_indexers, **sel_kwargs)
                else:
                    darr = darr.sel(dropped_shallow_indexers, **sel_kwargs)

            # Finally, check for conflicting non-dimension coordinates in the
            # processed data-array (which were ignored for the selections
            # above).
            confl_coords = {
                d: (darr.coords[d].values, coords[d].values)
                for d in darr.coords
                if d not in darr.dims
                and d in coords
                and not all_equal(darr.coords[d], coords[d])
            }

            if confl_coords:
                # Provide some information on the conflicting coords
                infostr = (
                    "\n"
                    r"Name: {name}"
                    "\n"
                    r"Group-level coordinate: {c_grp}"
                    "\n"
                    r"Member-level coordinate: {c_mem}"
                )

                suffix = "\n".join(
                    [
                        infostr.format(name=k, c_grp=v[1], c_mem=v[0])
                        for k, v in confl_coords.items()
                    ]
                )

                raise ValueError(
                    "Conflicting non-dimension coordinate"
                    f"{'s' if len(confl_coords) > 1 else ''} found within a "
                    f"member of {self.logstr}:\n" + suffix
                )

            return darr, coords

        # If the pre-selected member map contains only a single entry or single
        # invalid entry, no data combination will have to be done, i.e. no
        # `combination_method` needs to be applied. This can speed up the
        # computation time a lot.
        if cont_names.size == 1 or (cont_names != "").sum() == 1:
            if cont_names.size == 1:
                # Select the group member directly.
                cont = self[cont_names.item()]
                # Applying `.squeeze()` here is equivalent to selecting the
                # single entry, since no dimension is of size > 1.
                coords = cont_names.squeeze().coords
            else:
                # Find container name and respective index in `cont_names`
                name = cont_names.where(cont_names != "", drop=True).item()
                idx = tuple(e.item() for e in np.where(cont_names != ""))
                # Extract container and coords
                cont = self[name]
                coords = cont_names[idx].coords

            # Process the container into a DataArray.
            darr, coords = process_cont(cont, coords)

            # Pass on 1D dimensions with the coordinate information
            darr = darr.expand_dims(
                {
                    d: ([c.values] if c.ndim == 0 else c.values)
                    for d, c in coords.items()
                    if d in cont_names.dims
                }
            )

            # Pass on non-dimension coordinates
            darr = darr.assign_coords(
                {
                    d: c.values
                    for d, c in coords.items()
                    if d not in cont_names.dims and d not in darr.coords
                }
            )

            return darr

        dsets = np.zeros(cont_names.shape, dtype="object")
        dsets.fill(dict())  # placeholders, ignored in xr.merge

        # Create an iterator over the container names (mirrors dsets iteration)
        names_iter = np.nditer(cont_names, flags=("multi_index", "refs_ok"))

        for name in names_iter:
            # Get the corresponding member container, potentially changing the
            # combination method
            cont, combination_method = get_cont(
                name.item(), combination_method
            )

            # Might not have been found; go to the next iteration
            if cont is None:
                continue

            # Get the coordinates for this member container and further process
            # the container into a DataArray.
            coords = cont_names[names_iter.multi_index].coords
            darr, coords = process_cont(cont, coords)

            # As it's easier to work on xr.Datasets than on xr.DataArray-like
            # objects, create a dataset from the container, using a temporary
            # name which will later be used to resolve it back to a DataArray
            dset = darr.to_dataset(name="_tmp_dset_name")

            # Now, need to expand the dimensions to accomodate the coordinates.
            # (Important for concatenation!)
            # Add the new dimensions in front and assign coordinates to them.
            dset = dset.expand_dims(
                {
                    d: ([c.values] if c.ndim == 0 else c.values)
                    for d, c in coords.items()
                    if d in cont_names.dims  # only for actual dimensions
                }
            )
            # NOTE While this creates a non-shallow copy of the data, there is
            #      no other way of doing this: a copy can only be avoided if
            #      the DataArray can re-use the existing variables – for the
            #      changes it needs to do to expand the dims, however, it will
            #      necessarily need to create a copy of the original data.
            #      Thus, we might as well let xarray take care of that instead
            #      of bothering with that ourselves ...

            # For coordinates that are not linked to a dimension in
            # `cont_names`, do not expand the dimensions but assign them as
            # non-dimension coordinates.
            dset = dset.assign_coords(
                {
                    d: c.values
                    for d, c in coords.items()
                    if d not in cont_names.dims
                    # If coord already there, don't try assigning it. The check
                    # for conflicting coords ensures that it's correct.
                    and d not in dset.coords
                }
            )

            # Done. Store it in the object-array of datasets
            dsets[names_iter.multi_index] = dset

        # Now ready to combine them.
        if combination_method == "concat":
            dset = self._combine_by_concatenation(dsets, dims=cont_names.dims)

        elif combination_method == "merge":
            dset = self._combine_by_merge(dsets)

        elif combination_method == "try_concat":
            try:
                dset = self._combine_by_concatenation(
                    dsets, dims=cont_names.dims
                )

            except Exception as exc:
                # NOTE The exception is now something other than a member
                #      missing, i.e. some numerical issue during concatenation
                # Try again with merging ...
                log.warning(
                    "Failed concatenation with %s: %s. Attempting merge ...",
                    exc.__class__.__name__,
                    exc,
                )
                dset = self._combine_by_merge(dsets)

        else:
            raise ValueError(
                f"Invalid combination_method argument: {combination_method}! "
                "Available methods: try_concat, concat, merge."
            )

        # Combined into one dataset now, with '_tmp_dset_name' data variable...
        # Convert back into a DataArray; can drop the temporary name now.
        darr = dset["_tmp_dset_name"]
        darr.name = None
        return darr

    @classmethod
    def _combine_by_merge(cls, dsets: np.ndarray) -> xr.Dataset:
        """Combine the given datasets by merging using `xarray.merge`.

        Args:
            dsets (np.ndarray): The object-dtype array of xr.Datasets that are
                to be combined.

        Returns:
            xr.Dataset: All datasets, aligned and combined via `xarray.merge`
        """
        log.debug("Combining %d datasets by merging ...", dsets.size)

        dset = xr.merge(dsets.flat)

        log.debug("Merge successful.")
        return dset

    @classmethod
    def _combine_by_concatenation(
        cls, dsets: np.ndarray, *, dims: TDims
    ) -> xr.Dataset:
        """Combine the given datasets by concatenation using `xarray.concat`
        and subsequent application along all dimensions specified in ``dims``.

        Args:
            dsets (np.ndarray): The object-dtype array of xr.Dataset objects
                that are to be combined by concatenation.
            dims (TDims): The dimension names corresponding to _all_ the
                dimensions of the ``dsets`` array.

        Returns:
            xr.Dataset: The dataset resulting from the concatenation
        """
        log.debug(
            "Combining %d datasets by concatenation along %d dimension%s ...",
            dsets.size,
            len(dsets.shape),
            "s" if len(dsets.shape) != 1 else "",
        )

        # Go over all dimensions and concatenate
        # This effectively reduces the dsets array by one dimension in each
        # iteration by applying the xr.concat function along the axis
        # NOTE np.apply_along_axis would be what is desired here, but that
        #      function unfortunately tries to cast objects to np.arrays
        #      which is not what we want here at all!
        #      Thus, there is one implemented in dantro.tools ...
        for dim_idx, dim_name in reversed(list(enumerate(dims))):
            log.debug(
                "Concatenating along axis '%s' (axis # %d) ...",
                dim_name,
                dim_idx,
            )

            dsets = apply_along_axis(
                xr.concat, axis=dim_idx, arr=dsets, dim=dim_name
            )

        log.debug("Concatenation successful.")

        # The single item in the now scalar array is the combined xr.Dataset
        return dsets.item()
