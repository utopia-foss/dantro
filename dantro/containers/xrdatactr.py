"""This module implements specializations of the
:py:class:`~dantro.base.BaseDataContainer` class that make use of the xarray
package to represent the underlying data.
"""

import copy
import logging
from typing import Sequence, Tuple, Union

import numpy as np

from .._import_tools import LazyLoader
from ..abc import AbstractDataProxy
from ..base import BaseDataContainer, CheckDataMixin, ItemAccessMixin
from ..mixins import ComparisonMixin, ForwardAttrsToDataMixin, NumbersMixin
from ..utils import Link, extract_coords, extract_dim_names

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")

# -----------------------------------------------------------------------------


class XrDataContainer(
    ForwardAttrsToDataMixin,
    NumbersMixin,
    ComparisonMixin,
    CheckDataMixin,
    ItemAccessMixin,
    BaseDataContainer,
):
    """The XrDataContainer stores numerical :py:class:`xarray.DataArray` data
    associated with dimensions, coordinates, and attributes.
    """

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (
        "xarray.DataArray",
        np.ndarray,
    )
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = "raise"

    # Custom class variables for customizing XrDataContainer ..................
    _XRC_DIMS_ATTR = "dims"
    """Define as class variable the name of the attribute that determines the
    dimensions of the :py:class:`xarray.DataArray`"""

    _XRC_DIM_NAME_PREFIX = "dim_name__"
    """Attributes prefixed with this string can be used to set names for
    specific dimensions. The prefix should be followed by an integer-parsable
    string, e.g. ``dim_name__0`` would be the dimension name for the 0th dim.
    """

    _XRC_COORDS_ATTR_PREFIX = "coords__"
    """Attributes prefixed with this string determine the coordinate values for
    a specific dimension. The prefix should be followed by the *name* of the
    dimension, e.g. ``coord__time``. The values are interpreted according to
    the default coordinate mode or, if given, the ``coord_mode__*`` attribute.
    """

    _XRC_COORDS_MODE_DEFAULT = "values"
    """The default mode by which coordinates are interpreted"""

    _XRC_COORDS_MODE_ATTR_PREFIX = "coords_mode__"
    """Prefix for the coordinate mode if a custom mode is to be used"""

    _XRC_INHERIT_CONTAINER_ATTRIBUTES = True
    """Whether to inherit the other container attributes"""

    _XRC_STRICT_ATTR_CHECKING = True
    """Whether to use strict attribute checking; throws errors if there are
    container attributes available that match the prefix but don't match a
    valid dimension name. Can be disabled for speed improvements."""

    # .........................................................................

    def __init__(
        self,
        *,
        name: str,
        data: Union[np.ndarray, "xarray.DataArray"],
        dims: Sequence[str] = None,
        coords: dict = None,
        extract_metadata: bool = True,
        apply_metadata: bool = True,
        **dc_kwargs,
    ):
        """Initialize a XrDataContainer and extract dimension and coordinate
        labels.

        Args:
            name (str): which name to give to the XrDataContainer
            data (Union[numpy.ndarray, xarray.DataArray]): The data to store;
                anything that an :py:class:`xarray.DataArray` can take.
            dims (Sequence[str], optional): The dimension names.
            coords (dict, optional): The coordinates. The keys of this dict
                have to correspond to the dimension names.
            extract_metadata (bool, optional): If True, missing ``dims`` or
                ``coords`` arguments are tried to be populated from the
                container attributes.
            apply_metadata (bool, optional): Whether to apply the extracted
                or passed ``dims`` and ``coords`` to the underlying data.
                This might not be desired in cases where the given ``data``
                already is a labelled :py:class:`xarray.DataArray` or where
                the data is a proxy and the labelling should be postponed.
            **dc_kwargs: passed to parent
        """

        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug(
                "Received a list as `data` argument to %s '%s'. "
                "Calling np.array on it ...",
                self.classname,
                name,
            )
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
        return "{dtype:}, {shape:}, {sup:}".format(
            dtype=self.dtype,
            shape=self._format_shape(),
            sup=super()._format_info(),
        )

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
            return "({})".format(
                ", ".join(["{}: {}".format(*kv) for kv in sizes])
            )
        return f"shape {self.shape}"

    def __len__(self) -> int:
        """Length of the underlying data, i.e. first entry in shape"""
        return self.shape[0]

    def copy(self, *, deep: bool = True):
        """Return a new object with a copy of the data. The copy is deep if not
        specified otherwise.

        Args:
            deep (bool, optional): Whether the copy is deep

        Returns:
            XrDataContainer: A (deep) copy of this object.
        """
        log.debug("Creating copy of %s ...", self.logstr)

        return self.__class__(
            name=self.name + "_copy",
            data=(
                copy.deepcopy(self._data) if deep else copy.copy(self._data)
            ),
            attrs=copy.deepcopy(self.attrs),
            # Carry over cache attributes, needed for proxy
            dims=self._dim_names,
            coords=self._dim_to_coords_map,
            # No need to extract or apply; if it is a proxy,
            # the metadata is passed; if it is an xarray, it
            # is already labelled and passed via `data`
            extract_metadata=False,
            apply_metadata=False,
        )

    def save(self, path: str, **save_kwargs):
        """Saves the XrDataContainer to a file by invoking the ``.to_netcdf``
        method of the underlying data.

        The recommended file extension is ``.xrdc`` or ``.nc_da``, which are
        compatible with the xarray-based data loader.

        .. warning::

            This does NOT store container attributes!

        Args:
            path (str): The path to save the file at
            **save_kwargs: Passed to ``.no_netcdf`` method call
        """
        self.to_netcdf(path, **save_kwargs)

    # Methods to extract and apply metadata ...................................

    def _extract_metadata(self):
        """Extracts metadata from the container attributes and stores them
        in the ``_dim_names`` and ``_dim_to_coords_map`` cache attributes.
        """
        log.trace("Extracting metadata for labelling %s ...", self.logstr)

        # First: the dimension names
        if self._dim_names is None:
            try:
                dims = extract_dim_names(
                    self.attrs,
                    ndim=self.ndim,
                    attr_name=self._XRC_DIMS_ATTR,
                    attr_prefix=self._XRC_DIM_NAME_PREFIX,
                )

            except Exception as exc:
                raise type(exc)(
                    "Failed extracting dimension names from the "
                    f"attributes of {self.logstr}! {exc}"
                ) from exc

            else:
                self._dim_names = dims

        # With dimension names being cached, try extracting coordinates.
        if self._dim_to_coords_map is None:
            coords = extract_coords(
                self,
                mode="attrs",
                dims=self._dim_names,
                # Attribute names and prefixes
                strict=self._XRC_STRICT_ATTR_CHECKING,
                coords_attr_prefix=self._XRC_COORDS_ATTR_PREFIX,
                mode_attr_prefix=self._XRC_COORDS_MODE_ATTR_PREFIX,
                default_mode=self._XRC_COORDS_MODE_DEFAULT,
            )
            self._dim_to_coords_map = coords

    def _inherit_attrs(self):
        """Carry over container attributes to the data array attributes.

        This does not include container attributes that are used for extracting
        metadata; it makes no sense to have them in the attributes of the
        already labelled :py:class:`xarray.DataArray`.
        """

        def skip(attr_name: str) -> bool:
            return (
                attr_name == self._XRC_DIMS_ATTR
                or attr_name.startswith(self._XRC_DIM_NAME_PREFIX)
                or attr_name.startswith(self._XRC_COORDS_ATTR_PREFIX)
                or attr_name.startswith(self._XRC_COORDS_MODE_ATTR_PREFIX)
            )

        for attr_name, attr_val in self.attrs.items():
            if not skip(attr_name):
                self.data.attrs[attr_name] = attr_val

    def _apply_metadata(self):
        """Applies the cached metadata to the underlying
        :py:class:`xarray.DataArray`
        """
        # Make sure that data is an xarray
        if not isinstance(self.data, xr.DataArray):
            self._data = xr.DataArray(self.data)

        # Carry over the name (if the data itself is unnamed)
        if not self._data.name:
            self._data.name = self.name

        # Set the dimension names
        if self._dim_names:
            # Create a mapping from old to new names, then apply it
            new_names = {
                old: new
                for old, new in zip(self.data.dims, self._dim_names)
                if new is not None
            }

            log.trace("Renaming dimensions:  %s", new_names)
            self._data = self.data.rename(new_names)

        # Set the coordinates
        if self._dim_to_coords_map:
            log.trace("Associating coordinates:  %s", self._dim_to_coords_map)

            for dim_name, coords in self._dim_to_coords_map.items():
                # Need to handle links differently
                if isinstance(coords, Link):
                    # The target object is another DataContainer, which can not
                    # be used for association. Thus, just pass the raw data...
                    coords = np.array(coords.target_object)

                # Can associate now.
                try:
                    self.data.coords[dim_name] = coords

                except Exception as err:
                    raise ValueError(
                        f"Could not associate coordinates {coords} for "
                        f"dimension '{dim_name}' due to a "
                        f"{err.__class__.__name__}: {err}."
                    ) from err

        # Now write the rest of the attributes of the dataset to the xarray
        if self._XRC_INHERIT_CONTAINER_ATTRIBUTES:
            self._inherit_attrs()

        # Now set the flag that metadata was applied
        self._metadata_was_applied = True

    def _postprocess_proxy_resolution(self):
        """Only invoked from
        :py:class:`~dantro.mixins.proxy_support.ProxySupportMixin`, which have
        to be added to the class specifically. This function takes care to
        apply the potentially existing metadata *after* the proxy was resolved.
        """
        self._apply_metadata()

    def _parse_sizes_from_metadata(self) -> Sequence[Tuple[str, int]]:
        """Invoked from _format_shape when no metadata was applied but the
        dimension names are available. Should return data in the same form as
        ``xr.DataArray.sizes.items()`` does.
        """
        # Iterate over dimension names and shapes ...
        it = enumerate(zip(self._dim_names, self.shape))

        # ... and use the name from the metadata unless the name was None
        # which is a placeholder for "don't rename this dimension", in which
        # case it should be named via default names
        return tuple((n if n else f"dim_{i}", l) for i, (n, l) in it)
