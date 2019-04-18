"""Specializations of IndexedDataGroup that let groups behave as if the members
were labelled.

.. note::

  The interface here is not stable. In the long term it will be attempted to
  make these classes use an xr.DataArray representation internally such that
  the capabilities of that class can be re-used. This is currently only
  hindered by difficulties with storing array-like data _as objects_ in an
  object array, i.e. without xr.Variable attempting to coerce types.
  See https://github.com/pydata/xarray/issues/2097 for more info.
"""

import abc
import logging
from typing import Tuple, Dict, Sequence

import xarray as xr

from .ordered import IndexedDataGroup

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class LabelledDataGroup(IndexedDataGroup):
    """A group that assumes that the members it contains can be labelled
    with dimension names and coordinates.
    """
    # The dimensions of the underlying members. Can also be set dynamically
    LDG_DIMS = None

    def __init__(self, *args, **kwargs):
        """Initialize a TimeSeriesGroup"""
        super().__init__(*args, **kwargs)

        # Initialize cache members, populated later
        self._dims = self.LDG_DIMS
        self._coords = None

    # Interface ...............................................................

    @property
    def dims(self) -> Tuple[str]:
        """The dimension names of this group"""
        return self._dims

    @property
    @abc.abstractmethod
    def coords(self) -> Dict[str, Sequence[int]]:
        """Returns a mapping from dimension names to a sequence of coordinates
        for each dimension.
        """

    @abc.abstractmethod
    def sel(self, **indexer_kwargs) -> xr.DataArray:
        """Return a labelled array with a coordinate-selected subset of
        members of this group.
        """
    
    @abc.abstractmethod
    def isel(self, **indexer_kwargs) -> xr.DataArray:
        """Return a labelled array with a index-selected subset of
        members of this group.
        """


# -----------------------------------------------------------------------------

class TimeSeriesGroup(LabelledDataGroup):
    """A time-series group assumes that each stored member refers to one
    point in time, where the name is to be interpreted as the time coordinate.

    It allows basic selection features, but currently has a limited interface.
    See the note on the `~dantro.groups.labelled` module for more info.
    """
    # There is only one dimension: time
    LDG_DIMS = ('time',)


    # Implement abstract methods ..............................................

    @property
    def coords(self) -> Dict[str, Sequence[int]]:
        """Returns a mapping from dimension names to a sequence of coordinates
        for each dimension; in this case: time.
        """
        return dict(time=list(self.keys_as_int()))

    def sel(self, *, time: int):
        """Returns a _single_ element for the given time coordinate.

        .. note::

            This function has a limited feature set as it does not return a
            properly labelled array but only a single element or a sequence of
            elements. In later versions, this method will behave in the same
            way a `xr.DataArray.sel` method does.
        """
        if not isinstance(time, int):
            raise NotImplementedError("Cannot yet do more fancy things in the "
                                      "{}.sel method. Sorry!"
                                      "".format(self.classname))
            
        return self[time]
    
    def isel(self, *, time: int):
        """Return a _single_ element for the given time index.

        .. note::

            This function has a limited feature set as it does not return a
            properly labelled array but only a single element or a sequence of
            elements. In later versions, this method will behave in the same
            way a `xr.DataArray.isel` method does.
        """
        if not isinstance(time, int):
            raise NotImplementedError("Cannot yet do more fancy things in the "
                                      "{}.isel method. Sorry!"
                                      "".format(self.classname))

        return self[self.key_at_idx(time)]
