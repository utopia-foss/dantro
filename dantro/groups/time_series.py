"""Implements LabelledDataGroup specializations for time series"""

from .labelled import LabelledDataGroup
from .ordered import IndexedDataGroup

# -----------------------------------------------------------------------------

class TimeSeriesGroup(LabelledDataGroup, IndexedDataGroup):
    """A time-series group assumes that each stored member refers to one
    point in time, where the name is to be interpreted as the time coordinate.

    For more information on selection methods, see:

        * ~`dantro.groups.LabelledDataGroup.sel`
        * ~`dantro.groups.LabelledDataGroup.isel`
    """
    # There is only one dimension in a TimeSeriesGroup: time
    LDG_DIMS = ('time',)

    # Names are expected to be the time coordinates of each container
    LDG_EXTRACT_COORDS_FROM = 'name'


class HeterogeneousTimeSeriesGroup(TimeSeriesGroup):
    """This extends the ~`dantro.groups.TimeSeriesGroup` by configuring it
    such that it retrieves its coordinates not from the name of the members
    contained in it but from _their_ data.

    It still manages only the ``time`` dimension, which is now overlapping with
    the ``time`` dimension in the members of this group. However, the
    ~`dantro.groups.LabelledDataGroup` can handle this overlap and provides a
    uniform selection interface that allows combining this heterogeneously
    stored data.

    This becomes especially useful in cases where the members of this group
    store data with the following properties:

        * Potentially different coordiantes than the coordinates of other
          members of the group.
        * Containing time information for more than a single time coordinate
        * No guarantee for overlaps between ``time`` dimension or any other
          dimension.

    As such it is suitable to work with data that represents ensembles that
    frequently change not only their size but also their identifying labels.
    Additionally, it supports them not being stored in regular intervals but
    only upon a change in coordinates.
    """
    # Coordinates are extracted from the data directly, inspecting only the
    # group-level dimensions (here: ``time``)
    LDG_EXTRACT_COORDS_FROM = 'data'
