"""In this module, the BaseDataGroup is specialized for holding members in a
specific order.
"""

import logging
import collections
from typing import Generator

from ..base import BaseDataGroup
from ..mixins import IntegerItemAccessMixin
from ..utils import IntOrderedDict

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class OrderedDataGroup(BaseDataGroup, collections.abc.MutableMapping):
    """The OrderedDataGroup class manages groups of data containers, preserving
    the order in which they were added to this group.

    It uses an OrderedDict to associate containers with this group.
    """
    # Use OrderedDict for storage in insertion order
    _STORAGE_CLS = collections.OrderedDict

# -----------------------------------------------------------------------------

class IndexedDataGroup(IntegerItemAccessMixin, OrderedDataGroup):
    """The IndexedDataGroup class holds members that are of the same type and
    have names that can directly be interpreted as positive integers.

    Especially, this group maintains the correct order of members according to
    integer ordering.
    """
    # Use an orderable dict for storage, i.e. something like OrderedDict, but
    # where it's not sorted by insertion order but by key.
    _STORAGE_CLS = IntOrderedDict

    # The child class should not necessarily be of the same type as this class.
    _NEW_GROUP_CLS = OrderedDataGroup


    @property
    def min_key(self) -> str:
        """The smallest available key in this group"""
        for k in self.keys():
            return k
    
    @property
    def max_key(self) -> str:
        """The largest available key in this group"""
        for k in reversed(self.keys()):
            return k

    def keys_as_int(self) -> Generator[int, None, None]:
        """Returns an iterator over keys as integer values"""
        for k in self.keys():
            yield int(k)
