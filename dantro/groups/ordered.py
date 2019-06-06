"""In this module, the BaseDataGroup is specialized for holding members in a
specific order.
"""

import logging
import collections
from typing import Generator, List

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

    .. note::

      Albeit the members of this group are ordered, item access still refers
      to the _names_ of the members, not their index within the sequence!
    """
    # Use an orderable dict for storage, i.e. something like OrderedDict, but
    # where it's not sorted by insertion order but by key.
    _STORAGE_CLS = IntOrderedDict

    # The child class should not necessarily be of the same type as this class.
    _NEW_GROUP_CLS = OrderedDataGroup


    # Advanced key access .....................................................
    
    def key_at_idx(self, idx: int) -> str:
        """Get a key by its index within the container. Can be negative.
        
        Args:
            idx (int): The index within the member sequence
        
        Returns:
            str: The desired key
        
        Raises:
            IndexError: Index out of range
        """
        # Imitate indexing behaviour of lists, tuples, ...
        if not isinstance(idx, int):
            raise TypeError("Expected integer, got {} '{}'!"
                            "".format(type(idx), idx))

        if idx >= len(self) or idx < -len(self):
            raise IndexError("Index {:d} out of range for {} with {} members!"
                             "".format(idx, self.logstr, len(self)))

        # Wraparound negative
        idx = idx if idx >= 0 else idx%len(self)

        for i, k in enumerate(self.keys()):
            if i == idx:
                return k

    def keys_as_int(self) -> Generator[int, None, None]:
        """Returns an iterator over keys as integer values"""
        for k in self.keys():
            yield int(k)


    # Customizations of parent methods ........................................

    def _ipython_key_completions_(self) -> List[int]:
        """For ipython integration, return a list of available keys.

        Unlike the BaseDataGroup method, which returns a list of strings, this
        returns a list of integers.
        """
        return list(self.keys_as_int())
