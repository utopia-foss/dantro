"""In this module, the BaseDataGroup is specialized for holding members in a
specific order.
"""

import collections
import logging
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

    It uses an :py:class:`collections.OrderedDict` to associate containers
    with this group.
    """

    # Use OrderedDict for storage in insertion order
    _STORAGE_CLS = collections.OrderedDict


# -----------------------------------------------------------------------------


class IndexedDataGroup(IntegerItemAccessMixin, OrderedDataGroup):
    """The IndexedDataGroup class holds members that are of the same type and
    have names that can directly be interpreted as positive integers.

    Especially, this group maintains the correct order of members according to
    integer ordering.

    To speed up element insertion, this group keeps track of recently added
    container names, which are then used as hints for subsequent insertions.

    .. note::

        Albeit the members of this group being ordered, item access still
        refers to the *names* of the members, not their index within the
        sequence!

    .. warning::

        With the underlying ordering mechanism of
        :py:class:`~dantro.utils.ordereddict.KeyOrderedDict`, the performance
        of this data structure is sensitive to the insertion order of elements.

        It is fastest for **in-order** insertions, where the complexity per
        insertion is constant (regardless of whether insertion order is
        ascending or descending).
        For **out-of-order** insertions, the whole key map may have to be
        searched, in which case the complexity scales with the number of
        elements in this group.

    .. hint::

        If experiencing trouble with the performance of this data structure,
        **sort elements before adding them to this group**.
    """

    # A dict of (key length -> last key inserted of that length), which is used
    # as an insertion hint when adding a container to this group
    __last_keys = None

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
            raise TypeError(f"Expected integer, got {type(idx)} '{idx}'!")

        if idx >= len(self) or idx < -len(self):
            raise IndexError(
                "Index {:d} out of range for {} with {} members!".format(
                    idx, self.logstr, len(self)
                )
            )

        # Wraparound negative
        idx = idx if idx >= 0 else idx % len(self)

        for i, k in enumerate(self.keys()):
            if i == idx:
                return k

    def keys_as_int(self) -> Generator[int, None, None]:
        """Returns an iterator over keys as integer values"""
        for k in self.keys():
            yield int(k)

    # Customizations of parent methods ........................................

    def _add_container_to_data(self, cont) -> None:
        """Adds a container to the underlying integer-ordered dictionary.

        Unlike the parent method, this uses
        :py:meth:`~dantro.utils.ordereddict.KeyOrderedDict.insert` in order to
        provide hints regarding the insertion position. It is optimised for
        insertion in *ascending* order.
        """
        # Keep track of insertion hints
        if self.__last_keys is None:
            self.__last_keys = dict()

        # Insert it, using hint information for names of this length
        self._data.insert(
            cont.name, cont, hint_after=self.__last_keys.get(len(cont.name))
        )

        # Update the hints for names of this length
        self.__last_keys[len(cont.name)] = cont.name

    def _ipython_key_completions_(self) -> List[int]:
        """For ipython integration, return a list of available keys.

        Unlike the BaseDataGroup method, which returns a list of strings, this
        returns a list of integers.
        """
        return list(self.keys_as_int())
