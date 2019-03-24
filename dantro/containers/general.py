"""This module implements general specialisations of the BaseDataContainer"""

import logging
from collections.abc import MutableSequence, MutableMapping

from ..base import BaseDataContainer, CheckDataMixin
from ..base import ItemAccessMixin, CollectionMixin, MappingAccessMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ObjectContainer(ItemAccessMixin, BaseDataContainer):
    """Generically stores a Python object"""

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize an ObjectContainer, storing any Python object.
        
        Args:
            name (str): The name of this container
            data (list): The object to store
            **dc_kwargs: Additional arguments for container initialization
        """
        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the stored data"""
        return "{} stored, {}".format(type(self.data).__name__,
                                      super()._format_info())


class MutableSequenceContainer(CheckDataMixin, ItemAccessMixin,
                               CollectionMixin, BaseDataContainer,
                               MutableSequence):
    """The MutableSequenceContainer stores data that is sequence-like"""

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (MutableSequence, list)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'warn'


    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize a MutableSequenceContainer, storing data that is
        sequence-like.
        
        Args:
            name (str): The name of this container
            data (list): The sequence-like data to store
            **dc_kwargs: Additional arguments for container initialization
        """
        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.

    def insert(self, idx: int, val) -> None:
        """Insert an item at a given position. The first argument is the index 
        of the element before which to insert, so a.insert(0, x) inserts at 
        the front of the list, and a.insert(len(a), x) is equivalent to 
        a.append(x).
        
        Args:
            idx (int): The index before which to insert
            val: The value to insert
        """
        self.data.insert(idx, val)


class MutableMappingContainer(CheckDataMixin, MappingAccessMixin,
                              BaseDataContainer, MutableMapping):
    """The MutableMappingContainer stores mutable mapping data, e.g. dicts"""

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (MutableMapping, dict)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'warn'

    def __init__(self, *, name: str, data=None, **dc_kwargs):
        """Initialize a MutableMappingContainer, storing mapping data.
        
        Args:
            name (str): The name of this container
            data: The mapping-like data to store. If not given, an empty dict
                is created
            **dc_kwargs: Additional arguments for container initialization
        """
        # Supply a default value for the data, if none was given
        if data is None:
            data = {}

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
