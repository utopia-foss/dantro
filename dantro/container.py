"""This module implements specialisations of the BaseDataContainer class."""

import warnings
import logging
from collections.abc import MutableSequence, MutableMapping

from dantro.base import BaseDataContainer, ItemAccessMixin, CollectionMixin, MappingAccessMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class MutableSequenceContainer(ItemAccessMixin, CollectionMixin, BaseDataContainer, MutableSequence):
    """The MutableSequenceContainer stores data that is sequence-like"""

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialise a MutableSequenceContainer, storing data that is sequence-like.

        NOTE: There is no check if the given data is actually a sequence!
        
        Args:
            name (str): The name of this container
            data (list): The sequence-like data to store
            **dc_kwargs: Description
        """

        log.debug("MutableSequenceContainer.__init__ called.")

        # Perform a check whether the data is actually a mutable sequence
        if not isinstance(data, MutableSequence):
            warnings.warn("The data given to {} '{}' was not identified as a "
                          "MutableSequence, but as '{}'. Initialisation will "
                          "work, but be informed that there might be errors "
                          "later on.".format(self.classname, name, type(data)),
                          UserWarning)

        # Initialise with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("MutableSequenceContainer.__init__ finished.")

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

    def convert_to(self, TargetCls, **target_init_kwargs):
        """With this method, a TargetCls object can be created from this
        particular container instance.
        
        Conversion might not be possible if TargetCls requires more information
        than is available in this container.
        """
        return TargetCls(name=self.name, attrs=self.attrs, data=self.data,
                         **target_init_kwargs)

# -----------------------------------------------------------------------------

class MutableMappingContainer(MappingAccessMixin, BaseDataContainer, MutableMapping):

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialise a MutableMappingContainer, storing mapping data.

        NOTE: There is no check if the given data is actually a mapping!
        
        Args:
            name (str): The name of this container
            data: The mapping-like data to store
            **dc_kwargs: Additional arguments for container initialisation
        """

        log.debug("MutableMappingContainer.__init__ called.")

        # Perform a check whether the data is actually a mutable sequence
        if not isinstance(data, MutableMapping):
            warnings.warn("The data given to {} '{}' was not identified as a "
                          "MutableMapping, but as '{}'. Initialisation will "
                          "work, but be informed that there might be errors "
                          "later on.".format(self.classname, name, type(data)),
                          UserWarning)

        # Initialise with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("MutableMappingContainer.__init__ finished.")
