"""This module implements specialisations of the BaseDataContainer class."""

import warnings
import logging
from collections.abc import MutableSequence
from numpy import ndarray
from dantro.base import BaseDataContainer, ItemAccessMixin, CollectionMixin
from dantro.mixins import NumpyMixin
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

class NumpyDataContainer(NumpyMixin, BaseDataContainer):
    """The NumpyDataContainer stores data that is numpy.ndarray-like"""

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize a NumpyDataContainer, storing data that is like a numpy.ndarray.
        
        Arguments:
            name {str} -- The name of this container
            data {np.ndarray} -- The numpy.ndarray-like data to store
            **dc_kwargs: Description
        """

        log.debug("NumpyDataConainer.__init__ called.")

        # check whether the data is a numpy.ndarray.dtype
        if not isinstance(data, ndarray):
            warnings.warn("The data given to {} '{}' was not identified as a "
                          "MutableSequence, but as '{}'. Initialisation will "
                          "work, but be informed that there might be errors "
                          "later on.".format(self.classname, name, type(data)),
                          UserWarning)

        #initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("NumpyDataContainer.__init__ finished")

    def convert_to(self, TargetCls, **target_init_kwargs):
        """With this method, a TargetCls object can be created from this
        particular container instance.
        
        Conversion might not be possible if TargetCls requires more information
        than is available in this container.
        """
        return TargetCls(name=self.name, attrs=self.attrs, data=self.data,
                         **target_init_kwargs)


    def copy(self):
        return NumpyDataContainer(name=self.name, data=self.data)