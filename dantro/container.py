"""This module implements specialisations of the BaseDataContainer class."""

import warnings
import logging
from collections.abc import MutableSequence, MutableMapping
import numpy as np
from dantro.base import BaseDataContainer, ItemAccessMixin, CollectionMixin, MappingAccessMixin
from dantro.mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin
# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# General container specialisations

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


class MutableMappingContainer(MappingAccessMixin, BaseDataContainer, MutableMapping):

    def __init__(self, *, name: str, data=None, **dc_kwargs):
        """Initialise a MutableMappingContainer, storing mapping data.
        
        NOTE: There is no check if the given data is actually a mapping!
        
        Args:
            name (str): The name of this container
            data: The mapping-like data to store. If not given, an empty dict
                is created
            **dc_kwargs: Additional arguments for container initialisation
        """

        log.debug("MutableMappingContainer.__init__ called.")

        # Perform a check whether the data is actually a mutable sequence
        if data is None:
            data = {}
            
        elif not isinstance(data, MutableMapping):
            warnings.warn("The data given to {} '{}' was not identified as a "
                          "MutableMapping, but as '{}'. Initialisation will "
                          "work, but be informed that there might be errors "
                          "later on.".format(self.classname, name, type(data)),
                          UserWarning)

        # Initialise with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("MutableMappingContainer.__init__ finished.")

    def convert_to(self, TargetCls, **target_init_kwargs):
        """With this method, a TargetCls object can be created from this
        particular container instance.
        
        Conversion might not be possible if TargetCls requires more information
        than is available in this container.
        """
        return TargetCls(name=self.name, attrs=self.attrs, data=self.data,
                         **target_init_kwargs)


# -----------------------------------------------------------------------------
# Specialised containers

class NumpyDataContainer(ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin, ItemAccessMixin, BaseDataContainer):
    """The NumpyDataContainer stores data that is numpy.ndarray-like"""

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize a NumpyDataContainer, storing data that is like a numpy.ndarray.
        
        Arguments:
            name (str) -- The name of this container
            data (np.ndarray) -- The numpy.ndarray-like data to store
            **dc_kwargs: Description
        """

        log.debug("NumpyDataConainer.__init__ called.")

        # check whether the data is a numpy.ndarray.dtype
        if not isinstance(data, np.ndarray):
            warnings.warn("The data given to {} '{}' was not identified as a "
                          "np.ndarray data, but as '{}'. Initialisation will "
                          "work, but be informed that there might be errors "
                          "later on.".format(self.classname, name, type(data)),
                          UserWarning)
        
        #initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("NumpyDataContainer.__init__ finished")

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item

        In this case, the dtype and shape of the stored data is returned. Note
        that this relies on the ForwardAttrsToDataMixin.
        """
        return "{}, shape {}".format(self.dtype, self.shape)

    def convert_to(self, TargetCls, **target_init_kwargs):
        """With this method, a TargetCls object can be created from this
        particular container instance.
        
        Conversion might not be possible if TargetCls requires more information
        than is available in this container.
        """
        return TargetCls(name=self.name, attrs=self.attrs, data=self.data,
                         **target_init_kwargs)

    def copy(self):
        """Return a copy of this NumpyDataContainer.

        NOTE that this will create copies of the stored data.
        """
        log.debug("Creating copy of %s ...", self.logstr)
        return self.__class__(name=self.name + "_copy",
                              data=self.data.copy(),
                              attrs={k:v for k, v in self.attrs})

    # Disallow usage of some unary functions (added by NumbersMixin) which
    # don't make sense with the np.ndarray data

    def __invert__(self):
        """Inverse value"""
        raise NotImplementedError("__invert__ not supported for {}!"
                                  "".format(self.logstr))

    def __complex__(self):
        """Complex value"""
        raise NotImplementedError("__complex__ not supported for {}!"
                                  "".format(self.logstr))

    def __int__(self):
        """Inverse value"""
        raise NotImplementedError("__int__ not supported for {}!"
                                  "".format(self.logstr))

    def __float__(self):
        """Float value"""
        raise NotImplementedError("__float__ not supported for {}!"
                                  "".format(self.logstr))

    def __round__(self):
        """Round value"""
        raise NotImplementedError("__round__ not supported for {}!"
                                  "".format(self.logstr))

    def __ceil__(self):
        """Ceil value"""
        raise NotImplementedError("__ceil__ not supported for {}!"
                                  "".format(self.logstr))
    
    def __floor__(self):
        """Floor value"""
        raise NotImplementedError("__floor__ not supported for {}!"
                                  "".format(self.logstr))
    
    def __trunc__(self):
        """Truncated to the nearest integer toward 0"""
        raise NotImplementedError("__trunc__ not supported for {}!"
                                  "".format(self.logstr))
