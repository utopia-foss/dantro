"""This module implements specialisations of the BaseDataContainer class."""

import warnings
import logging
from collections.abc import MutableSequence, MutableMapping

import numpy as np

from dantro.base import BaseDataContainer, ItemAccessMixin, CollectionMixin, MappingAccessMixin, CheckDataMixin
from dantro.mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# General container specialisations

class ObjectContainer(ItemAccessMixin, BaseDataContainer):
    """Generically stores a Python object"""

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize an ObjectContainer, storing any Python object.
        
        Args:
            name (str): The name of this container
            data (list): The object to store
            **dc_kwargs: Additional arguments for container initialization
        """

        log.debug("ObjectContainer.__init__ called.")

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("ObjectContainer.__init__ finished.")

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the stored data"""
        return "stored type: " + str(type(self.data))


class MutableSequenceContainer(CheckDataMixin, ItemAccessMixin, CollectionMixin, BaseDataContainer, MutableSequence):
    """The MutableSequenceContainer stores data that is sequence-like"""

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (MutableSequence, list)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'warn'


    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialize a MutableSequenceContainer, storing data that is sequence-like.
        
        Args:
            name (str): The name of this container
            data (list): The sequence-like data to store
            **dc_kwargs: Additional arguments for container initialization
        """

        log.debug("MutableSequenceContainer.__init__ called.")

        # Initialize with parent method
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


class MutableMappingContainer(CheckDataMixin, MappingAccessMixin, BaseDataContainer, MutableMapping):
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

        log.debug("MutableMappingContainer.__init__ called.")

        # Supply a default value for the data, if none was given
        if data is None:
            data = {}

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("MutableMappingContainer.__init__ finished.")


# -----------------------------------------------------------------------------
# Specialised containers

class NumpyDataContainer(ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin, CheckDataMixin, ItemAccessMixin, BaseDataContainer):
    """The NumpyDataContainer stores numerical array-shaped data.

    Specifically: it is made for use with the np.ndarray class.
    """

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (np.ndarray,)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'raise'

    def __init__(self, *, name: str, data: np.ndarray, **dc_kwargs):
        """Initialize a NumpyDataContainer, storing data that is ndarray-like.
        
        Arguments:
            name (str): The name of this container
            data (np.ndarray): The numpy data to store
            **dc_kwargs: Additional arguments for container initialisation
        """

        log.debug("NumpyDataConainer.__init__ called.")

        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug("Received a list as `data` argument to %s '%s'. "
                      "Calling np.array on it ...", self.classname, name)
            data = np.array(data)

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

    def copy(self):
        """Return a copy of this NumpyDataContainer.

        NOTE that this will create copies of the stored data.
        """
        log.debug("Creating copy of %s ...", self.logstr)
        return self.__class__(name=self.name + "_copy",
                              data=self.data.copy(),
                              attrs={k:v for k, v in self.attrs})

    # .........................................................................
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
