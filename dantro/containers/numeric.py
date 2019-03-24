"""This module implements specializations of the BaseDataContainer class
that focus on holding numerical, array-like data"""

import logging

import numpy as np

from ..base import BaseDataContainer, ItemAccessMixin, CheckDataMixin
from ..mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class NumpyDataContainer(ForwardAttrsToDataMixin, NumbersMixin,
                         ComparisonMixin, CheckDataMixin, ItemAccessMixin,
                         BaseDataContainer):
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
        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug("Received a list as `data` argument to %s '%s'. "
                      "Calling np.array on it ...", self.classname, name)
            data = np.array(data)

        #initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item

        In this case, the dtype and shape of the stored data is returned. Note
        that this relies on the ForwardAttrsToDataMixin.
        """
        return "{}, shape {}, {}".format(self.dtype, self.shape,
                                         super()._format_info())

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


# .............................................................................

class XrContainer(BaseDataContainer):

    def __init__(self, **kwargs):
        raise NotImplementedError
