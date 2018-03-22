"""This module implements classes that serve as mixins for classes derived from base classes."""

import logging

import dantro.base

log = logging.getLogger(__name__)

# Local variables

# -----------------------------------------------------------------------------

class NumpyMixin(dantro.base.CollectionMixin, dantro.base.ItemAccessMixin):
    """This Mixin class implements the methods needed for getting the functionality of
    a numpy.ndarray. 
    """

    def __getattr__(self, attr_name):
        """Forward attributes that were not available in this class to data attribute, i.e. np.ndarray ..."""
        return getattr(self.data, attr_name)

    def __dir__(self):
        """List all attributes and methods"""
        return dir(self.data)

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item"""
        return "{} elements are in the {} dimensional data".format(self.data.size, self.data.ndim)

