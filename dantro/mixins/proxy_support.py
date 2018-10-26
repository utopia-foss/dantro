"""This module implements mixins that provide proxy support"""

import logging

import numpy as np

from ..base import BaseDataProxy

# Local variables
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ProxyMixin:
    """This Mixin class overwrites the `data` property to allow proxy objects.

    It should be used to add support for certain proxy types to a container.

    A proxy object is a place keeper for data that is not yet loaded. It will
    only be loaded if `data` is directly accessed.
    """

    # If the CheckDataMixin is used, it should also allow proxy data
    DATA_ALLOW_PROXY = True
    # NOTE Depending on the order of how the mixins are given, this might
    # not have an effect. Make sure the proxy mixin is loaded after the
    # CheckDataMixin or the container class that uses the CheckDataMixin


    @property
    def data(self):
        """The container data. If the data is a proxy, this call will lead
        to the resolution of the proxy.
        
        Returns:
            The data stored in this container
        """
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug("Resolving %s for %s ...",
                      self.proxy.classname, self.logstr)
            self._data = self.proxy.resolve()

        # Now, the data should be loaded and can be returned
        return self._data

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data
        
        Returns:
            bool: Whether the _currently_ stored data is a proxy object
        """
        return isinstance(self._data, BaseDataProxy)

    @property
    def proxy(self) -> BaseDataProxy:
        """If the data is proxy, returns the proxy data object without using 
        the .data attribute (which would trigger resolving the proxy); else 
        returns None.
        
        Returns:
            Union[BaseDataProxy, None]: If the data is proxy, return the
                proxy object; else None.
        """
        if self.data_is_proxy:
            return self._data
        return None


class Hdf5ProxyMixin(ProxyMixin):
    """Specialises the ProxyMixin to the capabilities of a Hdf5 Proxy, i.e. it
    allows access to the cached `dtype` and `shape` properties of the
    Hdf5DataProxy without resolving the proxy.
    """

    @property
    def dtype(self) -> np.dtype:
        """Returns the NumpyDCs dtype, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.dtype
        return self.data.dtype
    
    @property
    def shape(self) -> tuple:
        """Returns the NumpyDCs shape, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.shape
        return self.data.shape


# -----------------------------------------------------------------------------
# For extending containers
