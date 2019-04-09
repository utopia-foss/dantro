"""This module implements mixins that provide proxy support"""

import logging

import numpy as np

from ..abc import AbstractDataProxy

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

    # Which type to resolve the proxy to
    PROXY_RESOLVE_ASTYPE = None

    # Whether to retain the proxy object after resolving
    PROXY_RETAIN = False

    # Make sure the attribute where a retained proxy is stored is available
    _retained_proxy = None

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
                      self._data.classname, self.logstr)

            # Optionally, retain the proxy object. If not doing this, will go
            # out of scope
            if self.PROXY_RETAIN:
                log.debug("...retaining proxy object...")
                self._retained_proxy = self._data

            # Now, overwrite the _data attribute where the proxy used to be
            self._data = self.proxy.resolve(astype=self.PROXY_RESOLVE_ASTYPE)

            # Postprocess the resolved proxy with optional method
            if hasattr(self, '_postprocess_proxy_resolution'):
                self._postprocess_proxy_resolution()

        # Now, the data should be loaded and can be returned
        return self._data

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data
        
        Returns:
            bool: Whether the _currently_ stored data is a proxy object
        """
        return isinstance(self._data, AbstractDataProxy)

    @property
    def proxy(self) -> AbstractDataProxy:
        """If the data is proxy, returns the proxy data object without using 
        the .data attribute (which would trigger resolving the proxy); else 
        returns None.
        
        Returns:
            Union[AbstractDataProxy, None]: If the data is proxy, return the
                proxy object; else None.
        """
        if self.data_is_proxy:
            return self._data
        return self._retained_proxy

    def _format_info(self) -> str:
        """Adds an indicator to whether data is proxy to the info string"""
        if self.data_is_proxy:
            return "proxy, " + super()._format_info()
        return super()._format_info()


class Hdf5ProxyMixin(ProxyMixin):
    """Specialises the ProxyMixin to the capabilities of a Hdf5 Proxy, i.e. it
    allows access to the cached properties of the Hdf5DataProxy without
    resolving the proxy.
    """

    # Which type to resolve the proxy to
    PROXY_RESOLVE_ASTYPE = np.array

    @property
    def dtype(self) -> np.dtype:
        """Returns dtype, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.dtype
        return self.data.dtype
    
    @property
    def shape(self) -> tuple:
        """Returns shape, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.shape
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Returns ndim, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.ndim
        return self.data.ndim

    @property
    def size(self) -> int:
        """Returns size, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.size
        return self.data.size

    @property
    def chunks(self) -> tuple:
        """Returns chunks, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.chunks
        return self.data.chunks
