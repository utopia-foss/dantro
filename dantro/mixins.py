"""This module implements classes that serve as mixins for classes derived from base classes."""

import logging

import dantro.base

log = logging.getLogger(__name__)

# Local variables

# -----------------------------------------------------------------------------


class ProxyMixin:
    """This Mixin class overwrites the `data` property to allow proxy objects.

    A proxy object is a place keeper for data that is not yet loaded. It will
    only be loaded if `data` is directly accessed.
    """

    @property
    def data(self):
        """The container data. If the data is a proxy, this call will lead
        to the resolution of the proxy.
        
        Returns:
            The data stored in this container
        """
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug("Resolving %s for %s '%s' ...",
                      self._data.__class__.__name__,
                      self.classname, self.name)
            self._data = self._data.resolve()

        # Now, the data should be loaded and can be returned
        return self._data

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data
        
        Returns:
            bool: Whether the _currently_ stored data is a proxy object
        """
        return isinstance(self._data, dantro.base.BaseDataProxy)

    @property
    def proxy(self):
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


class NumpyProxyMixin(ProxyMixin):
    """Provides some numpy-specific proxy capabilities, i.e. an info string
    that takes care to not resolve the data."""
