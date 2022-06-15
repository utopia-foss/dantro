"""This module implements mixins that provide proxy support"""

import copy
import logging
import warnings
from typing import Union

import numpy as np

from ..abc import AbstractDataProxy

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class ProxySupportMixin:
    """This mixin class overwrites the ``data`` property to allow and resolve
    proxy objects.

    It should be used to add support for certain proxy types to a container.

    A proxy object is a place holder for data that is not yet loaded. It will
    only be loaded if the ``data`` property is directly or indirectly accessed.
    """

    # If the CheckDataMixin is used, it should also allow proxy data
    DATA_ALLOW_PROXY = True
    # NOTE Depending on the order of how the mixins are given, this might
    # not have an effect. Make sure the proxy mixin is loaded after the
    # CheckDataMixin or the container class that uses the CheckDataMixin

    PROXY_RESOLVE_ASTYPE = None
    """Which type to resolve the proxy to"""

    PROXY_RETAIN = False
    """Whether to retain the proxy object after resolving"""

    PROXY_REINSTATE_FAIL_ACTION = "raise"
    """Behaviour upon failure of reinstating a proxy.

    Can be: ``raise``, ``warn``, ``log_warning``, ``log_debug``"""

    PROXY_REINSTATE_FOR_PICKLING = True
    """If true, populates the pickling state with the proxy instead of the data
    """

    # The attribute where a retained proxy is stored (individually for each
    # object, of course).
    _retained_proxy = None

    def __getstate__(self) -> dict:
        """If the data is no longer a proxy, but a proxy was retained, this
        overload adjusts the pickling state such that the proxy object is
        returned instead of the data that was resolved from it. This hels to
        reduce the file size of the pickle.
        """
        if (
            self.data_is_proxy
            or not self.PROXY_REINSTATE_FOR_PICKLING
            or (not self.data_is_proxy and self._retained_proxy is None)
        ):
            return super().__getstate__()

        # else: retrieve the state and adjust it to contain the retained proxy.
        # Need a shallow copy here because state might be __dict__ and its
        # mutability would change the state of self!
        state = copy.copy(super().__getstate__())
        log.debug("Using retained proxy for pickling of %s ...", self.logstr)
        state["_data"] = self._retained_proxy
        state["_retained_proxy"] = None
        return state

    @property
    def data(self):
        """The container data. If the data is a proxy, this call will lead
        to the resolution of the proxy.

        Returns:
            The data stored in this container
        """
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug(
                "Resolving %s for %s ...", self._data.classname, self.logstr
            )

            # Optionally, retain the proxy object. If not doing this, will go
            # out of scope
            if self.PROXY_RETAIN:
                log.debug("...retaining proxy object...")
                self._retained_proxy = self._data

            # Now, overwrite the _data attribute where the proxy used to be
            self._data = self.proxy.resolve(astype=self.PROXY_RESOLVE_ASTYPE)

            # Postprocess the resolved proxy with optional method
            if hasattr(self, "_postprocess_proxy_resolution"):
                log.debug("Calling proxy resolution postprocessing ...")
                self._postprocess_proxy_resolution()

        # Now, the data should be loaded and can be returned
        return self._data

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data

        Returns:
            bool: Whether the *currently* stored data is a proxy object
        """
        return isinstance(self._data, AbstractDataProxy)

    @property
    def proxy(self) -> Union[AbstractDataProxy, None]:
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

    def reinstate_proxy(self):
        """Re-instate a previously retained proxy object, discarding ``_data``."""
        if self.data_is_proxy:
            return

        if self._retained_proxy is None:
            msg = (
                f"Could not reinstate a proxy for {self.logstr} because there "
                "was no proxy retained. Was there one in the first place? "
                "Make sure the `PROXY_RETAIN` class variable is set to True. "
                "To control the behaviour of this message, change the "
                "PROXY_REINSTATE_FAIL_ACTION class variable."
            )
            action = self.PROXY_REINSTATE_FAIL_ACTION

            if action == "raise":
                raise ValueError(msg)

            elif action == "warn":
                warnings.warn(msg, RuntimeWarning)

            elif action in ("log_warning", "log_warn"):
                log.warning(msg)

            elif action == "log_debug":
                log.debug(msg)

            else:
                raise ValueError(
                    "Invalid PROXY_REINSTATE_FAIL_ACTION value "
                    "'{}'! Possible values: raise, warn, "
                    "log_warning, log_debug".format(action)
                )

        else:
            # All good. Reinstate the proxy
            self._data = self._retained_proxy
            log.debug(
                "Reinstated %s for data of %s.",
                self.proxy.classname,
                self.logstr,
            )

    def _format_info(self) -> str:
        """Adds an indicator to whether data is proxy to the info string.
        Additionally, the proxy tags are appended.
        """
        if self.data_is_proxy:
            tags = (
                " ({})".format(", ".join(self.proxy.tags))
                if self.proxy.tags
                else ""
            )
            return f"proxy{tags}, {super()._format_info()}"
        return super()._format_info()


class Hdf5ProxySupportMixin(ProxySupportMixin):
    """Specializes the
    :py:class:`~dantro.mixins.proxy_support.ProxySupportMixin` to the
    capabilities of :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy`, i.e. it
    allows access to the cached properties of the proxy object without
    resolving it.
    """

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
