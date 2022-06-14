"""This module implements general mixin classes for containers and groups"""

import logging
from typing import Sequence

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class ForwardAttrsMixin:
    """This Mixin class forwards all calls to unavailable attributes to a
    certain other attribute, specified by ``FORWARD_ATTR_TO`` class variable.

    By including naive ``__getstate__`` and ``__setstate__`` methods, classes
    that include this mixin remain pickleable.
    """

    FORWARD_ATTR_TO: str = None
    """The name of the existing attribute to forward to. For None, this behaves
    as if no forwarding would occur, i.e. as if ``__getattr__`` was not called.
    """

    FORWARD_ATTR_ONLY: Sequence[str] = None
    """If set, the only attributes to be forwarded"""

    FORWARD_ATTR_EXCLUDE: Sequence[str] = ()
    """Attributes to *not* forward. Evaluated after ``FORWARD_ATTR_ONLY``"""

    def __getstate__(self) -> dict:
        """Returns the object's ``__dict__``"""
        return self.__dict__

    def __setstate__(self, d: dict):
        """Sets the object's ``__dict__`` to the given one"""
        self.__dict__ = d

    def __getattr__(self, attr_name: str):
        """Forward attributes that were not available in this class to some
        other attribute of the group or container.

        Args:
            attr_name (str): The name of the attribute that was tried to be
                accessed but was not available in ``self``.

        Returns:
            The attribute ``attr_name`` of
            ``getattr(self, self.FORWARD_ATTR_TO)``
        """
        if self.FORWARD_ATTR_TO is None:
            raise AttributeError(attr_name)

        if (
            self.FORWARD_ATTR_ONLY is not None
            and attr_name not in self.FORWARD_ATTR_ONLY
        ):
            raise AttributeError(attr_name)

        if attr_name in self.FORWARD_ATTR_EXCLUDE:
            raise AttributeError(attr_name)

        # Invoke the pre-hook
        self._forward_attr_pre_hook(attr_name)

        # Get the attribute
        a = getattr(self._forward_attr_get_forwarding_target(), attr_name)

        # Pass it through the post-hook
        return self._forward_attr_post_hook(a)

    def _forward_attr_pre_hook(self, attr_name: str = None):
        """Invoked before attribute forwarding occurs"""
        pass

    def _forward_attr_get_forwarding_target(self):
        """Get the object that the attribute call is to be forwarded to"""
        return getattr(self, self.FORWARD_ATTR_TO)

    def _forward_attr_post_hook(self, attr):
        """Invoked before attribute forwarding occurs"""
        return attr


class ForwardAttrsToDataMixin(ForwardAttrsMixin):
    """This mixin class forwards all calls to unavailable attributes to the
    ``data`` attribute (a property) and thus allows to replace all behaviour
    that is not implemented in the group or container with that of the stored
    data.
    """

    FORWARD_ATTR_TO = "data"
