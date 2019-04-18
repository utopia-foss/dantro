"""This module implements general mixin classes for containers and groups"""

import logging

# Local variables
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ForwardAttrsMixin:
    """This Mixin class forwards all calls to unavailable attributes to a
    certain other attribute, specified by ``FORWARD_ATTR_TO`` class variable.
    """
    # The name of the existing attribute to forward to. NOTE: Cannot be None.
    FORWARD_ATTR_TO = None

    # Attributes to not forward
    FORWARD_ATTR_EXCLUDE = ()

    # If set, the only attributes to be forwarded
    FORWARD_ATTR_ONLY = None

    def __getattr__(self, attr_name: str):
        """Forward attributes that were not available in this class to some
        other attribute of the group or container.
        
        Args:
            attr_name (str): The name of the attribute that was tried to be
                accessed but was not available in ``self``.
        
        Returns:
            The attribute ``attr_name`` of getattr(self, self.FORWARD_ATTR_TO)
        """
        if self.FORWARD_ATTR_TO is None:
            raise AttributeError(attr_name)

        if attr_name in self.FORWARD_ATTR_EXCLUDE:
            raise AttributeError(attr_name)

        if (    self.FORWARD_ATTR_ONLY is not None
            and attr_name not in self.FORWARD_ATTR_ONLY):
            raise AttributeError(attr_name)

        # Invoke the pre-hook
        self._forward_attr_pre_hook(attr_name)

        # Get the attribute
        a = getattr(getattr(self, self.FORWARD_ATTR_TO), attr_name)

        # Pass it through the post-hook
        return self._forward_attr_post_hook(a)

    def _forward_attr_pre_hook(self, attr_name: str=None):
        """Invoked before attribute forwarding occurs"""
        pass
    
    def _forward_attr_post_hook(self, attr):
        """Invoked before attribute forwarding occurs"""
        return attr

class ForwardAttrsToDataMixin(ForwardAttrsMixin):
    """This Mixin class forwards all calls to unavailable attributes to the
    ``data`` attribute (a property) and thus allows to replace all behaviour
    that is not implemented in the group or container with that of the stored
    data.
    """

    FORWARD_ATTR_TO = 'data'

