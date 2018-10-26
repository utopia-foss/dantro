"""This module implements general mixin classes for containers and groups"""

import logging

# Local variables
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ForwardAttrsToDataMixin():
    """This Mixin class forwards all calls to attributes to the .data attribute
    and thus allows to replace all behaviour that is not implemented in the
    group or container with that of the stored data.
    """

    def __getattr__(self, attr_name: str):
        """Forward attributes that were not available in this class to the
        group's or container's data attribute.
        
        Args:
            attr_name (str): The name of the attribute that is tried to access
        
        Returns:
            Determined by the stored data's behaviour
        """
        return getattr(self.data, attr_name)
