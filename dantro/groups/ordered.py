"""In this module, the BaseDataGroup is specialized for ordered members."""

import logging
import collections

from ..base import BaseDataGroup

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class OrderedDataGroup(BaseDataGroup, collections.abc.MutableMapping):
    """The OrderedDataGroup class manages groups of data containers, preserving
    the order in which they were added to this group.

    It uses an OrderedDict to associate containers with this group.
    """
    
    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("OrderedDataGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers,
                         StorageCls=collections.OrderedDict, **kwargs)

        # Done.
        log.debug("OrderedDataGroup.__init__ finished.")
