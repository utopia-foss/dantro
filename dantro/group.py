"""In this module, BaseDataContainer specialisations that group data containers are implemented."""

import collections
import logging

from dantro.base import BaseDataGroup, BaseDataContainer

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class DataGroup(BaseDataGroup):
    """The DataGroup class manages groups of data containers.

    It uses an OrderedDict to associate containers with this group.
    """
    
    def __init__(self, *, name: str, containers: list=None, **dc_kwargs):
        """Initialise a DataGroup."""

        log.debug("DataGroup.__init__ called.")

        # Initialise with parent method, which will call the _prepare_data
        super().__init__(name=name, containers=containers,
                         StorageCls=collections.OrderedDict, **dc_kwargs)

        # Fill the data

        # Done.
        log.debug("DataGroup.__init__ finished.")
