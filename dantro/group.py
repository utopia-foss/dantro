"""In this module, BaseDataContainer specialisations that group data containers are implemented."""

import collections
import logging
import warnings

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
                         data=None, **dc_kwargs)

        # Done.
        log.debug("DataGroup.__init__ finished.")

    @staticmethod
    def _prepare_data(*, data, containers) -> collections.OrderedDict:
        """ """

        if containers is not None:
            # Read the container names and generate an ordered dict from it.

            if not all([isinstance(c, BaseDataContainer) for c in containers]):
                raise TypeError("The given `containers` list can only have "
                                "BaseDataContainer-derived objects as "
                                "contents, but there were some of other type!")
            elif data is not None:
                warnings.warn("")

            # Build an OrderedDict with the container names
            names = [c.name for c in containers]
            data = collections.OrderedDict()

            for _name, container in zip(names, containers):
                data[_name] = container

        else:
            # Use an empty dict
            data = collections.OrderedDict()

        return data
