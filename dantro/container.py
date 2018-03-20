"""This module implements specialisations of the BaseDataContainer class."""

import logging

import dantro.base

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ItemContainer(dantro.base.ItemAccessMixin, dantro.base.BaseDataContainer):

    def __init__(self, *, name: str, data, **dc_kwargs):
        """Initialise an ItemContainer."""

        log.debug("ItemContainer.__init__ called.")

        # Initialise with parent method, which will call the _prepare_data
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.
        log.debug("ItemContainer.__init__ finished.")

    def _prepare_data(self, *, data):
        # TODO Actually check the data here?
        return data

    def convert_to(self):
        raise NotImplementedError
