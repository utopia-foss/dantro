"""Implements the Link class and the LinkContainer."""

import logging

from ..utils import Link
from ..mixins import CheckDataMixin
from .general import ObjectContainer, PassthroughContainer

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class LinkContainer(CheckDataMixin, PassthroughContainer):
    """A LinkContainer is a container containing a Link object.

    It forwards all attribute calls to the Link object, which in turn forwards
    all attribute calls to the linked object, thereby emulating the behaviour
    of the linked object.
    """
    DATA_EXPECTED_TYPES = (Link,)

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item

        In this case, the anchor and relative path of the associated link is
        returned.
        """
        return ("{} -> {}, {}"
                "".format(self.data.anchor_object.name,
                          self.data.target_rel_path,
                          super(ObjectContainer, self)._format_info()))
