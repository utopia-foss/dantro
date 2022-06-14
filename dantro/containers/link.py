"""Implements the :py:class:`dantro.containers.link.LinkContainer` which holds
a :py:class:`~dantro.utils.link.Link` object and can be used to link to
another position in the data tree.
"""

import logging

from ..mixins import CheckDataMixin
from ..utils import Link
from .general import ObjectContainer, PassthroughContainer

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class LinkContainer(CheckDataMixin, PassthroughContainer):
    """A LinkContainer is a container containing a
    :py:class:`~dantro.utils.link.Link` object.

    It forwards all attribute calls to the :py:class:`~dantro.utils.link.Link`
    object, which in turn forwards all attribute calls to the linked object,
    thereby emulating the behaviour of the linked object.
    """

    DATA_EXPECTED_TYPES = (Link,)

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item.

        In this case, the anchor and relative path of the associated link is
        returned.
        """
        return "{} -> {}, {}".format(
            self.data.anchor_object.name,
            self.data.target_rel_path,
            super(ObjectContainer, self)._format_info(),
        )
