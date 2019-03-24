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

    # Use OrderedDict for storage
    _STORAGE_CLS = collections.OrderedDict
