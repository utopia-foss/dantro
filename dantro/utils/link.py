"""Implements the Link class"""

import weakref
import logging

from ..abc import AbstractDataContainer, PATH_JOIN_CHAR
from ..mixins import ForwardAttrsMixin

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class Link(ForwardAttrsMixin):
    """A link is a connection between two objects in the data tree, i.e. a
    data group and a data container.

    It has a source object that it is coupled to and a relative path from that
    object to the target object.

    Whenever attribute access occurs, an object of this class will resolve the
    linked object (if not already cached) and then forward the attribute call
    to that object.
    """
    FORWARD_ATTR_TO = "target_object"  # ...but see …_forwarding_target method!

    def __init__(self, *, anchor: AbstractDataContainer, rel_path: str):
        """Initialize a link from an anchor and a relative path to a target"""
        # Use name-mangling to not take up any 
        self.__anchor = weakref.ref(anchor)
        self.__rel_path = rel_path
        self.__target_ref_cache = None

        log.debug("Created link with anchor %s and relative path '%s'.",
                  anchor.logstr, rel_path)

    @property
    def target_weakref(self) -> weakref:
        """Resolve the target and return the weak reference to it"""
        if self.__target_ref_cache is None:
            self.__resolve_target_ref()
        return self.__target_ref_cache

    @property
    def target_object(self) -> AbstractDataContainer:
        """Return a (non-weak) reference to the actual target object"""
        return self.target_weakref()

    @property
    def anchor_weakref(self) -> weakref:
        """Resolve the weak reference to the anchor and return it, i.e.:
        return a reference to the actual object.
        """
        return self.__anchor

    @property
    def anchor_object(self) -> AbstractDataContainer:
        """Return a (non-weak) reference to the anchor object"""
        return self.__anchor()

    @property
    def target_rel_path(self) -> str:
        """Returns the relative path to the target"""
        return self.__rel_path

    def __resolve_target_ref(self) -> None:
        """Resolves the weak reference to the target object and caches it"""
        # Start at the anchor, resolving the weak reference
        obj = self.__anchor()

        # Traverse the path, going to the parent or a child
        try:
            for segment in self.__rel_path.split(PATH_JOIN_CHAR):
                # Skip empty segments
                if not segment:
                    continue

                obj = obj.parent if segment == ".." else obj[segment]

        except Exception as err:
            raise RuntimeError("Failed resolving target of link '{}' relative "
                               "to anchor {} @ {}. Is the anchor embedded in "
                               "a data tree?"
                               "".format(self.__rel_path,
                                         self.__anchor().logstr,
                                         self.__anchor().path)
                               ) from err

        log.debug("Resolved link '%s' relative to anchor %s @ %s",
                  self.__rel_path, self.__anchor().logstr,
                  self.__anchor().path)

        self.__target_ref_cache = weakref.ref(obj)

    def _forward_attr_get_forwarding_target(self):
        """Get the object that the attribute call is to be forwarded to, i.e.
        the resolved target object. This invokes resolution of the target and
        caching of the corresponding weakref, but the returned (strong) ref
        will not be cached.
        """
        return self.target_object
