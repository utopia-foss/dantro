"""Implements the Link class"""

import logging
import weakref
from typing import TypeVar

from ..abc import PATH_JOIN_CHAR
from ..base import BaseDataContainer, BaseDataGroup
from ..mixins import ForwardAttrsMixin

# Local constants
log = logging.getLogger(__name__)

# Type definitions
TGroupOrContainer = TypeVar(
    "TGroupOrContainer", BaseDataContainer, BaseDataGroup
)

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

    def __init__(self, *, anchor: TGroupOrContainer, rel_path: str):
        """Initialize a link from an anchor and a relative path to a target"""
        # Use name-mangling to not take up any attributes that might get in the
        # way of attribute forwarding ...
        self.__anchor = weakref.ref(anchor)
        self.__rel_path = rel_path
        self.__target_ref_cache = None

        log.debug(
            "Created link with anchor %s and relative path '%s'.",
            anchor.logstr,
            rel_path,
        )

    def __eq__(self, other) -> bool:
        """Evaluates equality by making the following comparisons: identity,
        strict type equality, and finally: equality of the ``anchor_weakref``
        and ``target_rel_path`` properties.

        If types do not match exactly, ``NotImplemented`` is returned, thus
        referring the comparison to the other side of the ``==``.
        """
        if other is self:
            return True

        if type(other) is not type(self):
            return NotImplemented

        return (
            self.anchor_weakref == other.anchor_weakref
            and self.target_rel_path == other.target_rel_path
        )

    @property
    def target_weakref(self) -> weakref:
        """Resolve the target and return the weak reference to it"""
        if self.__target_ref_cache is None:
            self.__resolve_target_ref()
        return self.__target_ref_cache

    @property
    def target_object(self) -> TGroupOrContainer:
        """Return a (non-weak) reference to the actual target object"""
        return self.target_weakref()  # calling property to resolve the weakref

    @property
    def anchor_weakref(self) -> weakref:
        """Resolve the weak reference to the anchor and return it, i.e.:
        return a reference to the actual object.
        """
        return self.__anchor

    @property
    def anchor_object(self) -> TGroupOrContainer:
        """Return a (non-weak) reference to the anchor object"""
        return self.__anchor()

    @property
    def target_rel_path(self) -> str:
        """Returns the relative path to the target"""
        return self.__rel_path

    def __resolve_target_ref(self) -> None:
        """Resolves the weak reference to the target object and caches it"""
        # Start at the anchor, resolving the weak reference via '()'
        obj = self.__anchor()

        # If there is no relative path given, this is simply a link to itself
        if not self.__rel_path:
            self.__target_ref_cache = self.__anchor
            return obj

        # Distinguish between anchors that are groups and anchors that are
        # containers; the latter need to be embedded and the _parent_ object is
        # then where the path should start from.
        if not isinstance(obj, BaseDataGroup):
            # Assume it's a container-like anchor, although it might also be a
            # whole other type. We don't care as long as a parent is defined.
            if obj.parent is None:
                raise ValueError(
                    f"The anchor object {obj.logstr} is not embedded into a "
                    "data tree; cannot resolve the target "
                    f"'{self.__rel_path}'! Either choose a group-like dantro "
                    "object as an anchor or embed the container-like object "
                    "into a group."
                )

            obj = obj.parent

        # Try traversing the path, going to the parent or a child depending on
        # which kind of segment is encountered.
        try:
            for segment in self.__rel_path.split(PATH_JOIN_CHAR):
                # Skip empty segments, i.e. `foo//bar` paths. This makes path
                # traversal more robust and mirrors UNIX behaviour (try it).
                if not segment:
                    continue

                obj = obj.parent if segment == ".." else obj[segment]

        except Exception as err:
            raise ValueError(
                f"Failed resolving target of link '{self.__rel_path}' "
                f"relative to anchor {self.__anchor().logstr} "
                f"@ {self.__anchor().path}. Are anchor and target part of the "
                "same data tree?"
            ) from err

        log.debug(
            "Resolved link '%s' relative to anchor %s @ %s",
            self.__rel_path,
            self.__anchor().logstr,
            self.__anchor().path,
        )

        self.__target_ref_cache = weakref.ref(obj)

    def _forward_attr_get_forwarding_target(self):
        """Get the object that the attribute call is to be forwarded to, i.e.
        the resolved target object. This invokes resolution of the target and
        caching of the corresponding weakref, but the returned (strong) ref
        will not be cached.
        """
        return self.target_object
