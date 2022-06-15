"""Implements the :py:class:`~dantro.utils.link.Link` class and specializations
for it."""

import logging
import weakref
from typing import Any, Callable

from ..abc import PATH_JOIN_CHAR
from ..base import BaseDataContainer, BaseDataGroup
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

    .. note::

        The references are internally stored as weak references
        :py:class:`weakref.ref`; note that this limits the picklability of
        objects of this class.

        See `the weakref docs <https://docs.python.org/3/library/weakref.html#weakref.ref>`_
        for more information.
    """

    _REF_TYPE: type = weakref.ref
    """The referencing type for linking, here using weak references"""

    FORWARD_ATTR_TO: str = "target_object"
    """Forward attributes to the target object property (...but see
    :py:meth:`._forward_attr_get_forwarding_target` method for more information
    on why this is done.
    """

    def __init__(self, *, anchor: BaseDataContainer, rel_path: str):
        """Initialize a link from an anchor and a relative path to a target"""
        # Use name-mangling to not take up any attributes that might get in the
        # way of attribute forwarding ...
        self.__anchor = self._REF_TYPE(anchor)
        self.__rel_path = rel_path
        self.__target_ref_cache = None

        log.debug(
            "Created link with anchor %s and relative path '%s'.",
            anchor.logstr,
            rel_path,
        )

    def __eq__(self, other) -> bool:
        """Evaluates equality by making the following comparisons: identity,
        strict type equality, and finally: equality of the
        :py:attr:`.anchor_weakref` and :py:attr:`.target_rel_path` properties.

        If types do not match exactly, ``NotImplemented`` is *returned*,
        thus referring the comparison to the other side of the ``==``.
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
    def target_weakref(self) -> "weakref.ref":
        """Resolve the target and return the weak reference to it

        Returns:
            weakref.ref: A weak reference to the target object
        """
        if self.__target_ref_cache is None:
            self.__resolve_target_ref()
        return self.__target_ref_cache

    @property
    def target_object(self) -> BaseDataContainer:
        """Return a (non-weak) reference to the actual target object"""
        return self.target_weakref()  # calling property to resolve the weakref

    @property
    def anchor_weakref(self) -> "weakref.ref":
        """Resolve the weak reference to the anchor and return it, i.e.
        return a reference to the actual anchor object.

        Returns:
            weakref.ref: The weak reference to the anchor object
        """
        return self.__anchor

    @property
    def anchor_object(self) -> BaseDataContainer:
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

        # Try traversing the path
        try:
            obj = obj[self.__rel_path]

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

        self.__target_ref_cache = self._REF_TYPE(obj)

    def _forward_attr_get_forwarding_target(self):
        """Get the object that the attribute call is to be forwarded to, i.e.
        the resolved target object. This invokes resolution of the target and
        caching of the corresponding :py:class:`weakref.ref`, but the returned
        (strong) reference will not be cached.
        """
        return self.target_object


# -----------------------------------------------------------------------------


class _strongref:
    """Emulates part of the :py:class:`weakref.ref` interface but uses regular
    references instead of weak references.

    This is used *internally* by :py:class:`~dantro.utils.link.StrongLink` and
    improves picklability.
    """

    def __init__(self, obj: Any):
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj

    def __eq__(self, other) -> bool:
        """Two strong references are equal if and only if they point to the
        identical object.
        """
        if type(self) is not type(other):
            return False
        return self._obj is other._obj


class StrongLink(Link):
    """Like a :py:class:`~dantro.utils.link.Link`, but not using regular
    (non-weak) references instead of weak references, which improves the
    pickleability of these objects.
    """

    _REF_TYPE = _strongref
