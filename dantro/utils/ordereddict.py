"""Definition of an OrderedDict-subclass that maintains the order key values
rather than the insertion order. It can be specialized to use a specific
comparison function for ordering keys.
"""

import collections
import logging

# Imports needed for replicating OrderedDict behaviour
import sys as _sys
from collections import (
    _Link,
    _OrderedDictItemsView,
    _OrderedDictKeysView,
    _OrderedDictValuesView,
    _proxy,
)
from collections.abc import MutableMapping
from operator import eq as _eq
from reprlib import recursive_repr as _recursive_repr
from typing import Callable

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class KeyOrderedDict(dict):
    """This dict maintains the order of keys not by their insertion but by
    their value. It is a re-implementation of collections.OrderedDict, because
    subclassing that class is really difficult.

    Ordering is maintained by adjusting those methods of OrderedDict that
    take care of building and maintaining the doubly-linked list that provides
    the ordering. See OrderedDict for details, e.g. the weak-referencing.

    In effect, this relates only to ``__setitem__``; all other methods rely on
    this to add elements to the mapping.

    For comparison, the ``key`` callable given at initialisation can be used to
    perform a operation on keys, the result of which is used in comparison.
    If this is not given, the ``DEFAULT_KEY_COMPARATOR`` class variable is
    used; note that this needs to be a binary function, where the first
    argument is equivalent to ``self`` and the second is the actual key to
    perform the unary operation on.
    """

    # A counter to keep track of the total number of comparisons made during
    # element ordering, e.g. when adding elements to this dict.
    __num_comparisons = 0

    # The default key comparator
    DEFAULT_KEY_COMPARATOR = lambda _, k: k

    def __init__(self, *args, key: Callable = None, **kwds):
        """Initialize a KeyOrderedDict, which maintains key ordering. If no
        custom ordering function is given, orders by simple "smaller than"
        comparison.

        Apart from that, the interface is the same as for regular dictionaries.

        Args:
            *args: a single sequence of (key, value) pairs to insert
            key (Callable, optional): The callable used to
            **kwds: Passed on to ``update`` method

        Raises:
            TypeError: on len(args) > 1
        """
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")

        # Set the key comparison function, passing through if nothing is set
        self._key = key if key is not None else self.DEFAULT_KEY_COMPARATOR

        # Set up internally used attributes
        try:
            self.__root

        except AttributeError:
            self.__hardroot = _Link()
            self.__root = root = _proxy(self.__hardroot)
            root.prev = root.next = root
            self.__map = {}

        # Populate the dict
        self.__update(*args, **kwds)

    # .........................................................................
    # Custom implementations that ascertain key ordering

    def _key_comp_lt(self, k1, k2) -> bool:
        """The key comparator. Returns true for k1 < k2.

        Before comparison, the unary ``self._key`` method is invoked on both
        keys.

        Args:
            k1: lhs of comparison
            k2: rhs of comparison

        Returns:
            bool: result of ``k1 < k2``

        Raises:
            ValueError: Upon failed key comparison
        """
        # Create the actual keys to compare
        try:
            _k1, _k2 = self._key(k1), self._key(k2)

        except Exception as exc:
            raise ValueError(
                "Could not apply key transformation method on "
                f"one or both of the keys '{k1}' and '{k2}'!"
            ) from exc

        # Increment the comparisons counter
        self.__num_comparisons += 1

        # Compare
        try:
            return _k1 < _k2

        except Exception as exc:
            raise ValueError(
                f"Failed comparing '{_k1}' (type {type(_k1)}, from key '{k1}')"
                f" to '{_k2}' (type {type(_k2)}, from key '{k2}')! "
                f"Keys of {self.__class__.__name__} need to be comparable "
                f"after the key transformation function {self._key.__name__} "
                "was applied to them."
            ) from exc

    def __setitem__(
        self,
        key,
        value,
        *,
        dict_setitem=dict.__setitem__,
        proxy=_proxy,
        Link=_Link,
        start=None,
    ):
        """Set the item with the provided key to the given value, maintaining
        the ordering specified by ``_key_comp_lt``.

        If the key is not available, this takes care of finding the right place
        to insert the new link in the doubly-linked list.

        Unlike the regular ``__setitem__``, this allows specifying a start
        element at which to begin the search for an insertion point, which may
        greatly speed up insertion. By default, it is attempted to insert
        after the last element; if that is not possible, a full search is done.

        .. warning::

            This operation does not scale well with out-of-order insertion!

            This behavior is inherent to this data structure, where key
            ordering has to be maintained during every insertion. While a best
            guess is made regarding the insertion points (see above), inserting
            elements completely out of order will require a search time that is
            proportional to the number of elements *for each insertion*.

        .. hint::

            If you have information about where the element should be stored,
            use :py:meth:`~dantro.utils.ordereddict.KeyOrderedDict.insert` and
            provide the ``hint_after`` argument.
        """
        if key not in self:  # complexity: O(1)
            # Create a new link in the inherited mapping
            self.__map[key] = link = Link()

            # Find the link in the doubly-linked list after which the new
            # element is to be inserted, i.e.: the first element that compares
            # true when invoking the key comparator
            element = self.__find_element_to_insert_after(key, start=start)

            # element is now the element _after_ which to insert the new link
            link.prev, link.next, link.key = element, element.next, key

            # Update neighbouring links such that the new link is in between
            link.prev.next = link
            link.next.prev = proxy(link)  # need be a weak link

        # Key is now available
        # Set the value of the item using the method implemented by dict
        dict_setitem(self, key, value)

    def insert(self, key, value, *, hint_after=None):
        """Inserts a ``(key, value)`` pair using hint information to speed up
        the search for an insertion point.

        If hint information is available, it is highly beneficial to add this

        Args:
            key: The key at which to insert
            value: The value to insert
            hint_after (optional): A best guess after which key to insert.
                The requirement here is that the key compares
        """
        if hint_after is not None:
            self.__setitem__(key, value, start=self.__map.get(hint_after))

        else:
            # No hints, use regular insertion
            self[key] = value

    def __find_element_to_insert_after(self, key, *, start=None) -> _Link:
        """Finds the link in the doubly-linked list after which a new element
        with the given key may be inserted, i.e. the last element that
        compares False when invoking the key comparator.

        If inserting an element to the back or the front of the key list, the
        complexity of this method is constant. Otherwise, it scales with the
        number of already existing elements.

        Args:
            key: The key to find the insertion spot for
            start (None, optional): A key to use to start looking, if not given
                will use the last element as a best guess.
        """
        log.trace("Looking for insertion point for key '%s' ...", key)

        # Set default value: the last element; this is a best guess which is
        # very successful if inserting in ascending order and not very costly
        # (one additional comparison) if inserting in descending order.
        # This allows constant insertion time for both these scenarios.
        if start is None:
            start = self.__root.prev

        # For unpopulated maps, can directly insert at this point
        if start.next is start:
            log.trace("  Map is empty; inserting in the beginning.")
            return start

        # For populated maps, need to check whether the start element is really
        # suitable:
        #   * The start element needs to compare false in key < start.key,
        #     otherwise we don't know whether the next "stricter greater than"
        #     comparison is really the first (which is the requirement for a
        #     valid starting point).
        #   * Also, we can't do a key comparison for the root element, in which
        #     case we can also use this fallback case.
        if start is self.__root or self._key_comp_lt(key, start.key):
            log.trace("  Start element not suitable; using sentinel instead.")
            start = self.__root

        # Now begin element search at the specified start element
        element = start

        # Otherwise, need to iterate over the linked list until back at root
        while element.next is not self.__root:
            # Check if the next element's key would compare strictly
            # greater than the key to be inserted, in which case we found the
            # element to insert after
            if self._key_comp_lt(key, element.next.key):
                break

            # Did not find it. Go to the next one
            element = element.next

        log.trace(
            "  Found insertion point: after '%s'. Total comparisons: %d",
            element.key if element is not self.__root else "<root>",
            self._num_comparisons,
        )
        return element

    @property
    def _num_comparisons(self) -> int:
        """Total number of comparisons performed between elements, e.g. when
        finding an insertion point

        This number is low, if the insertion order is sequential.
        For out-of-order insertion, it may become large.
        """
        return self.__num_comparisons

    # .........................................................................
    # Remaining dict interface, oriented at collections.OrderedDict

    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        """``kod.__delitem__(y) <==> del kod[y]``

        Deleting an existing item uses self.__map to find the link which gets
        removed by updating the links in the predecessor and successor nodes.
        """
        dict_delitem(self, key)
        link = self.__map.pop(key)
        link_prev = link.prev
        link_next = link.next
        link_prev.next = link_next
        link_next.prev = link_prev
        link.prev = None
        link.next = None

    def __iter__(self):
        """``kod.__iter__() <==> iter(kod)``

        Traverse the linked list in order.
        """
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self):
        """``kod.__reversed__() <==> reversed(kod)``

        Traverse the linked list in reverse order.
        """
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev

    def clear(self) -> None:
        """Remove all items from this dict"""
        root = self.__root
        root.prev = root.next = root
        self.__map.clear()
        dict.clear(self)

    def __sizeof__(self) -> int:
        """Get the size of this object"""
        sizeof = _sys.getsizeof

        # number of links including root
        n = len(self) + 1

        # instance dictionary
        size = sizeof(self.__dict__)

        # internal dict and inherited dict
        size += sizeof(self.__map) * 2

        # link and proxy objects
        size += sizeof(self.__hardroot) * n
        size += sizeof(self.__root) * n

        # key comparator
        size += sizeof(self._key)

        return size

    update = __update = MutableMapping.update

    def keys(self):
        """Returns a set-like object providing a view on this dict's keys"""
        return _OrderedDictKeysView(self)

    def items(self):
        """Returns a set-like object providing a view on this dict's items"""
        return _OrderedDictItemsView(self)

    def values(self):
        """Returns an object providing a view on this dict's values"""
        return _OrderedDictValuesView(self)

    __ne__ = MutableMapping.__ne__

    __marker = object()

    def pop(self, key, default=__marker):
        """Removes the specified key and returns the corresponding value.
        If ``key`` is not found, ``default`` is returned if given, otherwise a
        ``KeyError`` is raised.
        """
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default

    def setdefault(self, key, default=None):
        """Retrieves a value, otherwise sets that value to a default.
        Calls ``kod.get(k,d)``, setting ``kod[k]=d`` if ``k not in kod``.
        """
        if key in self:
            return self[key]
        self[key] = default
        return default

    @_recursive_repr()
    def __repr__(self) -> str:
        """Returns a string representation of this dict"""
        if not self:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self.items())!r})"

    def __reduce__(self):
        """Return state information for pickling"""
        inst_dict = vars(self).copy()
        for k in vars(KeyOrderedDict()):
            inst_dict.pop(k, None)
        return self.__class__, (), inst_dict or None, None, iter(self.items())

    def copy(self):
        """Returns a shallow copy of kod, maintaining the key comparator"""
        return self.__class__(self, key=self._key)

    @classmethod
    def fromkeys(
        cls, iterable, value=None, *, key: Callable = None
    ) -> "KeyOrderedDict":
        """A call like ``KOD.fromkeys(S[, v])`` returns a new key-ordered
        dictionary with keys from ``S``. If not specified, the value defaults
        to None.

        Args:
            iterable: The iterable over keys
            value (None, optional): Default value for the key
            key (Callable, optional): Passed to the class initializer

        Returns:
            KeyOrderedDict: The resulting key-ordered dict.
        """
        self = cls(key=key)
        for key in iterable:
            self[key] = value
        return self

    def __eq__(self, other) -> bool:
        """``kod.__eq__(y) <==> kod==y``: Comparison to another
        KeyOrderedDict or OrderedDict is order-sensitive while comparison to a
        regular mapping is order-insensitive.

        Args:
            other: The object to compare to

        Returns:
            bool: Whether the two objects can be considered equal.
        """
        if isinstance(other, (KeyOrderedDict, collections.OrderedDict)):
            return dict.__eq__(self, other) and all(map(_eq, self, other))
        return dict.__eq__(self, other)


# -----------------------------------------------------------------------------
# Specializations


class IntOrderedDict(KeyOrderedDict):
    """A :py:class:`~dantro.utils.ordereddict.KeyOrderedDict` specialization
    that assumes keys to be castable to integer and using the comparison of
    the resulting integer values for maintaining the order
    """

    DEFAULT_KEY_COMPARATOR = lambda _, k: int(k)
