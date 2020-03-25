"""Definition of an OrderedDict-subclass that maintains the order key values
rather than the insertion order. It can be specialized to use a specific
comparison function for ordering keys.
"""

import collections
from typing import Callable

# Imports needed for replicating OrderedDict behaviour
import sys as _sys
from operator import eq as _eq
from collections import _proxy, _Link, _OrderedDictKeysView
from collections import _OrderedDictValuesView, _OrderedDictItemsView
from collections.abc import MutableMapping
from reprlib import recursive_repr as _recursive_repr

# -----------------------------------------------------------------------------

class KeyOrderedDict(dict):
    """This dict maintains the order of keys not by their insertion but by
    their value. It is a re-implementation of collections.OrderedDict, because
    subclassing that class is really difficult.

    Ordering is maintained by adjusting those methods of OrderedDict that
    take care of building and maintaining the doubly-linked list that provides
    the ordering. See OrderedDict for details, e.g. the weak-referencing.

    In effect, this relates only to __setitem__; all other methods rely on this
    to add elements to the mapping.

    For comparison, the ``key`` callable given at initialisation can be used to
    perform a operation on keys, the result of which is used in comparison.
    If this is not given, the ``DEFAULT_KEY_COMPARATOR`` class variable is
    used; note that this needs to be a binary function, where the first
    argument is equivalent to ``self`` and the second is the actual key to
    perform the unary operation on.
    """

    # The default key comparator
    DEFAULT_KEY_COMPARATOR = lambda _, k: k

    def __init__(self, *args, key: Callable=None, **kwds):
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
            raise TypeError("expected at most 1 arguments, got {}"
                            "".format(len(args)))

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
            bool: k1 < k2
        """
        # Create the actual keys to compare
        try:
            _k1, _k2 = self._key(k1), self._key(k2)
        
        except Exception as exc:
            raise ValueError("Could not apply key transformation method on "
                             "one or both of the keys '{}' and '{}'!"
                             "".format(k1, k2)
                             ) from exc
        
        # Compare
        try:
            return _k1 < _k2

        except Exception as exc:
            raise ValueError("Failed comparing '{}' (type {}, from key '{}') "
                             "to '{}' (type {}, from key '{}')! Keys of {} "
                             "need to be comparable after the key "
                             "transformation function {} was applied to them."
                             "".format(_k1, type(_k1), k1,
                                       _k2, type(_k2), k2,
                                       self.__class__.__name__,
                                       self._key.__name__)
                             ) from exc

    def __setitem__(self, key, value, *,
                    dict_setitem=dict.__setitem__, proxy=_proxy, Link=_Link):
        """Set the item with the provided key to the given value, maintaining
        the ordering specified by NEW_KEY_GT_NEXT_KEY

        If the key is not available, this takes care of finding the right place
        to insert the new link in the doubly-linked list.
        """
        if key not in self:
            # Create a new link in the inherited mapping
            self.__map[key] = link = Link()

            # Get the sentinel element and the one at the end
            root = self.__root
            last = root.prev

            # Find the link in the doubly-linked list after which the new
            # element is to be inserted, i.e.: the first element that compares
            # true when invoking the key comparator

            # Start with root element. For empty maps, this is already the
            # element after which the link can be inserted.
            element = root
            
            # For populated maps, need to iterate over the linked list
            if last is not root:
                # Start at root and iterate until again reaching root
                while element.next is not root:
                    # Check if the next element's key would compare strictly
                    # greater than the key to be inserted.
                    # NOTE root.key is not available!
                    if self._key_comp_lt(key, element.next.key):
                        # Found it.
                        break
                    # Did not find it. Go to the next one
                    element = element.next

            # element is now the element _after_ which to insert the new link
            link.prev, link.next, link.key = element, element.next, key

            # Update neighbouring links such that the new link is in between
            link.prev.next = link
            link.next.prev = proxy(link)  # need be a weak link

        # Key is now available
        # Set the value of the item using the method implemented by dict
        dict_setitem(self, key, value)

    def insert(self, key, value, *, hint_after=None, hint_before=None):
        """Inserts a (key, value) pair using hint information to speed up
        key search.
        """
        raise NotImplementedError

    # .........................................................................
    # Remaining dict interface, oriented at collections.OrderedDict

    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        """kod.__delitem__(y) <==> del kod[y]
        
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
        """kod.__iter__() <==> iter(kod)

        Traverse the linked list in order.
        """
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self):
        """kod.__reversed__() <==> reversed(kod)
        
        Traverse the linked list in reverse order.
        """
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev

    def clear(self):
        """Remove all items"""
        root = self.__root
        root.prev = root.next = root
        self.__map.clear()
        dict.clear(self)


    def __sizeof__(self):
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
        """D.keys() -> a set-like object providing a view on D's keys"""
        return _OrderedDictKeysView(self)

    def items(self):
        """D.items() -> a set-like object providing a view on D's items"""
        return _OrderedDictItemsView(self)

    def values(self):
        """D.values() -> an object providing a view on D's values"""
        return _OrderedDictValuesView(self)

    __ne__ = MutableMapping.__ne__

    __marker = object()

    def pop(self, key, default=__marker):
        """kod.pop(k[,d]) -> v, remove specified key and return the
        corresponding value.  If key is not found, default is returned if
        given, otherwise KeyError is raised.
        """
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default


    def setdefault(self, key, default=None):
        """kod.setdefault(k[,d]) -> kod.get(k,d), also set kod[k]=d if k not
        in kod
        """
        if key in self:
            return self[key]
        self[key] = default
        return default

    @_recursive_repr()
    def __repr__(self):
        """kod.__repr__() <==> repr(kod)"""
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.items()))

    def __reduce__(self):
        """Return state information for pickling"""
        inst_dict = vars(self).copy()
        for k in vars(KeyOrderedDict()):
            inst_dict.pop(k, None)
        return self.__class__, (), inst_dict or None, None, iter(self.items())

    def copy(self):
        """kod.copy() -> a shallow copy of kod, maintaining the key comparator"""
        return self.__class__(self, key=self._key)

    @classmethod
    def fromkeys(cls, iterable, value=None, *, key: Callable=None):
        """KOD.fromkeys(S[, v]) -> New key-ordered dictionary with keys from S.
        If not specified, the value defaults to None.
        """
        self = cls(key=key)
        for key in iterable:
            self[key] = value
        return self

    def __eq__(self, other):
        """kod.__eq__(y) <==> kod==y.
        Comparison to another KeyOrderedDict or OrderedDict is order-sensitive
        while comparison to a regular mapping is order-insensitive.
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
