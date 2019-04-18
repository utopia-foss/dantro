"""This sub-module implements the basic mixin classes that are required
in the dantro.base module"""

import logging
import warnings

from ..abc import AbstractDataProxy, PATH_JOIN_CHAR
from ..tools import TTY_COLS

# Local constants
log = logging.getLogger(__name__)


class UnexpectedTypeWarning(UserWarning):
    """Given when there was an unexpected type passed to a data container."""
    pass


# -----------------------------------------------------------------------------

class AttrsMixin:
    """This Mixin class supplies the `attrs` property getter and setter and
    the private `_attrs` attribute.

    Hereby, the setter function will initialize a BaseDataAttrs-derived object
    and store it as an attribute.
    This relays the checking of the correct attribute format to the actual
    BaseDataAttrs-derived class.

    For changing the class that is used for the attributes, an overwrite of the
    _ATTRS_CLS class variable suffices.
    """
    # The class attribute that the attributes will be stored to
    _attrs = None

    # Define the class to use for storing attributes
    _ATTRS_CLS = None

    @property
    def attrs(self):
        """The container attributes."""
        return self._attrs

    @attrs.setter
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        if self._ATTRS_CLS is None:
            raise ValueError("Need to declare the class variable _ATTRS_CLS "
                             "in order to use the AttrsMixin!")

        # Perform the initialisation
        self._attrs = self._ATTRS_CLS(name='attrs', attrs=new_attrs)


class PathMixin:
    """This Mixin class implements path capabilities for groups or containers.

    That means, that each object can re-create the path at which it can be
    accessed _if_ it knows its parent object."""
    
    # The parent object
    _parent = None

    @property
    def parent(self):
        """The group this container is contained in or None if on its own."""
        return self._parent

    @parent.setter
    def parent(self, cont):
        """Associate a parent object with this container."""
        if self.parent is not None and cont is not None:
            raise ValueError("A parent was already associated with {cls:} "
                             "'{}'! Instead of manually setting the parent, "
                             "use the functions supplied to manipulate "
                             "members of this {cls:}."
                             "".format(self.name, cls=self.classname))
        
        log.debug("Setting %s as parent of %s ...",
                  cont.logstr if cont else None, self.logstr)
        self._parent = cont

    @property
    def path(self) -> str:
        """Return the path to get to this container"""
        if self.parent is None:
            # At the top or no parent associated -> no reasonable path to give
            return self.name
        # else: not at the top, also need the parent's path
        return self.parent.path + PATH_JOIN_CHAR + self.name

    def _format_path(self) -> str:
        """A __format__ helper function: returns the path to this container"""
        return self.path


class LockDataMixin:
    """This Mixin class provides a flag for marking the data of a group or
    container as locked.
    """
    # Whether the data is regarded as locked. Note name-mangling here.
    __locked = False

    @property
    def locked(self) -> bool:
        """Whether this object is locked"""
        return self.__locked

    def lock(self):
        """Locks the data of this object"""
        self.__locked = True
        self._lock_hook()
    
    def unlock(self):
        """Unlocks the data of this object"""
        self.__locked = False
        self._unlock_hook()

    def raise_if_locked(self, *, prefix: str=None):
        """Raises an exception if this object is locked; does nothing otherwise
        """
        if self.locked:
            raise RuntimeError("{}Cannot modify {} because it was already "
                               "marked locked."
                               "".format(prefix + " " if prefix else "",
                                         self.logstr))

    def _lock_hook(self):
        """Invoked upon locking."""
        pass
    
    def _unlock_hook(self):
        """Invoked upon unlocking."""
        pass
    

class CollectionMixin:
    """This Mixin class implements the methods needed for being a Collection.
    
    It relays all calls forward to the data attribute.
    """

    def __contains__(self, key) -> bool:
        """Whether the given key is contained in the items."""
        return bool(key in self.data)

    def __len__(self) -> int:
        """The number of items."""
        return len(self.data)

    def __iter__(self):
        """Iterates over the items."""
        return iter(self.data)


class ItemAccessMixin:
    """This Mixin class implements the methods needed for getting, setting,
    and deleting items. It relays all calls forward to the data attribute, but
    if given a list (passed down from above), it extracts it
    """

    def __getitem__(self, key):
        """Returns an item."""
        return self.data[self._item_access_convert_list_key(key)]

    def __setitem__(self, key, val):
        """Sets an item."""
        self.data[self._item_access_convert_list_key(key)] = val

    def __delitem__(self, key):
        """Deletes an item"""
        del self.data[self._item_access_convert_list_key(key)]

    def _item_access_convert_list_key(self, key):
        """If given something that is not a list, just return that key"""
        if isinstance(key, list):
            if len(key) > 1:
                return tuple(key)
            return key[0]
        return key


class MappingAccessMixin(ItemAccessMixin, CollectionMixin):
    """Supplies all methods that are needed for Mapping access.

    All calls are relayed to the data attribute.
    """

    def keys(self):
        """Returns an iterator over the attribute names."""
        return self.data.keys()

    def values(self):
        """Returns an iterator over the attribute values."""
        return self.data.values()

    def items(self):
        """Returns an iterator over the (keys, values) tuple of the attributes."""
        return self.data.items()

    def get(self, key, default=None):
        """Return the value at `key`, or `default` if `key` is not available."""
        return self.data.get(key, default)


class CheckDataMixin:
    """This mixin class extends a BaseDataContainer-derived class to check the
    provided data before storing it in the container.
    
    It implements a general _check_data method, overwriting the placeholder 
    method in the BaseDataContainer, and can be controlled via class variables.
    
    Attributes:
        DATA_ALLOW_PROXY (bool): Whether to allow _all_ proxy types, i.e.
            classes derived from AbstractDataProxy
        DATA_EXPECTED_TYPES (tuple, None): Which types to allow. If None, all
            types are allowed.
        DATA_UNEXPECTED_ACTION (str): The action to take when an unexpected
            type was supplied. Can be: raise, warn, ignore
    """

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = None       # as tuple or None (allow all)
    DATA_ALLOW_PROXY = False         # to check for AbstractDataProxy
    DATA_UNEXPECTED_ACTION = 'warn'  # Can be: raise, warn, ignore

    def _check_data(self, data, *, name: str) -> None:
        """A general method to check the received data for its type
        
        Args:
            data: The data to check
            name (str): The name of the data container
        
        Returns:
            bool: True if the data is of an expected type
        
        Raises:
            TypeError: If the type was unexpected and the action was 'raise'
            ValueError: Illegal value for DATA_UNEXPECTED_ACTION class variable
        """
        if self.DATA_EXPECTED_TYPES is None:
            # All types allowed
            return

        # Compile tuple of allowed types
        expected_types = self.DATA_EXPECTED_TYPES

        if self.DATA_ALLOW_PROXY:
            expected_types += (AbstractDataProxy,)

        # Perform the check
        if isinstance(data, expected_types):
            # Is of the expected type
            return

        # else: was not of the expected type

        # Create a base message
        msg = ("Unexpected type {} for data passed to {} '{}'! "
               "Expected types are: {}.".format(type(data), self.classname,
                                                name, expected_types))

        # Handle according to the specified action
        if self.DATA_UNEXPECTED_ACTION == 'raise':
            raise TypeError(msg)

        elif self.DATA_UNEXPECTED_ACTION == 'warn':
            warnings.warn(msg + "\nInitialization will work, but be informed "
                          "that there might be errors at runtime.",
                          UnexpectedTypeWarning)
        
        elif self.DATA_UNEXPECTED_ACTION == 'ignore':
            log.debug(msg + " Ignoring ...")

        else:
            raise ValueError("Illegal value '{}' for class variable "
                             "DATA_UNEXPECTED_ACTION of {}. "
                             "Allowed values are: raise, warn, ignore"
                             "".format(self.DATA_UNEXPECTED_ACTION,
                                       self.classname))
