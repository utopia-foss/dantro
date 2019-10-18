"""This sub-module implements the basic mixin classes that are required
in the dantro.base module"""

import sys
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


class SizeOfMixin:
    """Provides the __sizeof__ magic method and attempts to take into account
    the size of the attributes.
    """

    def __sizeof__(self) -> int:
        """Returns the size of the data (in bytes) stored in this container's
        data and its attributes.

        Note that this value is approximate. It is computed by calling the
        ``sys.getsizeof`` function on the data, the attributes, the name and
        some caching attributes that each dantro data tree class contains.
        Importantly, this is not a recursive algorithm.

        Also, derived classes might implement further attributes that are not
        taken into account either. To be more precise in a subclass, create a
        specific __sizeof__ method and invoke this parent method additionally.

        For more information, see the documentation of ``sys.getsizeof``:

            https://docs.python.org/3/library/sys.html#sys.getsizeof
        """
        nbytes =  sys.getsizeof(self._data)
        nbytes += sys.getsizeof(self._attrs)
        nbytes += sys.getsizeof(self._name)
        nbytes += sys.getsizeof(self._logstr)

        return nbytes


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

    .. note::

        This is not suitable for checking containers that are added to an
        object of a BaseDataGroup-derived class!
    
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

    def _check_data(self, data) -> None:
        """A general method to check the received data for its type
        
        Args:
            data: The data to check
        
        Raises:
            TypeError: If the type was unexpected and the action was 'raise'
            ValueError: Illegal value for DATA_UNEXPECTED_ACTION class variable
        
        Returns:
            None
        """
        if self.DATA_EXPECTED_TYPES is None:
            # All types allowed
            return

        # Compile tuple of allowed types
        expected_types = self.DATA_EXPECTED_TYPES

        if self.DATA_ALLOW_PROXY:
            expected_types += (AbstractDataProxy,)

        # Check for expected types
        if isinstance(data, expected_types):
            return

        # else: was not of the expected type
        # Create a base message
        msg = ("Unexpected type {} for data passed to {}! "
               "Expected types are: {}.".format(type(data), self.logstr,
                                                expected_types))

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
