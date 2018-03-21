"""This module holds the abstract base classes needed for dantro:

AbstractDataContainer: define a general data container interface
AbstractDataGroup: define a Mapping-like group interface
AbstractDataAttr: define a dict-like attribute interface to both of the above
"""

import abc
import collections
from typing import Union
import logging

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class AbstractDataContainer(metaclass=abc.ABCMeta):
    """The AbstractDataContainer is the class defining the data container
    interface. It holds the bare basics of methods that _all_ classes should
    have in common.

    Attributes:
        name: The name of the container
        data: The stored data
    """

    @abc.abstractmethod
    def __init__(self, *, name: str, data):
        """Initialise the AbstractDataContainer, which holds the bare essentials of what a data container should have.
        
        Args:
            name (str): The name of this container
            data: The data that is to be stored
        """
        
        # Pass name and data to read-only attributes
        self._name = str(name)
        self._data = data

        # Caching variables
        self._logstr = None

        log.debug("Initialising %s ...", self.logstr)

    # .........................................................................
    # Properties

    @property
    def name(self) -> str:
        """The name of this DataContainer-derived object."""
        return self._name

    @property
    def classname(self) -> str:
        """Returns the name of this DataContainer-derived class"""
        return self.__class__.__name__

    @property
    def logstr(self) -> str:
        """Returns the classname and name of this object; a combination often
        used in logging..."""
        if not self._logstr:
            self._logstr = "{} '{}'".format(self.classname, self.name)
        return self._logstr

    @property
    def data(self):
        """The stored data."""
        return self._data

    # .........................................................................
    # Item access

    @abc.abstractmethod
    def __getitem__(self, key):
        """Gets an item from the container."""
        pass

    @abc.abstractmethod
    def __setitem__(self, key, val) -> None:
        """Sets an item in the container."""
        pass

    @abc.abstractmethod
    def __delitem__(self, key) -> None:
        """Deletes an item from the container."""
        pass

    # .........................................................................
    # Formatting

    def __str__(self) -> str:
        """An info string, that describes the object. Each class should implement this to return an informative response."""
        return "<{}, {}>".format(self.logstr, self._format_info())

    def __format__(self, spec_str: str) -> str:
        """Creates a formatted string from this """
        if not spec_str:
            return str(self)

        specs = spec_str.split(",")
        parts = []
        join_char = ", "

        for spec in specs:
            try:
                format_func = getattr(self, '_format_'+spec)
            except AttributeError as err:
                raise ValueError("No format string specification '{}', part "
                                 "of '{}', is available for {}!"
                                 "".format(spec, specs,
                                           self.logstr)) from err
            else:
                parts.append(format_func())

        return join_char.join(parts)

    def _format_name(self) -> str:
        """A __format__ helper function: returns the name"""
        return self.name
    
    def _format_cls_name(self) -> str:
        """A __format__ helper function: returns the class name"""
        return self.classname

    @abc.abstractmethod
    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used to characterise this object. Should NOT include name and classname!"""
        pass


# -----------------------------------------------------------------------------

class AbstractDataGroup(AbstractDataContainer, collections.abc.MutableMapping):
    """The AbstractDataGroup is the abstract basis of all data groups.

    It enforces a MutableMapping interface with a focus on _setting_ abilities
    and less so on deletion."""

    @abc.abstractmethod
    def add(self, *conts, overwrite: bool=False) -> None:
        """Adds the given containers to the group."""
        pass

    @abc.abstractmethod
    def __contains__(self, cont: Union[str, AbstractDataContainer]) -> bool:
        """Whether the given container is a member of this group"""
        pass

    @abc.abstractmethod
    def keys(self):
        """Returns an iterator over the container names in this group."""
        pass

    @abc.abstractmethod
    def values(self):
        """Returns an iterator over the containers in this group."""
        pass

    @abc.abstractmethod
    def items(self):
        """Returns an iterator over the (name, data container) tuple of this group."""
        pass

    @abc.abstractmethod
    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name `key` is not available."""
        pass

    @abc.abstractmethod
    def setdefault(self, key, default=None):
        """If `key` is in the dictionary, return its value. If not, insert `key` with a value of `default` and return `default`. `default` defaults to None."""
        pass

    @abc.abstractmethod
    def _format_tree(self) -> str:
        """A __format__ helper function: tree representation of this group"""
        pass

    @abc.abstractmethod
    def _tree_repr(self, level: int=0) -> str:
        """Recursively creates a multi-line string tree representation of this
        group. This is used by, e.g., the _format_tree method."""
        pass

# -----------------------------------------------------------------------------

class AbstractDataAttrs(collections.abc.Mapping, AbstractDataContainer):
    """The BaseDataAttrs class defines the interface for the `.attrs` attribute of a data container.

    This class derives from the abstract class as otherwise there would be 
    circular inheritance. It stores the attributes as mapping and need not be 
    subclassed.
    """
    
    # .........................................................................
    # Specify the attrs interface, dict-like

    @abc.abstractmethod
    def __contains__(self, key) -> bool:
        """Whether the given key is contained in the attributes."""
        pass
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of attributes."""
        pass

    @abc.abstractmethod
    def keys(self):
        """Returns an iterator over the attribute names."""
        pass

    @abc.abstractmethod
    def values(self):
        """Returns an iterator over the attribute values."""
        pass

    @abc.abstractmethod
    def items(self):
        """Returns an iterator over the (keys, values) tuple of the attributes."""
        pass

# -----------------------------------------------------------------------------

class AbstractDataProxy(metaclass=abc.ABCMeta):
    """A data proxy fills in for the place of a data container, e.g. if data should only be loaded on demand. It needs to supply the resolve method."""

    @abc.abstractmethod
    def resolve(self):
        """Resolve the proxy object, returning the actual object."""
        pass
