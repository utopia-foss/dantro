"""This module implements the base classes of dantro."""

import abc
import logging
from typing import Union

# Setup logging for this file
log = logging.getLogger(__name__)

# Local constants

# -----------------------------------------------------------------------------

class AbstractDataContainer(metaclass=abc.ABCMeta):
    """The AbstractDataContainer is the base class for the whole DataContainer
    construct. It holds the basic methods and properties that are common
    to all data container constructs.

    Attributes:
        name: The name of the container
        data: The stored data
    """

    def __init__(self, *, name: str, data):
        """Initialise the AbstractDataContainer, which holds the bare essentials of what a data container should have.
        Args:
            name (str): The name of this container
            data: The data that is to be stored
        """
        self._name = name  # always read-only
        self._data = data  # can be read-only, if there is no data.setter

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
    def data(self):
        """The stored data. If it is a proxy, it will be resolved now."""
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug("Resolving %s for %s '%s' ...",
                      self._data.__class__.__name__,
                      self.classname, self.name)
            self._data = self._data.resolve()

        # Now, the data should be loaded and can be returned
        return self._data

    @data.setter
    def data(self, val):
        """Set the data attribute. If it is a Proxy, it is first resolved."""
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug("Resolving %s for %s '%s' ...",
                      self._data.__class__.__name__,
                      self.classname, self.name)
            self._data = self._data.resolve()

        # Now can set the value
        self._data = val

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data"""
        return isinstance(self._data, BaseDataProxy)

    @property
    def proxy_data(self):
        """If the data is proxy, returns the proxy data object without using the .data attribute (which would trigger resolving the proxy); else returns None."""
        if self.data_is_proxy:
            return self._data
        return None

    # .........................................................................
    # Item access

    @abc.abstractmethod
    def __getitem__(self, key):
        """Get an item from the data"""
        pass

    @abc.abstractmethod
    def __setitem__(self, key, val) -> None:
        """Set an item in the data."""
        pass

    # .........................................................................
    # Formatting

    @abc.abstractmethod
    def __str__(self) -> str:
        """An info string, that describes the object. Each class should implement this to return an informative response."""
        pass

    def __format__(self, spec_str: str) -> str:
        """Creates a formatted string from this """
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
                                           self.classname)) from err
            else:
                parts.append(format_func())

        return join_char.join(parts)

    def _format_name(self) -> str:
        """A __format__ helper function: returns the name"""
        return self.name
    
    def _format_cls_name(self) -> str:
        """A __format__ helper function: returns the class name"""
        return self.classname

    def _format_info(self) -> str:
        """A __format__ helper function: returns the info string"""
        if not self.data_is_proxy:
            return str(self)
        return str(self.proxy_data)


# -----------------------------------------------------------------------------

class BaseDataAttrs(AbstractDataContainer):
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

class BaseDataContainer(AbstractDataContainer):
    """The BaseDataContainer extends the base class by its ability to holds attributes."""

    def __init__(self, *, name: str, data, attrs=None):
        """Initialise a BaseDataContainer, which can store data and attributes."""
        log.debug("BaseDataContainer.__init__ called.")

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Property-managed attributes
        self._attrs = None

        # Store the attributes object
        self.attrs = attrs

        log.debug("BaseDataContainer.__init__ finished.")

    # .........................................................................
    # Methods needed for attribute access

    @property
    def attrs(self) -> BaseDataAttrs:
        """The container attributes."""
        return self._attrs

    @attrs.setter
    @abc.abstractmethod
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        pass

    # .........................................................................
    # Methods needed for location relative to other groups

    @abc.abstractmethod
    @property
    def parent(self) -> Union[None, BaseDataGroup]:
        """The parent data group or None if on the highest level."""
        pass

    @abc.abstractmethod
    @property
    def path(self) -> str:
        """Return the path to get to this data group"""
        pass

    # .........................................................................
    # Methods needed for data container conversion

    @abc.abstractmethod
    def convert_to(self, TargetCls, **target_init_kwargs):
        """With this method, a TargetCls object can be created from this
        particular container instance.
        
        Conversion might not be possible if TargetCls requires more information
        than is available in this container.
        """
        pass

# -----------------------------------------------------------------------------

class BaseDataGroup(BaseDataContainer):
    """The BaseDataGroup serves as base group for all data groups."""

    @abc.abstractmethod
    @property
    def tree_repr(self) -> dict:
        """Returns a tree representation of the contents of this group, stored in a dict.

        The returned dict has as keys the entries and as values either dicts
        (for group-like object) or parsed strings (for container-like objects)
        """
        pass

    @abc.abstractmethod
    def _format_tree(self) -> str:
        """Returns a multi-line string tree representation of this group."""
        pass

# -----------------------------------------------------------------------------

class BaseDataProxy(metaclass=abc.ABCMeta):
    """A data proxy fills in for the place of a data container, e.g. if data should only be loaded on demand."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """Resolve the proxy object, returning the actual object."""
        pass

    @abc.abstractmethod
    def resolve(self):
        """Resolve the proxy object, returning the actual object."""
        pass
