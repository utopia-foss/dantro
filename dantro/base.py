"""This module implements the base classes of dantro."""

import abc
import collections
import logging
from typing import Union

# Setup logging for this file
log = logging.getLogger(__name__)

# Local constants
PATH_JOIN_CHAR = "/"

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
        
        # Pass name and data to read-only attributes
        self._name = str(name)
        self._data = data

        log.debug("Initialising %s '%s' ...", self.classname, self.name)

        # Nothing else to do here ...

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
        return "<{} '{}', {}>".format(self.classname, self.name,
                                      self._format_info())

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

    @abc.abstractmethod
    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used to characterise this object. Should NOT include name and classname!"""
        pass


# -----------------------------------------------------------------------------

class BaseDataAttrs(collections.abc.Mapping, AbstractDataContainer):
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

class DataAttrs(BaseDataAttrs):
    """A class to store attributes that belong to a data container.

    This implements a dict-like interface and serves as default attribute class."""

    def __init__(self, attrs: dict=None, **dc_kwargs):
        """Initialise a DataAttributes object.
        
        Args:
            attrs (dict, optional): The attributes to store
            **dc_kwargs: Further kwargs to the parent DataContainer
        """
        # Make sure it is a dict; initialise empty if empty
        attrs = dict(attrs) if attrs else {}

        # Store them via the parent method.
        super().__init__(data=attrs, **dc_kwargs)

        log.debug("BaseDataAttrs.__init__ finished.")
    
    # .........................................................................
    # Magic methods and iterators for convenient dict-like access

    def __str__(self) -> str:
        return "{} attributes".format(len(self))

    def __getitem__(self, key):
        """Returns an attribute."""
        return self.data[key]

    def __setitem__(self, key, val):
        """Sets an attribute."""
        log.debug("Setting attribute '%s' to '%s' ...", key, val)
        self.data[key] = val

    def __delitem__(self, key):
        """Deletes an attribute"""
        del self.data[key]

    def __contains__(self, key) -> bool:
        """Whether the given key is contained in the attributes."""
        return bool(key in self.data)

    def __len__(self) -> int:
        """The number of attributes."""
        return len(self.data)

    def __iter__(self):
        """Iterates over the attribute keys."""
        return iter(self.data)

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

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about these attributes"""
        return str(len(self)) + " attributes"


# -----------------------------------------------------------------------------

class BaseDataContainer(AbstractDataContainer):
    """The BaseDataContainer extends the base class by its ability to holds attributes."""

    def __init__(self, *, name: str, data, parent, attrs=None):
        """Initialise a BaseDataContainer, which can store data and attributes.
        
        Args:
            name (str): The name of this data container
            data (TYPE): The data to store in this container
            parent (TYPE): The parent object (or None if at the top)
            attrs (None, optional): A mapping that is stored as attributes
        """
        log.debug("BaseDataContainer.__init__ called.")

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Property-managed attributes
        self._attrs = None
        self._parent = parent

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
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        self._attrs = DataAttrs(name='attrs', attrs=new_attrs)

    # .........................................................................
    # Methods needed for location relative to other groups

    @property
    def parent(self):
        """The group this container is contained in."""
        return self._parent

    @property
    def path(self) -> str:
        """Return the path to get to this container"""
        if self.parent is None:
            return PATH_JOIN_CHAR + self.name
        # else: not at the top, also need the parent's path
        return self.parent.path + PATH_JOIN_CHAR + self.name

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

    # .........................................................................
    # Formatting

    def _format_path(self) -> str:
        """A __format__ helper function: returns the path to this container"""
        return self.path

# -----------------------------------------------------------------------------

class BaseDataGroup(collections.abc.MutableMapping, BaseDataContainer):
    """The BaseDataGroup serves as base group for all data groups.

    It enforces a MutableMapping interface with a focus on setting abilities
    and less so on deletion, i.e.
    """

    @abc.abstractmethod
    def __contains__(self, cont: Union[str, BaseDataContainer]) -> bool:
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
        """Returns a multi-line string tree representation of this group."""
        pass

    # Can already resolve the conversion method; this should always be the same
    def convert_to(self, TargetCls, **target_init_kwargs):
        """ """
        return TargetCls(name=self.name, data=self.data)


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
