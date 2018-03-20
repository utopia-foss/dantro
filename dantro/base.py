"""This module implements the base classes of dantro based on the abstract classes.

The base classes are classes that combine features of the abstract classes. For
example, the data group gains attribute functionality by being a combination
of the AbstractDataGroup and the BaseDataContainer.
In turn, the BaseDataContainer uses the BaseDataAttrs class as an attribute and
thereby extends the AbstractDataContainer class.

NOTE: These classes are not meant to be instantiated.
"""

import abc
import logging

import dantro.abc

# Setup logging for this file
log = logging.getLogger(__name__)

# Local constants
PATH_JOIN_CHAR = "/"

# -----------------------------------------------------------------------------

class BaseDataAttrs(dantro.abc.AbstractDataAttrs):
    """A class to store attributes that belong to a data container.

    This implements a dict-like interface and serves as default attribute class.

    NOTE: Unlike the other base classes, this can already be instantiated. That
    is required as it is needed in BaseDataContainer where no previous
    subclassing or mixin is reasonable.
    """

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

class BaseDataContainer(dantro.abc.AbstractDataContainer):
    """The BaseDataContainer extends the base class by its ability to holds attributes.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    def __init__(self, *, name: str, data, attrs=None):
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
        self._parent = None

        # Store the attributes object
        self.attrs = attrs

        log.debug("BaseDataContainer.__init__ finished.")

    # .........................................................................
    # Methods needed for handling data that is proxy

    @property
    def data(self):
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
    # Methods needed for attribute access

    @property
    def attrs(self) -> BaseDataAttrs:
        """The container attributes."""
        return self._attrs

    @attrs.setter
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        self._attrs = BaseDataAttrs(name='attrs', attrs=new_attrs)

    # .........................................................................
    # Methods needed for location relative to other groups

    @property
    def parent(self):
        """The group this container is contained in or None if on its own."""
        return self._parent

    @parent.setter
    def parent(self, cont):
        """Associate a parent object with this container."""
        if self.parent is not None and cont is not None:
            log.warning("A parent was already associated with %s '%s'! Will "
                        "ignore this assignment.", self.classname, self.name)
        else:
            log.debug("Setting %s as parent of %s ...",
                      cont.logstr if cont else None, self.logstr)
            self._parent = cont

    @property
    def path(self) -> str:
        """Return the path to get to this container"""
        if self.parent is None:
            # At the top or no parent associated -> no reasonable path to give
            return PATH_JOIN_CHAR
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

class BaseDataGroup(BaseDataContainer, dantro.abc.AbstractDataGroup):
    """The BaseDataGroup serves as base group for all data groups.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    def _link_child(self, child_cont: BaseDataContainer) -> None:
        """Links the child container to this group by setting the child's
        parent attribute."""
        child_cont.parent = self

    @staticmethod
    def _unlink_child(child_cont: BaseDataContainer) -> None:
        """Upon removal of a child from this container, unlinks the associated
        parent attribute."""
        child_cont.parent = None

    # Conversion between groups should always be the same; define here:
    def convert_to(self, TargetCls, **target_init_kwargs):
        """Convert this BaseDataGroup to TargetCls by passing data and attrs"""
        log.debug("Converting %s '%s' to %s ...", self.classname, self.name,
                  TargetCls.__name__)
        return TargetCls(name=self.name, parent=self.parent,
                         data=self.data, attrs=self.attrs,
                         **target_init_kwargs)


# -----------------------------------------------------------------------------

class BaseDataProxy(dantro.abc.AbstractDataProxy):
    """The base class for data proxies.

    NOTE: This is still an abstract class and needs to be subclassed.
    """
    # Nothing to define here; the resolve method needs to be data-specific
    pass
