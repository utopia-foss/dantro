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
from typing import Union

import dantro.abc

# Local constants
log = logging.getLogger(__name__)
PATH_JOIN_CHAR = "/"

# -----------------------------------------------------------------------------
# Mixins ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

class ProxyMixin:

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


class PathMixin:

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

    def _format_path(self) -> str:
        """A __format__ helper function: returns the path to this container"""
        return self.path


class AttrsMixin:

    @property
    def attrs(self):
        """The container attributes."""
        return self._attrs

    @attrs.setter
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        self._attrs = BaseDataAttrs(name='attrs', attrs=new_attrs)


# -----------------------------------------------------------------------------
# Base classes ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class BaseDataProxy(dantro.abc.AbstractDataProxy):
    """The base class for data proxies.

    NOTE: This is still an abstract class and needs to be subclassed.
    """
    # Nothing to define here; the resolve method needs to be data-specific
    pass


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

class BaseDataContainer(PathMixin, ProxyMixin, AttrsMixin, dantro.abc.AbstractDataContainer):
    """The BaseDataContainer extends the base class by its ability to holds attributes.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    @abc.abstractmethod
    def __init__(self, *, name: str, data, attrs=None):
        """Initialise a BaseDataContainer, which can store data and attributes.
        
        Args:
            name (str): The name of this data container
            data (TYPE): The data to store in this container
            attrs (None, optional): A mapping that is stored as attributes
        """
        log.debug("BaseDataContainer.__init__ called.")

        # Prepare the data (via abstract helper method)
        data = self._prepare_data(data=data)

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Property-managed attributes
        self._attrs = None
        self._parent = None

        # Store the attributes object
        self.attrs = attrs

        log.debug("BaseDataContainer.__init__ finished.")

    @abc.abstractmethod
    def _prepare_data(self, *, data):
        """Called by __init__, this method should parse the given `data`
        argument to a desirable form.
        """
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

class BaseDataGroup(PathMixin, ProxyMixin, AttrsMixin, dantro.abc.AbstractDataGroup):
    """The BaseDataGroup serves as base group for all data groups.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    def __init__(self, *, name: str, containers: list=None, attrs=None):
        """Initialise a BaseDataGroup, which can store other containers and attributes.
        
        Args:
            name (str): The name of this data container
            data (TYPE): The data to store in this container
            attrs (None, optional): A mapping that is stored as attributes
        """
        log.debug("BaseDataGroup.__init__ called.")

        # Prepare the data (via abstract helper method)
        data = self._prepare_data(containers=containers)

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Property-managed attributes
        self._attrs = None
        self._parent = None

        # Store the attributes object
        self.attrs = attrs

        log.debug("BaseDataGroup.__init__ finished.")

    @abc.abstractmethod
    def _prepare_data(self, *, containers: list) -> dict:
        """Called by __init__, this method should parse the arguments `data`
        and `containers` which are passed to __init__. As return value,
        a dict-like object is expected.
        """
        pass

    # .........................................................................
    # Item access

    def __getitem__(self, key: str):
        """Returns the container in this group with the given name.
        
        Args:
            key (str): The object to retrieve. If this is a path, will recurse
                down until at the end.
        
        Returns:
            The object at `key`
        """
        if not isinstance(key, list):
            # Assuming this is a string ...
            key = key.split(PATH_JOIN_CHAR)

        # Can be sure that this is a list now
        # If there is more than one entry, need to call this recursively
        if len(key) > 1:
            return self.data[key[0]][key[1:]]
        # else: end of recursion
        return self.data[key[0]]

    def __setitem__(self, key: str, val: BaseDataContainer) -> None:
        """Set the value at the given `key` of the group.
        
        Args:
            key (str): The key to which to set the value. If this is a path,
                will recurse down to the lowest level. Note that all inter-
                mediate keys need to be present.
            val: The value to set
        
        """
        if not isinstance(key, list):
            key = key.split(PATH_JOIN_CHAR)

        # Depending on length of the key sequence, start recursion or not
        if len(key) > 1:
            self.data[key[0]][key[1:]] = val
        
        # else: end of recursion, set the value
        old_val = self.data.get(key[0])
        self.data[key[0]] = val

        # Update the links
        self._link_child(new_child=val, old_child=old_val)

    def __delitem__(self, key: str) -> None:
        """Deletes an item from the group"""

        if not isinstance(key, list):
            # Assuming this is a string ...
            key = key.split(PATH_JOIN_CHAR)

        # Can be sure that this is a list now
        # If there is more than one entry, need to call this recursively
        if len(key) > 1:
            # Continue recursion
            del self.data[key[0]][key[1:]]
        # else: end of recursion: delete and unlink this container
        cont = self.data[key[0]]
        del self.data[key[0]]

        self._unlink_child(cont)

    # .........................................................................
    # Linking

    # For correct child-parent linking, some helper methods
    def _link_child(self, *, new_child: BaseDataContainer, old_child: BaseDataContainer=None):
        """Links the new_child to this class, unlinking the old one.

        This method should be called from any method that changes which items
        are associated with this group.
        """
        # Check that it was already associated
        if new_child not in self:
            raise ValueError("{} needs to be a child of {} _before_ it can "
                             "be linked.".format(new_child.logstr,
                                                 self.logstr))
        new_child.parent = self

        if old_child is not None:
            self._unlink_child(old_child)

    def _unlink_child(self, child: BaseDataContainer):
        """Unlink a child from this class.

        This method should be called from any method that removes an item from
        this group, be it through deletion or through 
        """
        if child not in self:
            raise ValueError("{} is no child of {}!".format(child.logstr,
                                                            self.logstr))
        child.parent = None

    # .........................................................................
    # Information

    def __len__(self) -> int:
        """The length of the data."""
        return len(self.data)

    def __contains__(self, cont: Union[str, BaseDataContainer]) -> bool:
        """Whether the given container is in this group or not.
        
        Args:
            cont (Union[str, BaseDataContainer]): The name of the container or 
                an object reference. 
        
        Returns:
            bool: Whether the given container is in this group.
        """
        if isinstance(cont, BaseDataContainer):
            return bool(cont in self.values())

        elif not isinstance(cont, list):
            # assume it is a string
            key_seq = cont.split(PATH_JOIN_CHAR)

        else:
            key_seq = cont

        # is a list of keys, might have to check recursively
        if len(key_seq) > 1:
            return bool(key_seq[1:] in self[key_seq[0]])
        return bool(key_seq[0] in self.keys())

    # .........................................................................
    # Iteration

    def __iter__(self):
        """Returns an iterator over the OrderedDict"""
        return iter(self.data)

    def keys(self):
        """Returns an iterator over the container names in this group."""
        return self.data.keys()

    def values(self):
        """Returns an iterator over the containers in this group."""
        return self.data.values()

    def items(self):
        """Returns an iterator over the (name, data container) tuple of this group."""
        return self.data.items()

    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name `key` is not available."""
        return self.data.get(key, default)

    def setdefault(self, key, default=None):
        """If `key` is in the dictionary, return its value. If not, insert `key` with a value of `default` and return `default`. `default` defaults to None."""
        if key in self:
            return self[key]
        # else: not available
        self.data[key] = default
        return default

    # .........................................................................
    # Formatting

    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used to characterise this object. Does NOT include name and classname!"""
        return str(len(self)) + " members"

    def _format_tree(self) -> str:
        """Returns a multi-line string tree representation of this group."""
        raise NotImplementedError

    # .........................................................................
    # Conversion
    
    def convert_to(self, TargetCls, **target_init_kwargs):
        """Convert this BaseDataGroup to TargetCls by passing data and attrs"""
        log.debug("Converting %s '%s' to %s ...", self.classname, self.name,
                  TargetCls.__name__)
        return TargetCls(name=self.name, parent=self.parent,
                         data=self.data, attrs=self.attrs,
                         **target_init_kwargs)
