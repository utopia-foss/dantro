"""This module implements the base classes of dantro based on the abstract classes.

The base classes are classes that combine features of the abstract classes. For
example, the data group gains attribute functionality by being a combination
of the AbstractDataGroup and the BaseDataContainer.
In turn, the BaseDataContainer uses the BaseDataAttrs class as an attribute and
thereby extends the AbstractDataContainer class.

NOTE: These classes are not meant to be instantiated.
"""

import abc
import copy
import logging
import warnings
import inspect
from typing import Union, List

import dantro.abc
from dantro.abc import PATH_JOIN_CHAR
import dantro.tools as tools

# Local constants
log = logging.getLogger(__name__)


class UnexpectedTypeWarning(UserWarning):
    """Given when there was an unexpected type passed to a data container."""
    pass

# -----------------------------------------------------------------------------
# Mixins ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


class AttrsMixin:
    """This Mixin class supplies the `attrs` property getter and setter and the private `_attrs` attribute.

    Hereby, the setter function will initialize a BaseDataAttrs-derived object
    and store it as an attribute.
    This relays the checking of the correct attribute format to the actual
    BaseDataAttrs-derived class.

    For changing the class that is used for the attributes, an overwrite of the
    _ATTRS_CLS class variable suffices.
    """
    # Define the class variables
    _attrs = None
    _ATTRS_CLS = None

    @property
    def attrs(self):
        """The container attributes."""
        return self._attrs

    @attrs.setter
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        # Decide which class to use for attributes
        AttrsCls = (BaseDataAttrs
                    if self._ATTRS_CLS is None
                    else self._ATTRS_CLS)

        # Perform the initialisation
        log.debug("Using %s for attributes of %s",
                  AttrsCls.__name__, self.logstr)
        self._attrs = AttrsCls(name='attrs', attrs=new_attrs)


class PathMixin:
    """This Mixin class implements path capabilities for groups or containers.

    That means, that each object can re-create the path at which it can be
    accessed _if_ it knows its parent object."""
    # Define the needed class variables
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
        key = self.__item_convert_key(key)
        return self.data[key]

    def __setitem__(self, key, val):
        """Sets an item."""
        key = self.__item_convert_key(key)
        self.data[key] = val

    def __delitem__(self, key):
        """Deletes an item"""
        key = self.__item_convert_key(key)
        del self.data[key]

    def __item_convert_key(self, key):
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
            classes derived from BaseDataProxy
        DATA_EXPECTED_TYPES (tuple, None): Which types to allow. If None, all
            types are allowed.
        DATA_UNEXPECTED_ACTION (str): The action to take when an unexpected
            type was supplied. Can be: raise, warn, ignore
    """

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = None       # as tuple or None (allow all)
    DATA_ALLOW_PROXY = False         # to check for BaseDataProxy
    DATA_UNEXPECTED_ACTION = 'warn'  # Can be: raise, warn, ignore

    def _check_data(self, data, *, name: str) -> bool:
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
            return True

        # Compile tuple of allowed types
        expected_types = self.DATA_EXPECTED_TYPES

        if self.DATA_ALLOW_PROXY:
            expected_types += (BaseDataProxy,)

        # Perform the check
        if isinstance(data, expected_types):
            # Is of the expected type
            return True

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
            pass

        else:
            raise ValueError("Illegal value '{}' for class variable "
                             "DATA_UNEXPECTED_ACTION of {}. "
                             "Allowed values are: raise, warn, ignore"
                             "".format(self.DATA_UNEXPECTED_ACTION,
                                       self.classname))
        
        return False

# -----------------------------------------------------------------------------
# Base classes ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class BaseDataProxy(dantro.abc.AbstractDataProxy):
    """The base class for data proxies.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    @abc.abstractmethod
    def __init__(self, obj):
        """Initialize a proxy object for the given object."""
        super().__init__(obj)
        log.debug("Initialising %s for %s ...", self.classname, type(obj))


# -----------------------------------------------------------------------------

class BaseDataAttrs(MappingAccessMixin, dantro.abc.AbstractDataAttrs):
    """A class to store attributes that belong to a data container.

    This implements a dict-like interface and serves as default attribute class.

    NOTE: Unlike the other base classes, this can already be instantiated. That
    is required as it is needed in BaseDataContainer where no previous
    subclassing or mixin is reasonable.
    """

    def __init__(self, attrs: dict=None, **dc_kwargs):
        """Initialize a DataAttributes object.
        
        Args:
            attrs (dict, optional): The attributes to store
            **dc_kwargs: Further kwargs to the parent DataContainer
        """
        # Make sure it is a dict; initialize empty if empty
        attrs = dict(attrs) if attrs else {}

        # Store them via the parent method.
        super().__init__(data=attrs, **dc_kwargs)

        log.debug("BaseDataAttrs.__init__ finished.")

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about these attributes"""
        return "{} attribute(s)".format(len(self))


# -----------------------------------------------------------------------------

class BaseDataContainer(PathMixin, AttrsMixin, dantro.abc.AbstractDataContainer):
    """The BaseDataContainer extends the abstract base class by the ability to
    hold attributes and be path-aware.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    @abc.abstractmethod
    def __init__(self, *, name: str, data, attrs=None):
        """Initialize a BaseDataContainer, which can store data and attributes.
        
        Args:
            name (str): The name of this data container
            data: The data to store in this container
            attrs (None, optional): A mapping that is stored as attributes
        """
        log.debug("BaseDataContainer.__init__ called.")

        # Supply the data to the _check_data method, which can be adjusted
        # to test the provided data. In this base class, the method does
        # nothing and serves only as a placeholder
        self._check_data(data, name=name)

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Store the attributes object
        self.attrs = attrs

        log.debug("BaseDataContainer.__init__ finished.")

    def _check_data(self, data, *, name: str) -> bool:
        """This method can be used to check the data provided to __init__.
        
        It is called before the data is stored via the parent's __init__ and
        should raise an exception or create a warning if the data is not as
        desired.
        
        NOTE: The CheckDataMixin provides a generalised function to perform
        some type checks and react to unexpected types.
        
        Args:
            data: The data to check
        
        Returns:
            bool: Whether the data passed the check.
        """
        return True

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the items"""
        return "{} stored, {} attributes".format(type(self.data),
                                                 len(self.attrs))
        

# -----------------------------------------------------------------------------

class BaseDataGroup(PathMixin, AttrsMixin, dantro.abc.AbstractDataGroup):
    """The BaseDataGroup serves as base group for all data groups.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    def __init__(self, *, name: str, containers: list=None, attrs=None, StorageCls: type=dict):
        """Initialize a BaseDataGroup, which can store other containers and
        attributes.
        
        Args:
            name (str): The name of this data container
            containers (list, optional): The containers to store in this group
            attrs (None, optional): A mapping that is stored as attributes
            StorageCls (type, optional): in which type of object to store the
                containers.
        """
        log.debug("BaseDataGroup.__init__ called.")

        # Prepare the storage class that is used as `data` attribute
        data = StorageCls()

        # Basic initialisation via parent method
        super().__init__(name=name, data=data)

        # Store the attributes object
        self.attrs = attrs

        # Now fill the storage
        if containers is not None:
            self.add(*containers)

        log.debug("BaseDataGroup.__init__ finished.")

    # .........................................................................
    # Item access

    def __getitem__(self, key: Union[str, List[str]]):
        """Returns the container in this group with the given name.
        
        Args:
            key (Union[str, List[str]]): The object to retrieve.
                If this is a path, will recurse down until at the end.
        
        Returns:
            The object at `key`
        
        Raises:
            KeyError: If no such key can be found
        """
        if not isinstance(key, list):
            # Assuming this is a string, i.e.: the only other type allowed
            key = key.split(PATH_JOIN_CHAR)

        # Can be sure that this is a list now
        try:
            # If there is more than one entry, need to call this recursively
            if len(key) > 1:
                return self.data[key[0]][key[1:]]
            # else: end of recursion
            return self.data[key[0]]

        except (KeyError, IndexError) as err:
            raise KeyError("No key or key sequence '{}' in {}! "
                           "Available keys at top level: {}"
                           "".format(key, self.logstr,
                                     ", ".join([k for k in self.keys()]))
                           ) from err

    def __setitem__(self, key: Union[str, List[str]], val: BaseDataContainer) -> None:
        """This method is used to allow access to the content of containers of
        this group. For adding an element to this group, use the `add` method!
        
        Args:
            key (Union[str, List[str]]): The key to which to set the value.
                If this is a path, will recurse down to the lowest level.
                Note that all intermediate keys need to be present.
            val (BaseDataContainer): The value to set
        
        Returns:
            None
        
        Raises:
            ValueError: If trying to add an element to this group, which should
                be done via the `add` method.
        
        """
        if not isinstance(key, list):
            key = key.split(PATH_JOIN_CHAR)

        # Depending on length of the key sequence, start recursion or not
        if len(key) > 1:
            print(key, self.data)
            self.data[key[0]][key[1:]] = val
            return
        
        # else: end of recursion, i.e. the path led to an item of this group
        # This operation is not allowed, as the add method should be used
        # That method takes care that the name this element is registered with
        # is equal to that of the registered object
        raise ValueError("{} cannot carry out __setitem__ operation for the "
                         "given key '{}'. Note that to add a group or "
                         "container to the group, the `add` method should "
                         "be used.".format(self.logstr, key))

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
            return

        # else: end of recursion: delete and unlink this container
        cont = self.data[key[0]]
        del self.data[key[0]]

        self._unlink_child(cont)

    def add(self, *conts, overwrite: bool=False):
        """Add the given containers to this group."""
        for cont in conts:
            if not isinstance(cont, (BaseDataGroup, BaseDataContainer)):
                raise TypeError("Can only add BaseDataGroup- or "
                                "BaseDataContainer-derived objects to {}, "
                                "got {}!".format(self.logstr, type(cont)))

            # else: is of correct type
            # Get the name and check if one like this already exists
            if cont.name in self:
                if not overwrite:
                    raise ValueError("{} already has a member with "
                                     "name '{}', cannot add {}."
                                     "".format(self.logstr, cont.name, cont))
                log.debug("Overwriting member '%s' of %s ...",
                          cont.name, self.logstr)
                old_cont = self[cont.name]
            
            else:
                old_cont = None

            # Write to data, assuring that the name is that of the container
            self.data[cont.name] = cont

            # Re-link
            self._link_child(new_child=cont, old_child=old_cont)

        log.debug("Added %d container(s) to %s.", len(conts), self.logstr)

    def new_container(self, path: Union[str, list], *, Cls: type, **kwargs):
        """Creates a new container of class `Cls` and adds it at the given path
        relative to this group.
        
        Args:
            path (Union[str, list]): Where to add the container. Note that the
                intermediates of this path need to already exist.
            Cls (type): The class of the container to add
            **kwargs: kwargs to pass on to Cls.__init__
        
        Returns:
            Cls: the created container
        
        Raises:
            KeyError: Description
            TypeError: Description
        """
        # Check the class to create the container with
        if not inspect.isclass(Cls):
            raise TypeError("Argument `Cls` needs to be a class, but "
                            "was of type {} with value '{}'."
                            "".format(type(Cls), Cls))

        elif not issubclass(Cls, (BaseDataContainer, BaseDataGroup)):
            raise TypeError("Argument `Cls` needs to be a subclass of "
                            "BaseDataContainer or BaseDataGroup, was '{}'."
                            "".format(Cls))
        # Class is checked now

        # Make sure the path is a list
        if not isinstance(path, list):
            path = path.split(PATH_JOIN_CHAR)

        # Check whether recursion ends here, i.e.: the path ends here
        if len(path) == 1:
            # Yes, and it's a string: create container, add, return
            cont = Cls(name=path[0], **kwargs)
            self.add(cont)
            return cont

        # Recursive branch: need to split off the front section and continue
        grp_name, new_path = path[0], path[1:]

        try:
            return self[grp_name].new_container(new_path, Cls=Cls, **kwargs)

        except KeyError as err:
            raise KeyError("Could not create {} at '{}'! Check that all "
                           "intermediate groups have already been created. "
                           "Entries available in {}: {}"
                           "".format(Cls.__name__,
                                     PATH_JOIN_CHAR.join(path), self.logstr,
                                     ", ".join([k for k in self.keys()]))
                           ) from err

    def new_group(self, path: Union[str, list], *, Cls: type=None, **kwargs):
        """Creates a new group at the given path.
        
        Args:
            path (Union[str, list]): The path to create the group at. Note
                that the whole intermediate path needs to already exist.
            Cls (type, optional): If given, use this type to create the
                group. If not given, uses the type of this instance.
            **kwargs: Passed on to Cls.__init__
        
        Returns:
            Cls: the created group
        
        Raises:
            TypeError: For the given class not being derived from BaseDataGroup
        """
        # If no Cls is given, use this instance's type
        if Cls is None:
            Cls = type(self)

        # Need to catch the case where a non-group class was given
        elif inspect.isclass(Cls) and not issubclass(Cls, BaseDataGroup):
            raise TypeError("Argument `Cls` needs to be a subclass of "
                            "BaseDataGroup, was '{}'.".format(Cls))

        # Use container method to create the entry. Recursion happens there.
        return self.new_container(path, Cls=Cls, **kwargs)

    def recursive_update(self, other):
        """Recursively updates the contents of this data group with the entries
        of the given data group
        
        NOTE This will create shallow copies of those elements in `other` that
        are added to this object.
        
        Args:
            other (BaseDataGroup): The group to update with
        
        Raises:
            TypeError: If `other` was of invalid type
        """

        if not isinstance(other, BaseDataGroup):
            raise TypeError("Can only update {} with objects of classes that "
                            "are derived from BaseDataGroup. Got: {}"
                            "".format(self.logstr, type(other)))

        # Loop over the given DataGroup
        for name, obj in other.items():
            # Distinguish between the case where it is another group and where
            # it is a container
            if isinstance(obj, BaseDataGroup):
                # Already a group -> if a group with the same name is already
                # present, continue recursion. If not, just create an entry
                # and add it to this group
                if name in self:
                    # Continue recursion
                    self[name].recursive_update(obj)
                else:
                    # Can add the object, but need to detach the parent first
                    # and thus need to work on a copy
                    obj = copy.copy(obj)
                    obj.parent = None
                    self.add(obj)

            else:
                # Not a group; add a shallow copy
                obj = copy.copy(obj)
                obj.parent = None
                self.add(obj)

        log.debug("Finished recursive update of %s.", self.logstr)

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
                             "be linked. Use the add method to add the child "
                             "to the group.".format(new_child.logstr,
                                                    self.logstr))
        new_child.parent = self

        if old_child is not None:
            self._unlink_child(old_child)

    def _unlink_child(self, child: BaseDataContainer):
        """Unlink a child from this class.

        This method should be called from any method that removes an item from
        this group, be it through deletion or through 
        """
        if child.parent is not self:
            raise ValueError("{} was not linked to {}. Refuse to unlink."
                             "".format(child.logstr, self.logstr))
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
        if isinstance(cont, (BaseDataGroup, BaseDataContainer)):
            # Implement a contains-check via is-relationships
            for obj in self.values():
                if obj is cont:
                    return True 
            return False

        # Check via the string, if such a key is available
        elif isinstance(cont, str):
            key_seq = cont.split(PATH_JOIN_CHAR)

        else:
            # Need to check its type
            if not isinstance(cont, list):
                raise TypeError("Can only check content of {} against the "
                                "container name or path (as string), or the "
                                "explicit reference to the object! "
                                "Got {} with value '{}'."
                                "".format(self.logstr, type(cont), cont))

            key_seq = cont

        # key_seq is now a list of keys
        if len(key_seq) > 1:
            # More than one element.
            # If the target element exists and is a group, continue recursively
            if (key_seq[0] in self
                and isinstance(self[key_seq[0]], BaseDataGroup)):
                return key_seq[1:] in self[key_seq[0]]

            # else: does not exist or is not a group
            return False
        
        elif len(key_seq) == 1:
            return bool(key_seq[0] in self.keys())

        return False

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
        """Returns an iterator over the (name, data container) tuple of this
        group."""
        return self.data.items()

    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name
        `key` is not available."""
        return self.data.get(key, default)

    def setdefault(self, key, default=None):
        """This method is not supported for a data group"""
        raise NotImplementedError("setdefault is not supported by {}! Use the "
                                  "`add` method or `new_group` and "
                                  "`new_container` to add elements."
                                  "".format(self.classname))

    # .........................................................................
    # Formatting

    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used
        to characterise this object. Does NOT include name and classname!"""
        return "{} members, {} attributes".format(len(self), len(self.attrs))

    def _format_tree(self) -> str:
        """Returns a multi-line string tree representation of this group."""
        return self._tree_repr()

    def _tree_repr(self, level: int=0, info_fstr="<{:cls_name,info}>", info_ratio: float=0.5) -> str:
        """Recursively creates a multi-line string tree representation of this
        group. This is used by, e.g., the _format_tree method.
        
        Args:
            level (int, optional): The depth within the tree
            info_fstr (str, optional): The format string for the info string
            info_ratio (float, optional): The width ratio of the whole line
                width that the info string takes
        
        Returns:
            str: The (multi-line) tree representation of this group
        """
        # Mark symbols
        first_mark = r' └┬'
        base_mark =  r'  ├'
        last_mark =  r'  └'
        only_mark =  r' └─'

        # Offset
        offset = "   " * level
            
        # Format string
        fstr = "{offset:}{mark:>3s} {name:<{name_width}s}  {info:}"
        
        # Calculations
        num_items = len(self)
        num_cols = tools.TTY_COLS

        info_width = int(num_cols * info_ratio)
        name_width = (num_cols - info_width) - (len(offset) + 3 + 1 + 2)       

        # Helper function
        def get_mark(n: int) -> str:
            """Helper function that returns the mark symbol depending on the
            iteration number."""
            if n == 0:
                if num_items == 1:
                    return only_mark
                return first_mark
            elif n == num_items - 1:
                return last_mark
            return base_mark

        # Create the list to gather the lines in; add a description on level 0
        lines = []
        if level == 0:
            lines.append("")
            lines.append("Tree of {:logstr,info}".format(self))

        # Go over the entries on this level and format the lines
        for n, (key, obj) in enumerate(self.items()):
            # Get key and info, truncate if necessary
            name = key if len(key) <= name_width else key[:name_width-1]+"…"
            info = info_fstr.format(obj)
            info = info if len(info) <= info_width else info[:info_width-1]+"…"

            # Get the mark, depending on the item number
            mark = get_mark(n)

            # Format the line and add to list of lines
            line = fstr.format(offset=offset, mark=mark, name_width=name_width,
                               name=name, info=info)
            lines.append(line)

            # If it was a group and it is not empty...
            if isinstance(obj, BaseDataGroup) and len(obj) > 0:
                # ...continue recursion
                lines += obj._tree_repr(level=level+1, info_fstr=info_fstr)

        # Done, return them.
        if level > 0:
            # Within recursion: return the list of lines
            return lines
        
        # Highest level; join the lines together and return that string
        lines.append("")
        return "\n".join(lines)
