"""This module implements the base classes of dantro, based on the abstract
classes.

The base classes are classes that combine features of the abstract classes. For
example, the data group gains attribute functionality by being a combination
of the AbstractDataGroup and the BaseDataContainer.
In turn, the BaseDataContainer uses the BaseDataAttrs class as an attribute and
thereby extends the AbstractDataContainer class.

NOTE: These classes are not meant to be instantiated but used as a basis to
implement more specialized BaseDataGroup- or BaseDataContainer-derived classes.
"""

import abc
import copy
import logging
import inspect
from typing import Union, List, Tuple, Callable

import dantro.abc
from .abc import PATH_JOIN_CHAR
from .mixins import AttrsMixin, SizeOfMixin, CheckDataMixin, LockDataMixin
from .mixins import CollectionMixin, ItemAccessMixin, MappingAccessMixin
from .tools import TTY_COLS

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class BaseDataProxy(dantro.abc.AbstractDataProxy):
    """The base class for data proxies.

    NOTE: This is still an abstract class and needs to be subclassed.
    """

    @abc.abstractmethod
    def __init__(self, obj):
        """Initialize a proxy object for the given object."""
        log.trace("Initialising %s for %s ...", self.classname, type(obj))

        # Specify the tags of this proxy; none by default
        self._tags = tuple()

    @property
    def tags(self) -> Tuple[str]:
        """The tags describing this proxy object"""
        return self._tags


# -----------------------------------------------------------------------------

class BaseDataAttrs(MappingAccessMixin, dantro.abc.AbstractDataAttrs):
    """A class to store attributes that belong to a data container.

    This implements a dict-like interface and serves as default attribute
    class.

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
    
    # .........................................................................
    
    def as_dict(self) -> dict:
        """Returns a shallow copy of the attributes as a dict"""
        return {k:v for k, v in self.items()}

    # .........................................................................
    # Magic methods and iterators for convenient dict-like access

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about these attributes"""
        return "{} attribute(s)".format(len(self))


# -----------------------------------------------------------------------------

class BaseDataContainer(AttrsMixin, SizeOfMixin,
                        dantro.abc.AbstractDataContainer):
    """The BaseDataContainer extends the abstract base class by the ability to
    hold attributes and be path-aware.
    """
    # Define which class to use for storing attributes
    _ATTRS_CLS = BaseDataAttrs

    def __init__(self, *, name: str, data, attrs=None):
        """Initialize a BaseDataContainer, which can store data and attributes.
        
        Args:
            name (str): The name of this data container
            data: The data to store in this container
            attrs (None, optional): A mapping that is stored as attributes
        """
        # Initialize via parent, then additionally store attributes
        super().__init__(name=name, data=data)
        self.attrs = attrs

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the content of this
        data container.
        """
        return ("{} attribute{}"
                "".format(len(self.attrs),
                          "s" if len(self.attrs) != 1 else ""))
        

# -----------------------------------------------------------------------------

class BaseDataGroup(LockDataMixin, SizeOfMixin, AttrsMixin,
                    dantro.abc.AbstractDataGroup):
    """The BaseDataGroup serves as base group for all data groups.

    It implements all functionality expected of a group, which is much more
    than what is expected of a general container.
    """
    # Define which class to use for storing attributes
    _ATTRS_CLS = BaseDataAttrs
    
    # The mapping type that is used to store the members of this group.
    _STORAGE_CLS = dict

    # Define which class to use in the new_group method. If None, the type of
    # this instance is used
    _NEW_GROUP_CLS = None
    
    # Define which class to use in the new_container method. If None, the type
    # needs to be specified explicitly in the call to new_container
    _NEW_CONTAINER_CLS = None

    # Define the types that are allowed to be stored in this group. If None,
    # the dantro base classes are allowed
    _ALLOWED_CONT_TYPES = None

    # Condensed tree representation: maximum level
    _COND_TREE_MAX_LEVEL = 10

    # Condensed tree representation: threshold parameter
    _COND_TREE_CONDENSE_THRESH = 10

    # .........................................................................

    def __init__(self, *, name: str, containers: list=None, attrs=None):
        """Initialize a BaseDataGroup, which can store other containers and
        attributes.
        
        Args:
            name (str): The name of this data container
            containers (list, optional): The containers that are to be stored
                as members of this group. If given, these are added one by one
                using the `.add` method.
            attrs (None, optional): A mapping that is stored as attributes
        """
        # Prepare the storage class that is used to store the members
        data = self._STORAGE_CLS()

        # Initialize via parent and store attributes
        super().__init__(name=name, data=data)
        self.attrs = attrs

        # Now add the member containers
        if containers is not None:
            self.add(*containers)

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
                return self._data[key[0]][key[1:]]
            # else: end of recursion
            return self._data[key[0]]

        except (KeyError, IndexError) as err:
            raise KeyError("No key or key sequence '{}' in {}! "
                           "Available keys at top level: {}"
                           "".format(key, self.logstr,
                                     ", ".join([k for k in self.keys()]))
                           ) from err

    def __setitem__(self, key: Union[str, List[str]],
                    val: BaseDataContainer) -> None:
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
            self._data[key[0]][key[1:]] = val
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
            del self._data[key[0]][key[1:]]
            return

        # else: end of recursion: delete and unlink this container
        # ... if it is not locked
        self.raise_if_locked()
        
        cont = self._data[key[0]]
        del self._data[key[0]]

        self._unlink_child(cont)

    def add(self, *conts, overwrite: bool=False):
        """Add the given containers to this group."""
        for cont in conts:
            self._add_container(cont, overwrite=overwrite)

        log.trace("Added %d container(s) to %s.", len(conts), self.logstr)

    def _add_container(self, cont, *, overwrite: bool):
        """Private helper method to add a container to this group."""
        # Data may not be locked
        self.raise_if_locked()

        # Check the allowed types
        if (self._ALLOWED_CONT_TYPES is None
            and not isinstance(cont, (BaseDataGroup, BaseDataContainer))):
            raise TypeError("Can only add BaseDataGroup- or "
                            "BaseDataContainer-derived objects to {}, got {}!"
                            "".format(self.logstr, type(cont)))

        elif (self._ALLOWED_CONT_TYPES is not None
              and not isinstance(cont, self._ALLOWED_CONT_TYPES)):
            raise TypeError("Can only add objects derived from the following "
                            "classes: {}. Got: {}"
                            "".format(self._ALLOWED_CONT_TYPES, type(cont)))

        # else: is of correct type
        # Check if one like this already exists
        old_cont = None
        if cont.name in self:
            if not overwrite:
                raise ValueError("{} already has a member with name '{}', "
                                 "cannot add {}."
                                 "".format(self.logstr, cont.name, cont))
            log.debug("A member '%s' of %s already exists and will be "
                      "overwritten ...", cont.name, self.logstr)
            old_cont = self[cont.name]

        # Allow for subclasses to perform further custom checks on the
        # container object before adding it
        self._check_cont(cont)

        # Write to data, assuring that the name matches that of the container
        self._add_container_to_data(cont)
        self._add_container_callback(cont)
        
        # Re-link the containers
        self._link_child(new_child=cont, old_child=old_cont)

    def _check_cont(self, cont) -> None:
        """Can be used by a subclass to check a container before adding it to
        this group. Is called by _add_container before checking whether the
        object exists or not.
        
        This is not expected to return, but can raise errors, if something
        did not work out as expected.
        
        Args:
            cont: The container to check
        """
        pass

    def _add_container_to_data(self, cont) -> None:
        """Performs the operation of adding the container to the _data. This
        can be used by subclasses to make more elaborate things while adding
        data, e.g. specify ordering ...

        NOTE This method should NEVER be called on its own, but only via the
             _add_container method, which takes care of properly linking the
             container that is to be added.

        NOTE After adding, the container need be reachable under its .name!
        
        Args:
            cont: The container to add
        """
        # Just add it via _data.__setitem__, using the container's name
        self._data[cont.name] = cont

    def _add_container_callback(self, cont) -> None:
        """Called after a container was added."""
        pass

    def new_container(self, path: Union[str, list], *,
                      Cls: type=None, **kwargs):
        """Creates a new container of class `Cls` and adds it at the given path
        relative to this group.
        
        Args:
            path (Union[str, list]): Where to add the container. Note that the
                intermediates of this path need to already exist.
            Cls (type, optional): The class of the container to add. If None,
                the _NEW_CONTAINER_CLS class variable's value is used; if not
                given, this will raise a ValueError.
            **kwargs: kwargs to pass on to Cls.__init__
        
        Returns:
            Cls: the created container
        
        Raises:
            KeyError: When intermediate groups to `path` are missing
            TypeError: When the given Cls is invalid
        """
        # Resolve the Cls argument, if possible from the class variable
        if Cls is None:
            if self._NEW_CONTAINER_CLS is None:
                raise ValueError("Got neither argument `Cls` nor class "
                                 "variable _NEW_CONTAINER_CLS, at least one "
                                 "of which is needed to determine the type "
                                 "of the new container!")

            Cls = self._NEW_CONTAINER_CLS

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
                group. If not given, uses the class specified in the
                _NEW_GROUP_CLS class variable or, as last resort, the type of
                this instance.
            **kwargs: Passed on to Cls.__init__
        
        Returns:
            Cls: the created group
        
        Raises:
            TypeError: For the given class not being derived from BaseDataGroup
        """
        # If no Cls is given, use this instance's type
        if Cls is None:
            Cls = self._NEW_GROUP_CLS if self._NEW_GROUP_CLS else type(self)

        # Need to catch the case where a non-group class was given
        if inspect.isclass(Cls) and not issubclass(Cls, BaseDataGroup):
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
    def _link_child(self, *, new_child: BaseDataContainer,
                    old_child: BaseDataContainer=None):
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
        return len(self._data)

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

    def _ipython_key_completions_(self) -> List[str]:
        """For ipython integration, return a list of available keys"""
        return list(self.keys())

    # .........................................................................
    # Iteration

    def __iter__(self):
        """Returns an iterator over the OrderedDict"""
        return iter(self._data)

    def keys(self):
        """Returns an iterator over the container names in this group."""
        return self._data.keys()

    def values(self):
        """Returns an iterator over the containers in this group."""
        return self._data.values()

    def items(self):
        """Returns an iterator over the (name, data container) tuple of this
        group."""
        return self._data.items()

    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name
        `key` is not available."""
        return self._data.get(key, default)

    def setdefault(self, key, default=None):
        """This method is not supported for a data group"""
        raise NotImplementedError("setdefault is not supported by {}! Use the "
                                  "`add` method or `new_group` and "
                                  "`new_container` to add elements."
                                  "".format(self.classname))

    # .........................................................................
    # Formatting

    @property
    def tree(self) -> str:
        """Returns the default (full) tree representation of this group"""
        return self._tree_repr()
    
    @property
    def tree_condensed(self) -> str:
        """Returns the condensed tree representation of this group. Uses the
        ``_COND_TREE_*`` prefixed class attributes as parameters.
        """
        return self._tree_repr(max_level=self._COND_TREE_MAX_LEVEL,
                               condense_thresh=self._COND_TREE_CONDENSE_THRESH)

    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used
        to characterize this object. Does NOT include name and classname!
        """
        return ("{} member{}, {} attribute{}"
                "".format(len(self), "s" if len(self) != 1 else "",
                          len(self.attrs), "s" if len(self.attrs) != 1 else "")
                )

    def _format_tree(self) -> str:
        """Returns the default tree representation of this group by invoking
        the .tree property
        """
        return self.tree

    def _format_tree_condensed(self) -> str:
        """Returns the default tree representation of this group by invoking
        the .tree property
        """
        return self.tree_condensed

    def _tree_repr(self, *,
                   level: int=0,
                   max_level: int=None,
                   info_fstr="<{:cls_name,info}>",
                   info_ratio: float=0.6,
                   condense_thresh: Union[int, Callable[[int, int], int]]=None,
                   total_item_count: int=0
                   ) -> Union[str, List[str]]:
        """Recursively creates a multi-line string tree representation of this
        group. This is used by, e.g., the _format_tree method.
        
        Args:
            level (int, optional): The depth within the tree
            max_level (int, optional): The maximum depth within the tree;
                recursion is not continued beyond this level.
            info_fstr (str, optional): The format string for the info string
            info_ratio (float, optional): The width ratio of the whole line
                width that the info string takes
            condense_thresh (Union[int, Callable[[int, int], int]], optional):
                If given, this specifies the threshold beyond which the tree
                view for the current element becomes condensed by hiding the
                output for some elements.
                The minimum value for this is 3, indicating that there should
                be at most 3 lines be generated from this level (excluding the
                lines coming from recursion), i.e.: two elements and one line
                for indicating how many values are hidden.
                If a smaller value is given, this is silently brought up to 3.
                Half of the elements are taken from the beginning of the
                item iteration, the other half from the end.
                If given as integer, that number is used.
                If a callable is given, the callable will be invoked with the
                current level, number of elements to be added at this level,
                and the current total item count along this recursion branch.
                The callable should then return the number of lines to be
                shown for the current element.
            total_item_count (int, optional): The total number of items
                already created in this recursive tree representation call.
                Passed on between recursive calls.
        
        Returns:
            Union[str, List[str]]: The (multi-line) tree representation of
                this group. If this method was invoked with ``level == 0``, a
                string will be returned; otherwise, a list of strings will be
                returned.
        """
        def get_offset_str(level: int) -> str:
            """Returns an offst string, depending on level"""
            return "   " * level

        def truncate(s: str, *, max_length: int, suffix: str="…") -> str:
            """Truncates the given string to the desired length"""
            return s if len(s) <= max_length else s[:max_length-1] + suffix

        # Offset
        offset = get_offset_str(level)

        # Mark symbols
        first_mark = r' └┬'
        base_mark =  r'  ├'
        last_mark =  r'  └'
        only_mark =  r' └─'
        
        # Evaluate the condensation threshold, i.e. the maximum number of lines
        # to allow originating from this object (excluding recursion)
        num_items = len(self)
        total_item_count += num_items
        num_skipped = 0
        
        if callable(condense_thresh):
            max_lines = condense_thresh(level=level, num_items=num_items,
                                        total_item_count=total_item_count)
        else:
            max_lines = condense_thresh
        
        if max_lines is not None:
            # Additional check for lower bound; makes visualization much easier
            max_lines = max(3, int(max_lines))

            # If there are too few items, the variable is set to None to
            # indicate regular behavior.
            if num_items - max_lines < 1:
                max_lines = None

        # Calculations that make the output line fit into one terminal line
        num_cols = TTY_COLS
        info_width = int(num_cols * info_ratio)
        name_width = (num_cols - info_width) - (len(offset) + 3 + 1 + 2)

        def get_mark(n: int, *, max_n: int) -> str:
            """Returns the mark symbol depending on the iteration number.

            NOTE This uses variables from the outer scope!
            """
            if n == 0:
                if max_n == 0:
                    return only_mark
                return first_mark
            elif n == max_n:
                return last_mark
            return base_mark

        # The format string that's used to compose the whole output line
        fstr = "{offset:}{mark:>3s} {name:<{name_width}s}  {info:}"
        
        # Create the list to gather the lines in; add a description on level 0
        lines = []
        if level == 0:
            lines.append("")
            lines.append("Tree of {:logstr,info}".format(self))

        # Go over the entries on this level and format the lines
        for n, (key, obj) in enumerate(self.items()):
            # Determine whether to show this line of the tree or not. The lines
            # in the middle of the iteration are not shown.
            # If it is not shown, the first line that is then to be shown also
            # adds a line that indicates how many items were skipped.
            if max_lines is not None:
                if max_lines//2 <= n < (num_items - (max_lines-1)//2):
                    num_skipped += 1
                    continue

                elif n == (num_items - (max_lines-1)//2):
                    # Add the indicator line
                    lines.append(fstr.format(offset=offset, mark=base_mark,
                                             name_width=name_width,
                                             name="...",
                                             info=("... ({:d} more) ..."
                                                   "".format(num_skipped))))

            # Get key and info, truncating if necessary, and the mark
            name = truncate(key, max_length=name_width)
            info = truncate(info_fstr.format(obj), max_length=info_width)
            mark = get_mark(n, max_n=num_items-1)

            # Format the line and add to list of lines
            lines.append(fstr.format(offset=offset, mark=mark,
                                     name_width=name_width, name=name,
                                     info=info))

            # If it was a group and it is not empty...
            if isinstance(obj, BaseDataGroup) and len(obj) > 0:
                # ... and maximum recursion depth is not reached:
                if max_level is None or level < max_level:
                    # Continue recursion
                    lines += obj._tree_repr(level=level+1,
                                            max_level=max_level,
                                            info_fstr=info_fstr,
                                            info_ratio=info_ratio,
                                            condense_thresh=condense_thresh,
                                            total_item_count=total_item_count)

                else:
                    # Only indicate that it _would_ continue here, but do not
                    # actually continue with the recursion.
                    lines.append(fstr.format(offset=get_offset_str(level+1),
                                             mark=only_mark, name_width=3,
                                             name="...", info=""))

        # Done, depending on whether this is within the recursion or not,
        # return as list of lines or as combined multi-line string
        if level > 0:
            return lines
        return "\n".join(lines) + "\n"
