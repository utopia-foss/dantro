"""This module holds the abstract base classes needed for dantro"""

import abc
import collections
import collections.abc
from typing import Union
import logging

# Local constants
log = logging.getLogger(__name__)

# The character used for separating hierarchies in the path
PATH_JOIN_CHAR = "/"

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
        """Initialize the AbstractDataContainer, which holds the bare essentials of what a data container should have.
        
        Args:
            name (str): The name of this container
            data: The data that is to be stored
        """
        # Require strings as name
        if not isinstance(name, str):
            raise TypeError("Name for {} needs to be a string, was of type "
                            "{} with value '{}'.".format(self.classname,
                                                         type(name), name))

        # Ensure name does not contain path join character
        if PATH_JOIN_CHAR in name:
            raise ValueError("Name for {} cannot contain the path separator "
                             "'{}'! Got: '{}'"
                             "".format(self.classname, PATH_JOIN_CHAR, name))
        
        # Pass name and data to read-only attributes
        self._name = name
        self._data = data

        # Set caching variables
        self._logstr = None

        # Done.

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
        # Store the cache value, if not already happened
        if self._logstr is None:
            self._logstr = "{} '{}'".format(self.classname, self.name)

        # Return the cache value
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

    @abc.abstractmethod
    def __setitem__(self, key, val) -> None:
        """Sets an item in the container."""

    @abc.abstractmethod
    def __delitem__(self, key) -> None:
        """Deletes an item from the container."""

    # .........................................................................
    # Formatting

    def __str__(self) -> str:
        """An info string, that describes the object. Each class should implement this to return an informative response."""
        return "<{:logstr,info}>".format(self)

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

    def _format_logstr(self) -> str:
        """A __format__ helper function: returns the log string, a combination
        of class name and name"""
        return self.logstr

    @abc.abstractmethod
    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used to characterise this object. Should NOT include name and classname!"""


# -----------------------------------------------------------------------------

class AbstractDataGroup(AbstractDataContainer, collections.abc.MutableMapping):
    """The AbstractDataGroup is the abstract basis of all data groups.

    It enforces a MutableMapping interface with a focus on _setting_ abilities
    and less so on deletion."""

    @property
    def data(self):
        """The stored data."""
        raise AttributeError("Cannot directly access group data!")

    @abc.abstractmethod
    def add(self, *conts, overwrite: bool=False) -> None:
        """Adds the given containers to the group."""

    @abc.abstractmethod
    def __contains__(self, cont: Union[str, AbstractDataContainer]) -> bool:
        """Whether the given container is a member of this group"""

    @abc.abstractmethod
    def keys(self):
        """Returns an iterator over the container names in this group."""

    @abc.abstractmethod
    def values(self):
        """Returns an iterator over the containers in this group."""

    @abc.abstractmethod
    def items(self):
        """Returns an iterator over the (name, data container) tuple of this group."""

    @abc.abstractmethod
    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name `key` is not available."""

    @abc.abstractmethod
    def setdefault(self, key, default=None):
        """If `key` is in the dictionary, return its value. If not, insert `key` with a value of `default` and return `default`. `default` defaults to None."""

    @abc.abstractmethod
    def recursive_update(self, other):
        """Updates the group with the contents of another group."""

    @abc.abstractmethod
    def _format_tree(self) -> str:
        """A __format__ helper function: tree representation of this group"""

    @abc.abstractmethod
    def _tree_repr(self, level: int=0) -> str:
        """Recursively creates a multi-line string tree representation of this
        group. This is used by, e.g., the _format_tree method."""

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
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of attributes."""

    @abc.abstractmethod
    def keys(self):
        """Returns an iterator over the attribute names."""

    @abc.abstractmethod
    def values(self):
        """Returns an iterator over the attribute values."""

    @abc.abstractmethod
    def items(self):
        """Returns an iterator over the (keys, values) tuple of the attributes."""

# -----------------------------------------------------------------------------

class AbstractDataProxy(metaclass=abc.ABCMeta):
    """A data proxy fills in for the place of a data container, e.g. if data should only be loaded on demand. It needs to supply the resolve method."""

    @abc.abstractmethod
    def __init__(self, obj):
        """Initialize the proxy object, being supplied with the object that
        this proxy is to be proxy for.
        """

    @property
    def classname(self) -> str:
        """Returns this proxy's class name"""
        return self.__class__.__name__

    @abc.abstractmethod
    def resolve(self, *, astype: type=None):
        """Get the data that this proxy is a placeholder for and return it.

        Note that this method does not place the resolved data in the
        container of which this proxy object is a placeholder for! This only
        returns the data.
        """


# -----------------------------------------------------------------------------

class AbstractPlotCreator(metaclass=abc.ABCMeta):
    """This class defines the interface for PlotCreator classes"""

    @abc.abstractmethod
    def __init__(self, name: str, *, dm, **plot_cfg):
        """Initialize the PlotCreator, given a data manager, the plot name, and
        the default plot configuration.
        """

    @abc.abstractmethod
    def __call__(self, *, out_path: str=None, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling _plot.
        """

    @abc.abstractmethod
    def plot(self, out_path: str=None, **cfg):
        """Given a specific configuration, perform a plot.

        This method should always be private and only be called from __call__.
        """

    @abc.abstractmethod
    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot"""

    @abc.abstractmethod
    def prepare_cfg(self, *, plot_cfg: dict, pspace) -> tuple:
        """Prepares the plot configuration for the PlotManager.

        This function is called by the plot manager before the first plot
        is created.
        
        The base implementation just passes the given arguments through.
        However, it can be re-implemented by derived classes to change the
        behaviour of the plot manager, e.g. by converting a plot configuration
        to a parameter space.
        """
        
    @abc.abstractmethod
    def can_plot(self, creator_name: str, **plot_cfg) -> bool:
        """Whether this plot creator is able to make a plot for the given plot
        configuration.

        This function is used by the PlotManager's auto-detect feature.
        
        Args:
            creator_name (str): The name for this creator used within the
                PlotManager.
            **plot_cfg: The plot configuration with which to decide this.
        
        Returns:
            bool: Whether this creator can be used for the given plot config
        """
        return False

    @abc.abstractmethod
    def _prepare_path(self, out_path: str) -> str:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.

        This is called from __call__ and is meant to postpone directory
        creation as far as possible.
        """
