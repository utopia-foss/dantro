"""This module holds the abstract base classes needed for dantro"""

import abc
import collections
import collections.abc
import logging
from typing import Any, Tuple, Union

log = logging.getLogger(__name__)

PATH_JOIN_CHAR = "/"
"""The character used for separating hierarchies in the path"""

BAD_NAME_CHARS = ("*", "?", "[", "]", "!", ":", "(", ")", PATH_JOIN_CHAR, "\\")
"""Substrings that may not appear in names of data containers"""

# -----------------------------------------------------------------------------


class AbstractDataContainer(metaclass=abc.ABCMeta):
    """The AbstractDataContainer is the class defining the data container
    interface. It holds the bare basics of methods and attributes that _all_
    dantro data tree classes should have in common: a name, some data, and some
    association with others via an optional parent object.

    Via the parent and the name, path capabilities are provided. Thereby, each
    object in a data tree has some information about its location relative to
    a root object.
    Objects that have *no* parent are regarded to be an object that is located
    "next to" root, i.e. having the path ``/<container_name>``.
    """

    @abc.abstractmethod
    def __init__(self, *, name: str, data: Any):
        """Initialize the AbstractDataContainer, which holds the bare
        essentials of what a data container should have.

        Args:
            name (str): The name of this container
            data (Any): The data that is to be stored
        """
        self._parent = None

        self._name = None
        self.name = name

        self._check_data(data)
        self._data = data

    # .........................................................................
    # Properties

    @property
    def name(self) -> str:
        """The name of this DataContainer-derived object."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """Rename this object; not always possible!"""
        # Check if this is a renaming operation; allow only if currently orphan
        if self._name is not None:
            if self.parent is not None:
                raise ValueError(
                    f"Cannot rename {self.logstr} to '{new_name}', because a "
                    "parent was already associated with it."
                )
            log.debug("Renaming %s to '%s' ...", self.logstr, new_name)

        # Require string as name
        if not isinstance(new_name, str):
            raise TypeError(
                f"Name for {self.classname} needs to be a string, was of type "
                f"{type(new_name)} with value '{new_name}'."
            )

        # Ensure name does not contain any bad substrings. Do this here rather
        # than in _check_name because these checks are crucial.
        if any([c in new_name for c in BAD_NAME_CHARS]):
            _bad_name_chars = ", ".join(repr(s) for s in BAD_NAME_CHARS)
            raise ValueError(
                f"Invalid name '{new_name}' for new {self.classname}! "
                "The name may not contain any of the following "
                f"substrings: {_bad_name_chars}"
            )

        # Allow further checks by an additional method
        self._check_name(new_name)

        # Everything ok, store the attribute and invalidate cached logstring
        self._name = new_name

    @property
    def classname(self) -> str:
        """Returns the name of this DataContainer-derived class"""
        return self.__class__.__name__

    @property
    def logstr(self) -> str:
        """Returns the classname and name of this object"""
        return f"{self.classname} '{self.name}'"

    @property
    def data(self) -> Any:
        """The stored data."""
        return self._data

    @property
    def parent(self):
        """The associated parent of this container or group"""
        return self._parent

    @parent.setter
    def parent(self, cont):
        """Associate a new parent object with this container or group"""
        if self.parent is not None and cont is not None:
            raise ValueError(
                "A parent was already associated with {cls:} "
                "'{}'! Instead of manually setting the parent, "
                "use the functions supplied to manipulate "
                "members of this {cls:}."
                "".format(self.name, cls=self.classname)
            )

        log.trace(
            "Setting %s as parent of %s ...",
            cont.logstr if cont else None,
            self.logstr,
        )
        self._parent = cont

    @property
    def path(self) -> str:
        """The path to get to this container or group from some root path"""
        if self.parent is None:
            # Is at the root, thus prefix it with the root character
            return PATH_JOIN_CHAR + self.name

        # else: not at the top, thus also need the parent's path
        return self.parent.path + PATH_JOIN_CHAR + self.name

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
    # Helper functions

    def _check_name(self, new_name: str) -> None:
        """Called from name.setter and can be used to check the name that the
        container is supposed to have. On invalid name, this should raise.

        This method can be subclassed to implement more specific behaviour. To
        propagate the parent classes' behaviour the subclassed method should
        always call its parent method using super().

        Args:
            new_name (str): The new name, which is to be checked.
        """
        pass

    def _check_data(self, data: Any) -> None:
        """This method can be used to check the data provided to this container

        It is called before the data is stored in the ``__init__`` method and
        should raise an exception or create a warning if the data is not as
        desired.

        This method can be subclassed to implement more specific behaviour. To
        propagate the parent classes' behaviour the subclassed method should
        always call its parent method using ``super()``.

        .. note::

            The :py:class:`~dantro.mixins.base.CheckDataMixin` provides a
            generalised implementation of this method to perform some type
            checks and react to unexpected types.

        Args:
            data (Any): The data to check
        """
        pass

    # .........................................................................
    # Formatting

    def __str__(self) -> str:
        """An info string, that describes the object. This invokes the
        formatting helpers to show the log string (type and name) as well as
        the info string of this object.
        """
        return f"<{self:logstr,info}>"

    def __repr__(self) -> str:
        """Same as __str__"""
        return str(self)

    def __format__(self, spec_str: str) -> str:
        """Creates a formatted string from the given specification.

        Invokes further methods which are prefixed by ``_format_``.
        """
        if not spec_str:
            return str(self)

        specs = spec_str.split(",")
        parts = []
        join_char = ", "

        for spec in specs:
            try:
                format_func = getattr(self, "_format_" + spec)
            except AttributeError as err:
                raise ValueError(
                    "No format string specification '{}', part "
                    "of '{}', is available for {}!"
                    "".format(spec, specs, self.logstr)
                ) from err
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

    def _format_path(self) -> str:
        """A __format__ helper function: returns the path to this container"""
        return self.path

    @abc.abstractmethod
    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used
        to characterise this object. Should NOT include name and classname!
        """


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
    def add(self, *conts, overwrite: bool = False) -> None:
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
        """Returns an iterator over the (name, data container) tuple of this
        group."""

    @abc.abstractmethod
    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name
        `key` is not available."""

    @abc.abstractmethod
    def setdefault(self, key, default=None):
        """If `key` is in the dictionary, return its value. If not, insert
        `key` with a value of `default` and return `default`. `default`
        defaults to None."""

    @abc.abstractmethod
    def recursive_update(self, other):
        """Updates the group with the contents of another group."""

    @abc.abstractmethod
    def _format_tree(self) -> str:
        """A __format__ helper function: tree representation of this group"""

    @abc.abstractmethod
    def _tree_repr(self, level: int = 0) -> str:
        """Recursively creates a multi-line string tree representation of this
        group. This is used by, e.g., the _format_tree method."""


# -----------------------------------------------------------------------------


class AbstractDataAttrs(collections.abc.Mapping, AbstractDataContainer):
    """The BaseDataAttrs class defines the interface for the `.attrs`
    attribute of a data container.

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
    """A data proxy fills in for the place of a data container, e.g. if data
    should only be loaded on demand. It needs to supply the resolve method.
    """

    @abc.abstractmethod
    def __init__(self, obj: Any = None):
        """Initialize the proxy object, being supplied with the object that
        this proxy is to be proxy for.
        """

    @property
    def classname(self) -> str:
        """Returns this proxy's class name"""
        return self.__class__.__name__

    @abc.abstractmethod
    def resolve(self, *, astype: type = None):
        """Get the data that this proxy is a placeholder for and return it.

        Note that this method does not place the resolved data in the
        container of which this proxy object is a placeholder for! This only
        returns the data.
        """

    @property
    @abc.abstractmethod
    def tags(self) -> Tuple[str]:
        """The tags describing this proxy object"""


# -----------------------------------------------------------------------------


class AbstractPlotCreator(metaclass=abc.ABCMeta):
    """This class defines the interface for PlotCreator classes"""

    @abc.abstractmethod
    def __init__(
        self, name: str, *, dm: "dantro.data_mngr.DataManager", **plot_cfg
    ):
        """Initialize the plot creator, given a
        :py:class:`~dantro.data_mngr.DataManager`, the plot name, and the
        default plot configuration.
        """

    @abc.abstractmethod
    def __call__(self, *, out_path: str = None, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling :py:meth:`.plot`.

        This method essentially takes care of parsing the configuration, while
        :py:meth:`.plot` expects *parsed* arguments.
        """

    @abc.abstractmethod
    def plot(self, *, out_path: str = None, **cfg) -> None:
        """Given a specific configuration, performs a plot.

        To parse plot configuration arguments, use :py:meth:`.__call__`, which
        will call this method.
        """

    @abc.abstractmethod
    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot"""

    @abc.abstractmethod
    def prepare_cfg(
        self, *, plot_cfg: dict, pspace: "paramspace.paramspace.ParamSpace"
    ) -> tuple:
        """Prepares the plot configuration for the plot.

        This function is called by the plot manager before the first plot
        is created.

        The base implementation just passes the given arguments through.
        However, it can be re-implemented by derived classes to change the
        behaviour of the plot manager, e.g. by converting a plot configuration
        to a :py:class:`~paramspace.paramspace.ParamSpace`.
        """

    @abc.abstractmethod
    def _prepare_path(self, out_path: str) -> str:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.

        This is called from :py:meth:`.__call__` and is meant to postpone
        directory creation as far as possible.
        """
