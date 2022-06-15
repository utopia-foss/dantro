"""This sub-module implements the basic mixin classes that are required
in the :py:mod:`dantro.base` module"""

import contextlib
import logging
import sys
import warnings
from typing import Union

from ..abc import PATH_JOIN_CHAR, AbstractDataProxy
from ..exceptions import UnexpectedTypeWarning

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class AttrsMixin:
    """This Mixin class supplies the ``attrs`` property getter and setter and
    the private ``_attrs`` attribute.

    Hereby, the setter function will initialize a
    :py:class:`~dantro.base.BaseDataAttrs` -derived object and store it as an
    attribute. This relays the checking of the correct attribute format to the
    actual :py:class:`~dantro.base.BaseDataAttrs`-derived class.

    For changing the class that is used for the attributes, an overwrite of the
    ``_ATTRS_CLS`` class variable suffices.
    """

    _attrs = None
    """The class attribute that the attributes will be stored to"""

    _ATTRS_CLS = None
    """The class to use for storing attributes"""

    @property
    def attrs(self):
        """The container attributes."""
        return self._attrs

    @attrs.setter
    def attrs(self, new_attrs):
        """Setter method for the container `attrs` attribute."""
        if self._ATTRS_CLS is None:
            raise ValueError(
                "Need to declare the class variable _ATTRS_CLS "
                "in order to use the AttrsMixin!"
            )

        self._attrs = self._ATTRS_CLS(name="attrs", attrs=new_attrs)


class SizeOfMixin:
    """Provides the ``__sizeof__`` magic method and attempts to take into
    account the size of the attributes.
    """

    def __sizeof__(self) -> int:
        """Returns the size of the data (in bytes) stored in this container's
        data and its attributes.

        Note that this value is approximate. It is computed by calling the
        :py:func:`sys.getsizeof` function on the data, the attributes, the
        name and some caching attributes that each dantro data tree class
        contains. *Importantly,* this is *not* a recursive algorithm.

        Also, derived classes might implement further attributes that are not
        taken into account either. To be more precise in a subclass, create a
        specific __sizeof__ method and invoke this parent method additionally.
        """
        nbytes = sys.getsizeof(self._data)
        nbytes += sys.getsizeof(self._attrs)
        nbytes += sys.getsizeof(self._name)

        return nbytes


class LockDataMixin:
    """This mixin class provides a flag for marking the data of a group or
    container as locked.
    """

    __locked = False
    """Whether the data is regarded as locked. Note name-mangling here."""

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

    def raise_if_locked(self, *, prefix: str = None):
        """Raises an exception if this object is locked; does nothing otherwise"""
        if self.locked:
            raise RuntimeError(
                "{}Cannot modify {} because it was already "
                "marked locked."
                "".format(prefix + " " if prefix else "", self.logstr)
            )

    def _lock_hook(self):
        """Invoked upon locking."""
        pass

    def _unlock_hook(self):
        """Invoked upon unlocking."""
        pass


class BasicComparisonMixin:
    """Provides a (very basic) ``__eq__`` method to compare equality."""

    def __eq__(self, other) -> bool:
        """Evaluates equality by making the following comparisons: identity,
        strict type equality, and finally: equality of the ``_data`` and
        ``_attrs`` attributes, i.e. the *private* attribute. This ensures that
        comparison does not trigger any downstream effects like resolution of
        proxies.

        If types do not match exactly, ``NotImplemented`` is returned, thus
        referring the comparison to the other side of the ``==``.
        """
        if other is self:
            return True

        if type(other) is not type(self):
            return NotImplemented

        return self._data == other._data and self._attrs == other._attrs


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
    if given a list (passed down from above), it extracts it.
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
        """Returns an iterator over the data's keys."""
        return self.data.keys()

    def values(self):
        """Returns an iterator over the data's values."""
        return self.data.values()

    def items(self):
        """Returns an iterator over data's ``(key, value)`` tuples"""
        return self.data.items()

    def get(self, key, default=None):
        """Return the value at ``key``, or ``default`` if ``key`` is not
        available.
        """
        return self.data.get(key, default)


class CheckDataMixin:
    """This mixin class extends a BaseDataContainer-derived class to check the
    provided data before storing it in the container.

    It implements a general :py:meth:`._check_data` method, overwriting the
    placeholder method in the :py:class:`~dantro.base.BaseDataContainer`, and
    can be controlled via class variables.

    .. note::

        This is not suitable for checking containers that are added to an
        object of a :py:class:`~dantro.base.BaseDataGroup`-derived class!
    """

    DATA_EXPECTED_TYPES: tuple = None
    """Which types to allow. If None, all types are allowed."""

    DATA_ALLOW_PROXY: bool = False
    """Whether to allow *all* proxy types, i.e. classes derived from
    :py:class:`~dantro.abc.AbstractDataProxy`."""

    DATA_UNEXPECTED_ACTION = "warn"
    """The action to take when an unexpected type was supplied.
    Can be: ``raise``, ``warn``, ``ignore``."""

    def _check_data(self, data) -> None:
        """A general method to check the received data for its type

        Args:
            data: The data to check

        Raises:
            TypeError: If the type was unexpected and the action was 'raise'
            ValueError: Illegal value for ``DATA_UNEXPECTED_ACTION`` class
                variable

        Returns:
            None
        """
        from .._import_tools import resolve_types

        if self.DATA_EXPECTED_TYPES is None:
            # All types allowed
            return

        # Compile tuple of allowed types, importing those that were supplied
        # as module strings
        expected_types = resolve_types(self.DATA_EXPECTED_TYPES)

        if self.DATA_ALLOW_PROXY:
            expected_types += (AbstractDataProxy,)

        # Check for expected types
        if isinstance(data, expected_types):
            return

        # else: was not of the expected type
        # Create a base message
        msg = (
            f"Unexpected type {type(data)} for data passed to {self.logstr}! "
            f"Expected types are: {expected_types}."
        )

        # Handle according to the specified action
        if self.DATA_UNEXPECTED_ACTION == "raise":
            raise TypeError(msg)

        elif self.DATA_UNEXPECTED_ACTION == "warn":
            warnings.warn(
                f"{msg}\nInitialization will work, but be informed "
                "that there might be errors at runtime.",
                UnexpectedTypeWarning,
            )

        elif self.DATA_UNEXPECTED_ACTION == "ignore":
            log.debug(msg + " Ignoring ...")

        else:
            raise ValueError(
                f"Illegal value '{self.DATA_UNEXPECTED_ACTION}' for class "
                f"variable DATA_UNEXPECTED_ACTION of {self.classname}. "
                "Allowed values are: raise, warn, ignore"
            )


class DirectInsertionModeMixin:
    """A mixin class that provides a context manager, within which insertion
    into the mixed-in class (think: group or container) can happen more
    directly. This is useful in cases where more assumptions can be made about
    the to-be-inserted data, thus allowing to make fewer checks during
    insertion (think: duplicates, key order, etc.).

    .. note::

        This direct insertion mode is not (yet) part of the public interface,
        as it has to be evaluated how robust and error-prone it is.
    """

    __in_direct_insertion_mode = False
    """A name-mangled state flag that determines the state of the object."""

    @property
    def with_direct_insertion(self) -> bool:
        """Whether the class this mixin is mixed into is currently in direct
        insertion mode.
        """
        return self.__in_direct_insertion_mode

    @contextlib.contextmanager
    def _direct_insertion_mode(self, *, enabled: bool = True):
        """A context manager that brings the class this mixin is used in into
        direct insertion mode. While in that mode, the
        :py:meth:`.with_direct_insertion` property will return true.

        This context manager additionally invokes two callback functions, which
        can be specialized to perform certain operations when entering or
        exiting direct insertion mode: *Before* entering,
        :py:meth:`._enter_direct_insertion_mode` is called. *After* exiting,
        :py:meth:`_exit_direct_insertion_mode` is called.

        Args:
            enabled (bool, optional): whether to actually use direct insertion
                mode. If False, will yield directly without setting the toggle.
                This is equivalent to a null-context.
        """
        if not enabled:
            self.__in_direct_insertion_mode = False
            yield
            return

        log.trace(
            "Entering direct insertion mode of %s @ %s ...",
            self.logstr,
            self.path,
        )

        # Perform the entering callback
        try:
            self._enter_direct_insertion_mode()

        except Exception as exc:
            raise RuntimeError(
                "Error in callback while entering direct insertion mode of "
                f"{self.logstr} @ {self.name}! {type(exc).__name__}: {exc}"
            ) from exc

        # Now inside direct insertion mode
        self.__in_direct_insertion_mode = True

        try:
            # Yield control to the with-context now
            yield

        finally:
            # Will end up here if there was an exception within the context.
            log.trace(
                "Exiting direct insertion mode of %s @ %s ...",
                self.logstr,
                self.path,
            )
            self.__in_direct_insertion_mode = False

            # NOTE Important to NOT have a return here or handle any other
            #      error, otherwise exceptions from the context are discarded.

        # Perform the exiting callback
        try:
            self._exit_direct_insertion_mode()

        except Exception as exc:
            raise RuntimeError(
                "Error in callback while exiting direct insertion mode of "
                f"{self.logstr} @ {self.name}! {type(exc).__name__}: {exc}"
            ) from exc

    def _enter_direct_insertion_mode(self):
        """Called after entering direct insertion mode; can be overwritten to
        attach additional behaviour.
        """

        pass

    def _exit_direct_insertion_mode(self):
        """Called before exiting direct insertion mode; can be overwritten to
        attach additional behaviour.
        """

        pass
