"""Implements an object registry that can be specialized for certain use
cases, e.g. to store all available container types."""
import logging
from typing import Any, Optional, Union

from .exceptions import *
from .tools import make_columns
from .utils import KeyOrderedDict

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class ObjectRegistry:
    """"""

    _DESC: str = "object"
    """A description string for the entries of this registry"""

    _SKIP: bool = False
    """Default behavior for ``skip_existing`` argument"""

    _OVERWRITE: bool = False
    """Default behavior for ``overwrite_existing`` argument"""

    _EXPECTED_TYPE: Optional[Union[tuple, type]] = None
    """If set, will check for expected types"""

    # .........................................................................

    def __init__(self):
        self._d = KeyOrderedDict()

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    @property
    def desc(self) -> str:
        return self._DESC

    def __getitem__(self, key: str) -> Any:
        try:
            return self._d[key]
        except KeyError:
            raise MissingRegistryEntry(
                f"Missing {self.desc} named '{key}'! "
                f"Available entries:\n{make_columns(self.keys())}"
            )

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def __iter__(self):
        return self._d.__iter__()

    def __contains__(self, obj_or_key: Union[Any, str]) -> bool:
        """Whether the given argument is part of the keys or values of this
        registry."""
        return obj_or_key in self._d.keys() or obj_or_key in self._d.values()

    def __len__(self) -> int:
        return len(self._d)

    # .........................................................................

    def _determine_name(self, obj: Any, *, name: Optional[str]) -> str:
        """Determines the object name, using a potentially given name"""
        if name is not None:
            return name

        if hasattr(obj, "__name__"):
            return obj.__name__

        raise MissingNameError(
            f"Need `name` argument to register {self.desc} {obj} in "
            f"{self.classname}."
        )

    def _check_object(self, obj: Any) -> None:
        """Checks whether the object is valid.
        If not, raises :py:exc:`~dantro.exceptions.InvalidRegistryEntry`.
        """
        if self._EXPECTED_TYPE is not None:
            if not isinstance(obj, self._EXPECTED_TYPE):
                raise InvalidRegistryEntry(
                    f"Expected type for {self.desc} in {self.classname} is "
                    f"{self._EXPECTED_TYPE}, but got {type(obj)} "
                    f"with value: '{obj}'"
                )

    def register(
        self,
        obj: Any,
        name: Optional[str] = None,
        *,
        skip_existing: bool = None,
        overwrite_existing: bool = None,
    ) -> str:
        """Adds an entry to the registry.

        Args:
            obj (Any): The object to add to the registry.
            name (Optional[str], optional): The name to use. If not given, will
                deduce a name from the given object.
            skip_existing (bool, optional): Whether to skip registration if an
                object of that name already exists. If None, the classes
                default behavior (see :py:attr:`._SKIP`) is used.
            overwrite_existing (bool, optional): Whether to overwrite an
                entry if an object with that name already exists. If None, the
                classes default behavior (see :py:attr:`._OVERWRITE`)
                is used.
        """
        if skip_existing is None:
            skip_existing = self._SKIP

        if overwrite_existing is None:
            overwrite_existing = self._OVERWRITE
        self._check_object(obj)

        name = self._determine_name(obj, name=name)
        if not isinstance(name, str):
            raise TypeError(
                f"{self.classname} `name` argument needs to be a string, but "
                f"was {type(name).__name__} with value:  {name}"
            )

        if name in self and not overwrite_existing:
            if skip_existing:
                log.debug(
                    "A %s named '%s' is already registered and will not be "
                    "registered again. Choose a different name or unset the "
                    "`skip_existing` flag.",
                    self.desc,
                    name,
                )
                return

            elif self[name] is obj:
                log.debug(
                    "A %s named '%s' is already registered, but is identical "
                    "to the existing one. Not setting it anew."
                )
                return

            else:
                raise RegistryEntryExists(
                    f"A {self.desc} named '{name}' is already registered in "
                    f"{self.classname}! Not overwriting.\n"
                    f"  Existing:  {self[name]}\n"
                    f"  New:       {obj}\n"
                    "Set the `overwrite_existing` flag to force overwriting "
                    "or choose a different `name`. If no name was given, "
                    "consider specifying it explicitly instead of letting it "
                    "be deduced."
                )

        self._d[name] = obj
        log.debug("Added %s '%s' to registry:\n%s", self.desc, name, obj)

        return name

    # .. Decorator ............................................................

    def _register_via_decorator(self, obj, name: Optional[str] = None, **kws):
        """Performs the registration operations when the decorator is used to
        register an object."""
        self.register(obj, name=name, **kws)

    def _decorator(
        self,
        arg: Union[Any, str] = None,
        /,
        **kws,
    ):
        """Method that can be used as a decorator for registering objects
        with this registry.

        Args:
            arg (Union[Any, str], optional): The name that should be used or
                the object that is to be added. If not a string, this refers
                to the ``@is_container`` call syntax
            **kws: Passed to :py:meth:`.register`
        """

        def wrapper(obj):
            # Need some nonlocal names, see PEP3104
            nonlocal arg

            if isinstance(arg, str):
                # Invocation via @is_operation() -- without arguments
                # The name was given as `arg` argument to _decorator
                kws["name"] = arg

            else:
                # Invocation via @is_operation -- without parentheses
                # No `arg` argument, thus no name information
                pass

            self._register_via_decorator(obj, **kws)
            return obj

        # Allow both invocation styles
        if arg is not None and not isinstance(arg, str):
            # @is_operation
            return wrapper(arg)

        # @is_operation(*args, **kwargs)
        return wrapper
