"""Implements a registry for dantro group types based on
:py:class:`~dantro._registry.ObjectRegistry`."""

import logging
from typing import Any, Dict, Optional, Union

from .._registry import ObjectRegistry as _ObjectRegistry
from ..base import BaseDataGroup
from ..exceptions import *

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class GroupRegistry(_ObjectRegistry):

    _DESC = "group"
    _SKIP = False
    _OVERWRITE = True
    _EXPECTED_TYPE = (type,)

    def _check_object(self, obj: Any) -> None:
        """Checks whether the object is valid."""
        super()._check_object(obj)

        if not issubclass(obj, BaseDataGroup):
            raise TypeError(
                f"The given {self.desc} type {obj} needs to be a subclass of "
                "the dantro BaseDataContainer but was not!"
            )

    def _register_via_decorator(
        self, obj: type, name: Optional[str] = None, **kws
    ):
        """Performs the registration operations when the decorator is used to
        register an object."""
        self.register(obj, name=obj.__name__)
        self.register(obj, name=f"{obj.__module__}.{obj.__name__}")
        if name is not None:
            self.register(obj, name=name, **kws)


# -----------------------------------------------------------------------------

GROUPS = GroupRegistry()
"""The dantro data group registry object."""


def register_group(
    Cls: type,
    name: str,
    *,
    skip_existing: bool = False,
    overwrite_existing: bool = True,
) -> None:
    """Adds an entry to the shared group registry.

    Args:
        Cls (type): The class that is to be registered as a group.
        name (str): The name to use for registration.
        skip_existing (bool, optional): Whether to skip registration if the
            group name is already registered. This suppresses the
            ValueError raised on existing group name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing group of the same name. If set, this takes
            precedence over ``skip_existing``.
    """
    return GROUPS.register(
        Cls,
        name=name,
        skip_existing=skip_existing,
        overwrite_existing=overwrite_existing,
    )


def is_group(
    arg: Union[str, type] = None,
    /,
    **kws,
):
    """Decorator for registering groups with the container type registry.

    As an alternative to :py:func:`.register_group`, this decorator can be
    used to register a container right where its defined:

    .. testcode::

        from dantro.groups import BaseDataGroup, is_group

        # Group name deduced from class name
        @is_group
        class MyDataGroup(BaseDataGroup):
            # ... do stuff here ...
            pass

        # Custom group name
        @is_group("my_group")
        class MyDataGroup(BaseDataGroup):
            # ... do stuff here ...
            pass

        # Overwriting a registered group of the same name
        @is_group("my_group", overwrite_existing=True)
        class MyDataGroup(BaseDataGroup):
            # ... do stuff here ...
            pass

    .. testcode::
        :hide:

        from dantro.groups._registry import GROUPS

        assert MyDataGroup in GROUPS
        assert "MyDataGroup" in GROUPS
        assert "my_group" in GROUPS
        assert "OrderedDataGroup" in GROUPS
    """
    return GROUPS._decorator(arg, **kws)
