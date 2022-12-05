"""Implements a registry for dantro container types based on
:py:class:`~dantro._registry.ObjectRegistry`."""

import logging
from typing import Any, Dict, Optional, Union

from .._registry import ObjectRegistry as _ObjectRegistry
from ..base import BaseDataContainer
from ..exceptions import *

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class ContainerRegistry(_ObjectRegistry):

    _DESC = "container"
    _SKIP = False
    _OVERWRITE = True
    _EXPECTED_TYPE = (type,)

    def _check_object(self, obj: Any) -> None:
        """Checks whether the object is valid."""

        super()._check_object(obj)

        if not issubclass(obj, BaseDataContainer):
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

CONTAINERS = ContainerRegistry()
"""The dantro data container registry object."""


def register_container(
    Cls: type,
    name: str,
    *,
    skip_existing: bool = False,
    overwrite_existing: bool = True,
) -> None:
    """Adds an entry to the shared container registry.

    Args:
        Cls (type): The class that is to be registered as a container.
        name (str): The name to use for registration.
        skip_existing (bool, optional): Whether to skip registration if the
            container name is already registered. This suppresses the
            ValueError raised on existing container name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing container of the same name. If set, this takes
            precedence over ``skip_existing``.
    """
    return CONTAINERS.register(
        Cls,
        name=name,
        skip_existing=skip_existing,
        overwrite_existing=overwrite_existing,
    )


def is_container(
    arg: Union[str, type] = None,
    /,
    **kws,
):
    """Decorator for registering containers with the container type registry.

    As an alternative to :py:func:`.register_container`, this decorator can be
    used to register a container right where its defined:

    .. testcode::

        from dantro.containers import BaseDataContainer, is_container

        # Container name deduced from class name
        @is_container
        class MyDataContainer(BaseDataContainer):
            # ... do stuff here ...
            pass

        # Custom container name
        @is_container("my_container")
        class MyDataContainer(BaseDataContainer):
            # ... do stuff here ...
            pass

        # Overwriting a registered container of the same name
        @is_container("my_container", overwrite_existing=True)
        class MyDataContainer(BaseDataContainer):
            # ... do stuff here ...
            pass

    .. testcode::
        :hide:

        from dantro.containers._registry import CONTAINERS

        assert MyDataContainer in CONTAINERS
        assert "MyDataContainer" in CONTAINERS
        assert "my_container" in CONTAINERS
        assert "PathContainer" in CONTAINERS
    """
    return CONTAINERS._decorator(arg, **kws)
