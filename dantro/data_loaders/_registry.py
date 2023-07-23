"""Implements registration of data loaders, including a decorator to ensure
correct loader function signature (which also automatically keeps track of the
data loader function).
"""

import logging
from typing import Callable, List

from .._registry import ObjectRegistry as _ObjectRegistry

log = logging.getLogger(__name__)

LOAD_FUNC_PREFIX: str = "_load_"
"""The prefix that all load functions need to start with"""

# -----------------------------------------------------------------------------


class DataLoaderRegistry(_ObjectRegistry):
    """Specialization of :py:class:`~dantro._registry.ObjectRegistry` for the
    purpose of keeping track of data loaders.
    """

    _DESC = "data loader"
    _SKIP = False
    _OVERWRITE = False
    _EXPECTED_TYPE = None


DATA_LOADERS = DataLoaderRegistry()
"""The dantro data loaders registry.

The :py:class:`~dantro.data_mngr.DataManager` and derived classes have access
to all data loaders via this registry (in addition to method-based access
they have via potentially used mixins).

To register a new loader, use the :py:func:`.add_loader` decorator:
"""


def _register_loader(
    wrapped_func: Callable,
    name: str,
    *,
    skip_existing: bool = False,
    overwrite_existing: bool = True,
) -> None:
    """Internally used method to add an entry to the shared loader registry.

    Args:
        wrapped_func (Callable): The wrapped callable that is to be registered
            as a loader. This is what the :py:func:`.add_loader` decorator
            generates.
        name (str, optional): The name to use for registration.
        skip_existing (bool, optional): Whether to skip registration if the
            loader name is already registered. This suppresses the
            ValueError raised on existing loader name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing loader of the same name. If set, this takes
            precedence over ``skip_existing``.
    """
    return DATA_LOADERS.register(
        wrapped_func,
        name=name,
        skip_existing=skip_existing,
        overwrite_existing=overwrite_existing,
    )


def add_loader(
    *,
    TargetCls: type,
    omit_self: bool = True,
    overwrite_existing: bool = True,
    register_aliases: List[str] = None,
):
    """This decorator should be used to specify loader methods in mixin classes
    to the :py:class:`~dantro.data_mngr.DataManager`.

    All decorated methods where ``omit_self is True`` will additinoally be
    registered in the :py:data:`.DATA_LOADERS` registry.

    Example:

    .. testcode::

        from dantro.containers import ObjectContainer
        from dantro.data_loaders import add_loader

        class MyDataLoaderMixin:

            @add_loader(TargetCls=ObjectContainer)
            def _load_foobar(path: str, *, TargetCls: type, **kws):
                # load something from the given file path
                with open(path, **kws) as f:
                    data = f.read()

                return TargetCls(data=data)

        # Define a DataManager that has the custom loader mixed-in

        from dantro import DataManager

        class MyDataManager(MyDataLoaderMixin, DataManager):
            pass

    .. testcode::
        :hide:

        from dantro.data_loaders._registry import DATA_LOADERS

        assert "foobar" in DATA_LOADERS
        del DATA_LOADERS._d["foobar"]
        assert "foobar" not in DATA_LOADERS

    .. note::

        Loader methods need to be named ``_load_<name>`` and are then
        accessible via ``<name>``.

        **Important:** Loader methods may not be named ``_load_file``!

    .. hint::

        This decorator **can also be used on standalone functions**, without
        the need to define a mixin class.
        In such a case, ``omit_self`` can still be set to False, leading to the
        first positional argument that the decorated function needs to accept
        to be the :py:class:`~dantro.data_mngr.DataManager` instance that the
        loader is used in.

        Note that these standalone function should still begin with ``_load_``.

    Args:
        TargetCls (type): The return type of the load function. This is stored
            as an attribute of the decorated function.
        omit_self (bool, optional): If True (default), the decorated method
            will not be supplied with the ``self`` object instance, thus being
            equivalent to a class method.
        overwrite_existing (bool, optional): If False, will not overwrite the
            existing registry entry in :py:data:`.DATA_LOADERS` but raise an
            error instead.
        register_aliases (List[str], optional): If given, will additionally
            register this method under the given name
    """

    def load_func_decorator(func):
        """This decorator sets the load function's ``TargetCls`` attribute."""

        func_name = func.__name__
        assert func_name.startswith(LOAD_FUNC_PREFIX)
        assert func_name != "_load_file"  # used in DataManager

        name = func_name[len(LOAD_FUNC_PREFIX) :]

        # Wrap the load function such that the call signature is handled
        def load_func(dm: "DataManager", *args, **kwargs):
            """Calls the load function, either with or without ``self``."""
            load_func.__doc__ = func.__doc__
            if omit_self:
                # class method or standalone function
                return func(*args, **kwargs)
            # regular method
            return func(dm, *args, **kwargs)

        # Set the target class as function attribute and also make the captured
        # function more easily available
        load_func.TargetCls = TargetCls
        load_func._func = func

        # Carry over the docstring of the decorated function
        load_func.__doc__ = func.__doc__

        # Keep track of it via the registry dict.
        # NOTE Can only do this for class methods, which can act as standalone
        #      functions. Load functions that require access to other
        #      attributes or methods of the DataManager *object* can not be
        #      registered in a sensible way because these methods may not be
        #      available.
        if omit_self:
            _register_loader(
                load_func, name=name, overwrite_existing=overwrite_existing
            )
            for alias in register_aliases if register_aliases else ():
                _register_loader(
                    load_func,
                    name=alias,
                    overwrite_existing=overwrite_existing,
                )

        return load_func

    return load_func_decorator
