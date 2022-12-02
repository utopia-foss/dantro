"""Implements registration of data loaders, including a decorator to ensure
correct loader function signature (which also automatically keeps track of the
data loader function).
"""
import logging
from typing import Callable, Dict, List

log = logging.getLogger(__name__)

DATA_LOADERS: Dict[str, type] = dict()
"""The dantro data loaders registry.

The :py:class:`~dantro.data_mngr.DataManager` and derived classes have access
to all data loaders via this registry (in addition to method-based access
they have via potentially used mixins).

To register a new loader, use the :py:func:`.add_loader` decorator.
"""

LOAD_FUNC_PREFIX: str = "_load_"
"""The prefix that all load functions need to start with"""

# -----------------------------------------------------------------------------


def _register_loader(
    wrapped_func: Callable,
    *,
    name: str = None,
    overwrite_existing: bool = True,
):
    """Registers a loader function in :py:data:`.DATA_LOADERS`.

    Args:
        wrapped_func (Callable): The load function, wrapped by whatever the
            :py:func:`.add_loader` decorator does.
        name (str, optional): The name to use in registration
        overwrite_existing (bool, optional): Whether to overwrite an existing
            entry. If False and the wrapped function not being identical, there
            will be an error.
    """
    if name in DATA_LOADERS and not overwrite_existing:
        # May be identical though, in which case we need no error
        existing_func = DATA_LOADERS[name]._func
        new_func = wrapped_func._func

        if existing_func is not new_func:
            raise ValueError(
                f"A loader function with the name '{name}' is already "
                "registered!\n"
                f"  Existing:  {existing_func.__name__}  {existing_func}\n"
                f"  New:       {new_func.__name__}  {new_func}\n"
                "Either change the name or set the `overwrite_existing` flag."
            )

    # All good, can register
    DATA_LOADERS[name] = wrapped_func
    log.debug("Registered data loader:  %s", name)


def add_loader(
    *,
    TargetCls: type,
    omit_self: bool = True,
    overwrite_existing: bool = True,
    register_aliases: List[str] = None,
):
    """This decorator should be used to specify loader methods in mixin classes
    to the :py:class:`~dantro.data_mngr.DataManager`.

    All decorated methods will additinoally be registered in the data
    loaders registry.

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

        # Keep track of it via the registry dict
        _register_loader(
            load_func, name=name, overwrite_existing=overwrite_existing
        )
        for alias in register_aliases if register_aliases else ():
            _register_loader(
                load_func, name=alias, overwrite_existing=overwrite_existing
            )

        return load_func

    return load_func_decorator
