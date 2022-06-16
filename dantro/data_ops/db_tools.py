"""Tools that help to monitor and manipulate the operations database"""

import logging
from difflib import get_close_matches as _get_close_matches
from typing import Callable, Sequence, Union

from .db import _OPERATIONS

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def register_operation(
    *,
    name: str,
    func: Callable,
    skip_existing: bool = False,
    overwrite_existing: bool = False,
    _ops: dict = None,
) -> None:
    """Adds an entry to the shared operations registry.

    Args:
        name (str): The name of the operation
        func (Callable): The callable
        skip_existing (bool, optional): Whether to skip registration if the
            operation name is already registered. This suppresses the
            ValueError raised on existing operation name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing operation of the same name. If given, this takes
            precedence over ``skip_existing``.
        _ops (dict, optional): The operations database object to use; if None,
            uses the dantro operations database

    Raises:
        TypeError: On invalid name or non-callable for the func argument
        ValueError: On already existing operation name and no skipping or
            overwriting enabled.
    """
    if _ops is None:
        _ops = _OPERATIONS

    if name in _ops and not overwrite_existing:
        if skip_existing:
            log.debug(
                "Operation '%s' is already registered and will not be "
                "registered again.",
                name,
            )
            return
        raise ValueError(
            f"Operation name '{name}' already exists! Refusing to register a "
            "new one. Set the overwrite_existing flag to force overwriting."
        )

    elif not callable(func):
        raise TypeError(
            f"The given {func} for operation '{name}' is not callable! "
        )

    elif not isinstance(name, str):
        raise TypeError(
            f"Operation name need be a string, was {type(name)} with "
            f"value {name}!"
        )

    _ops[name] = func
    log.debug("Registered operation '%s'.", name)


def is_operation(
    arg: Union[str, Callable] = None,
    /,
    *,
    _ops: dict = None,
    _reg_func: Callable = None,
    **kws,
):
    """Decorator for registering operations with the operations database.

    Usage examples:

    .. code-block:: python

        from dantro.data_ops import is_operation

        # Operation name deduced from function name
        @is_operation
        def my_operation(data, *args):
            # ...

        # Custom operation name
        @is_operation("do_stuff")
        def my_operation_with_a_custom_name(foo, bar):
            # ...

        # Overwriting an operation of the same name
        @is_operation("do_stuff", overwrite_existing=True)
        def actually_do_stuff(spam, fish):
            # ...

    If you want to provide different default values for the decorator used in
    your package, consider the following implementation:

    .. code-block:: python

        from dantro.data_ops import register_operation as _register_operation
        from dantro.data_ops import is_operation as _is_operation

        # Your operations database object that is used as the default database.
        MY_OPERATIONS = dict()

        # Define a registration function with `skip_existing = True` as default
        # and evaluation of the default database
        def my_reg_func(*args, skip_existing=True, _ops=None, **kwargs):
            _ops = _ops if _ops is not None else MY_OPERATIONS

            return _register_operation(
                *args, skip_existing=skip_existing, _ops=_ops, **kwargs
            )

        # Define a custom decorator that uses the custom registration function
        def my_decorator(
            arg: Union[str, Callable] = None, /, **kws
        ):
            return _is_operation(arg, _reg_func=my_reg_func, **kws)

        # Usage examples
        @my_decorator
        def some_operation():
            pass

        @my_decorator("my_operation_name")
        def some_other_operation():
            pass

    Args:
        arg (Union[str, Callable], optional): The name that should be used in
            the operation registry. If not given, will use the name of the
            decorated function instead. If a callable, this refers to the
            ``@is_operation`` call syntax and will use that as a function.
        _ops (dict, optional): The operations database to use. If not given, ´
            will use the dantro default operations database.
        _reg_func (Callable, optional): If given, uses that callable for
            registration, which should have the same signature as
            :py:func:`.register_operation`. If None, uses dantro's registration
            function, :py:func:`.register_operation`.
        **kws: Passed to :py:func:`.register_operation` or a potentially given
            custom ``_reg_func``.
    """
    if _reg_func is None:
        _reg_func = register_operation

    def wrapper(func):
        # Need some nonlocal names, see PEP3104
        nonlocal arg, _reg_func

        name = arg
        if callable(name):
            # Invocation via @is_operation -- without parentheses
            name = func.__name__
        elif name is None:
            # Invocation via @is_operation() -- without arguments
            name = func.__name__

        _reg_func(name=name, func=func, _ops=_ops, **kws)
        return func

    # Allow both invocation styles
    if callable(arg):
        # @is_operation
        return wrapper(arg)

    # @is_operation(*args, **kwargs)
    return wrapper


def available_operations(
    *, match: str = None, n: int = 5, _ops: dict = None
) -> Sequence[str]:
    """Returns all available operation names or a fuzzy-matched subset of them.

    Args:
        match (str, optional): If given, fuzzy-matches the names and only
            returns close matches to this name.
        n (int, optional): Number of close matches to return. Passed on to
            :py:func:`difflib.get_close_matches`
        _ops (dict, optional): The operations database object to use; if None,
            uses the dantro operations database

    Returns:
        Sequence[str]: All available operation names or the matched subset.
            The sequence is sorted alphabetically.
    """
    if _ops is None:
        _ops = _OPERATIONS

    if match is None:
        return _ops.keys()

    # Use fuzzy matching to return close matches
    return _get_close_matches(match, _ops.keys(), n=n)
