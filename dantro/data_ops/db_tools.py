"""Tools that help to monitor and manipulate the operations database"""

import logging
from difflib import get_close_matches as _get_close_matches
from typing import Callable, Dict, Sequence, Union

from ..exceptions import *
from ..tools import make_columns as _make_columns
from .db import _OPERATIONS

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def register_operation(
    func: Callable,
    name: str = None,
    *,
    skip_existing: bool = False,
    overwrite_existing: bool = False,
    _ops: dict = None,
) -> None:
    """Adds an entry to the shared operations registry.

    Args:
        func (Callable): The callable that is to be registered as operation.
        name (str, optional): The name of the operation. If not given (and the
            callable not being a lambda), will use the function name instead.
        skip_existing (bool, optional): Whether to skip registration if the
            operation name is already registered. This suppresses the
            ValueError raised on existing operation name.
        overwrite_existing (bool, optional): Whether to overwrite a potentially
            already existing operation of the same name. If given, this takes
            precedence over ``skip_existing``.
        _ops (dict, optional): The operations database object to use; if None,
            uses the :ref:`dantro operations database <data_ops_available>`.

    Raises:
        TypeError: On invalid name or non-callable for the func argument
        ValueError: On already existing operation name and no skipping or
            overwriting enabled. Also if no ``name`` was given but the given
            callable is a lambda (which only has ``<lambda>`` as name).
    """
    if _ops is None:
        _ops = _OPERATIONS

    if not callable(func):
        raise TypeError(
            f"The given {func} for operation '{name}' is not callable!"
        )

    if name is None:
        name = func.__name__
        if name == (lambda: 0).__name__:
            raise ValueError(
                "Could not automatically deduce an operation name because the "
                "given callable appears to be a lambda! Explicitly specify an "
                "operation name using the `name` argument."
            )
    elif not isinstance(name, str):
        raise TypeError(
            f"Operation name need be a string, was {type(name)} with "
            f"value {name}!"
        )

    if name in _ops and not overwrite_existing:
        if skip_existing:
            log.debug(
                "Operation '%s' is already registered and will not be "
                "registered again. Choose a different name or unset the "
                "`skip_existing` flag.",
                name,
            )
            return
        raise ValueError(
            f"Operation name '{name}' already exists! Not overwriting. "
            "Set the `overwrite_existing` flag to force overwriting or choose "
            "a different `name`. If no name was given, consider specifying it "
            "explicitly instead of letting it be deduced."
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
    """Decorator for registering functions with the operations database.

    As an alternative to :py:func:`.register_operation`, this decorator can be
    used to register a function with the operations database right where its
    defined:

    .. testcode::

        from dantro.data_ops import is_operation

        # Operation name deduced from function name
        @is_operation
        def my_operation(data, *args):
            # ... do stuff here ...
            return data

        # Custom operation name
        @is_operation("do_something")
        def my_operation_with_a_custom_name(foo, bar):
            pass

        # Overwriting an operation of the same name
        @is_operation("do_something", overwrite_existing=True)
        def actually_do_something(spam, fish):
            pass

    .. testcode::
        :hide:

        from dantro.data_ops.db import _OPERATIONS
        assert "my_operation" in _OPERATIONS
        assert "do_something" in _OPERATIONS

    See :ref:`register_data_ops` for general information.
    For instructions on how to overwrite this decorator with a custom one, see
    :ref:`customize_db_tools`.

    Args:
        arg (Union[str, Callable], optional): The name that should be used in
            the operation registry. If not given, will use the name of the
            decorated function instead. If a callable, this refers to the
            ``@is_operation`` call syntax and will use that as a function.
        _ops (dict, optional): The operations database to use. If not given,
            uses the :ref:`dantro operations database <data_ops_available>`.
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
            name = None
        elif name is None:
            # Invocation via @is_operation() -- without arguments
            pass

        _reg_func(func, name=name, _ops=_ops, **kws)
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

    Also see :ref:`data_ops_available` for an overview.

    Args:
        match (str, optional): If given, fuzzy-matches the names and only
            returns close matches to this name.
        n (int, optional): Number of close matches to return. Passed on to
            :py:func:`difflib.get_close_matches`
        _ops (dict, optional): The operations database object to use; if None,
            uses the :ref:`dantro operations database <data_ops_available>`.

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


def get_operation(
    op_name: str, *, _ops: Dict[str, Callable] = None
) -> Callable:
    """Retrieve the operation's callable

    Args:
        op_name (str): Name of the operation
        _ops (Dict[str, Callable], optional): The operations database object
            to use; if None, uses the
            :ref:`dantro operations database <data_ops_available>`.

    Raises:
        BadOperationName: Upon invalid operation name
    """
    try:
        return _ops[op_name]

    except KeyError as err:
        # Find some close matches to make operation discovery easier
        _close_matches = available_operations(match=op_name, n=8, _ops=_ops)
        _did_you_mean = (
            f" Did you mean: {', '.join(_close_matches)} ?"
            if _close_matches
            else ""
        )
        _available = _make_columns(available_operations(_ops=_ops))

        raise BadOperationName(
            f"No operation '{op_name}' registered!{_did_you_mean} "
            f"\nAvailable operations:\n{_available}If you need to register "
            "a new operation, use dantro.utils.register_operation."
        ) from err
