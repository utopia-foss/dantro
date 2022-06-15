"""Tools that help to monitor and manipulate the operations database"""

import logging
from difflib import get_close_matches as _get_close_matches
from typing import Callable, Sequence

from .db import _OPERATIONS

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def register_operation(
    *,
    name: str,
    func: Callable,
    skip_existing: bool = False,
    overwrite_existing: bool = False,
    _ops: dict = _OPERATIONS,
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
        _ops (dict, optional): The operations database object to use

    Raises:
        TypeError: On invalid name or non-callable for the func argument
        ValueError: On already existing operation name and no skipping or
            overwriting enabled.
    """
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


def available_operations(
    *, match: str = None, n: int = 5, _ops: dict = _OPERATIONS
) -> Sequence[str]:
    """Returns all available operation names or a fuzzy-matched subset of them.

    Args:
        match (str, optional): If given, fuzzy-matches the names and only
            returns close matches to this name.
        n (int, optional): Number of close matches to return. Passed on to
            :py:func:`difflib.get_close_matches`
        _ops (dict, optional): The operations database object to use

    Returns:
        Sequence[str]: All available operation names or the matched subset.
            The sequence is sorted alphabetically.
    """
    if match is None:
        return _ops.keys()

    # Use fuzzy matching to return close matches
    return _get_close_matches(match, _ops.keys(), n=n)
