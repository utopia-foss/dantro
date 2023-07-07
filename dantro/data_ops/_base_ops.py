"""Implements operations that need to be importable from other modules or that
are so basic that they apply to a wide range of applications."""

import operator
from typing import Callable

# -----------------------------------------------------------------------------

# fmt: off
BOOLEAN_OPERATORS = {
    "==": operator.eq,  "eq": operator.eq,
    "<":  operator.lt,  "lt": operator.lt,
    "<=": operator.le,  "le": operator.le,
    ">":  operator.gt,  "gt": operator.gt,
    ">=": operator.ge,  "ge": operator.ge,
    "!=": operator.ne,  "ne": operator.ne,
    "^":  operator.xor, "xor": operator.xor,
    #
    # Expecting an iterable as second argument
    "contains":         operator.contains,
    "in":               (lambda x, y: x in y),
    "not in":           (lambda x, y: x not in y),
    #
    # Performing bitwise boolean operations to support numpy logic
    "in interval":      (lambda x, y: x >= y[0] & x <= y[1]),
    "not in interval":  (lambda x, y: x < y[0] | x > y[1]),
} # End of boolean operator definitions
"""Boolean binary operators"""
# fmt: on


# -----------------------------------------------------------------------------


def _make_passthrough(func: Callable) -> Callable:
    """Wraps a callable such that it returns its first positional argument.

    This is meant for functions that operate on an object (conventionally the
    first argument) and do not have a return value.
    By constructing a callable using this function, it can be made compatible
    with the data transformation framework.

    .. code-block:: python

        f = setattr                 # f has no return value
        g = _make_passtrough(f)     # g will return the first argument
    """

    def wrapped(d, *args, **kwargs):
        func(d, *args, **kwargs)
        return d

    return wrapped
