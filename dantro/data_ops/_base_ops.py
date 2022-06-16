"""Implements operations that need to be importable from other modules"""

import operator

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
