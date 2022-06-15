"""This module implements mixin classes which provide numeric interfaces for
containers
"""

import logging
import math
import operator

import numpy as np

from ..abc import AbstractDataContainer

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class UnaryOperationsMixin:
    """This mixin class implements the methods needed for unary operations.

    It leaves out those that expect that return values are of a certain type,
    e.g. ``__complex__``, ``__int__``, ...
    """

    def __neg__(self):
        """Make negative

        Returns:
            A new object with negative elements
        """
        return apply_func_to_copy(self, operator.neg)

    def __pos__(self):
        """Make positive

        Returns:
            A new object with negative elements
        """
        return apply_func_to_copy(self, operator.pos)

    def __abs__(self):
        """Absolute value

        Returns:
            A new object with the absolute value of the elements
        """
        return apply_func_to_copy(self, operator.abs)

    def __invert__(self):
        """Inverse value

        Returns:
            A new object with the inverted values of the elements
        """
        return apply_func_to_copy(self, operator.invert)

    def __round__(self):
        """Rounds number to nearest integer

        Returns:
            A new object as rounded number to nearest integer
        """
        return apply_func_to_copy(self, round)

    def __ceil__(self):
        """Smallest integer

        Returns:
            A new object containing the smallest integer
        """
        return apply_func_to_copy(self, math.ceil)

    def __floor__(self):
        """Largest integer

        Returns:
            A new object containing the largest element
        """
        return apply_func_to_copy(self, math.floor)

    def __trunc__(self):
        """Truncated to the nearest integer toward 0

        Returns:
            A new object containing the truncated element
        """
        return apply_func_to_copy(self, math.trunc)


class NumbersMixin(UnaryOperationsMixin):
    """This mixin implements the methods needed for calculating with numbers."""

    def __add__(self, other):
        """Add two objects

        Returns:
            A new object containing the summed data
        """
        return apply_func_to_copy(self, operator.add, other)

    def __sub__(self, other):
        """Subtract two objects

        Returns:
            A new object containing the subtracted data
        """
        return apply_func_to_copy(self, operator.sub, other)

    def __mul__(self, other):
        """Multiply two objects

        Returns:
            A object containing the multiplied data
        """
        return apply_func_to_copy(self, operator.mul, other)

    def __truediv__(self, other):
        """Divide two objects

        Returns:
            A new object containing the divided data
        """
        return apply_func_to_copy(self, operator.truediv, other)

    def __floordiv__(self, other):
        """Floor divide two objects

        Returns:
            A new object containing the floor divided data
        """
        return apply_func_to_copy(self, operator.floordiv, other)

    def __mod__(self, other):
        """Calculate the modulo of two objects

        Returns:
            A new object containing the summed data
        """
        return apply_func_to_copy(self, operator.mod, other)

    def __divmod__(self, other):
        """Calculate the floor division and modulo of two objects

        Returns:
            A new object containing the floor divided data and its modulo
        """
        return apply_func_to_copy(self, divmod, other)

    def __pow__(self, other):
        """Calculate the self data to the power of other data

        Returns:
            A new object containing the result
        """
        return apply_func_to_copy(self, operator.pow, other)

    # inplace operations
    def __iadd__(self, other):
        """Add two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.iadd, other)

    def __isub__(self, other):
        """Subtract two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.isub, other)

    def __imul__(self, other):
        """Multiply two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.imul, other)

    def __itruediv__(self, other):
        """Divide two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.itruediv, other)

    def __ifloordiv__(self, other):
        """Floor divide two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.ifloordiv, other)

    def __imod__(self, other):
        """Calculate the modulo of two objects

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.imod, other)

    def __ipow__(self, other):
        """Calculate the self data to the power of other data

        Returns:
            Self with modified data
        """
        return apply_func_inplace(self, operator.ipow, other)


class ComparisonMixin:
    """This Mixin implements functions to compare objects"""

    def __eq__(self, other):
        """Equality"""
        return self.data == get_data(other)

    def __ne__(self, other):
        """Inequality"""
        return self.data != get_data(other)

    def __lt__(self, other):
        """Less than"""
        return self.data < get_data(other)

    def __le__(self, other):
        """Less than or equal"""
        return self.data <= get_data(other)

    def __gt__(self, other):
        """Greater than"""
        return self.data > get_data(other)

    def __ge__(self, other):
        """Greater than or equal"""
        return self.data >= get_data(other)

    def __bool__(self):
        """Truth value"""
        return bool(self.data)


# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def get_data(obj):
    """Get the data of ``obj`` depending on whether it is part of dantro or
    not.

    Args:
        obj: The object to check

    Returns:
        Either the ``.data`` attribute of a dantro-based object or otherwise
            the object itself.
    """
    if isinstance(obj, AbstractDataContainer):
        return obj.data
    # Not dantro-based, just return the object itself.
    return obj


def apply_func_to_copy(obj, func, other=None):
    """Apply a given function to a copy for all datatypes

    Returns:
        An object with the data on which the function was applied
    """
    # Work on a copy
    new = obj.copy()

    # Change the data of the new object
    if other is None:
        new._data = func(new.data)
    else:
        if isinstance(other, AbstractDataContainer):
            new._data = func(new.data, other.data)
        else:
            new._data = func(new.data, other)

    return new


def apply_func_inplace(obj, func, other=None):
    """Apply a given function inplace for all data types.

    Returns:
        An object with the data on which the function was applied
    """
    # Change the data of the new object
    if other is None:
        func(obj._data)
    else:
        if isinstance(other, AbstractDataContainer):
            func(obj._data, other.data)
        else:
            func(obj._data, other)

    return obj
