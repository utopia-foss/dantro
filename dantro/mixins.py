"""This module implements classes that serve as mixins for classes derived from base classes."""

import logging

import dantro.base

log = logging.getLogger(__name__)

# Local variables

# -----------------------------------------------------------------------------

class ForwardAttrsToDataMixin():
    """This Mixin class implements the method to forward attributes to the data attribute.
    """

    def __getattr__(self, attr_name):
        """Forward attributes that were not available in this class to data attribute."""
        return getattr(self.data, attr_name)


class NumbersMixin():
    """This Mixin class implements the methods needed for calculating with numbers
    """

    def __add__(self, other):
        """Add two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data + other.data
        else:
            new._data = new.data + other
        return new

    def __sub__(self, other):
        """Subtract two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data - other.data
        else:
            new._data = new.data - other
        return new

    def __mul__(self, other):
        """Multiply two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data * other.data
        else:
            new._data = new.data * other
        return new

    def __truediv__(self, other):
        """Divide two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data / other.data
        else:
            new._data = new.data / other
        return new

    def __floordiv__(self, other):
        """Floor divide two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data // other.data
        else:
            new._data = new.data // other
        return new

    def __mod__(self, other):
        """Calculate the modulo of two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data % other.data
        else:
            new._data = new.data % other
        return new

    def __divmod__(self, other):
        """Calculate thefloor division and modulo of two objects"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = divmod(new.data, other.data)
        else:
            new._data = divmod(new.data, other)
        return new

    def __pow__(self, other):
        """Calculate self to the power of other"""
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = new.data ** other.data
        else:
            new._data = new.data ** other
        return new
