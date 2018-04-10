"""This module implements classes that serve as mixins for classes derived from base classes."""

import logging

import numpy as np

import dantro.base
import operator
import math
log = logging.getLogger(__name__)

# Local variables

# -----------------------------------------------------------------------------

class ForwardAttrsToDataMixin():
    """This Mixin class implements the method to forward attributes to the data attribute.
    """

    def __getattr__(self, attr_name):
        """Forward attributes that were not available in this class to data attribute."""
        return getattr(self.data, attr_name)


class UnaryOperationsMixin():
    """This Mixin class implements the methods needed for unary operations
    """

    def __neg__(self):
        """Negative numbers
        
        Returns:
            A new object with negative elements
        """
        return apply_func_to_copy(self, operator.neg)

    def __pos__(self):
        """Negative numbers
        
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

    def __complex__(self):
        """Complex numbers

        Returns:
            A new object as complex number
        """
        return apply_func_to_copy(self, complex)

    def __int__(self):
        """Integer numbers

        Returns:
            A new object as integer
        """
        return apply_func_to_copy(self, int)

    def __float__(self):
        """Float numbers

        Returns:
            A new object as float
        """
        return apply_func_to_copy(self, float)

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
    """This Mixin class implements the methods needed for calculating with numbers
    """

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


class ComparisonMixin():
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
    """Get the data 
    """
    if isinstance(obj, dantro.base.BaseDataContainer):
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
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = func(new.data, other.data)
        else:
            new._data = func(new.data, other)
    return new


def apply_func_inplace(obj, func, other=None):
    """Apply a given function inplace for all datatypes
    
    Returns:
        An object with the data on which the function was applied
    """
    # Change the data of the new object
    if other is None:
        func(obj._data)
    else:
        if isinstance(other, dantro.base.BaseDataContainer):
            func(obj._data, other.data)
        else:
            func(obj._data, other)
    return obj
class ProxyMixin:
    """This Mixin class overwrites the `data` property to allow proxy objects.

    It should be used to add support for certain proxy types to a container.

    A proxy object is a place keeper for data that is not yet loaded. It will
    only be loaded if `data` is directly accessed.
    """

    @property
    def data(self):
        """The container data. If the data is a proxy, this call will lead
        to the resolution of the proxy.
        
        Returns:
            The data stored in this container
        """
        # Have to check whether the data might be a proxy. If so, resolve it.
        if self.data_is_proxy:
            log.debug("Resolving %s for %s ...",
                      self.proxy.classname, self.logstr)
            self._data = self.proxy.resolve()

        # Now, the data should be loaded and can be returned
        return self._data

    @property
    def data_is_proxy(self) -> bool:
        """Returns true, if this is proxy data
        
        Returns:
            bool: Whether the _currently_ stored data is a proxy object
        """
        return isinstance(self._data, dantro.base.BaseDataProxy)

    @property
    def proxy(self) -> dantro.base.BaseDataProxy:
        """If the data is proxy, returns the proxy data object without using 
        the .data attribute (which would trigger resolving the proxy); else 
        returns None.
        
        Returns:
            Union[BaseDataProxy, None]: If the data is proxy, return the
                proxy object; else None.
        """
        if self.data_is_proxy:
            return self._data
        return None


class Hdf5ProxyMixin(ProxyMixin):
    """Specialises the ProxyMixin to the capabilities of a Hdf5 Proxy, i.e. it
    allows access to the cached `dtype` and `shape` properties of the
    Hdf5DataProxy without resolving the proxy.
    """

    @property
    def dtype(self) -> np.dtype:
        """Returns the NumpyDCs dtype, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.dtype
        return self.data.dtype
    
    @property
    def shape(self) -> tuple:
        """Returns the NumpyDCs shape, proxy-aware"""
        if self.data_is_proxy:
            return self.proxy.shape
        return self.data.shape
