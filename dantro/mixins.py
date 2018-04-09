"""This module implements classes that serve as mixins for classes derived from base classes."""

import logging

import dantro.base
import operator
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
    
    def _apply_to_copy_numbersmixin(self, func, other):
        """Apply a given function to a copy for all datatypes
        
        Returns:
            An object with the data on which the function was applied
        """
        # Work on a copy
        new = self.copy()
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            new._data = func(new.data, other.data)
        else:
            new._data = func(new.data, other)
        return new

    def _apply_inplace_numbersmixin(self, func, other):
        """Apply a given function inplace for all datatypes
        
        Returns:
            An object with the data on which the function was applied
        """
        # Change the data of the new object
        if isinstance(other, dantro.base.BaseDataContainer):
            func(self._data, other.data)
        else:
            func(self._data, other)
        return self

    def __add__(self, other):
        """Add two objects
        
        Returns:
            A new object containing the summed data
        """
        return self._apply_to_copy_numbersmixin(operator.add, other)

    def __sub__(self, other):
        """Subtract two objects
        
        Returns:
            A new object containing the subtracted data
        """
        return self._apply_to_copy_numbersmixin(operator.sub, other)

    def __mul__(self, other):
        """Multiply two objects
        
        Returns:
            A object containing the multiplied data
        """
        return self._apply_to_copy_numbersmixin(operator.mul, other)

    def __truediv__(self, other):
        """Divide two objects
        
        Returns:
            A new object containing the divided data
        """
        return self._apply_to_copy_numbersmixin(operator.truediv, other)

    def __floordiv__(self, other):
        """Floor divide two objects
        
        Returns:
            A new object containing the floor divided data
        """
        return self._apply_to_copy_numbersmixin(operator.floordiv, other)

    def __mod__(self, other):
        """Calculate the modulo of two objects
        
        Returns:
            A new object containing the summed data
        """
        return self._apply_to_copy_numbersmixin(operator.mod, other)

    def __divmod__(self, other):
        """Calculate the floor division and modulo of two objects
        
        Returns:
            A new object containing the floor divided data and its modulo
        """
        return self._apply_to_copy_numbersmixin(divmod, other)

    def __pow__(self, other):
        """Calculate the self data to the power of other data
        
        Returns:
            A new object containing the result
        """
        return self._apply_to_copy_numbersmixin(operator.pow, other)

    def __iadd__(self, other):
        """Add two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.iadd, other)

    def __isub__(self, other):
        """Subtract two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.isub, other)

    def __imul__(self, other):
        """Multiply two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.imul, other)

    def __itruediv__(self, other):
        """Divide two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.itruediv, other)

    def __ifloordiv__(self, other):
        """Floor divide two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.ifloordiv, other)

    def __imod__(self, other):
        """Calculate the modulo of two objects
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.imod, other)

    def __ipow__(self, other):
        """Calculate the self data to the power of other data
        
        Returns:
            Self with modified data
        """
        return self._apply_inplace_numbersmixin(operator.ipow, other)

class ComparisonMixin():
    """This Mixin implements functions to compare objects"""

    def _apply_inplace_comparison(self, func, other):
        """Apply a given function for all datatypes
        
        Returns:
            The result of the applied function applied on the correct datatype 
        """
        if isinstance(other, dantro.base.BaseDataContainer):
            return func(self.data, other.data)
        else:
            return func(self.data, other)

    def __eq__(self, other):
        """Equality"""
        return self._apply_inplace_comparison(operator.eq, other)

    def __ne__(self, other):
        """Inequality"""
        return self._apply_inplace_comparison(operator.ne, other)
    
    def __lt__(self, other):
        """Less than"""
        return self._apply_inplace_comparison(operator.lt, other)

    def __le__(self, other):
        """Less than or equal"""
        return self._apply_inplace_comparison(operator.le, other)

    def __gt__(self, other):
        """Greater than"""
        return self._apply_inplace_comparison(operator.gt, other)

    def __ge__(self, other):
        """Greater than or equal"""
        return self._apply_inplace_comparison(operator.ge, other)

    def __bool__(self):
        """Truth value"""
        return bool(self.data)