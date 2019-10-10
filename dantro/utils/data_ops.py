"""This module implements data processing operations for dantro objects"""

import logging
from typing import Callable, Any

from ..base import BaseDataContainer
import numpy as np

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# The Operations Database
# NOTE If a single "object to act upon" can be reasonably defined for an
#      operation, it should be accepted as the first positional argument.
OPERATIONS = {
    # General operations - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Unary ...................................................................
    
    # Binary ..................................................................
    # Item access
    'getitem':      lambda obj, key: obj[key],

    # N-ary ...................................................................
    # Attribute-related
    'getattr':      getattr,
    'callattr':     lambda obj, attr, *a, **k: getattr(obj, attr)(*a, **k),
    

    # Numerical operations - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # These are mainly attribute-call based, to
    # Unary ...................................................................
    'increment':    lambda obj: obj + 1,
    'decrement':    lambda obj: obj - 1,
    
    '.T':           lambda obj: obj.T,

    # Binary ..................................................................

    # N-ary ...................................................................
    'invert':       lambda obj, **kws: np.invert(obj, **kws),
    'transpose':    lambda obj, **kws: np.transpose(obj, **kws)
}

# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation(*, name: str, func: Callable,
                       overwrite_existing: bool=False) -> None:
    """Sets the shared OPERATIONS registry"""
    if name in OPERATIONS and not overwrite_existing:
        raise ValueError("Operation name '{}' already exists! Refusing to "
                         "register a new one. Set the overwrite_existing flag "
                         "to force overwriting.".format(name))

    elif not callable(func):
        raise TypeError("The given {} for operation '{}' is not callable! "
                        "".format(func, name))

    OPERATIONS[name] = func

def apply_operation(op_name: str, *args, _maintain_container_type: bool=False,
                    **kwargs) -> Any:
    """Apply an operation with the given arguments and then return it.

    This function is also capable to attempt to maintain the result type of
    the operation. For that, the type of the first positional argument is
    inspected before the application of the operation. If it is a dantro
    container type and the result is not, it is attempted to feed the result
    to a new instance of the type.
    """
    op = OPERATIONS[op_name]

    if _maintain_container_type:
        if args and isinstance(args[0], BaseDataContainer):
            _ContCls = type(args[0])
            _cont_name = args[0].name
            _cont_attrs = args[0].attrs
        else:
            # Not applicable; turn back the flag
            _maintain_container_type = False

    # Compute the results
    res = op(*args, **kwargs)

    # If container type is to be maintained but does not match, attempt to
    # change it, carrying over name and attributes ...
    if _maintain_container_type and not isinstance(res, _ContCls):
        try:
            res = _ContCls(name="{}_after_{}".format(_cont_name, op_name),
                           data=res, attrs=_cont_attrs)
        except:
            pass

    return res
