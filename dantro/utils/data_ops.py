"""This module implements data processing operations for dantro objects"""

import logging

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

# The operations database
OPERATIONS = {
    # Attribute access
    'getattr':  getattr,
    
    # Item access
    'getitem':  lambda obj, key: obj[key],

    # Numerical operations
    'increment':    lambda obj: obj + 1,
    'decrement':    lambda obj: obj + 1
}

# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation():
    raise NotImplementedError()

def apply_operation():
    raise NotImplementedError()
