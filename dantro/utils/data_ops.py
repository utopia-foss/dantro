"""This module implements data processing operations for dantro objects"""

import logging

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

# The operations database
OPERATIONS = {}

# -----------------------------------------------------------------------------
# Registering and applying operations

def register_operation():
    raise NotImplementedError()

def apply_operation():
    raise NotImplementedError()
