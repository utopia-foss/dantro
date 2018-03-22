"""This module implements loaders mixin classes for use with the DataManager"""

import logging
import warnings
from typing import Union

import yaml

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class YamlLoaderMixin:
    """Supplies functionalit to load yaml files in the data manager"""
