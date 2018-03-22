"""This module implements loaders mixin classes for use with the DataManager"""

import logging
import warnings
from typing import Union

from dantro.container import MutableMappingContainer
import dantro.tools as tools

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class YamlLoaderMixin:
    """Supplies functionality to load yaml files in the data manager"""


    def _load_yaml():
        pass

    # Also make available under `yml`
    _load_yml = _load_yaml
