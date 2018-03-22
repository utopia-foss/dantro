"""This module implements loaders mixin classes for use with the DataManager.

All these mixin classes should follow the pattern:
class LoadernameLoaderMixin:
    def _load_loadername(filepath: str, *, TargetCls):
        # ...
        return TargetCls(...)
    _load_loadername.TargetCls = TheTargetContainerClass

Each `_load_loadername` method gets supplied with path to a file and the
`TargetCls` argument, which creates a TargetClass of the correct type and name
"""

import logging

from dantro.container import MutableMappingContainer
import dantro.tools as tools

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class YamlLoaderMixin:
    """Supplies functionality to load yaml files in the data manager"""

    @staticmethod
    def _load_yaml(filepath: str, *, TargetCls: MutableMappingContainer):
        """Load a yaml file from the given path and creates a container to
        store that data in."""
        # Load the dict
        d = tools.load_yml(filepath)

        # Populate the target container with the data
        return TargetCls(data=d, attrs=dict(filepath=filepath))
    _load_yaml.TargetCls = MutableMappingContainer

    # Also make available under `yml`
    _load_yml = _load_yaml
