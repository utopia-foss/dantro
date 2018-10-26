"""Supplies loading functions for YAML files"""

import logging

from ..containers import MutableMappingContainer, ObjectContainer
from ..tools import load_yml
from ._tools import add_loader

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class YamlLoaderMixin:
    """Supplies functionality to load yaml files in the data manager"""

    @add_loader(TargetCls=MutableMappingContainer)
    def _load_yaml(filepath: str, *, TargetCls: type) -> MutableMappingContainer:
        """Load a yaml file from the given path and creates a container to
        store that data in.
        
        Args:
            filepath (str): Where to load the yaml file from
            TargetCls (type): The class constructor
        
        Returns:
            MutableMappingContainer: The loaded yaml file as a container
        """
        # Load the dict
        d = load_yml(filepath)

        # Populate the target container with the data
        return TargetCls(data=d, attrs=dict(filepath=filepath))
    
    @add_loader(TargetCls=ObjectContainer)
    def _load_yaml_to_object(filepath: str, *, TargetCls: type) -> ObjectContainer:
        """Load a yaml file from the given path and creates a container to
        store that data in.
        
        Args:
            filepath (str): Where to load the yaml file from
            TargetCls (type): The class constructor
        
        Returns:
            ObjectContainer: The loaded yaml file as a container
        """
        # Load the dict
        d = load_yml(filepath)

        # Populate the target container with the data
        return TargetCls(data=d, attrs=dict(filepath=filepath))

    # Also make available under `yml`
    _load_yml = _load_yaml
    _load_yml_to_object = _load_yaml_to_object
