"""This module implements loaders mixin classes for use with the DataManager.

All these mixin classes should follow the pattern:
class LoadernameLoaderMixin:
    def _load_loadername(filepath: str, *, TargetCls):
        # ...
        return TargetCls(...)
    _load_loadername.TargetCls = TheTargetContainerClass

Each `_load_loadername` method gets supplied with path to a file and the
`TargetCls` argument, which can be called to create an object of the correct
type and name.
"""

import logging

from dantro.container import MutableMappingContainer
from dantro.group import OrderedDataGroup
import dantro.tools as tools

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Decorator to ensure correct loader function signature

def add_loader(*, TargetCls, omit_self: bool=True):
    """This decorator should be used to specify loader functions.
    
    Args:
        TargetCls: Description
        omit_self (bool, optional): If True (default), the decorated method
            will not be supplied with the `self` object instance
    """
    def load_func_decorator(func):
        """This decorator sets the load function's `TargetCls` attribute."""
        def load_func(instance, *args, **kwargs):
            """Calls the load function, either as with or without `self`."""
            if omit_self:
                return func(*args, **kwargs)
            # not as static method
            return func(instance, *args, **kwargs)

        # Set the target class as function attribute 
        load_func.TargetCls = TargetCls
        return load_func
    return load_func_decorator

# -----------------------------------------------------------------------------

class YamlLoaderMixin:
    """Supplies functionality to load yaml files in the data manager"""

    @add_loader(TargetCls=MutableMappingContainer)
    def _load_yaml(filepath: str, *, TargetCls) -> MutableMappingContainer:
        """Load a yaml file from the given path and creates a container to
        store that data in.
        
        Args:
            filepath (str): Where to load the yaml file from
            TargetCls (TYPE): The MutableMappingContainer class
        
        Returns:
            MutableMappingContainer: The loaded yaml file
        """
        # Load the dict
        d = tools.load_yml(filepath)

        # Populate the target container with the data
        return TargetCls(data=d, attrs=dict(filepath=filepath))

    # Also make available under `yml`
    _load_yml = _load_yaml

# -----------------------------------------------------------------------------

class Hdf5LoaderMixin:
    """Supplies functionality to load hdf5 files into the data manager.

    It resolves the hdf5 groups into corresponding data groups and the datasets
    into NumpyDataContainers."""

    @add_loader(TargetCls=OrderedDataGroup)
    def _load_hdf5(filepath: str, *, TargetCls) -> OrderedDataGroup:
        raise NotImplementedError

    @add_loader(TargetCls=OrderedDataGroup)
    def _load_hdf5_proxy(filepath: str, *, TargetCls) -> OrderedDataGroup:
        raise NotImplementedError
