"""This module implements loaders mixin classes for use with the DataManager.

All these mixin classes should follow the pattern:
class LoadernameLoaderMixin:

    @add_loader(TargetCls=TheTargetContainerClass)
    def _load_loadername(filepath: str, *, TargetCls):
        # ...
        return TargetCls(...)

Each `_load_loadername` method gets supplied with path to a file and the
`TargetCls` argument, which can be called to create an object of the correct
type and name.
By default, and to decouple the loader from the container, it should be
considered to be a staticmethod; in other words: the first positional argument
should _not_ be `self`!
If `self` is required, `omit_self=False` may be given to the decorator.
"""

import warnings
import logging

import numpy as np
import h5py as h5

from dantro.base import BaseDataGroup, BaseDataContainer
from dantro.container import MutableMappingContainer, NumpyDataContainer
from dantro.group import OrderedDataGroup
from dantro.proxy import Hdf5DataProxy
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

    # Define the container classes to use when loading data with this mixin
    _HDF5_DSET_DEFAULT_CLS = NumpyDataContainer

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5(self, filepath: str, *, TargetCls, load_as_proxy: bool=False, lower_case_keys: bool=False, print_params: dict=None) -> OrderedDataGroup:
        """Loads the specified hdf5 file into DataGroup and DataContainer-like
        object; this completely recreates the hierarchic structure of the hdf5
        file. The data can be loaded into memory completely, or be loaded as
        a proxy object.
        
        The h5py File and Group objects will be converted to the specified
        DataGroup-derived objects; the Dataset objects to the specified
        DataContainer-derived object.
        
        All attributes are carried over and are accessible under the `attrs`
        attribute.
        
        Args:
            filepath (str): hdf5 file to load
            TargetCls (OrderedDataGroup): the group object this is loaded into
            load_as_proxy (bool, optional): if True, the leaf datasets are
                loaded as Hdf5DataProxy objects. That way, they are only
                loaded when their .data attribute is accessed the first time.
                To do so, a reference to the hdf5 file is saved in a
                H5DataProxy
            lower_case_keys (bool, optional): whether to cast all keys to
                lower case
            print_params (dict, optional): parameters for the status report
                level: how verbose to print loading info; possible values are:
                    0: None, 1: on file level, 2: on dataset level
                fstrs1: format string level 1
                fstrs2: format string level 2
        
        Returns:
            OrderedDataGroup: The root level, corresponding to the file"""
        

        def recursively_load_hdf5(src, target: BaseDataGroup, *, load_as_proxy: bool, lower_case_keys: bool, GroupCls: BaseDataGroup, DsetCls: BaseDataContainer):
            """Recursively loads the data from the source hdf5 file into the target DataGroup object."""

            # Go through the elements of the source object
            for key, obj in src.items():
                if lower_case_keys and isinstance(key, str):
                    key = key.lower()

                if isinstance(obj, h5.Group):
                    # Need to continue recursion
                    # Create a group, carrying over its attributes manually
                    attrs = {k:v for k, v in obj.attrs.items()}
                    target.add(GroupCls(name=key, attrs=attrs))

                    # Continue recursion
                    recursively_load_hdf5(obj, target[key],
                                          load_as_proxy=load_as_proxy,
                                          lower_case_keys=lower_case_keys,
                                          GroupCls=GroupCls, DsetCls=DsetCls)

                elif isinstance(obj, h5.Dataset):
                    # Reached a leaf -> Import the data and attributes into a
                    # BaseDataContainer-derived object. This assumes that the
                    # given DsetCls supports np.ndarray-like data

                    # Progress information on loading this dataset
                    if plvl >= 2:
                        line = fstr2.format(name=target.name, key=key, obj=obj)
                        print(tools.fill_tty_line(line), end="\r")

                    if load_as_proxy:
                        # Instantiate a proxy object
                        data = Hdf5DataProxy(obj)
                    else:
                        # Import the data completely
                        data = np.array(obj)

                    # Save the data / the proxy to the target group, carrying
                    # over attrs manually
                    attrs = {k:v for k, v in obj.attrs.items()}
                    target.add(DsetCls(name=key, data=data, attrs=attrs))

                else:
                    warnings.warn("Object {} is neither a dataset nor a "
                                  "group, but of type {}. Cannot load this!"
                                  "".format(key, type(obj)),
                                  NotImplementedError)

        # Prepare print format strings
        print_params = print_params if print_params else {}
        plvl = print_params.get('level', 0)
        fstr1 = print_params.get('fstr1', "  Loading {name:} ... ")
        fstr2 = print_params.get('fstr2', "  Loading {name:} - {key:} ...")

        # Initialise the root group
        log.debug("Loading hdf5 file %s into %s ...",
                  filepath, TargetCls.__name__)
        root = TargetCls()

        # Get the classes to use for groups and/or containers
        DsetCls = self._HDF5_DSET_DEFAULT_CLS
        GroupCls = type(root)

        # Now recursively go through the hdf5 file and add them to the roo
        with h5.File(filepath, 'r') as h5file:
            if plvl >= 1:
                # Print information on the level of this file
                line = fstr1.format(name=root.name, file=filepath)
                print(tools.fill_tty_line(line), end="\r")

            # Load the file level attributes, manually re-creating the dict
            root.attrs = {k:v for k, v in h5file.attrs.items()}

            # Now recursively load the data into the root group
            recursively_load_hdf5(h5file, root,
                                  load_as_proxy=load_as_proxy,
                                  lower_case_keys=lower_case_keys,
                                  GroupCls=GroupCls, DsetCls=DsetCls)

        return root

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5_proxy(self, filepath: str, *, TargetCls, **loader_kwargs) -> OrderedDataGroup:
        """Loads the specified hdf5 file into DataGroup and DataContainer-like
        object; this completely recreates the hierarchic structure of the hdf5
        file. Instead of loading all data directly, this loader will create
        proxy objects for each dataset.
        
        The h5py File and Group objects will be converted to the specified
        DataGroup-derived objects; the Dataset objects to the specified
        DataContainer-derived object, but storing only proxies.
        
        All attributes are carried over and are accessible under the `attrs`
        attribute.
        
        Args:
            filepath (str): hdf5 file to load
            TargetCls (OrderedDataGroup): the group object this is loaded into
            lower_case_keys (bool, optional): whether to cast all keys to
                lower case
            print_params (dict, optional): parameters for the status report
                level: how verbose to print loading info; possible values are:
                    0: None, 1: on file level, 2: on dataset level
                fstrs1: format string level 1
                fstrs2: format string level 2
        
        Returns:
            OrderedDataGroup: The root level, corresponding to the file
        """
        # Use the other loader ...
        return self._load_hdf5(filepath, TargetCls=TargetCls,
                               load_as_proxy=True, **loader_kwargs)
