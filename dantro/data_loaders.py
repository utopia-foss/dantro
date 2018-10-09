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

import pickle as pkl

import numpy as np
import h5py as h5

from dantro.base import BaseDataGroup, BaseDataContainer
from dantro.container import ObjectContainer, MutableMappingContainer, NumpyDataContainer
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
        d = tools.load_yml(filepath)

        # Populate the target container with the data
        return TargetCls(data=d, attrs=dict(filepath=filepath))

    # Also make available under `yml`
    _load_yml = _load_yaml

# -----------------------------------------------------------------------------

class PickleLoaderMixin:
    """Supplies functionality to load pickled python objects"""

    # Define the load function such that it can be set in a flexible way
    _PICKLE_LOAD_FUNC = pkl.load

    @add_loader(TargetCls=ObjectContainer, omit_self=False)
    def _load_pickle(self, filepath: str, *, TargetCls: type, **pkl_kwargs) -> ObjectContainer:
        """Load a pickled object.

        This uses the load function defined under the _PICKLE_LOAD_FUNC class
        variable, which defaults to the pickle.load function.
        
        Args:
            filepath (str): Where the pickle-dumped file is located
            TargetCls (type): The class constructor
            **pkl_kwargs: Passed on to the load function
        
        Returns:
            ObjectContainer: The unpickled file, stored in a dantro container
        """
        # Open file in binary mode and unpickle with the given load function
        with open(filepath, mode='rb') as f:
            obj = self._PICKLE_LOAD_FUNC(f, **pkl_kwargs)

        # Populate the target container with the object
        return TargetCls(data=obj, attrs=dict(filepath=filepath))

    # Also make available under `pkl`
    _load_pkl = _load_pickle

# -----------------------------------------------------------------------------

class Hdf5LoaderMixin:
    """Supplies functionality to load hdf5 files into the data manager.
    
    It resolves the hdf5 groups into corresponding data groups and the datasets
    into NumpyDataContainers.

    If `enable_mapping` is set, the _HDF5_DSET_MAP and _HDF5_GROUP_MAP class
    variables are used to map from a string to a container type. The class
    variable _HDF5_MAP_FROM_ATTR determines the default value of the attribute
    to read and use as input string for the mapping. 
    """

    # Define the container classes to use when loading data with this mixin
    # The default class to use for containers
    _HDF5_DSET_DEFAULT_CLS = NumpyDataContainer

    # The mapping of types to data groups
    _HDF5_GROUP_MAP = None

    # The mapping of types to data containers
    _HDF5_DSET_MAP = None

    # The name of the attribute to read for mapping
    _HDF5_MAP_FROM_ATTR = None

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5(self, filepath: str, *, TargetCls: type, load_as_proxy: bool=False, lower_case_keys: bool=False, enable_mapping: bool=False, map_from_attr: str=None, print_params: dict=None) -> OrderedDataGroup:
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
            enable_mapping (bool, optional): If true, will use the class
                variables _HDF5_GROUP_MAP and _HDF5_DSET_MAP to map groups or
                datasets to a custom container class during loading. Which
                attribute to read is determined by the `map_from_attr` argument
            map_from_attr (str, optional): From which attribute to read the
                key that is used in the mapping. If nothing is given, the
                class variable _HDF5_MAP_FROM_ATTR is used.
            print_params (dict, optional): parameters for the status report
                level: how verbose to print loading info; possible values are:
                    0: None, 1: on file level, 2: on dataset level
                fstrs1: format string level 1
                fstrs2: format string level 2
        
        Returns:
            OrderedDataGroup: The root level, corresponding to the file
        
        Raises:
            ValueError: If `enable_mapping`, but no map attribute can be
                determined from the given argument or the class variable
                _HDF5_MAP_FROM_ATTR
        """
        

        def recursively_load_hdf5(src, target: BaseDataGroup, *, load_as_proxy: bool, lower_case_keys: bool, enable_mapping: bool, GroupCls: BaseDataGroup, DsetCls: BaseDataContainer, GroupMap: dict, DsetMap: dict, map_attr: str):
            """Recursively loads the data from the source hdf5 file into the 
            target DataGroup object.
            If given, each group or dataset is checked whether an attribute
            `container_type` or `dset_type` exists, which is then used to apply a
            mapping from that attribute to a certain type of DataGroup or
            DataContainer, respectively.
            """

            # Go through the elements of the source object
            for key, obj in src.items():
                if lower_case_keys and isinstance(key, str):
                    key = key.lower()

                if isinstance(obj, h5.Group):
                    # Need to continue recursion
                    # Extract attributes manually
                    attrs = {k:v for k, v in obj.attrs.items()}

                    # Determine the class to use for this group
                    if enable_mapping and GroupMap and attrs.get(map_attr):
                        # Try to resolve the mapping
                        try:
                            _GroupCls = GroupMap[attrs[map_attr]]

                        except KeyError:
                            # Fall back to default
                            log.warning("Could not find a mapping from map "
                                        "attribute %s='%s' to a DataGroup "
                                        "class. Available keys: %s. Falling "
                                        "back to default class %s...",
                                        map_attr, attrs[map_attr],
                                        ", ".join([k
                                                   for k in GroupMap.keys()]),
                                        GroupCls.__name__)
                            _GroupCls = GroupCls
                    
                    else:
                        # Use the default
                        _GroupCls = GroupCls

                    # Create and add the group, passing the attributes
                    target.add(_GroupCls(name=key, attrs=attrs))

                    # Continue recursion
                    recursively_load_hdf5(obj, target[key],
                                          load_as_proxy=load_as_proxy,
                                          lower_case_keys=lower_case_keys,
                                          enable_mapping=enable_mapping,
                                          GroupCls=GroupCls, DsetCls=DsetCls,
                                          GroupMap=GroupMap, DsetMap=DsetMap,
                                          map_attr=map_attr)

                elif isinstance(obj, h5.Dataset):
                    # Reached a leaf -> Import the data and attributes into a
                    # BaseDataContainer-derived object. This assumes that the
                    # given DsetCls supports np.ndarray-like data

                    # Progress information on loading this dataset
                    if plvl >= 2:
                        line = fstr2.format(name=target.name, key=key, obj=obj)
                        print(tools.fill_line(line), end="\r")

                    if load_as_proxy:
                        # Instantiate a proxy object
                        data = Hdf5DataProxy(obj)
                    else:
                        # Import the data completely
                        data = np.array(obj)

                    # Extract attributes manually
                    attrs = {k:v for k, v in obj.attrs.items()}

                    # Determine the class to use for this dataset
                    if enable_mapping and DsetMap and attrs.get(map_attr):
                        # Try to resolve the mapping
                        try:
                            _DsetCls = DsetMap[attrs[map_attr]]

                        except KeyError:
                            # Fall back to default
                            log.warning("Could not find a mapping from map "
                                        "attribute %s='%s' to a DataContainer "
                                        "class. Available keys: %s. Falling "
                                        "back to default class %s...",
                                        map_attr, attrs[map_attr],
                                        ", ".join([k for k in DsetMap.keys()]),
                                        DsetCls.__name__)
                            _DsetCls = DsetCls
                    
                    else:
                        # Use the default
                        _DsetCls = DsetCls

                    # Now create and add the dataset, passing data and attrs
                    target.add(_DsetCls(name=key, data=data, attrs=attrs))

                else:
                    raise NotImplementedError("Object {} is neither a dataset "
                                              "nor a group, but of type {}. "
                                              "Cannot load this!"
                                              "".format(key, type(obj)))

        # Prepare print format strings
        print_params = print_params if print_params else {}
        plvl = print_params.get('level', 0)
        fstr1 = print_params.get('fstr1', "  Loading {name:} ... ")
        fstr2 = print_params.get('fstr2', "  Loading {name:} - {key:} ...")

        # Initialize the root group
        log.debug("Loading hdf5 file %s into %s ...",
                  filepath, TargetCls.__name__)
        root = TargetCls()

        # Get the classes to use for groups and/or containers
        DsetCls = self._HDF5_DSET_DEFAULT_CLS
        GroupCls = type(root)

        # Determine from which attribute to read the mapping
        if not map_from_attr:
            # No custom value was given; use the class variable, if available
            if self._HDF5_MAP_FROM_ATTR:
                map_from_attr = self._HDF5_MAP_FROM_ATTR

            elif enable_mapping:
                # Mapping was enabled but it is unclear from which attribute
                # the map should be read. Need to raise an exception
                raise ValueError("Could not determine from which attribute "
                                 "to read the mapping. Either set the loader "
                                 "argument `map_from_attr`, the class "
                                 "variable _HDF5_MAP_FROM_ATTR, or disable "
                                 "mapping altogether via the `enable_mapping` "
                                 "argument.")

        # Now recursively go through the hdf5 file and add them to the roo
        with h5.File(filepath, 'r') as h5file:
            if plvl >= 1:
                # Print information on the level of this file
                line = fstr1.format(name=root.name, file=filepath)
                print(tools.fill_line(line), end="\r")

            # Load the file level attributes, manually re-creating the dict
            root.attrs = {k:v for k, v in h5file.attrs.items()}

            # Now recursively load the data into the root group
            recursively_load_hdf5(h5file, root,
                                  load_as_proxy=load_as_proxy,
                                  lower_case_keys=lower_case_keys,
                                  enable_mapping=enable_mapping,
                                  GroupCls=GroupCls, DsetCls=DsetCls,
                                  GroupMap=self._HDF5_GROUP_MAP,
                                  DsetMap=self._HDF5_DSET_MAP,
                                  map_attr=map_from_attr)

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
