"""Implements loading of Hdf5 files into the dantro data tree"""

import logging

import numpy as np
import h5py as h5

from ..base import BaseDataGroup, BaseDataContainer
from ..containers import NumpyDataContainer
from ..groups import OrderedDataGroup
from ..proxy import Hdf5DataProxy
from ..tools import fill_line, decode_bytestrings
from ._tools import add_loader

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class Hdf5LoaderMixin:
    """Supplies functionality to load hdf5 files into the data manager.

    It resolves the hdf5 groups into corresponding data groups and the datasets
    into NumpyDataContainers.

    If ``enable_mapping`` is set, the class variables ``_HDF5_DSET_MAP`` and
    ``_HDF5_GROUP_MAP`` are used to map from a string to a container type. The
    class variable ``_HDF5_MAP_FROM_ATTR`` determines the default value of the
    attribute to read and use as input string for the mapping.

    Attributes:
        _HDF5_DSET_DEFAULT_CLS (type): the default class to use for datasets.
            This should be a dantro :py:class:`~dantro.base.BaseDataContainer`
            -derived class. Note that certain data groups can overwrite the
            default class for underlying members.
        _HDF5_GROUP_MAP (Dict[str, type]): if mapping is enabled, the
            equivalent dantro types for HDF5 groups are determined from this
            mapping.
        _HDF5_DSET_MAP (Dict[str, type]): if mapping is enabled, the
            equivalent dantro types for HDF5 datasets are determined from this
            mapping.
        _HDF5_MAP_FROM_ATTR (str): the name of the HDF5 dataset or group
            attribute to read in order to determine the type mapping. For
            example, this could be ``"content"``. This is the fallback value
            if no ``map_from_attr`` argument is given to
            :py:meth:`dantro.data_loaders.load_hdf5.Hdf5LoaderMixin._load_hdf5`
        _HDF5_DECODE_ATTR_BYTESTRINGS (bool): if true (default), will attempt
            to decode HDF5 attributes that are stored as byte arrays into
            regular Python strings; this can make attribute handling much
            easier.
    """
    #Default values for class variables; see above for docstrings
    _HDF5_DSET_DEFAULT_CLS = NumpyDataContainer
    _HDF5_GROUP_MAP = None
    _HDF5_DSET_MAP = None
    _HDF5_MAP_FROM_ATTR = None
    _HDF5_DECODE_ATTR_BYTESTRINGS = True


    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5(self, filepath: str, *,
                   TargetCls: type,
                   load_as_proxy: bool=False,
                   proxy_kwargs: dict=None,
                   lower_case_keys: bool=False,
                   enable_mapping: bool=False,
                   map_from_attr: str=None,
                   print_params: dict=None
                   ) -> OrderedDataGroup:
        """Loads the specified hdf5 file into DataGroup- and DataContainer-like
        objects; this completely recreates the hierarchic structure of the hdf5
        file. The data can be loaded into memory completely, or be loaded as
        a proxy object.

        The h5py File and Group objects will be converted to the specified
        DataGroup-derived objects; the Dataset objects to the specified
        DataContainer-derived object.

        All HDF5 group or dataset attributes are carried over and are
        accessible under the ``attrs`` attribute of the respective dantro
        objects in the tree.

        Args:
            filepath (str): The path to the HDF5 file that is to be loaded
            TargetCls (type): The group type this is loaded into
            load_as_proxy (bool, optional): if True, the leaf datasets are
                loaded as :py:class:`dantro.proxy.hdf5.Hdf5DataProxy`
                objects. That way, the data is only loaded into memory when
                their ``.data`` property is accessed the first time, either
                directly or indirectly.
            proxy_kwargs (dict, optional): When loading as proxy, these
                parameters are unpacked in the ``__init__`` call. For available
                argument see :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy`.
            lower_case_keys (bool, optional): whether to use only lower-case
                versions of the paths encountered in the HDF5 file.
            enable_mapping (bool, optional): If true, will use the class
                variables ``_HDF5_GROUP_MAP`` and ``_HDF5_DSET_MAP`` to map
                groups or datasets to a custom container class during loading.
                Which attribute to read is determined by the ``map_from_attr``
                argument (see there).
            map_from_attr (str, optional): From which attribute to read the
                key that is used in the mapping. If nothing is given, the
                class variable ``_HDF5_MAP_FROM_ATTR`` is used.
            print_params (dict, optional): parameters for the status report.
                Available keys:

                level (int):
                    how verbose to print loading info; possible values are:
                    ``0``: None, ``1``: on file level, ``2``: on dataset level
                fstr1:
                    format string level 1, receives keys ``name`` and ``file``,
                    which is the file path.
                fstr2:
                    format string level 2, receives keys ``name``, ``file`` and
                    ``obj``, which is an ``h5py.Dataset``.

        Returns:
            OrderedDataGroup: The populated root-level group, corresponding to
                the base group of the file

        Raises:
            ValueError: If ``enable_mapping``, but no map attribute can be
                determined from the given argument or the class variable
                ``_HDF5_MAP_FROM_ATTR``
        """

        def recursively_load_hdf5(src: h5.Group, target: BaseDataGroup, *,
                                  load_as_proxy: bool,
                                  proxy_kwargs: dict,
                                  lower_case_keys: bool,
                                  DsetCls: BaseDataContainer,
                                  enable_mapping: bool, GroupMap: dict,
                                  DsetMap: dict,map_attr: str):
            """Recursively loads the data from the source hdf5 file into the
            target DataGroup object.
            If given, each group or dataset is checked whether an attribute
            `container_type` or `dset_type` exists, which is then used to
            apply a mapping from that attribute to a certain type of DataGroup
            or DataContainer, respectively.

            Args:
                src (h5.Group): The source group to iterate over
                target (BaseDataGroup): The target group where the content from
                    the source group is loaded into
                load_as_proxy (bool): Whether to load as
                    :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy`
                proxy_kwargs (dict): Upon proxy initialization, unpacked into
                    :py:meth:`dantro.proxy.hdf5.Hdf5DataProxy.__init__`
                lower_case_keys (bool): Whether to make keys lower-case
                DsetCls (BaseDataContainer): The type that is used to create
                    the dataset-equivalents in ``target``
                enable_mapping (bool): Whether type mapping should be used
                GroupMap (dict): Map of names to BaseDataGroup-derived types
                DsetMap (dict): Map of names to BaseDataContainer-derived types
                map_attr (str): The HDF5 attribute to inspect in order to
                    determine the name of the mapping

            Raises:
                NotImplementedError: When encountering objects other than
                    groups or datasets in the HDF5 file
            """

            def get_map_attr_val(attrs) -> str:
                """Make sure the map attribute isn't a 1-sized array!"""
                attr_val = attrs[map_attr] # map_attr and check in outer scope

                if isinstance(attr_val, np.ndarray):
                    # Need be single item and already decoded
                    attr_val = attr_val.item()

                return attr_val

            def decode_attr_val(attr_val) -> str:
                """Wrapper around decode_bytestrings"""
                # If feature not activated, return without doing anything
                if not self._HDF5_DECODE_ATTR_BYTESTRINGS:
                    return attr_val

                return decode_bytestrings(attr_val)

            # Go through the elements of the source object
            for key, obj in src.items():
                if lower_case_keys and isinstance(key, str):
                    key = key.lower()

                if isinstance(obj, h5.Group):
                    # Need to continue recursion
                    # Extract attributes manually
                    attrs = {k: decode_attr_val(v)
                             for k, v in obj.attrs.items()}

                    # Determine the class to use for this group
                    if enable_mapping and GroupMap and attrs.get(map_attr):
                        # Try to resolve the mapping
                        try:
                            _GroupCls = GroupMap[get_map_attr_val(attrs)]

                        except KeyError:
                            # Fall back to default
                            log.warning("Could not find a mapping from map "
                                        "attribute %s='%s' (originally %s) "
                                        "to a DataGroup class. Available "
                                        "keys: %s. Falling back to default "
                                        "class ...",
                                        map_attr, get_map_attr_val(attrs),
                                        attrs[map_attr],
                                        ", ".join([k
                                                   for k in GroupMap.keys()]))
                            _GroupCls = None

                    else:
                        # Use the default of the target group
                        _GroupCls = None
                        # None value will lead to new_group determining the
                        # type of this group. Unlike for datasets, this is
                        # always possible, because the type of the target
                        # group is used as fallback type.

                    # Create and add the group, passing the attributes
                    target.new_group(path=key, Cls=_GroupCls,
                                     attrs=attrs)

                    # Continue recursion
                    recursively_load_hdf5(obj, target[key],
                                          load_as_proxy=load_as_proxy,
                                          proxy_kwargs=proxy_kwargs,
                                          lower_case_keys=lower_case_keys,
                                          DsetCls=DsetCls,
                                          enable_mapping=enable_mapping,
                                          GroupMap=GroupMap, DsetMap=DsetMap,
                                          map_attr=map_attr)

                elif isinstance(obj, h5.Dataset):
                    # Reached a leaf -> Import the data and attributes into a
                    # BaseDataContainer-derived object. This assumes that the
                    # given DsetCls supports np.ndarray-like data

                    # Progress information on loading this dataset
                    if plvl >= 2:
                        line = fstr2.format(name=target.name, key=key, obj=obj)
                        print(fill_line(line), end="\r")

                    if load_as_proxy:
                        # Instantiate a proxy object
                        data = Hdf5DataProxy(obj,
                                             **(proxy_kwargs if proxy_kwargs
                                                else {}))
                    else:
                        # Import the data completely
                        data = np.array(obj)

                    # Extract attributes manually
                    attrs = {k: decode_attr_val(v)
                             for k, v in obj.attrs.items()}

                    # Determine the class to use for this dataset
                    if enable_mapping and DsetMap and attrs.get(map_attr):
                        # Try to resolve the mapping
                        try:
                            _DsetCls = DsetMap[get_map_attr_val(attrs)]

                        except KeyError:
                            # Fall back to default
                            log.warning("Could not find a mapping from map "
                                        "attribute %s='%s' (originally %s) to "
                                        "a DataContainer class. Available "
                                        "keys: %s. "
                                        "Falling back to default class %s...",
                                        map_attr, get_map_attr_val(attrs),
                                        attrs[map_attr],
                                        ", ".join([k for k in DsetMap.keys()]),
                                        DsetCls.__name__)
                            _DsetCls = DsetCls

                    else:
                        # If the target group supplies a default container
                        # type, use that one; otherwise fall back to default.
                        if target._NEW_CONTAINER_CLS is not None:
                            _DsetCls = None
                            # Leads to new_container taking care of the type

                        else:
                            _DsetCls = DsetCls

                    # Now create and add the dataset, passing data and attrs
                    target.new_container(path=key, Cls=_DsetCls,
                                         data=data, attrs=attrs)

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
                print(fill_line(line), end="\r")

            # Load the file level attributes, manually re-creating the dict
            root.attrs = {k:v for k, v in h5file.attrs.items()}

            # Now recursively load the data into the root group
            recursively_load_hdf5(h5file, root,
                                  load_as_proxy=load_as_proxy,
                                  proxy_kwargs=proxy_kwargs,
                                  lower_case_keys=lower_case_keys,
                                  DsetCls=DsetCls,
                                  enable_mapping=enable_mapping,
                                  GroupMap=self._HDF5_GROUP_MAP,
                                  DsetMap=self._HDF5_DSET_MAP,
                                  map_attr=map_from_attr)

        return root

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5_proxy(self, *args, **kwargs) -> OrderedDataGroup:
        """This is a shorthand for
        :py:meth:`~dantro.data_loaders.load_hdf5.Hdf5LoaderMixin._load_hdf5`
        with the ``load_as_proxy`` flag set.
        """
        return self._load_hdf5(*args, load_as_proxy=True, **kwargs)

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5_as_dask(self, *args, **kwargs) -> OrderedDataGroup:
        """This is a shorthand for
        :py:meth:`~dantro.data_loaders.load_hdf5.Hdf5LoaderMixin._load_hdf5`
        with the ``load_as_proxy`` flag set and ``resolve_as_dask`` passed as
        additional arguments to the proxy via ``proxy_kwargs``.
        """
        return self._load_hdf5(*args,
                               load_as_proxy=True,
                               proxy_kwargs=dict(resolve_as_dask=True),
                               **kwargs)
