"""Implements loading of Hdf5 files into the dantro data tree"""

import logging
import os
from typing import Dict, Union

import numpy as np

from .._import_tools import LazyLoader
from ..base import BaseDataContainer, BaseDataGroup
from ..containers import NumpyDataContainer
from ..groups import OrderedDataGroup
from ..proxy import Hdf5DataProxy
from ..tools import decode_bytestrings, print_line
from ._tools import add_loader

log = logging.getLogger(__name__)

h5 = LazyLoader("h5py")

# -----------------------------------------------------------------------------


class Hdf5LoaderMixin:
    """Supplies functionality to load HDF5 files into the
    :py:class:`~dantro.data_mngr.DataManager`.

    It resolves the HDF5 groups into corresponding data groups and the datasets
    (by default) into
    :py:class:`~dantro.containers.numeric.NumpyDataContainer` s.

    If ``enable_mapping`` is set, the class variables ``_HDF5_DSET_MAP`` and
    ``_HDF5_GROUP_MAP`` are used to map from a string to a container type. The
    class variable ``_HDF5_MAP_FROM_ATTR`` determines the default value of the
    attribute to read and use as input string for the mapping.
    """

    _HDF5_DSET_DEFAULT_CLS: type = NumpyDataContainer
    """the default class to use for datasets. This should be a dantro
    :py:class:`~dantro.base.BaseDataContainer`-derived class.
    Note that certain data groups can overwrite the default class for
    underlying members."""

    _HDF5_GROUP_MAP: Dict[str, type] = None
    """If mapping is enabled, the equivalent dantro types for HDF5 groups are
    determined from this mapping."""

    _HDF5_DSET_MAP: Dict[str, type] = None
    """If mapping is enabled, the equivalent dantro types for HDF5 datasets
    are determined from this mapping."""

    _HDF5_MAP_FROM_ATTR: str = None
    """The name of the HDF5 dataset or group attribute to read in order to
    determine the type mapping. For example, this could be ``"content"``.
    This is the fallback value if no ``map_from_attr`` argument is given to
    :py:meth:`dantro.data_loaders.hdf5.Hdf5LoaderMixin._load_hdf5`"""

    _HDF5_DECODE_ATTR_BYTESTRINGS: bool = True
    """If true (default), will attempt to decode HDF5 attributes that are
    stored as byte arrays into regular Python strings; this can make attribute
    handling much easier."""

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5(
        self,
        filepath: str,
        *,
        TargetCls: type,
        load_as_proxy: bool = False,
        proxy_kwargs: dict = None,
        lower_case_keys: bool = False,
        enable_mapping: bool = False,
        map_from_attr: str = None,
        direct_insertion: bool = False,
        progress_params: dict = None,
    ) -> OrderedDataGroup:
        """Loads the specified hdf5 file into DataGroup- and DataContainer-like
        objects; this completely recreates the hierarchic structure of the hdf5
        file. The data can be loaded into memory completely, or be loaded as
        a proxy object.

        The :py:class:`h5py.File` and :py:class:`h5py.Group` objects will be
        converted to the specified
        :py:class:`~dantro.base.BaseDataGroup`-derived objects and the
        :py:class:`h5py.Dataset` objects to the specified
        :py:class:`~dantro.base.BaseDataContainer`-derived object.

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
            direct_insertion (bool, optional): If True, some non-crucial checks
                are skipped during insertion and elements are inserted (more
                or less) directly into the data tree, thus speeding up the data
                loading process.
                This option should only be enabled if data is loaded into a yet
                unpopulated part of the data tree, otherwise existing elements
                might be overwritten silently.
                This option only applies to data groups, not to containers.
            progress_params (dict, optional): parameters for the progress
                indicator. Possible keys:

                level (int):
                    how verbose to print progress info; possible values are:
                    ``0``: None, ``1``: on file level, ``2``: on dataset level.
                    Note that this option and the ``progress_indicator`` of
                    the DataManager are independent from each other.
                fstr:
                    format string for progress report, receives the following
                    keys:

                        * ``progress_info`` (total progress indicator),
                        * ``fname`` (basename of current hdf5 file),
                        * ``fpath`` (full path of current hdf5 file),
                        * ``name`` (current dataset name),
                        * ``path`` (current path within the hdf5 file)

        Returns:
            OrderedDataGroup: The populated root-level group, corresponding to
                the base group of the file

        Raises:
            ValueError: If ``enable_mapping``, but no map attribute can be
                determined from the given argument or the class variable
                ``_HDF5_MAP_FROM_ATTR``
        """
        # Initialize the root group
        log.debug(
            "Loading hdf5 file %s into %s ...", filepath, TargetCls.__name__
        )
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
                raise ValueError(
                    "Could not determine from which attribute to read the "
                    "mapping. Either set the loader argument `map_from_attr`, "
                    "the class variable _HDF5_MAP_FROM_ATTR, or disable "
                    "mapping altogether via the `enable_mapping` argument."
                )

        # Prepare parameters
        GroupMap = self._HDF5_GROUP_MAP if enable_mapping else {}
        DsetMap = self._HDF5_DSET_MAP if enable_mapping else {}

        # Prepare progress information
        progress_params = progress_params if progress_params else {}
        plvl = progress_params.get("level", 0)
        pfstr = progress_params.get(
            "fstr", "  {progress_info:} {fname:} : {path:} ... "
        )

        # Now recursively go through the hdf5 file and add them to the roo
        with h5.File(filepath, "r") as h5file:
            if plvl >= 1:
                # Print information on the level of this file
                _info = pfstr.format(
                    progress_info=self._progress_info_str,
                    fpath=filepath,
                    fname=os.path.basename(filepath),
                    key="",
                    path="",
                )
                print_line(_info)

            # Load the file level attributes, manually re-creating the dict
            root.attrs = {
                k: self._decode_attr_val(v) for k, v in h5file.attrs.items()
            }

            # Now recursively load the data into the root group
            self._recursively_load_hdf5(
                h5file,
                target=root,
                load_as_proxy=load_as_proxy,
                proxy_kwargs=proxy_kwargs,
                lower_case_keys=lower_case_keys,
                DsetCls=DsetCls,
                GroupMap=GroupMap,
                DsetMap=DsetMap,
                map_attr=map_from_attr,
                direct_insertion=direct_insertion,
                plvl=plvl,
                pfstr=pfstr,
            )

        return root

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5_proxy(self, *args, **kwargs) -> OrderedDataGroup:
        """This is a shorthand for
        :py:meth:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin._load_hdf5`
        with the ``load_as_proxy`` flag set.
        """
        return self._load_hdf5(*args, load_as_proxy=True, **kwargs)

    @add_loader(TargetCls=OrderedDataGroup, omit_self=False)
    def _load_hdf5_as_dask(self, *args, **kwargs) -> OrderedDataGroup:
        """This is a shorthand for
        :py:meth:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin._load_hdf5`
        with the ``load_as_proxy`` flag set and ``resolve_as_dask`` passed as
        additional arguments to the proxy via ``proxy_kwargs``.
        """
        return self._load_hdf5(
            *args,
            load_as_proxy=True,
            proxy_kwargs=dict(resolve_as_dask=True),
            **kwargs,
        )

    # .........................................................................

    def _recursively_load_hdf5(
        self,
        src: Union["h5py.Group", "h5py.File"],
        *,
        target: BaseDataGroup,
        lower_case_keys: bool,
        direct_insertion: bool,
        **kwargs,
    ):
        """Recursively loads the data from a source object (an h5py.File or a
        h5py.Group) into the target dantro group.

        Args:
            src (Union[h5py.Group, h5py.File]): The HDF5 source object from
                which to load the data. This object it iterated over.
            target (BaseDataGroup): The target group to populate with the data
                from ``src``.
            lower_case_keys (bool): Whether to make keys lower-case
            direct_insertion (bool): Whether to use direct insertion mode on
                the target group (and all groups below)
            **kwargs: Passed on to the group and container loader methods,
                :py:meth:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin._container_from_h5dataset`
                and
                :py:meth:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin._group_from_h5group`.

        Raises:
            NotImplementedError: When encountering objects other than groups
                or datasets in the HDF5 file
        """
        # Go through the elements of the source object
        for key, obj in src.items():
            if lower_case_keys and isinstance(key, str):
                key = key.lower()

            with target._direct_insertion_mode(enabled=direct_insertion):
                if isinstance(obj, h5.Group):
                    # Create the new group
                    grp = self._group_from_h5group(
                        obj, target=target, name=key, **kwargs
                    )

                    # Continue recursion
                    self._recursively_load_hdf5(
                        obj,
                        target=grp,
                        lower_case_keys=lower_case_keys,
                        direct_insertion=direct_insertion,
                        **kwargs,
                    )

                elif isinstance(obj, h5.Dataset):
                    # Reached a leaf -> Import the data and attributes into a
                    # BaseDataContainer-derived object. This assumes that the
                    # given DsetCls supports np.ndarray-like data
                    self._container_from_h5dataset(
                        obj, target=target, name=key, **kwargs
                    )

                else:
                    raise NotImplementedError(
                        f"Object {key} is neither a dataset nor a group, but "
                        f"of type {type(obj)}. Cannot load this!"
                    )

    def _group_from_h5group(
        self,
        h5grp: "h5py.Group",
        target: BaseDataGroup,
        *,
        name: str,
        map_attr: str,
        GroupMap: dict,
        **_,
    ) -> BaseDataGroup:
        """Adds a new group from a h5.Group

        The group types may be mapped to different dantro types; this is
        controlled by the extracted HDF5 attribute with the name specified in
        the ``_HDF5_MAP_FROM_ATTR`` class attribute.

        Args:
            h5grp (h5py.Group): The HDF5 group to create a dantro group for in
                the ``target`` group.
            target (BaseDataGroup): The group in which to create a new group
                that represents ``h5grp``
            name (str): the name of the new group
            GroupMap (dict): Map of names to BaseDataGroup-derived types;
                always needed, but may be empty
            map_attr (str): The HDF5 attribute to inspect in order to determine
                the name of the mapping
            **_: ignored
        """
        # Extract attributes manually
        attrs = {k: self._decode_attr_val(v) for k, v in h5grp.attrs.items()}

        # Determine the mapping type, falling back to the group default if no
        # mapping was specified
        _GroupCls = self._evaluate_type_mapping(
            map_attr, attrs=attrs, tmap=GroupMap, fallback=None
        )

        # Create and add the group, passing the attributes
        return target.new_group(path=name, Cls=_GroupCls, attrs=attrs)

    def _container_from_h5dataset(
        self,
        h5dset: "h5py.Dataset",
        target: BaseDataGroup,
        *,
        name: str,
        load_as_proxy: bool,
        proxy_kwargs: dict,
        DsetCls: type,
        map_attr: str,
        DsetMap: dict,
        plvl: int,
        pfstr: str,
        **_,
    ) -> BaseDataContainer:
        """Adds a new data container from a h5.Dataset

        The group types may be mapped to different dantro types; this is
        controlled by the extracted HDF5 attribute with the name specified in
        the ``_HDF5_MAP_FROM_ATTR`` class attribute.

        Args:
            h5dset (h5py.Dataset): The source dataset to load into ``target``
                as a dantro data container.
            target (BaseDataGroup): The target group where the ``h5dset`` will
                be represented in as a new dantro data container.
            name (str): the name of the new container
            load_as_proxy (bool): Whether to load as
                :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy`
            proxy_kwargs (dict): Upon proxy initialization, unpacked into
                :py:meth:`dantro.proxy.hdf5.Hdf5DataProxy.__init__`
            DsetCls (BaseDataContainer): The type that is used to create
                the dataset-equivalents in ``target``. If mapping is enabled,
                this serves as the fallback type.
            map_attr (str): The HDF5 attribute to inspect in order to determine
                the name of the mapping
            DsetMap (dict): Map of names to BaseDataContainer-derived types;
                always needed, but may be empty
            plvl (int): the verbosity of the progress indicator
            pfstr (str): a format string for the progress indicator
        """
        # Progress information on loading this dataset
        if plvl >= 2:
            _info = pfstr.format(
                progress_info=self._progress_info_str,
                fpath=h5dset.file.filename,
                fname=os.path.basename(h5dset.file.filename),
                name=name,
                path=h5dset.name,
            )
            print_line(_info)

        # Extract attributes manually
        attrs = {k: self._decode_attr_val(v) for k, v in h5dset.attrs.items()}

        # Determine the class to use for this dataset.
        # If the target group supplies a default container type, specify None
        # as the fallback, delegating the choice to the `new_container` method.
        # Otherwise use the default specified here.
        _DsetCls = self._evaluate_type_mapping(
            map_attr,
            attrs=attrs,
            tmap=DsetMap,
            fallback=DsetCls if target._NEW_CONTAINER_CLS is None else None,
        )

        # Get the data, potentially as proxy.
        if load_as_proxy:
            data = Hdf5DataProxy(
                h5dset, **(proxy_kwargs if proxy_kwargs else {})
            )
        else:
            data = np.array(h5dset)

        # Now create and add the dataset
        return target.new_container(
            path=name, Cls=_DsetCls, data=data, attrs=attrs
        )

    # .........................................................................
    # Smaller helper methods

    def _decode_attr_val(self, attr_val) -> str:
        """Wrapper around decode_bytestrings"""
        # If feature not activated, return without doing anything
        if not self._HDF5_DECODE_ATTR_BYTESTRINGS:
            return attr_val

        return decode_bytestrings(attr_val)

    def _evaluate_type_mapping(
        self, key: str, *, attrs: dict, tmap: Dict[str, type], fallback: type
    ) -> type:
        """Given an attributes dict or group attributes, evaluates which type
        a target container should use.
        """

        def parse_map_attr(v) -> str:
            """Make sure the map attribute isn't a 1-sized array!"""
            if isinstance(v, np.ndarray):
                v = v.item()
            return str(v)

        # Easy cases first: no map or mapping attribute given
        if not tmap or not attrs.get(key):
            return fallback

        try:
            return tmap[parse_map_attr(attrs[key])]

        except KeyError:
            # Fall back to default
            log.warning(
                "Could not find a mapping from map attribute %s='%s' "
                "(originally %s) to a dantro container or group class. "
                "Available keys: %s. Using fallback type instead: %s .",
                key,
                parse_map_attr(attrs[key]),
                attrs[key],
                ", ".join([k for k in tmap]),
                fallback.__name__ if fallback is not None else "(none)",
            )
            return fallback
