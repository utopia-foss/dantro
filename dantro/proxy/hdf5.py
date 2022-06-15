"""This module implements a :py:class:`dantro.base.BaseDataProxy`
specialization for HDF5 data.
"""

import logging

import numpy as np

from .._import_tools import LazyLoader
from ..base import BaseDataProxy

log = logging.getLogger(__name__)

da = LazyLoader("dask.array")
h5 = LazyLoader("h5py")

# -----------------------------------------------------------------------------


class Hdf5DataProxy(BaseDataProxy):
    """The Hdf5DataProxy is a placeholder for a :py:class:`h5py.Dataset`.

    It saves the filename and dataset name needed to later load the dataset.
    Additionaly, it caches some values that give information on the shape and
    dtype of the dataset, thus further delaying the load to the time the
    *actual* data is required.

    Depending on the type that this proxy is resolved as via the
    :py:meth:`.resolve` method, the corresponding :py:class:`h5py.File` object
    needs to stay open and in memory; it is closed upon garbage-collection of
    this object.
    """

    def __init__(self, obj: "h5py.Dataset", *, resolve_as_dask: bool = False):
        """Initializes a proxy object for a :py:class:`h5py.Dataset` object.

        Args:
            obj (h5py.Dataset): The dataset object to be proxy for
            resolve_as_dask (bool, optional): Whether to resolve the dataset
                object as a delayed :py:class:`dask.array.Array` object, using
                an :py:class:`h5py.Dataset` to initialize it and passing over
                chunk information.
        """
        super().__init__(obj)

        # Information to later resolve the data
        self._fname = obj.file.filename
        self._name = obj.name  # is the full path within the file

        # If file objects need be kept in scope, this is the list to store them
        self._h5files = []

        # Extract some further information from the dataset before, basically
        # all information that can be known without loading the data
        self._shape = obj.shape
        self._dtype = obj.dtype
        self._ndim = obj.ndim
        self._size = obj.size
        self._chunks = obj.chunks

        # Whether to load the hdf5 data through dask.array.from_array
        self._resolve_as_dask = resolve_as_dask

        # Set the tags
        self._tags += ("hdf5",)

        if self._resolve_as_dask:
            self._tags += ("dask",)

    def resolve(self, *, astype: type = None):
        """Resolve the data of this proxy by opening the hdf5 file and loading
        the dataset into a :py:class:`numpy.ndarray` or a type specified by the
        ``astype`` argument.

        Args:
            astype (type, optional): As which type to return the data from the
                dataset this object is proxy for.
                If None, will return as :py:class:`numpy.ndarray`.
                For :py:class:`h5py.Dataset`, the :py:class:`h5py.File` object
                stays in memory until the proxy is deleted.
                Note that if ``resolve_as_dask`` was specified during proxy
                initialization, the data will be loaded as
                :py:class:`dask.array.Array` only if ``astype`` is **not**
                specified in this call!

        Returns:
            type specified by ``astype``: the resolved data.
        """
        if astype is h5.Dataset and not self._resolve_as_dask:
            log.debug(
                "Resolving %s as h5py.Dataset from dataset %s in file "
                "at %s ...",
                self.classname,
                self._name,
                self._fname,
            )

            # Open the file and keep it in scope
            h5file = self._open_h5file()

            # Return the dataset object, which remains valid until the file
            # object is closed, i.e. the proxy goes out of scope
            return h5file[self._name]

        elif astype is None and self._resolve_as_dask:
            log.debug(
                "Resolving %s as h5py.Dataset from dataset %s in file "
                "at %s into a delayed dask array object...",
                self.classname,
                self._name,
                self._fname,
            )

            # Open the file and keep it in scope
            h5file = self._open_h5file()

            # Build the delayed dask array from the h5.Dataset and chunk info
            return da.from_array(h5file[self._name], chunks=self._chunks)

        else:
            # By default, return as numpy array
            astype = astype if astype is not None else np.array

            log.debug(
                "Resolving %s as %s.%s from dataset %s in file at %s ...",
                self.classname,
                astype.__module__,
                astype.__name__,
                self._name,
                self._fname,
            )

            with h5.File(self._fname, "r") as h5file:
                return astype(h5file[self._name])

    # Handling of HDF5 files ..................................................

    def _open_h5file(self) -> "h5py.File":
        """Opens the associated HDF5 file and stores it in ``_h5files`` in
        order to keep it in scope. These file objects are only closed upon
        deletion of this proxy object!

        Returns:
            h5py.File: The newly opened HDF5 file
        """
        h5file = h5.File(self._fname, "r")
        self._h5files.append(h5file)
        return h5file

    def __del__(self):
        """Make sure all potentially still open h5py.File objects are closed"""
        for f in self._h5files:
            try:
                f.close()

            except Exception:
                # Can no longer close it; garbace collection probably already
                # took care of it ... which is fine.
                pass

    # Properties to access information without resolving ......................

    @property
    def shape(self):
        """The cached shape of the dataset, accessible without resolving"""
        return self._shape

    @property
    def dtype(self):
        """The cached dtype of the dataset, accessible without resolving"""
        return self._dtype

    @property
    def ndim(self):
        """The cached ndim of the dataset, accessible without resolving"""
        return self._ndim

    @property
    def size(self):
        """The cached size of the dataset, accessible without resolving"""
        return self._size

    @property
    def chunks(self):
        """The cached chunks of the dataset, accessible without resolving"""
        return self._chunks
