"""This module implements a BaseDataProxy specialization for Hdf5 data."""

import logging
from typing import Union

import numpy as np
import h5py as h5

from ..base import BaseDataProxy

# Local variables
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class Hdf5DataProxy(BaseDataProxy):
    """The Hdf5DataProxy is a placeholder for Hdf5 datasets.

    It saves the filename and dataset name needed to later load the dataset.
    Additionaly, it caches some values that give information on the shape and
    dtype of the dataset.
    """

    def __init__(self, obj: h5.Dataset):
        """Initializes a proxy object for Hdf5 datasets.
        
        Args:
            obj (h5.Dataset): The dataset object to be proxy for
        """
        super().__init__(obj)

        # Information to later resolve the data
        self._fname = obj.file.filename
        self._name = obj.name  # is the full path

        # If file objects need be kept in scope, this is the list to store them
        self._h5files = []

        # Extract some further information from the dataset before, basically
        # all information that can be known without loading the data
        self._shape = obj.shape
        self._dtype = obj.dtype
        self._ndim = obj.ndim
        self._size = obj.size
        self._chunks = obj.chunks

    def resolve(self, *, astype: type=None) -> Union[np.array, h5.Dataset]:
        """Resolve the data of this proxy by opening the hdf5 file and loading
        the dataset into a numpy array or a type specified by `astype`
        
        Args:
            astype (type, optional): As which type to return the data from the
                dataset this object is proxy for. If None, will return as
                np.array. For `h5py.Dataset`, the h5py.File object stays in
                memory until the proxy is deleted.
        
        Returns:
            Union[np.array, h5.Dataset] ... or 
        """
        # By default, return as numpy array
        astype = astype if astype is not None else np.array

        # Distinguish between the desired return type
        if astype is h5.Dataset:
            log.debug("Resolving %s as h5py.Dataset from dataset %s in file "
                      "at %s ...",
                      self.classname, self._name, self._fname)

            # Open the file and keep it in scope
            h5file = h5.File(self._fname, 'r')
            self._h5files.append(h5file)

            # Return the dataset object, which remains valid until the file
            # object is closed, i.e. the proxy goes out of scope
            return h5file[self._name]

        else:
            log.debug("Resolving %s as %s.%s from dataset %s in "
                      "file at %s ...", self.classname,
                      astype.__module__, astype.__name__,
                      self._name, self._fname)

            with h5.File(self._fname, 'r') as h5file:
                return astype(h5file[self._name])


    # Proper garbage collection ...............................................

    def __del__(self):
        """Make sure all potentially still open h5py.File objects are closed"""
        for f in self._h5files:
            f.close()


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
