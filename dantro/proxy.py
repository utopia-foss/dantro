"""This module implements specialisations of BaseDataProxy."""

import logging

import numpy as np
import h5py as h5

from dantro.base import BaseDataProxy

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
        """Initialises a proxy object for Hdf5 datasets.
        
        Args:
            obj (h5.Dataset): The dataset object to be proxy for
        """

        # Information to later resolve the data
        self.fname = obj.file.filename
        self.name = obj.name

        # Extract some further information of the dataset
        self.shape = obj.shape
        self.dtype = obj.dtype

    def __str__(self) -> str:
        """An info string that can be used to represent this object."""
        return "{shape:}, {dtype:}".format(shape=self.shape, dtype=self.dtype)

    def resolve(self) -> np.ndarray:
        """Resolve the proxy by opening the hdf5 file and loading the dataset
        into a numpy array.
        
        Returns:
            np.ndarray: The resolved dataset
        """

        with h5.File(self.fname, 'r') as h5file:
            return np.array(h5file[self.name])
