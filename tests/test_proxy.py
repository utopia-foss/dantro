"""Test the dantro.proxy modules

NOTE Parts of the tests for the proxies are done elsewhere.
"""
import copy

import pytest

import h5py as h5
import numpy as np

import dantro
from dantro.proxy import Hdf5DataProxy

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def tmp_h5file(tmpdir) -> h5.File:
    """Returns a temporary h5.File object with some datasets"""
    h5f = h5.File(tmpdir.join('testfile.h5'))

    h5f.create_dataset("foo", data=np.array([1,2,3], dtype=int))

    return h5f

# -----------------------------------------------------------------------------

def test_Hdf5DataProxy(tmp_h5file):
    """Tests the Hdf5DataProxy directly, without integration"""

    foo = Hdf5DataProxy(tmp_h5file["foo"])

    # Check resolution
    assert isinstance(foo.resolve(), np.ndarray)
    assert (foo.resolve() == [1,2,3]).all()

    # Access properties
    assert foo.shape == (3,)
    assert foo.dtype is np.dtype(int)
    assert foo.ndim == 1
    assert foo.size == 3
    assert foo.chunks is None

    # No h5files should have been retained now
    assert not foo._h5files

    # When loading with a type, that type is called on the data
    assert isinstance(foo.resolve(astype=list), list)

    # And still, no files retained
    assert not foo._h5files

    # However, when loading as h5.Dataset, the file is stored and kept open
    dset = foo.resolve(astype=h5.Dataset)
    assert isinstance(dset, h5.Dataset)
    assert len(foo._h5files) == 1


    # Deleting the proxy should close the file, invalidating the dataset
    del foo
    assert str(dset) == "<Closed HDF5 dataset>"
    with pytest.raises(ValueError, match="Not a dataset"):
        dset[()]
