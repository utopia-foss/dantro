"""Test the dantro.proxy modules

NOTE Parts of the tests for the proxies are done elsewhere.
"""

import dask as da
import dask.array
import h5py as h5
import numpy as np
import pytest

import dantro
from dantro.proxy import Hdf5DataProxy

# Fixtures --------------------------------------------------------------------


@pytest.fixture
def tmp_h5file(tmpdir) -> h5.File:
    """Returns a temporary h5.File object with some datasets"""
    h5f = h5.File(tmpdir.join("testfile.h5"), "w")

    h5f.create_dataset("foo", data=np.array([1, 2, 3], dtype=int))
    h5f.create_dataset(
        "foochunk", data=np.array([1, 2, 3], dtype=int), chunks=(1,)
    )

    # Some more chunked datasets
    grp = h5f.create_group("chunked")
    grp.create_dataset(
        "random", data=np.random.random((3, 4, 5, 6)), chunks=(3, 4, 5, 1)
    )
    grp.create_dataset(
        "zeros", data=np.zeros((3, 4, 5, 6), dtype=int), chunks=(3, 4, 5, 1)
    )

    return h5f


# -----------------------------------------------------------------------------
# Hdf5DataProxy


def test_Hdf5DataProxy(tmp_h5file):
    """Tests the Hdf5DataProxy directly, without integration"""

    foo = Hdf5DataProxy(tmp_h5file["foo"])

    # Check resolution
    assert isinstance(foo.resolve(), np.ndarray)
    assert (foo.resolve() == [1, 2, 3]).all()

    # Access properties
    assert foo.shape == (3,)
    assert foo.dtype is np.dtype(int)
    assert foo.ndim == 1
    assert foo.size == 3
    assert foo.chunks is None

    # Tags
    assert "hdf5" in foo.tags

    # No h5files should have been kept in memory
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
    # This may lead to different error types, depending on environment
    del foo
    assert str(dset) == "<Closed HDF5 dataset>"
    with pytest.raises((ValueError, RuntimeError)):
        dset[()]

    # NOTE Can't really test the try-except there, which only becomes relevant
    #      once the whole python session is ended.


def test_Hdf5DataProxy_resolve_as_dask(tmp_h5file):
    """Tests the Hdf5DataProxy that resolves as dask arrays"""

    foochunk = Hdf5DataProxy(tmp_h5file["foochunk"], resolve_as_dask=True)

    # Check resolution
    assert isinstance(foochunk.resolve(), da.array.core.Array)

    # Access properties
    assert foochunk.shape == (3,)
    assert foochunk.dtype is np.dtype(int)
    assert foochunk.ndim == 1
    assert foochunk.size == 3
    assert foochunk.chunks == (1,)

    # Check tags
    assert "hdf5" in foochunk.tags
    assert "dask" in foochunk.tags

    # There should be a h5file in memory
    assert foochunk._h5files

    # Read the data in as numpy array ...
    np_foochunk = Hdf5DataProxy(tmp_h5file["foochunk"], resolve_as_dask=False)

    # ... to check that elements are equal to the dask.array elements
    assert (foochunk.resolve() == np_foochunk.resolve()).all()
