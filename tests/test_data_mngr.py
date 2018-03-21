"""Test the DataManager"""

import os

import pytest

from dantro.data_mngr import DataManager

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def data_dir(tmpdir) -> str:
    """Writes some dummy data to a temporary directory and returns the path to that directory"""
    # TODO actually write data here

    return tmpdir

# Tests -----------------------------------------------------------------------

def test_init(data_dir):
    """Test the initialisation of a DataManager"""
    dm = DataManager(data_dir)

    # Assert folders are existing and correctly linked
    assert dm.dirs['data'] == data_dir
    assert os.path.isdir(dm.dirs['data'])
    assert os.path.isdir(dm.dirs['out'])

    # Without out_dir, this should be different
    assert DataManager(data_dir, out_dir=None).dirs['out'] is False
