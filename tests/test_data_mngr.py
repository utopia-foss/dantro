"""Test the DataManager"""

import os
import pkg_resources

import pytest

from dantro.data_mngr import DataManager

# Local constants
LOAD_CFG_PATH = pkg_resources.resource_filename('tests', 'cfg/load_cfg.yml')

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def data_dir(tmpdir) -> str:
    """Writes some dummy data to a temporary directory and returns the path to that directory"""
    # TODO actually write data here

    return tmpdir

@pytest.fixture
def load_cfg() -> dict:
    """Returns a dummy load configuration"""
    return dict(yaml=dict(loader="yaml", glob_str="*.yml"))

@pytest.fixture
def basic_dm(data_dir) -> DataManager:
    """Returns a basic configuration of a DataManager"""
    return DataManager(data_dir, load_cfg=LOAD_CFG_PATH)

# Tests -----------------------------------------------------------------------

def test_init(data_dir, load_cfg):
    """Test the initialisation of a DataManager"""
    dm = DataManager(data_dir, load_cfg=load_cfg)

    # Assert folders are existing and correctly linked
    assert dm.dirs['data'] == data_dir
    assert os.path.isdir(dm.dirs['data'])
    assert os.path.isdir(dm.dirs['out'])

    # Without out_dir, this should be different
    assert DataManager(data_dir, out_dir=None).dirs['out'] is False

def test_loading(basic_dm):
    """Tests whether loading works"""
    basic_dm.load_data()
