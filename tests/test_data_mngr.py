"""Test the DataManager"""

import os
import pkg_resources

import pytest

import dantro.data_mngr
from dantro.data_loaders import YamlLoaderMixin
from dantro.tools import write_yml

# Local constants
LOAD_CFG_PATH = pkg_resources.resource_filename('tests', 'cfg/load_cfg.yml')

# Test class ------------------------------------------------------------------

class DataManager(YamlLoaderMixin, dantro.data_mngr.DataManager):
    """A DataManager-derived class for testing the implementation"""
    pass


# Fixtures --------------------------------------------------------------------

@pytest.fixture
def data_dir(tmpdir) -> str:
    """Writes some dummy data to a temporary directory and returns the path to that directory"""
    # Create YAML dummy data and write it out
    foobar = dict(one=1, two=2,
                  go_deeper=dict(eleven=11),
                  a_list=list(range(10)))

    write_yml(foobar, path=tmpdir.join("foobar.yml"))

    return tmpdir

@pytest.fixture
def dm(data_dir) -> DataManager:
    """Returns a basic configuration of a DataManager"""
    return DataManager(data_dir, load_cfg=LOAD_CFG_PATH)

# Tests -----------------------------------------------------------------------

def test_init(data_dir):
    """Test the initialisation of a DataManager"""
    dm = DataManager(data_dir,
                     load_cfg=dict(test=dict(loader="yaml", glob_str="*.yml")))

    # Assert folders are existing and correctly linked
    assert dm.dirs['data'] == data_dir
    assert os.path.isdir(dm.dirs['data'])
    assert os.path.isdir(dm.dirs['out'])

    # Without out_dir, this should be different
    assert DataManager(data_dir, out_dir=None).dirs['out'] is False

def test_loading(dm):
    """Tests whether loading works

    NOTE this uses the load configuration specified in cfg/load_cfg.yml
    """
    dm.load_data()

    # Assert that the top level entries are all available
    assert 'foobar' in dm

    # Assert their content is right
    foobar = dm['foobar']
    assert foobar['one'] == 1
    assert foobar['two'] == 2
    assert foobar['go_deeper']['eleven'] == 11
    assert foobar['a_list'] == list(range(10))
