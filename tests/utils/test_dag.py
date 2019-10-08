"""Tests the utils.dag module"""

from pkg_resources import resource_filename

import pytest

import dantro
import dantro.utils
import dantro.utils.dag as dag
from dantro import DataManager
from dantro.groups import OrderedDataGroup
from dantro.containers import ObjectContainer
from dantro.tools import load_yml

# Local constants
TRANSFORMATIONS_PATH = resource_filename('tests', 'cfg/transformations.yml')

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def dm(tmpdir) -> DataManager:
    """A data manager with some basic testing data"""
    _dm = DataManager(tmpdir, name='TestDM', out_dir=False)

    # Create some groups
    _dm.new_group('some')
    _dm['some'].new_group('path')
    g_foo = _dm['some/path'].new_group('foo')
    g_bar = _dm['some/path'].new_group('bar')

    # Create some containers
    # TODO

    return _dm

# -----------------------------------------------------------------------------

def test_serialize():
    """Tests the serialization function"""
    _serialize = dag._serialize

def test_hash():
    """Test the hash function"""
    _hash = dag._hash

def test_Transformation():
    """Tests the Transformation class"""
    Transformation = dag.Transformation

def test_DAGObjects():
    """Tests the DAGObjects class."""
    DAGObjects = dag.DAGObjects

    objs = DAGObjects()

def test_TransformationDAG(dm):
    """Tests the TransformationDAG class"""
    TransformationDAG = dag.TransformationDAG

    transformations = load_yml(TRANSFORMATIONS_PATH)

    for name, trf_cfg in transformations.items():
        # Extract specification and expected values etc
        print("\nTesting transformation config '{}' ...".format(name))

        # Extract arguments
        trfs_kwargs = {k: v for k, v in trf_cfg.items()
                       if not k.startswith('_')}

        _may_fail = trf_cfg.get('_may_fail', False)
        _expected_num_fields = trf_cfg['_expected_num_fields']
        _expected_fields = trf_cfg['_expected_fields']

        # Initialize it
        tdag = TransformationDAG(dm=dm, **trfs_kwargs)
        # TODO try-except block

        # result = tdag.compute() # FIXME
        
        # TODO Compare with expected result...
