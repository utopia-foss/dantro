"""Tests the utils.dag module"""

from pkg_resources import resource_filename

import pytest

import numpy as np

import dantro
import dantro.utils
import dantro.utils.dag as dag
from dantro import DataManager
from dantro.groups import OrderedDataGroup
from dantro.containers import ObjectContainer, NumpyDataContainer
from dantro.tools import load_yml

# Local constants
TRANSFORMATIONS_PATH = resource_filename('tests', 'cfg/transformations.yml')
DAG_SYNTAX_PATH = resource_filename('tests', 'cfg/dag_syntax.yml')

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
    data = _dm.new_group('data')
    data.new_container('zeros', Cls=NumpyDataContainer,
                       data=np.zeros((2,3,4)))

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

def test_TransformationDAG_parsing():
    """Tests the TransformationDAG class"""
    TransformationDAG = dag.TransformationDAG

    syntax_test_cfgs = load_yml(DAG_SYNTAX_PATH)
    
    # Initialize an empty DAG object that will be re-used
    tdag = TransformationDAG(dm=dm)
    parse_func = tdag._parse_trfs

    for name, cfg in syntax_test_cfgs.items():
        # Extract specification and expected values etc
        print("Testing transformation syntax case '{}' ...".format(name))

        # Extract arguments
        params = cfg['params']
        expected = cfg['expected']

        # Error checking arguments
        _raises = cfg.get('_raises', False)
        _exp_exc = (Exception if not isinstance(_raises, str)
                    else getattr(__builtins__, _raises))
        _match = cfg.get('_match')

        # Invoke it
        if not _raises:
            output = parse_func(**params)

        else:
            with pytest.raises(_exp_exc, match=_match):
                output = parse_func(**params)

            print("Raised error as expected.\n")
            continue

        # Compare with expected result...
        assert output == expected
        print("Parsing output was as expected.\n")

def test_TransformationDAG_build_and_compute(dm):
    """Tests the TransformationDAG class"""
    TransformationDAG = dag.TransformationDAG

    transformation_test_cfgs = load_yml(TRANSFORMATIONS_PATH)

    for name, cfg in transformation_test_cfgs.items():
        # Extract specification and expected values etc
        print("\nTesting transformation DAG case '{}' ...".format(name))

        # Extract arguments
        params = cfg['params']
        expected = cfg['expected']

        # Error checking arguments
        _raises = cfg.get('_raises', False)
        _exp_exc = (Exception if not isinstance(_raises, str)
                    else getattr(__builtins__, _raises))
        _match = cfg.get('_match')

        # Initialize TransformationDAG object, which will build the DAGs
        if not _raises:
            tdag = TransformationDAG(dm=dm, **params)

        else:
            with pytest.raises(_exp_exc, match=_match):
                tdag = TransformationDAG(dm=dm, **params)

            print("Raised error as expected.\n")
            continue
        
        # Compare with expected tree structure and tags etc.
        if expected.get('num_nodes'):
            assert expected['num_nodes'] == len(tdag.nodes)
        
        if expected.get('num_objects'):
            assert expected['num_objects'] == len(tdag.objects)

        if expected.get('tags'):
            assert set(expected['tags']) == set(tdag.tags.keys())

        print("Tree structure and tags as expected.")

        # Compare with expected result...
        print("Computing results ...")
        results = tdag.compute()
        print(results.tree)

        # TODO check

        print("Computation results as expected.\n")

        
    assert False
