"""Tests the utils.dag module"""

from pkg_resources import resource_filename

import pytest

import numpy as np

import dantro
import dantro.utils
import dantro.utils.dag as dag

from dantro import DataManager
from dantro.base import BaseDataGroup
from dantro.groups import OrderedDataGroup
from dantro.containers import ObjectContainer, NumpyDataContainer
from dantro.tools import load_yml, write_yml
from dantro._hash import _hash

# Local constants
TRANSFORMATIONS_PATH = resource_filename('tests', 'cfg/transformations.yml')
DAG_SYNTAX_PATH = resource_filename('tests', 'cfg/dag_syntax.yml')

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def dm() -> DataManager:
    """A data manager with some basic testing data"""
    _dm = DataManager("/some/fixed/path", name='TestDM', out_dir=False)
    # NOTE This attaches to some (imaginary) fixed path, because the hashstr
    #      of the DataManager is computed from the name and the data directory
    #      path. By using a fixed value (instead of tmpdir), the hashes of all
    #      the DAG objects remain fixed as well, making testing much easier.

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

def test_hash():
    """Test that the hash function did not change"""
    assert _hash("I will not change.") == "cac42c9aeca87793905d257c1b1b89b8"


def test_Transformation():
    """Tests the Transformation class"""
    Transformation = dag.Transformation

    t0 = Transformation(operation="add", args=[1,2], kwargs=dict())
    assert t0.result is None
    assert t0.hashstr == "21c6675666732d9e6c6426ffb454e829"

    assert t0.compute() == 3
    assert t0.result == 3
    assert t0.compute() == 3  # to hit the cache
    assert t0.compute(discard_cache=True) == 3
    
    # Same arguments should lead to the same hash
    t1 = Transformation(operation="add", args=[1,2], kwargs=dict())
    assert t1.hashstr == t0.hashstr
    
    # Keyword argument order should not play a role for the hash
    t2 = Transformation(operation="foo", args=[], kwargs=dict(a=1, b=2))
    t3 = Transformation(operation="foo", args=[], kwargs=dict(b=2, a=1))
    assert t2.hashstr == t3.hashstr

    # Transformations with references need a DAG
    tfail = Transformation(operation="add",
                           args=[dag.DAGNode(-1)], kwargs=dict())

    with pytest.raises(TypeError, match="needed to resolve the references"):
        tfail.compute()


def test_DAGObjects(dm):
    """Tests the DAGObjects class."""
    DAGObjects = dag.DAGObjects
    Transformation = dag.Transformation

    # Initialize an empty database
    objs = DAGObjects()

    # Some objects to store in
    t0 = Transformation(operation="add", args=[1,2], kwargs=dict())
    t1 = Transformation(operation="add", args=[1,2], kwargs=dict())

    # Can store only certain objects in it
    hdm = objs.add_object(dm)
    ht0 = objs.add_object(t0)
    ht1 = objs.add_object(t1)

    # t1 was not added, because t0 was added first and they have the same hash
    assert ht0 == ht1
    assert len(objs) == 2
    assert t0 in objs.values()
    assert t1 not in objs.values()

    # Can't add just any hashable to it
    with pytest.raises(AttributeError, match="hashstr"):
        objs.add_object("123")

    # Can access them via item access, key being their hash
    assert objs[hdm] is dm
    assert objs[ht0] is t0

    # Can check if a hash exists
    assert "123" not in objs
    assert 123 not in objs
    assert hdm in objs
    assert ht0 in objs

    # Coverage test of iteration methods
    list(objs.keys())
    list(objs.values())
    list(objs.items())


def test_TransformationDAG_parsing(dm):
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

    # Make sure the DataManager hash is as expected
    assert dm.hashstr == "38518b2446b95e8834372949a8e9dfc2"

    # Get the configs
    transformation_test_cfgs = load_yml(TRANSFORMATIONS_PATH)

    for name, cfg in transformation_test_cfgs.items():
        # Extract specification and expected values etc
        print("Testing transformation DAG case '{}' ...".format(name))

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

        # Test node hashes
        if expected.get('node_hashes'):
            assert tdag.nodes == expected['node_hashes']            
            print("Node hashes consistent.")

        # Compare with expected result...
        print("Computing results ...")
        results = tdag.compute()
        print(results.tree)

        print("Checking results ...")
        
        # Should be a DataGroup
        assert isinstance(results, BaseDataGroup)

        # Check more explicitly
        for tag, to_check in expected.get('results', {}).items():
            print("  Tag:  {}".format(tag))

            # Check if the tag is in the results group
            assert tag in results.keys()

            # Get the result for this tag
            res = results[tag]
            print("    ID: {} \tdata type: {} \tdata ID: {}"
                  "".format(id(res), type(res.data), id(res.data)))

            # Check if the type of the object is as expected; do so by string
            # comparison to avoid having to do an import here ...
            if 'type' in to_check:
                assert type(res).__name__ == to_check['type']

            # Check that the linked object has the correct path and type; this
            # makes only sense for LinkContainer objects...
            if 'linked_path' in to_check:
                assert res.target_rel_path == to_check['linked_path']

            if 'linked_type' in to_check:
                assert type(res.target_object).__name__ == to_check['linked_type']

            # Check attribute values, calling callables
            if 'attributes' in to_check:
                for attr_name, exp_attr_val in to_check['attributes'].items():
                    attr = getattr(res, attr_name)
                    
                    if callable(attr):
                        assert attr() == exp_attr_val
                    else:
                        # Convert tuples to lists to allow yaml-comparison
                        attr = list(attr) if isinstance(attr, tuple) else attr
                        assert attr == exp_attr_val

        print("All computation results as expected.\n")
        print("------------------------------------\n")
