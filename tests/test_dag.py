"""Tests the utils.dag module"""

import os
from pkg_resources import resource_filename

import pytest

import numpy as np

import dantro
import dantro.dag as dag

from dantro import DataManager
from dantro.base import BaseDataGroup
from dantro.groups import OrderedDataGroup
from dantro.containers import (ObjectContainer, NumpyDataContainer,
                               XrDataContainer)
from dantro.data_loaders import (YamlLoaderMixin, PickleLoaderMixin,
                                 NumpyLoaderMixin, XarrayLoaderMixin)
from dantro.tools import load_yml, write_yml
from dantro._hash import _hash

# Local constants
TRANSFORMATIONS_PATH = resource_filename('tests', 'cfg/transformations.yml')
DAG_SYNTAX_PATH = resource_filename('tests', 'cfg/dag_syntax.yml')

# Class Definitios ------------------------------------------------------------

from .test_data_mngr import Hdf5DataManager

class FullDataManager(PickleLoaderMixin, NumpyLoaderMixin,
                      XarrayLoaderMixin, Hdf5DataManager):
    """A DataManager with all the loaders implemented"""


# Fixtures --------------------------------------------------------------------

@pytest.fixture
def dm() -> FullDataManager:
    """A data manager with some basic testing data"""
    _dm = FullDataManager("/some/fixed/path", name='TestDM', out_dir=False)
    # NOTE This attaches to some (imaginary) fixed path, because the hashstr
    #      of the DataManager is computed from the name and the data directory
    #      path. By using a fixed value (instead of tmpdir), the hashes of all
    #      the DAG objects remain fixed as well, making testing much easier.

    # Create some groups
    _dm.new_group('some')
    _dm['some'].new_group('path')
    g_foo = _dm['some/path'].new_group('foo')
    g_bar = _dm['some/path'].new_group('bar')

    # Create some regular numpy data
    data = _dm.new_group('data')
    data.new_container('zeros', Cls=NumpyDataContainer,
                       data=np.zeros((2,3,4)))
    data.new_container('random', Cls=NumpyDataContainer,
                       data=np.random.random((2,3,4)))

    # Create some xarray data
    ldata = _dm.new_group('labelled_data')
    ldata.new_container('zeros', Cls=XrDataContainer,
                        data=np.zeros((2,3,4)),
                        attrs=dict(dims=['x', 'y', 'z']))
    ldata.new_container('random', Cls=XrDataContainer,
                        data=np.zeros((2,3,4)),
                        attrs=dict(dims=['x', 'y', 'z']))

    return _dm

# -----------------------------------------------------------------------------

def test_hash():
    """Test that the hash function did not change"""
    assert _hash("I will not change.") == "cac42c9aeca87793905d257c1b1b89b8"


def test_Transformation():
    """Tests the Transformation class"""
    Transformation = dag.Transformation

    t0 = Transformation(operation="add", args=[1,2], kwargs=dict())
    assert t0.hashstr == "21c6675666732d9e6c6426ffb454e829"

    assert t0.compute() == 3
    assert t0.compute() == 3  # to hit the (memory) cache
    
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

@pytest.mark.skip
def test_Transformation_cache():
    """Test Transformation caching"""
    pass


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
    
    # Initialize an empty DAG object that will be used for the parsing
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

def test_TransformationDAG_life_cycle(dm, tmpdir):
    """Tests the TransformationDAG class"""
    TransformationDAG = dag.TransformationDAG

    # Make sure the DataManager hash is as expected
    assert dm.hashstr == "38518b2446b95e8834372949a8e9dfc2"

    # The temporary cache directory
    base_cache_dir = tmpdir

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

        # Custom cache directory
        cache_dir = str(base_cache_dir.join(name + "_cache"))

        # Initialize TransformationDAG object, which will build the DAGs
        if not _raises:
            tdag = TransformationDAG(dm=dm, **params,
                                     cache_dir=cache_dir)

        else:
            with pytest.raises(_exp_exc, match=_match):
                tdag = TransformationDAG(dm=dm, **params, cache_dir=cache_dir)

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

        # Cache directory should definitely not exist yet!
        assert not os.path.isdir(cache_dir)

        # Compare with expected result...
        print("\nComputing results ...")
        results = tdag.compute()
        print("\n".join(["  * {:<20s}  {:}".format(k, v)
                         for k, v in results.items()]))

        # Cache directory MAY exist after computation
        if not os.path.isdir(cache_dir):
            print("\nCache directory not available.")
        else:
            print("\nContent of cache directory:")
            print("  * " + "\n  * ".join(os.listdir(cache_dir)))

        if expected.get('cache_dir_available'):
            assert os.path.isdir(cache_dir)

            if expected.get('cache_files'):
                expected_files = expected['cache_files']
                assert set(expected_files) == set(os.listdir(cache_dir))

            print("Cache directory content as expected.")

        # Now, check the results ..............................................
        print("\nChecking results ...")
        
        # Should be a dict
        assert isinstance(results, dict)

        # Check more explicitly
        for tag, to_check in expected.get('results', {}).items():
            print("  Tag:  {}".format(tag))

            # Get the result for this tag
            res = results[tag]

            # Check if the type of the object is as expected; do so by string
            # comparison to avoid having to do an import here ...
            if 'type' in to_check:
                assert type(res).__name__ == to_check['type']

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

