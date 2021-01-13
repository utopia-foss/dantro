"""Tests the utils.dag module"""

import copy
import os
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pkg_resources import resource_filename

import dantro
import dantro._dag_utils as dag_utils
import dantro.dag as dag
from dantro import DataManager
from dantro._hash import _hash
from dantro._yaml import yaml_dumps
from dantro.base import BaseDataGroup
from dantro.containers import (
    NumpyDataContainer,
    ObjectContainer,
    StringContainer,
    XrDataContainer,
)
from dantro.data_loaders import (
    NumpyLoaderMixin,
    PickleLoaderMixin,
    XarrayLoaderMixin,
    YamlLoaderMixin,
)
from dantro.groups import OrderedDataGroup
from dantro.tools import load_yml, write_yml

# Local constants
TRANSFORMATIONS_PATH = resource_filename("tests", "cfg/transformations.yml")
DAG_SYNTAX_PATH = resource_filename("tests", "cfg/dag_syntax.yml")

# Class Definitios ------------------------------------------------------------

from .test_data_mngr import Hdf5DataManager


class FullDataManager(
    PickleLoaderMixin, NumpyLoaderMixin, XarrayLoaderMixin, Hdf5DataManager
):
    """A DataManager with all the loaders implemented"""


def some_func() -> str:
    return "I can be pickled (with dill)"


class UnpickleableString(StringContainer):
    def __getstate__(self):
        raise RuntimeError("I refuse to be pickled!")


class MockTransformationDAG:
    """A mock class for the TransformationDAG, only containing features needed
    for the placeholder resolution.
    """

    def __init__(self, **results):
        """Just stores the "results" as attributes, later to be returned"""
        self._results = results
        self.tags = tuple(results.keys())

    def compute(self, *, compute_only: list = None, **_):
        """Returns all or a subset of the results set"""
        if not compute_only:
            compute_only = self.tags
        return {k: self._results[k] for k in compute_only}


# Fixtures and Helpers --------------------------------------------------------


@pytest.fixture
def dm() -> FullDataManager:
    """A data manager with some basic testing data"""
    _dm = FullDataManager("/some/fixed/path", name="TestDM", out_dir=False)
    # NOTE This attaches to some (imaginary) fixed path, because the hashstr
    #      of the DataManager is computed from the name and the data directory
    #      path. By using a fixed value (instead of tmpdir), the hashes of all
    #      the DAG objects remain fixed as well, making testing much easier.

    # Create some groups
    _dm.new_group("some")
    _dm["some"].new_group("path")
    g_foo = _dm["some/path"].new_group("foo")
    g_bar = _dm["some/path"].new_group("bar")

    # Some data for documentation examples
    _dm.new_group("path")
    g_to = _dm["path"].new_group("to")
    g_to.new_container(
        "some_data", Cls=NumpyDataContainer, data=np.zeros((5, 5))
    )
    g_to.new_container(
        "more_data", Cls=NumpyDataContainer, data=np.ones((5, 5))
    )

    elephant_t = np.linspace(0, 1000, 1001)
    elephant_f = lambda t: 1.6 + 0.01 * t - 0.001 * t ** 2 + 2.3e-5 * t ** 3
    elephant_ts = xr.DataArray(
        data=(elephant_f(elephant_t) + np.random.random((1001,))),
        dims=("time",),
        coords=dict(time=elephant_t),
    )
    g_to.new_container("elephant_ts", Cls=XrDataContainer, data=elephant_ts)

    # Create some regular numpy data
    data = _dm.new_group("data")
    data.new_container(
        "zeros", Cls=NumpyDataContainer, data=np.zeros((2, 3, 4))
    )
    data.new_container(
        "random", Cls=NumpyDataContainer, data=np.random.random((2, 3, 4))
    )

    # Create some xarray data
    ldata = _dm.new_group("labelled_data")
    ldata.new_container(
        "zeros",
        Cls=XrDataContainer,
        data=np.zeros((2, 3, 4)),
        attrs=dict(dims=["x", "y", "z"]),
    )
    ldata.new_container(
        "ones",
        Cls=XrDataContainer,
        data=np.ones((2, 3, 4)),
        attrs=dict(dims=["x", "y", "z"]),
    )
    ldata.new_container(
        "random",
        Cls=XrDataContainer,
        data=np.zeros((2, 3, 4)),
        attrs=dict(dims=["x", "y", "z"]),
    )

    # Create some other objects, mainly for testing caching
    odata = _dm.new_group("objects")
    odata.new_container("some_dict", Cls=ObjectContainer, data=dict(foo="bar"))
    odata.new_container("some_list", Cls=ObjectContainer, data=[1, 2, 3])
    odata.new_container("some_func", Cls=ObjectContainer, data=some_func)

    bodata = _dm.new_group("bad_objects")
    bodata.new_container(
        "some_local_func",
        Cls=ObjectContainer,
        data=lambda: "i cannot be pickled (even with dill)",
    )
    bodata.new_container(
        "some_string",
        Cls=UnpickleableString,
        data="i cannot be pickled (even with dill)",
    )

    print(_dm.tree)
    return _dm


def yaml_roundtrip(obj: Any, *, path: str) -> Any:
    """Makes a YAML roundtrip to the given path"""
    from dantro._yaml import load_yml, write_yml

    write_yml(obj, path=path)
    return load_yml(path=path)


# -----------------------------------------------------------------------------


def test_hash():
    """Test that the hash function did not change"""
    assert _hash("I will not change.") == "cac42c9aeca87793905d257c1b1b89b8"


def test_Placeholder(tmpdir):
    """Test the Placeholder class"""
    ph = dag_utils.Placeholder("foo bar")

    assert ph == dag_utils.Placeholder("foo bar")
    assert ph != dag_utils.Placeholder("bar foo")

    assert "Placeholder" in repr(ph)
    assert "foo bar" in repr(ph)

    # YAML Roundtrip
    yaml_rt = lambda o: yaml_roundtrip(o, path=tmpdir.join(ph._data))
    assert yaml_rt(ph) == ph
    assert yaml_rt(ph) is not ph


def test_ResultPlaceholder_resolution(tmpdir):
    """Tests the placeholder resolution in an isolated setting"""
    from dantro._dag_utils import ResultPlaceholder, resolve_placeholders

    # Basics
    rph = ResultPlaceholder("some_result_name")
    assert rph.result_name == "some_result_name"

    # YAML roundtrip
    yaml_rt = lambda o: yaml_roundtrip(o, path=tmpdir.join(o.result_name))
    assert yaml_rt(rph) == rph
    assert yaml_rt(rph) is not rph

    # -- Test resolution
    # Basic case
    rph_foo = ResultPlaceholder("foo")
    rph_bar = ResultPlaceholder("bar")
    d = dict(
        one=rph_foo,
        two=dict(
            some_list=[0, 1, rph_bar, 2, rph_foo],
            more_nesting=[[], dict(foo=rph_foo)],
        ),
    )
    print("d before:", d)

    mdag = MockTransformationDAG(foo="FOO", bar="BAR")
    d_after = resolve_placeholders(copy.deepcopy(d), dag=mdag)
    print("d after:", d)

    assert d != d_after
    assert d_after["one"] == "FOO"
    assert d_after["two"]["some_list"] == [0, 1, "BAR", 2, "FOO"]
    assert d_after["two"]["more_nesting"][1]["foo"] == "FOO"

    # Without placeholders, nothing happens
    d = dict(foo="bar", spam=dict(fish="foobar"))
    d_after = resolve_placeholders(d, dag=mdag)
    assert d == d_after
    assert d is d_after  # ... because not a deepcopy here

    # With a bad placeholder, this will fail. As this is only the Mock DAG, we
    # don't need to care too much about the exception type ...
    rph_BAD = ResultPlaceholder("BAD")
    d = dict(foo=rph_foo, spam=dict(bad=rph_BAD))

    with pytest.raises(Exception, match="BAD"):
        resolve_placeholders(d, dag=mdag)


def test_argument_placeholders():
    """Tests the PositionalArgument and KeywordArgument specializations of the
    base Placeholder class
    """
    Arg = dag_utils.PositionalArgument
    Kwarg = dag_utils.KeywordArgument

    assert Arg(0).position == 0
    assert Arg(10).position == 10
    assert Arg("3").position == 3

    assert Kwarg("foo").name == "foo"

    # Errors
    with pytest.raises(TypeError, match="int-convertible"):
        Arg("abc")
    with pytest.raises(ValueError, match="non-negative"):
        Arg("-1")

    with pytest.raises(TypeError, match="requires a string"):
        Kwarg(123)


def test_DAGReference():
    """Test the DAGReference class

    NOTE Reference resolution cannot be tested without DAG
    """
    # Initialization
    some_hash = _hash("some")
    ref = dag.DAGReference(some_hash)
    assert ref.ref == some_hash

    assert ref == dag.DAGReference(some_hash)
    assert ref != dag.DAGReference(_hash("some_other_hash"))
    assert ref != some_hash

    assert some_hash in repr(ref)

    # Errors
    with pytest.raises(TypeError, match="requires a string-like argument"):
        dag.DAGReference(123)

    # Reference resolution
    assert ref._resolve_ref(dag=None) == some_hash
    assert id(ref) != id(ref.convert_to_ref(dag=None))

    # YAML representation
    assert "!dag_ref" in yaml_dumps(ref, register_classes=(dag.DAGReference,))


def test_DAGTag():
    """Test the DAGTag class

    NOTE Reference resolution cannot be tested without DAG
    """
    some_tag = "tag42"
    tag = dag.DAGTag(some_tag)
    assert tag.name == some_tag

    assert tag == dag.DAGTag(some_tag)
    assert tag != dag.DAGTag("some other tag")
    assert tag != some_tag

    assert some_tag in repr(tag)

    assert "!dag_tag" in yaml_dumps(tag, register_classes=(dag.DAGTag,))

    # Cannot use the DAGMetaOperationTag string separator in names
    with pytest.raises(ValueError, match="cannot include the '::' substring"):
        dag.DAGTag("foo::bar")


def test_DAGMetaOperationTag():
    """Test the DAGMetaOperationTag class

    NOTE Reference resolution cannot be tested without DAG
    """
    MOpTag = dag_utils.DAGMetaOperationTag

    some_tag = "foo::bar"
    tag = MOpTag(some_tag)
    assert tag.name == some_tag

    assert some_tag in repr(tag)

    assert "!mop_tag" in yaml_dumps(tag, register_classes=(MOpTag,))

    # There are restrictions on the name
    with pytest.raises(ValueError, match="Invalid name"):
        MOpTag("foo::bar::baz")

    with pytest.raises(ValueError, match="Invalid name"):
        MOpTag("foo:without-valid_substring")


def test_DAGNode():
    """Test the DAGNode class

    NOTE Reference resolution cannot be tested without DAG
    """
    some_node = 42
    node = dag.DAGNode(some_node)
    assert node.idx == some_node

    assert node == dag.DAGNode(some_node)
    assert node == dag.DAGNode("42")
    assert node != dag.DAGNode(-1)
    assert node != dag.DAGNode(2)
    assert node != some_node

    assert str(some_node) in repr(node)

    with pytest.raises(TypeError, match="requires an int-convertible"):
        dag.DAGNode("not int-convertible")

    assert "!dag_node" in yaml_dumps(node, register_classes=(dag.DAGNode,))


def test_DAGObjects(dm):
    """Tests the DAGObjects class."""
    DAGObjects = dag.DAGObjects
    Transformation = dag.Transformation

    # Initialize an empty database
    objs = DAGObjects()
    assert "0 entries" in str(objs)

    # Some objects to store in
    t0 = Transformation(operation="add", args=[1, 2], kwargs=dict())
    t1 = Transformation(operation="add", args=[1, 2], kwargs=dict())

    # Can store only certain objects in it
    hdm = objs.add_object(dm)
    assert "1 entry" in str(objs)
    ht0 = objs.add_object(t0)
    assert "2 entries" in str(objs)
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

    # Adding an object with a custom hash does not work if it can be hashed
    with pytest.raises(TypeError, match="Cannot use a custom hash for "):
        objs.add_object(t0, custom_hash="foobar")

    # Add an object
    assert "123_hash" == objs.add_object("123", custom_hash="123_hash")

    # Adding one with the same hash does not work
    with pytest.raises(ValueError, match="already exists! Refusing to add it"):
        objs.add_object("not_123", custom_hash="123_hash")


def test_Transformation():
    """Tests the Transformation class"""
    Transformation = dag.Transformation

    t0 = Transformation(operation="add", args=[1, 2], kwargs=dict())
    assert t0.hashstr == "23cf81f382bd65f15f9e22ab80923a3b"
    assert hash(t0.hashstr) == hash(t0)

    assert "operation: add, 2 args, 0 kwargs" in str(t0)
    expected_repr = (
        "<dantro.dag.Transformation, operation='add', "
        "args=[1, 2], kwargs={}, salt=None>"
    )
    assert repr(t0) == expected_repr

    assert t0.compute() == 3
    assert t0.compute() == 3  # to hit the (memory) cache

    # Test salting
    t0s = Transformation(operation="add", args=[1, 2], kwargs=dict(), salt=42)
    assert t0 != t0s
    assert t0.hashstr != t0s.hashstr

    # Same arguments should lead to the same hash
    t1 = Transformation(operation="add", args=[1, 2], kwargs=dict())
    assert t1.hashstr == t0.hashstr

    # Keyword argument order should not play a role for the hash
    t2 = Transformation(operation="foo", args=[], kwargs=dict(a=1, b=2, c=3))
    t3 = Transformation(operation="foo", args=[], kwargs=dict(b=2, c=3, a=1))
    assert t2.hashstr == t3.hashstr

    # Transformations with references need a DAG
    tfail = Transformation(
        operation="add", args=[dag.DAGNode(-1)], kwargs=dict()
    )

    with pytest.raises(ValueError, match="no DAG was associated with this"):
        tfail.compute()

    # Read the profile property
    assert isinstance(t0.profile, dict)

    # Serialize as yaml
    assert "!dag_trf" in yaml_dumps(t0, register_classes=(Transformation,))
    assert "salt" not in yaml_dumps(t0, register_classes=(Transformation,))
    assert "salt: 42" in yaml_dumps(t0s, register_classes=(Transformation,))


def test_Transformation_dependencies(dm):
    """Tests that the Transformation's are aware of their dependencies"""
    Transformation = dag.Transformation
    TransformationDAG = dag.TransformationDAG
    DAGTag = dag.DAGTag

    # Define some nested nodes, that don't actually do anything. The only
    # dependency will be the DataManager
    tdag = TransformationDAG(dm=dm)
    tdag.add_node(operation="pass", args=[[[[DAGTag("dm")]]]])
    tdag.add_node(
        operation="pass",
        kwargs=dict(foo=[[[[DAGTag("dm")]]]], bar=DAGTag("dm")),
    )
    tdag.add_node(operation="pass", args=[[[[DAGTag("dm")], DAGTag("dm")]]])

    # These should always find the DataManager
    for node_hash in tdag.nodes:
        trf = tdag.objects[node_hash]
        rdeps = trf.resolved_dependencies
        assert isinstance(rdeps, set)
        assert len(rdeps) == 1
        assert dm in rdeps

    # In a new DAG, there will be two custom dependencies
    tdag = TransformationDAG(dm=dm)
    ref1 = tdag.add_node(operation="define", args=[1])
    ref2 = tdag.add_node(operation="define", args=[2])
    print(ref1, ref2)

    tdag.add_node(operation="pass", args=[[[[ref1], ref2]]])
    tdag.add_node(operation="pass", args=[ref1], kwargs=dict(foo=ref2))
    tdag.add_node(operation="pass", kwargs=dict(foo=[1, 2, 3, [[ref2], ref1]]))

    for node_hash in tdag.nodes:
        trf = tdag.objects[node_hash]
        deps = trf.dependencies

        if node_hash in [ref1.ref, ref2.ref]:
            assert len(deps) == 0
            continue

        assert len(deps) == 2
        assert isinstance(deps, set)
        assert ref1 in deps
        assert ref2 in deps


def test_TransformationDAG_syntax(dm):
    """Tests the TransformationDAG class"""
    TransformationDAG = dag.TransformationDAG

    syntax_test_cfgs = load_yml(DAG_SYNTAX_PATH)

    for name, cfg in syntax_test_cfgs.items():
        # Extract specification and expected values etc
        print("Testing transformation syntax case '{}' ...".format(name))

        # Extract arguments
        init_kwargs = cfg.get("init_kwargs", {})
        params = cfg["params"]
        expected = cfg.get("expected", {})

        # Initialize a new empty DAG object that will be used for the parsing
        tdag = TransformationDAG(dm=dm, **init_kwargs)
        parse_func = tdag._parse_trfs

        # Error checking arguments
        _raises = cfg.get("_raises", False)
        _exp_exc = (
            Exception
            if not isinstance(_raises, str)
            else __builtins__[_raises]
        )
        _match = cfg.get("_match")

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
    """Tests the TransformationDAG class."""
    TransformationDAG = dag.TransformationDAG

    # Make sure the DataManager hash is as expected
    assert dm.hashstr == "38518b2446b95e8834372949a8e9dfc2"

    # The temporary cache directory
    base_cache_dir = tmpdir

    # Get the configs
    transformation_test_cfgs = load_yml(TRANSFORMATIONS_PATH)

    # Go over all configured tests
    for name, cfg in transformation_test_cfgs.items():
        # Extract specification and expected values etc
        print("-" * 80)
        print("Testing transformation DAG case '{}' ...".format(name))

        # Extract arguments
        params = cfg["params"]
        expected = cfg.get("expected", {})

        # Error checking arguments
        _raises = cfg.get("_raises", False)
        _raises_on_compute = cfg.get("_raises_on_compute", False)
        _exp_exc = (
            Exception
            if not isinstance(_raises, str)
            else __builtins__[_raises]
        )
        _match = cfg.get("_match")

        # Custom cache directory. If the parameter is given, it can be used to
        # have a shared cache directory ...
        cache_dir_name = cfg.get("cache_dir_name", name + "_cache")
        cache_dir = str(base_cache_dir.join(cache_dir_name))

        # Initialize TransformationDAG object, which will build the DAGs
        if not _raises or _raises_on_compute:
            tdag = TransformationDAG(dm=dm, **params, cache_dir=cache_dir)

        else:
            with pytest.raises(_exp_exc, match=_match):
                tdag = TransformationDAG(dm=dm, **params, cache_dir=cache_dir)

            print("Raised error as expected.\n")
            continue

        # Check some properties that are unspecific to the params
        assert tdag.dm is dm
        assert isinstance(tdag.objects, dag.DAGObjects)
        assert tdag.cache_dir == cache_dir
        assert isinstance(tdag.cache_files, dict)

        # String representation
        assert (
            "TransformationDAG, {:d} node(s), {:d} tag(s), {:d} object(s)"
            "".format(len(tdag.nodes), len(tdag.tags), len(tdag.objects))
            in str(tdag)
        )

        # Check the select_base property getter and setter
        tdag.select_base = "dm"
        assert tdag.select_base == dag.DAGReference(tdag.tags["dm"])

        tdag.select_base = dag.DAGReference(tdag.tags["dm"])
        assert tdag.select_base == dag.DAGReference(tdag.tags["dm"])

        with pytest.raises(
            KeyError, match="cannot be used to set `select_base`"
        ):
            tdag.select_base = "some_invalid_tag"

        # Compare with expected tree structure and tags etc.
        if expected.get("num_nodes"):
            assert expected["num_nodes"] == len(tdag.nodes)

        if expected.get("num_objects"):
            assert expected["num_objects"] == len(tdag.objects)

        if expected.get("tags"):
            assert set(expected["tags"]) == set(tdag.tags.keys())

        print("Tree structure and tags as expected.")

        # Test node hashes
        if expected.get("node_hashes"):
            # Debug information
            print("\nObjects:")
            print("\n".join([f"- {k}: {v}" for k, v in tdag.objects.items()]))

            assert tdag.nodes == expected["node_hashes"]
            print("Node hashes consistent.")

        # Test number of node dependencies
        if expected.get("node_dependencies"):
            for node_hash, deps in zip(
                tdag.nodes, expected["node_dependencies"]
            ):
                node = tdag.objects[node_hash]

                if isinstance(deps, int):
                    assert len(node.dependencies) == deps
                else:
                    # Only compare hash references (easier to specify in yaml)
                    assert set([r.ref for r in node.dependencies]) == set(deps)

        # Test meta operations and the extracted arguments
        if expected.get("meta_operations"):
            expected_mops = expected["meta_operations"]

            assert expected_mops.keys() == tdag._meta_ops.keys()
            print("\nMeta-operation names as expected.")

            print("Checking meta-operation properties for ...")
            for mop_name, mop_spec in tdag._meta_ops.items():
                print(f"  {mop_name} ...", end="")

                exp_spec = expected_mops[mop_name]
                assert exp_spec["num_nodes"] == len(mop_spec["specs"])
                assert exp_spec["num_args"] == mop_spec["num_args"]
                assert set(exp_spec["kwarg_names"]) == mop_spec["kwarg_names"]

                if "defined_tags" in exp_spec:
                    assert (
                        set(exp_spec["defined_tags"])
                        == mop_spec["defined_tags"]
                    )

                print("ok")
            print("Meta-operation properties as expected.")

        # The reference stack should always be empty
        assert sum([len(stack) for stack in tdag.ref_stacks.values()]) == 0

        # Compare with expected result...
        compute_only = cfg.get("compute_only")
        print(
            "\nComputing results (compute_only argument: {}) ...".format(
                compute_only
            )
        )

        if not _raises or not _raises_on_compute:
            # Compute normally
            results = tdag.compute(compute_only=compute_only)

            print(
                "\n".join(
                    [
                        "  * {:<20s}  {:}".format(k, v)
                        for k, v in results.items()
                    ]
                )
            )

        else:
            with pytest.raises(_exp_exc, match=_match):
                results = tdag.compute(compute_only=compute_only)

            print("Raised error as expected.\n")
            continue

        # Cache directory MAY exist after computation
        if not os.path.isdir(cache_dir):
            print("\nCache directory not available.")
        else:
            print("\nContent of cache directory ({})".format(cache_dir))
            print("  * " + "\n  * ".join(os.listdir(cache_dir)))

        if expected.get("cache_dir_available"):
            assert os.path.isdir(cache_dir)

            if expected.get("cache_files"):
                expected_files = expected["cache_files"]
                assert set(expected_files) == set(os.listdir(cache_dir))

                # Check that both the full path and the extension is available
                for chash, cinfo in tdag.cache_files.items():
                    assert "full_path" in cinfo
                    assert "ext" in cinfo
                    assert (
                        os.path.basename(cinfo["full_path"])
                        == chash + cinfo["ext"]
                    )

            print("Cache directory content as expected.")

            # Temporarily manipulate the cache directory content to check that
            # the cache_files property returns correct results
            tmp_foodir = os.path.join(tdag.cache_dir, "some_dir.foobar")
            os.mkdir(tmp_foodir)
            assert "some_dir" not in tdag.cache_files
            os.rmdir(tmp_foodir)

            tmp_file = os.path.join(tdag.cache_dir, "some_other_file.some_ext")
            open(tmp_file, "a").close()
            assert "some_other_file.some_ext" not in tdag.cache_files
            os.remove(tmp_file)

        # Check the profile information
        prof = tdag.profile
        print("Profile: ", prof)
        assert len(prof) == 2
        assert all([item in prof for item in ("add_node", "compute")])

        extd_prof = tdag.profile_extended
        _expected = (
            "add_node",
            "compute",
            "tags",
            "aggregated",
            "sorted",
            "operations",
            "slow_operations",
        )
        # print("Extended profile: ", extd_prof)
        assert set(extd_prof.keys()) == set(_expected)

        # If there are no nodes available, there should be nans in the profile.
        # Otherwise, values may be NaN or an actual number
        for item in (
            "compute",
            "hashstr",
            "cache_lookup",
            "cache_writing",
            "effective",
        ):
            assert all(
                [
                    np.isnan(v)
                    for v in extd_prof["aggregated"][item].values()
                    if not len(tdag.nodes)
                ]
            )

        # Now, check the results ..............................................
        print("\nChecking results ...")

        # Should be a dict with certain specified keys
        assert isinstance(results, dict)

        if expected.get("computed_tags"):
            assert expected["computed_tags"] == list(results.keys())

        # Check more explicitly
        for tag, to_check in expected.get("results", {}).items():
            print("  Tag:  {}".format(tag))

            # Get the result for this tag
            res = results[tag]

            # Check if the type of the object is as expected; do so by string
            # comparison to avoid having to do an import here ...
            if "type" in to_check:
                assert type(res).__name__ == to_check["type"]

            # Check attribute values, calling callables
            if "attributes" in to_check:
                for attr_name, exp_attr_val in to_check["attributes"].items():
                    attr = getattr(res, attr_name)

                    if callable(attr):
                        assert attr() == exp_attr_val
                    else:
                        # Convert tuples to lists to allow yaml-comparison
                        attr = list(attr) if isinstance(attr, tuple) else attr
                        assert attr == exp_attr_val

            if "compare_to" in to_check:
                assert res == to_check["compare_to"]

        print("All computation results as expected.\n")

    # All done.


def test_TransformationDAG_specifics(dm, tmpdir):
    """Tests the TransformationDAG class."""
    TransformationDAG = dag.TransformationDAG

    # Make sure the DataManager hash is as expected
    assert dm.hashstr == "38518b2446b95e8834372949a8e9dfc2"

    # Setup the temporary cache directory
    cache_dir = str(tmpdir.join(".cache"))

    # For some specific cases, need the config parameters
    test_cfgs = load_yml(TRANSFORMATIONS_PATH)

    # Start with an empty TransformationDAG
    tdag = TransformationDAG(dm=dm, cache_dir=cache_dir)

    # Check cache dir conflicts
    os.makedirs(cache_dir, exist_ok=True)

    HASH_LEN = dag.FULL_HASH_LENGTH
    fp_foo = os.path.join(cache_dir, "a" * HASH_LEN + ".foo")
    fp_bar = os.path.join(cache_dir, "a" * HASH_LEN + ".bar")

    with open(fp_foo, mode="w") as f:
        f.write("foo")
    with open(fp_bar, mode="w") as f:
        f.write("bar")

    with pytest.raises(ValueError, match="duplicate cache file.* hash aaaaa"):
        tdag.cache_files

    os.remove(fp_foo)
    assert tdag.cache_files["a" * HASH_LEN]
    os.remove(fp_bar)

    # Check cache retrieval from pickles, where a second computation should
    # lead to cache file retrieval
    tdag = TransformationDAG(
        dm=dm,
        cache_dir=cache_dir,
        **test_cfgs["file_cache_pkl_fallback"]["params"],
    )
    tdag.compute()

    tdag = TransformationDAG(
        dm=dm,
        cache_dir=cache_dir,
        **test_cfgs["file_cache_pkl_fallback"]["params"],
    )
    results = tdag.compute()
    assert isinstance(results["arr_uint64_read"], xr.DataArray)
