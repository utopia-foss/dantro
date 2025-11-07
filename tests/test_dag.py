"""Tests the utils.dag module"""

import copy
import os
import sys
import time
import timeit
from builtins import *  # to have Exception types available in globals
from typing import Any

import dill
import networkx as nx
import numpy as np
import pytest
import xarray as xr

import dantro
import dantro._dag_utils as dag_utils
import dantro.dag as dag
from dantro import DataManager
from dantro._hash import _hash
from dantro._import_tools import get_resource_path
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
from dantro.data_ops import is_operation
from dantro.exceptions import *
from dantro.groups import OrderedDataGroup
from dantro.tools import load_yml, write_yml
from dantro.utils.nx import ATTR_MAPPER_OP_PREFIX_DAG

from . import ON_WINDOWS

# Test files
DAG_SYNTAX_PATH = get_resource_path("tests", "cfg/dag_syntax.yml")
TRANSFORMATIONS_PATH = get_resource_path("tests", "cfg/dag.yml")  # life-cycle
DAG_NX_PATH = get_resource_path("tests", "cfg/dag_nx.yml")

# Class Definitions -----------------------------------------------------------

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


def create_dm():
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
    elephant_f = lambda t: 1.6 + 0.01 * t - 0.001 * t**2 + 2.3e-5 * t**3
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

    return _dm


@pytest.fixture
def dm_silent() -> FullDataManager:
    return create_dm()


@pytest.fixture
def dm(dm_silent):
    print(dm_silent.tree)
    return dm_silent


def yaml_roundtrip(obj: Any, *, path: str) -> Any:
    """Makes a YAML roundtrip to the given path"""
    from dantro._yaml import load_yml, write_yml

    write_yml(obj, path=path)
    return load_yml(path=path)


# -----------------------------------------------------------------------------


def test_hash():
    """Test that the hash function did not change"""
    fixed_hash = "cac42c9aeca87793905d257c1b1b89b8"
    assert _hash("I will not change.") == fixed_hash

    # Also works with byte strings
    assert _hash(b"I will not change.") == fixed_hash


def test_deepcopy(dm):
    """Tests the custom deepcopy function, based on pickle"""
    deepcopy = dag._deepcopy
    assert deepcopy("foo") == "foo"
    assert deepcopy(dict(foo=dict(bar="spam"))) == dict(foo=dict(bar="spam"))

    # also works with complex objects
    assert deepcopy(int)
    assert deepcopy(("some", "mixed", list, "objects", 123, _hash))
    assert deepcopy(dag_utils.Placeholder("foo bar"))
    assert deepcopy(DataManager)
    assert deepcopy(dag.Transformation)
    assert deepcopy(lambda x: x**2)

    with pytest.raises(RuntimeError):
        assert deepcopy(dm)

    del dm["bad_objects"]
    assert "bad_objects" not in dm
    assert deepcopy(dm)

    # ... and local objects as well as modules, which will trigger the fallback
    assert copy.deepcopy(lambda x: "foo")
    assert deepcopy(lambda x: "foo")  # will fall back to copy.deepcopy

    # The function should not work in places where copy.deepcopy also fails
    with pytest.raises(TypeError):
        copy.deepcopy(time)  # module
    with pytest.raises(TypeError):
        deepcopy(time)

    # The pickle-based function is significantly faster than regular deep copy.
    # For comparison, also include dill:
    large_list = list(range(100000))

    dill_deepcopy = lambda obj: dill.loads(dill.dumps(obj))

    t0 = timeit.default_timer()
    deepcopy(large_list)
    t1 = timeit.default_timer()
    copy.deepcopy(large_list)
    t2 = timeit.default_timer()
    dill_deepcopy(large_list)
    t3 = timeit.default_timer()

    dt = dict()
    dt["pickle"] = t1 - t0
    dt["regular"] = t2 - t1
    dt["dill"] = t3 - t2
    print("times and speedups (vs regular):")
    for name, _dt in dt.items():
        print(f"  {name:>10s}:  {_dt:.3g}s\t({dt['regular'] / _dt:.3g}x)")

    # For Python 3.14, this is not yet optimized ...
    # We can be happy if it is not slower.
    # TODO Remove this once it is optimized!
    if sys.version_info.minor >= 14:
        assert dt["regular"] / dt["pickle"] > 1.0
        assert dt["dill"] / dt["pickle"] > 1.0
    else:
        assert dt["regular"] / dt["pickle"] > 5
        assert dt["dill"] / dt["pickle"] > 10


def test_Placeholder(tmpdir):
    """Test the Placeholder class"""
    ph = dag_utils.Placeholder("foo bar")

    assert ph == dag_utils.Placeholder("foo bar")
    assert ph != dag_utils.Placeholder("bar foo")

    assert "Placeholder" in repr(ph)
    assert "'foo bar'" in repr(ph)

    # String formatting
    assert str(ph) == "<Placeholder, payload: 'foo bar'>"
    assert ph.PAYLOAD_DESC in str(ph)

    # ... can also be adjusted in subclasses
    class MyPlaceholder(dag_utils.Placeholder):
        PAYLOAD_DESC = "content"

        def _format_payload(self):
            return str(self._data)

    mph = MyPlaceholder("foo bar baz")
    assert str(mph) == "<MyPlaceholder, content: foo bar baz>"

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

    # YAML representation
    from dantro._yaml import yaml, yaml_dumps

    load_str = lambda s: yaml.load(s)
    dump = lambda obj: yaml_dumps(obj, yaml_obj=yaml)
    roundtrip = lambda obj: load_str(dump(obj))

    assert load_str("!arg 0") == Arg(0)
    assert load_str("!kwarg foo") == Kwarg("foo")
    assert roundtrip(Arg(123)) == Arg(123)
    assert roundtrip(Kwarg("foobar")) == Kwarg("foobar")


def test_argument_placeholders_fallback():
    """Tests the ability to store an arbitrary fallback *value*"""
    Arg = dag_utils.PositionalArgument
    Kwarg = dag_utils.KeywordArgument

    assert Arg(0, "zero").fallback == "zero"
    assert Arg(10, "ten").fallback == "ten"
    assert Arg("3", 3).fallback == 3

    assert Kwarg("foo", "bar").fallback == "bar"
    assert Kwarg("foo", None).fallback is None
    assert Kwarg("foo", None).fallback is None

    # Whether there was a fallback at all
    a = Arg(0)
    assert not a.has_fallback
    with pytest.raises(ValueError, match="has no fallback value defined"):
        a.fallback

    a2 = Arg(0, 1234)
    assert a2.has_fallback
    assert a2.fallback == 1234

    # Errors
    with pytest.raises(TypeError, match="only accepts a single"):
        Arg(1, "abc", 123)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Arg(1, "def", 123, 4, 5, bar="baz")

    with pytest.raises(TypeError, match="only accepts a single"):
        Kwarg("one", "abc", 123)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Kwarg("one", fallback="baz")

    # String representation
    with_fallback = Arg(42, 1234)
    with_None_fallback = Arg(42, None)
    without_fallback = Arg(42)

    assert "fallback: 1234" in str(with_fallback)
    assert "fallback" not in str(without_fallback)

    # Hashing
    assert hash(with_fallback) != hash(without_fallback)
    assert hash(with_fallback) != hash(with_None_fallback)

    # YAML representation
    import ruamel.yaml

    from dantro._yaml import yaml, yaml_dumps

    load_str = lambda s: yaml.load(s)
    dump = lambda obj: yaml_dumps(obj, yaml_obj=yaml)
    roundtrip = lambda obj: load_str(dump(obj))

    a3 = load_str("!arg [123, bar]")
    assert a3 == Arg(123, "bar")

    k1 = load_str("!kwarg [foo, bar]")
    print(k1)
    assert k1.name == "foo"
    assert k1.fallback == "bar"
    assert dump(k1) == "!kwarg [foo, bar]\n"
    assert roundtrip(k1) == k1

    k1._has_fallback = False
    assert roundtrip(k1) == k1
    assert "!kwarg foo" in dump(k1)
    assert "bar" not in dump(k1)

    with pytest.raises(TypeError, match="only accepts a single fallback"):
        load_str("!arg [1, bar, baz]")

    with pytest.raises(TypeError, match="only accepts a single fallback"):
        load_str("!kwarg [foo, bar, baz]")

    with pytest.raises(ruamel.yaml.constructor.ConstructorError):
        load_str("!kwarg {foo: bar}")


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

    # String representation contains shortened hash
    assert some_hash[:12] in str(ref)
    assert some_hash[:13] not in str(ref)

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

    assert f"tag: {some_tag}" in str(tag)

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
    assert f"node ID: {some_node}" in str(node)

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

    # Can query whether there are cached results
    assert t0.has_result
    assert not t1.has_result
    assert not t2.has_result
    assert not tfail.has_result

    # Read the profile property
    assert isinstance(t0.profile, dict)

    # Can pass some context, which is NOT part of the hash
    t0c = Transformation(
        operation="add", args=[1, 2], kwargs=dict(), context=dict(foo="bar")
    )
    assert t0.hashstr == t0c.hashstr
    assert t0.context != t0c.context

    # Serialize as yaml
    get_yaml = lambda o: yaml_dumps(o, register_classes=(Transformation,))
    assert "!dag_trf" in get_yaml(t0)
    assert "salt" not in get_yaml(t0)
    assert "salt: 42" in get_yaml(t0s)

    # Context can be passed on to Transformation's YAML representation
    t4 = Transformation(
        operation="add",
        args=[1, 2],
        kwargs=dict(),
        context=dict(foobar="my_custom_context"),
    )
    assert "foobar: my_custom_context" in get_yaml(t4)

    # Get layer
    assert t0.layer == 0
    assert t0s.layer == 0
    assert t1.layer == 0
    assert t2.layer == 0
    assert t3.layer == 0

    # Status
    assert t0.status == "computed"
    assert t1.status == "initialized"
    assert t2.status == "initialized"
    assert t3.status == "initialized"


def test_Transformation_status():
    """Tests Transformation.status property (as far as that's possible in a
    standalone scenario.
    """
    Transformation = dag.Transformation

    t0 = Transformation(operation="add", args=[1, 2], kwargs=dict())
    assert t0.status == "initialized"

    # Compute it
    t0.compute()
    assert t0.status == "computed"

    # Failure is correctly associated
    t0f = Transformation(operation="add", args=[1, "baz"], kwargs=dict())
    with pytest.raises(DataOperationFailed):
        t0f.compute()
    assert t0f.status == "failed_here"

    # Cannot set it to arbitrary values
    with pytest.raises(ValueError, match="Invalid status"):
        t0f.status = "foobar"


def test_Transformation_fallback():
    """Tests the fallback feature of the Transformation class"""
    Transformation = dag.Transformation

    # Without fallback
    t0 = Transformation(operation="add", args=[1, 2], kwargs=dict())
    assert t0.hashstr == "23cf81f382bd65f15f9e22ab80923a3b"
    assert hash(t0.hashstr) == hash(t0)

    assert "operation: add, 2 args, 0 kwargs" in str(t0)
    expected_repr = (
        "<dantro.dag.Transformation, operation='add', "
        "args=[1, 2], kwargs={}, salt=None>"
    )
    assert repr(t0) == expected_repr

    # With fallback: hash and repr changes
    t1 = Transformation(
        operation="add",
        args=[1, 2],
        kwargs=dict(),
        allow_failure=True,
        fallback=3,
    )
    assert t1.hashstr == "2c04f856d659dd1a70d75770e4e11610"
    assert t1.status == "initialized"
    assert hash(t1.hashstr) == hash(t1)

    assert "operation: add, 2 args, 0 kwargs, allows failure" in str(t1)
    expected_repr = (
        "<dantro.dag.Transformation, operation='add', "
        "args=[1, 2], kwargs={}, salt=None, fallback=3>"
    )
    assert repr(t1) == expected_repr

    # Initialization has requirements
    with pytest.raises(ValueError, match="may only be passed with"):
        Transformation(operation="add", args=[1, 2], kwargs=dict(), fallback=3)

    with pytest.raises(ValueError, match="Invalid.*Choose from"):
        Transformation(
            operation="add", args=[1, 2], kwargs=dict(), allow_failure="foo"
        )

    # Can use a trivial fallback
    t2 = Transformation(
        operation="div",
        args=[1, 0],
        kwargs=dict(),
        allow_failure=True,
        fallback=np.inf,
    )
    assert t2.compute() == np.inf
    assert t2.status == "used_fallback"

    # ... on an operation that would fail without fallback
    t2_fail = Transformation(operation="div", args=[1, 0], kwargs=dict())
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        t2_fail.compute()
    assert t2_fail.status == "failed_here"

    # Can get a warning when using the fallback
    t3 = Transformation(
        operation="div",
        args=[1, 0],
        kwargs=dict(),
        fallback=3,
        allow_failure="warn",
    )
    with pytest.warns(DataOperationWarning, match="ZeroDivisionError"):
        t3.compute()
    assert t3.status == "used_fallback"

    # YAML serialization includes fallback
    get_yaml = lambda o: yaml_dumps(o, register_classes=(Transformation,))
    assert "fallback: 3" in get_yaml(t1)
    assert "allow_failure: true" in get_yaml(t1)
    assert "allow_failure: warn" in get_yaml(t3)


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
        print(f"Testing transformation syntax case '{name}' ...")

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
            Exception if not isinstance(_raises, str) else globals()[_raises]
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
        print(f"Testing transformation DAG case '{name}' ...")

        if cfg.get("skip_on_windows", False) and ON_WINDOWS:
            print("Skipping this test case on Windows system.\n")
            continue

        # Extract arguments
        params = cfg["params"]
        expected = cfg.get("expected", {})

        # Error checking arguments
        _raises = cfg.get("_raises", False)
        _raises_on_compute = cfg.get("_raises_on_compute", False)
        _exp_exc = (
            Exception if not isinstance(_raises, str) else globals()[_raises]
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
                    assert {r.ref for r in node.dependencies} == set(deps)

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
        assert sum(len(stack) for stack in tdag.ref_stacks.values()) == 0

        # Check compute_only argument
        compute_only = cfg.get("compute_only")
        print(
            "\nComputing results (compute_only argument: {}) ...".format(
                compute_only
            )
        )

        if compute_only in (None, "all"):
            to_compute = tdag._parse_compute_only(compute_only)
            assert isinstance(to_compute, list)
            assert all(isinstance(t, str) for t in to_compute)

            # There shouldn't be private tags in there, nor special tags
            assert not [t for t in to_compute if t.startswith(".")]
            assert not [t for t in to_compute if t.startswith("_")]
            assert not [t for t in to_compute if t in tdag.SPECIAL_TAGS]

        # Compute the results
        if not _raises or not _raises_on_compute:
            # Compute normally
            results = tdag.compute(compute_only=compute_only)

            print(
                "\n".join([f"  * {k:<20s}  {v}" for k, v in results.items()])
            )

        else:
            with pytest.raises(_exp_exc, match=_match):
                tdag.compute(compute_only=compute_only)

            print("Raised error as expected.\n")
            continue

        # Cache directory MAY exist after computation
        if not os.path.isdir(cache_dir):
            print("\nCache directory not available.")
        else:
            print(f"\nContent of cache directory ({cache_dir})")
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
            print(f"  Tag:  {tag}")

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

            if to_check.get("find_tag"):
                ref = tdag.tags[tag]
                assert tdag._find_tag(ref)
                assert tdag._find_tag(tdag.objects[ref])

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
    assert isinstance(results["arr_read"], xr.DataArray)


# -----------------------------------------------------------------------------
# networkx Graph-related stuff


def test_generate_nx_graph(dm_silent):
    """Tests networkx Graph generation from a TransformationDAG"""
    dm = dm_silent
    TransformationDAG = dag.TransformationDAG

    test_cfgs = load_yml(DAG_NX_PATH)

    # ... Empty ...............................................................
    tdag = TransformationDAG(dm=dm, **test_cfgs["empty"])
    assert len(tdag.nodes) == 0

    g = tdag.generate_nx_graph()
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0

    tdag.compute()

    # Different arguments for tags to include
    assert tdag.generate_nx_graph(tags_to_include="all").number_of_nodes() == 0
    assert tdag.generate_nx_graph(tags_to_include=[]).number_of_nodes() == 0
    assert tdag.generate_nx_graph(tags_to_include=None).number_of_nodes() == 0

    # ... Simple ..............................................................
    tdag = TransformationDAG(dm=dm, **test_cfgs["simple"])
    assert len(tdag.nodes) == 3

    g = tdag.generate_nx_graph()
    assert g.number_of_nodes() == 3 + 1  # data manager dependency
    assert g.number_of_edges() == 3

    for node, data in g.nodes(data=True):
        print(f"- {node}:\n    {data}\n")

    for src, target in g.edges():
        print(f"{src}  ->  {target}")

    # In this case, check the graph structure manually
    assert g.nodes()[dm.hashstr]["obj"] is dm

    foo_dep = "275d726a23c0c8d1becb5a4798f25336"
    assert g.nodes()[tdag.tags["foo"]]["obj"]._args[0].ref == foo_dep
    assert g.nodes()[tdag.tags["foo"]]["obj"]._args[1] == "foo"

    assert g.nodes()[foo_dep]["obj"]._args[0].ref == dm.hashstr
    assert g.nodes()[foo_dep]["obj"]._args[1] == "some/path"

    assert g.nodes()[tdag.tags["bar"]]["obj"]._args[0].ref == dm.hashstr
    assert g.nodes()[tdag.tags["bar"]]["obj"]._args[1] == "some/path/bar"

    # Check that it computes and that results are available afterwards
    tdag.compute()
    assert g.nodes()[tdag.tags["bar"]]["obj"].has_result

    # ... With select and define ..............................................
    tdag = TransformationDAG(dm=dm, **test_cfgs["with_select_and_define"])

    # Reduced number of nodes if only selecting specific tags
    assert tdag.generate_nx_graph().number_of_nodes() == 23  # all
    assert tdag.generate_nx_graph(tags_to_include=[]).number_of_nodes() == 0
    assert (
        tdag.generate_nx_graph(tags_to_include=["foo"]).number_of_nodes() == 2
    )

    # Check length of some paths
    g = tdag.generate_nx_graph()

    dm_node = dm.hashstr
    fifty_node = tdag.tags["fifty"]
    foo_node = tdag.tags["foo"]

    assert len(nx.shortest_path(g, dm_node, fifty_node)) == 11
    assert len(nx.shortest_path(g, dm_node, foo_node)) == 2

    # ... cannot go the reverse way
    with pytest.raises(nx.exception.NetworkXNoPath):
        nx.shortest_path(g, fifty_node, dm_node)

    with pytest.raises(nx.exception.NetworkXNoPath):
        nx.shortest_path(g, foo_node, dm_node)

    # Have result available after computation
    assert not g.nodes()[fifty_node]["obj"].has_result
    tdag.compute()
    assert g.nodes()[fifty_node]["obj"].has_result

    # Can also include that information into the node attributes
    g = tdag.generate_nx_graph(
        tags_to_include=("fifty",), include_results=True
    )
    assert g.nodes()[fifty_node]["has_result"]
    assert g.nodes()[fifty_node]["result"] == 50

    assert not g.nodes()[dm.hashstr]["has_result"]
    assert g.nodes()[dm.hashstr]["result"] is None

    # The foo_node should not be in the graph
    assert foo_node not in g.nodes()

    # What about reverse edges, pointing towards dependencies?
    g = tdag.generate_nx_graph(edges_as_flow=False)

    assert len(nx.shortest_path(g, fifty_node, dm_node)) == 11
    assert len(nx.shortest_path(g, foo_node, dm_node)) == 2

    with pytest.raises(nx.exception.NetworkXNoPath):
        nx.shortest_path(g, dm_node, fifty_node)

    with pytest.raises(nx.exception.NetworkXNoPath):
        nx.shortest_path(g, dm_node, foo_node)


def test_generate_nx_graph_attr_mapping(dm_silent):
    """Tests node attribute mapping"""
    dm = dm_silent
    TransformationDAG = dag.TransformationDAG
    test_cfgs = load_yml(DAG_NX_PATH)

    tdag = TransformationDAG(dm=dm, **test_cfgs["with_select_and_define"])

    dm_node = dm.hashstr
    fifty_node = tdag.tags["fifty"]
    foo_node = tdag.tags["foo"]
    foo_path_node = tdag.tags["foo_path"]

    g = tdag.generate_nx_graph()

    # Labelled nodes should always have a tag
    assert g.nodes()[dm_node]["tag"] == "dm"
    assert g.nodes()[fifty_node]["tag"] == "fifty"
    assert g.nodes()[foo_node]["tag"] == "foo"

    # Now again and with some mappers
    prefix = ATTR_MAPPER_OP_PREFIX_DAG
    assert prefix == "attr_mapper.dag"

    @is_operation("my_test_operation")
    def my_operation(foo, *, bar, attrs):
        assert "tag" in attrs
        assert "obj" in attrs
        return f"{foo} :: {bar}"

    mappers = dict()

    # Default mappers: operation, layer, description
    # Add only additional mappers
    mappers["arguments"] = f"{prefix}.format_arguments"
    mappers["disabled"] = None  # --> not carried out
    mappers["my_attr"] = {
        f"my_test_operation": ["foo"],
        "kwargs": dict(bar="bar"),
    }

    g = tdag.generate_nx_graph(
        tags_to_include=("fifty", "foo_path"),
        manipulate_attrs=dict(
            map_node_attrs=mappers,
            keep_node_attrs=True,
        ),
        lookup_tags=False,
    )

    assert g.nodes()[fifty_node]["tag"] == "fifty"
    assert g.nodes()[fifty_node]["operation"] == "pass"
    assert g.nodes()[fifty_node]["layer"] == 11
    assert "pass" in g.nodes()[fifty_node]["description"]
    assert "fifty" in g.nodes()[fifty_node]["description"]
    assert "hash: 6d27ed1726a2…" in g.nodes()[fifty_node]["arguments"]
    assert g.nodes()[fifty_node]["my_attr"] == "foo :: bar"
    with pytest.raises(KeyError):
        g.nodes()[fifty_node]["disabled"]

    # DataManager should never have a value, but the attribute should be set
    # and there should be a tag and layer.
    assert g.nodes()[dm.hashstr]["tag"] == "dm"
    assert g.nodes()[dm.hashstr]["layer"] == 0
    assert "dm" in g.nodes()[dm.hashstr]["description"]
    assert g.nodes()[dm.hashstr]["operation"] == ""
    assert g.nodes()[dm.hashstr]["arguments"] == ""
    assert g.nodes()[dm.hashstr]["my_attr"] == "foo :: bar"
    with pytest.raises(KeyError):
        g.nodes()[dm.hashstr]["disabled"]

    # The foo_node should still be in the graph ...
    assert foo_node in g.nodes()
    assert foo_path_node in g.nodes()

    assert g.nodes()[foo_node]["tag"] is None  # because lookup_tags == False
    assert g.nodes()[foo_path_node]["tag"] == "foo_path"
    assert "foo_path" in g.nodes()[foo_path_node]["description"]

    # Error
    mappers["some_attr"] = "bad_operation"
    with pytest.raises(BadOperationName, match="Failed mapping node"):
        tdag.generate_nx_graph(manipulate_attrs=dict(map_node_attrs=mappers))


def test_visualize_DAG(dm_silent, tmpdir):
    """Tests visualization"""
    dm = dm_silent
    TransformationDAG = dag.TransformationDAG
    test_cfgs = load_yml(DAG_NX_PATH)

    tdag = TransformationDAG(dm=dm, **test_cfgs["with_select_and_define"])

    # This should just work
    out1 = tmpdir.join("out1.pdf")
    tdag.visualize(out_path=out1)
    assert os.path.isfile(out1)

    # Can also pass a graph explicitly
    g = tdag.generate_nx_graph()
    out2 = tmpdir.join("out2.pdf")
    tdag.visualize(out_path=out2, g=g)
    assert os.path.isfile(out2)

    # ... but then cannot pass generation arguments
    with pytest.raises(ValueError, match="argument is not allowed"):
        tdag.visualize(out_path=out2, g=g, generation=dict(foo="bar"))

    # More arguments
    out3 = tmpdir.join("out3.pdf")
    tdag.visualize(
        out_path=out3,
        generation=dict(include_results=False),
        drawing=dict(nodes=dict(node_color="b")),
        use_defaults=True,
        scale_figsize=True,
        figure_kwargs=dict(dpi=72),
        save_kwargs=dict(bbox_inches="tight"),
    )
    assert os.path.isfile(out3)
