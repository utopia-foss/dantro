"""Tests features of the base class.

NOTE Some of the tests are easier to carry out with PyPlotCreator; see there.
"""

import logging

import pytest

from dantro.dag import TransformationDAG, _ResultPlaceholder
from dantro.data_mngr import DataManager
from dantro.exceptions import *
from dantro.plot import BasePlotCreator, is_plot_func

logging.getLogger("dantro.plot.creators.base").setLevel(12)  # remark

# Test classes ----------------------------------------------------------------


class MockPlotCreator(BasePlotCreator):
    """Test class to test the base class implementation"""

    EXTENSIONS = ("one", "two", "ext")

    def plot(self, out_path: str = None, **cfg):
        """Does nothing but writing the content of cfg to a file at out_path"""
        with open(out_path, "w") as f:
            f.write(str(cfg))


class MockPlotCreator2(MockPlotCreator):
    """Test class to test the base class implementation"""

    DEFAULT_EXT_REQUIRED = False


class MockPlotCreator3(MockPlotCreator):
    """Test class to test the base class implementation"""

    DEFAULT_EXT = "zero"


# Fixtures --------------------------------------------------------------------


@pytest.fixture
def init_kwargs(tmpdir) -> dict:
    """Default initialisation kwargs"""
    return dict(dm=DataManager(data_dir=tmpdir), default_ext="ext")


@pytest.fixture
def tmp_module(tmpdir) -> str:
    """Creates a module file in a temporary directory"""
    write_something_funcdef = (
        "def write_something(dm, *, out_path, **kwargs):\n"
        "    '''Writes the kwargs to the given path'''\n"
        "    with open(out_path, 'w') as f:\n"
        "        f.write(str(kwargs))\n"
        "    return 42\n"
    )

    path = tmpdir.join("test_module.py")
    path.write(write_something_funcdef)

    return path


# Tests -----------------------------------------------------------------------


def test_init(init_kwargs):
    """Tests initialisation"""
    # Basic case
    MockPlotCreator("init", **init_kwargs)

    # Test check for existence of default extension
    init_kwargs.pop("default_ext")
    with pytest.raises(ValueError, match="requires a default extension"):
        MockPlotCreator("init", **init_kwargs)

    # Without a required default extension, this should work:
    MockPlotCreator2("init", **init_kwargs)

    # This one should create a ValueError, as the default extension does not
    # match the supported extensions
    with pytest.raises(ValueError, match="Extension 'zero' not supported in"):
        MockPlotCreator3("init", **init_kwargs)


def test_properties(init_kwargs):
    """Tests the properties"""
    mpc = MockPlotCreator("init", **init_kwargs)

    assert mpc.name
    assert mpc.classname
    assert mpc.logstr
    assert isinstance(mpc.dm, DataManager)
    assert mpc.plot_cfg == mpc._plot_cfg
    assert not mpc.plot_cfg is mpc._plot_cfg
    assert mpc.default_ext == "ext"

    # Assert setting of default extension checks work
    mpc.default_ext = "one"
    assert mpc.default_ext == "one"

    with pytest.raises(ValueError, match="Extension 'three' not supported"):
        mpc.default_ext = "three"
    assert mpc.default_ext == "one"

    # Assert the get_ext method works (does nothing here)
    assert mpc.get_ext() == mpc.default_ext

    # The BasePlotCreator should never mark itself as `can_plot`
    assert not mpc.can_plot(
        creator_name=None,  # need to specify this kwarg
        foo="bar",
        some_plot_param=123,
    )


def test_call(init_kwargs, tmpdir):
    """Test the call to the plot creator"""
    mpc = MockPlotCreator("test", **init_kwargs)

    mpc(out_path=tmpdir.join("call0"), foo="bar")
    mpc(out_path=tmpdir.join("call1"), foo="bar")

    # Same output path should fail:
    with pytest.raises(FileExistsError, match="There already exists a"):
        mpc(out_path=tmpdir.join("call1"), foo="bar")

    # ... unless the PlotCreator was initialized with the exist_ok argument
    mpc = MockPlotCreator("test", exist_ok=True, **init_kwargs)
    mpc(out_path=tmpdir.join("call1"), foo="bar")

    # ... or skipping is desired
    with pytest.raises(SkipPlot, match="Plot output already exists"):
        mpc(out_path=tmpdir.join("call1"), foo="bar", exist_ok="skip")


def test_data_selection_interface(init_kwargs, tmpdir):
    """Tests the data selection interface, using TransformationDAG"""
    mpc = MockPlotCreator("test", **init_kwargs)

    # Has no DAG
    with pytest.raises(ValueError, match="has no TransformationDAG"):
        mpc.dag

    # Some test parameters
    params0 = dict(foo="bar", baz=123)  # mock parameters for some plot config
    params1 = dict(**params0)
    params2 = dict(
        **params1, transform=[dict(operation="add", args=[1, 2], tag="sum")]
    )
    params3 = dict(
        transform=[
            dict(operation="add", args=[1, 2], tag="sum"),
            dict(operation="sub", args=[3, 2], tag="sub"),
        ],
        compute_only=["sub"],
    )
    params4 = dict(
        **params3,
        dag_options=dict(file_cache_defaults=dict(write=False, read=True)),
    )
    params5 = dict(
        **params3,
        dag_options=dict(
            file_cache_defaults=dict(write=False, read=True),
            select_base="nonexisting",
        ),
    )
    params6 = dict(
        **params3,
        foo=dict(bar=_ResultPlaceholder("sum")),
        spam=[
            "some",
            ["deeply", dict(nested=dict(one=_ResultPlaceholder("sub")))],
        ],
    )
    params7 = dict(**params3, foo=dict(bar=_ResultPlaceholder("BAD_TAG")))

    # Disabled DAG usage -> parameters should be passed through
    flg, ds0 = mpc._perform_data_selection(use_dag=False, plot_kwargs=params0)
    assert flg is False
    assert "use_dag" not in ds0
    assert ds0 is params0

    # ... still no DAG
    with pytest.raises(ValueError, match="has no TransformationDAG"):
        mpc.dag

    # Enabled DAG usage -> DAG should be created, but of course without result
    flg, ds1 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params1)
    assert flg is True
    assert "use_dag" not in ds1
    assert isinstance(ds1["dag"], TransformationDAG)
    assert ds1["dag_results"] == dict()
    assert ds1["foo"] == "bar"
    assert ds1["baz"] == 123

    # ... but now!
    assert mpc.dag is not None

    # Now with some actual transformations, results are generated
    _, ds2 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params2)
    assert "use_dag" not in ds2
    assert "transform" not in ds2
    assert ds2["dag_results"] == dict(sum=3)

    # It's possible to pass `compute_only`
    _, ds3 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params3)
    assert ds3["dag_results"] == dict(sub=1)
    assert "transform" not in ds3
    assert "compute_only" not in ds3

    # It's possible to pass file cache default values via DAG options
    _, ds4 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params4)
    assert ds4["dag_results"] == dict(sub=1)
    assert "transform" not in ds4
    assert "compute_only" not in ds4
    assert "file_cache_defaults" not in ds4
    assert ds4["dag"]._fc_opts == dict(
        write=dict(enabled=False), read=dict(enabled=True)
    )

    # It's possible to pass parameters through to TransformationDAG. If they
    # are bad, it will fail
    with pytest.raises(KeyError, match="cannot be used to set `select_base`"):
        mpc._perform_data_selection(use_dag=True, plot_kwargs=params5)

    # Placeholders get resolved
    _, ds6 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params6)
    assert ds6["foo"]["bar"] == 3
    assert ds6["spam"][1][1]["nested"]["one"] == 1

    # ... and for bad placeholder names, the error message propagates
    with pytest.raises(ValueError, match="Some of the tags specified in"):
        mpc._perform_data_selection(use_dag=True, plot_kwargs=params7)

    # Perform data selection via __call__ to test it is carried through
    # Need to change some class variables for that
    assert mpc.DAG_INVOKE_IN_BASE
    assert mpc.DAG_SUPPORTED
    mpc(out_path=tmpdir.join("foo"), use_dag=True, **params0)


def test_dag_object_cache(init_kwargs, tmpdir):
    """Tests the caching of TransformationDAG objects via the plot creator"""
    from dantro.plot.creators.base import _DAG_OBJECT_CACHE

    assert len(_DAG_OBJECT_CACHE) == 0

    mpc = MockPlotCreator("test", **init_kwargs)

    params = dict(
        transform=[
            dict(operation="add", args=[1, 2], tag="sum"),
            dict(operation="sub", args=[3, 2], tag="sub"),
        ],
    )

    # Initial call: should write to object cache. Suppress copies to allow
    # testing for identical objects
    _, ds1 = mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(
            **params, dag_object_cache=dict(write=True, use_copy=False)
        ),
    )
    assert len(_DAG_OBJECT_CACHE) == 1
    dag1 = ds1["dag"]

    # Do the same again: this time, the DAG object should be read from cache.
    # Because copying is disabled, it will be the identical object.
    _, ds2 = mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(
            **params, dag_object_cache=dict(read=True, use_copy=False)
        ),
    )
    assert len(_DAG_OBJECT_CACHE) == 1
    dag2 = ds2["dag"]
    assert dag1 is dag2

    # If copying is enabled, it will not be identical
    _, ds3 = mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(**params, dag_object_cache=dict(read=True)),
    )
    assert len(_DAG_OBJECT_CACHE) == 1
    dag3 = ds3["dag"]
    assert dag3 is not dag1

    # After clearing, the cache will be empty again but the DAG object will
    # have been read prior to that, so it's still identical to the initial one.
    # This will also implicitly invoke garbage collection.
    _, ds4 = mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(
            **params,
            dag_object_cache=dict(read=True, clear=True, use_copy=False),
        ),
    )
    assert len(_DAG_OBJECT_CACHE) == 0
    dag4 = ds4["dag"]
    assert dag4 is dag1

    # By default, the cached object is a deep copy of the returned one, so they
    # are not identical
    _, ds5 = mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(**params, dag_object_cache=dict(write=True)),
    )
    assert len(_DAG_OBJECT_CACHE) == 1
    dag5 = ds5["dag"]
    assert dag5 is not _DAG_OBJECT_CACHE[list(_DAG_OBJECT_CACHE.keys())[0]]

    # This interface can also be used to invoke (general) garbage collection
    mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(
            **params, dag_object_cache=dict(write=True, collect_garbage=True)
        ),
    )
    assert len(_DAG_OBJECT_CACHE) == 1

    # Can also clear without collecting garbage
    mpc._perform_data_selection(
        use_dag=True,
        plot_kwargs=dict(
            **params, dag_object_cache=dict(clear=True, collect_garbage=False)
        ),
    )
    assert len(_DAG_OBJECT_CACHE) == 0


def test_resolve_plot_func(init_kwargs, tmpdir, tmp_module):
    """Tests whether the _resolve_plot_func works as expected"""
    epc = MockPlotCreator("init", **init_kwargs)

    # Make a shortcut to the function
    resolve = epc._resolve_plot_func

    # Test with valid arguments
    # Directly passing a callable should just return it
    func = lambda foo: "bar"
    assert resolve(plot_func=func) is func

    # Giving a module file should load that module. Test by calling function
    wfunc = resolve(module_file=tmp_module, plot_func="write_something")
    wfunc("foo", out_path=tmpdir.join("wfunc_output")) == 42

    # ...but only for absolute paths
    with pytest.raises(ValueError, match="Need to specify `base_module_file_"):
        resolve(module_file="some/relative/path", plot_func="foobar")

    # Giving a module name works also
    assert callable(resolve(module=".basic", plot_func="lineplot"))

    # Not giving enough arguments will fail
    with pytest.raises(TypeError, match="neither argument"):
        resolve(plot_func="foo")

    # So will a plot_func of wrong type
    with pytest.raises(TypeError, match="needs to be a string or a callable"):
        resolve(plot_func=666, module="foo")

    # Can have longer plot_func modstr as well, resolved recursively
    assert callable(resolve(module=".basic", plot_func="plt.plot"))
    # NOTE that this would not work as a plot function; just for testing here


def test_can_plot(init_kwargs, tmp_module):
    """Tests the can_plot and _valid_plot_func_signature methods"""
    epc = MockPlotCreator("can_plot", **init_kwargs)

    # Should work for the .basic lineplot, which is decorated and specifies
    # the creator type
    assert epc.can_plot("external", module=".basic", plot_func="lineplot")
    assert not epc.can_plot("foobar", module=".basic", plot_func="lineplot")

    # Cases where no plot function can be resolved
    assert not epc.can_plot("external", **{})
    assert not epc.can_plot("external", plot_func="some_func")
    assert not epc.can_plot("external", plot_func="some_func", module="foo")
    assert not epc.can_plot("external", plot_func="some_func", module_file=".")

    # Test the decorator . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Define a shortcut
    def declared_pf_by_attrs(func, pc=epc, creator_name="external"):
        return pc._declared_plot_func_by_attrs(func, creator_name)

    # Define some functions to test
    @is_plot_func(creator_name="external")
    def pfdec_name():
        pass

    @is_plot_func(creator_type=MockPlotCreator)
    def pfdec_type():
        pass

    @is_plot_func(creator_type=MockPlotCreator, creator_name="foo")
    def pfdec_type_and_name():
        pass

    @is_plot_func(creator_type=MockPlotCreator2)
    def pfdec_subtype():
        pass

    @is_plot_func(creator_name="base")
    def pfdec_subtype_name():
        pass

    @is_plot_func(creator_type=int)
    def pfdec_bad_type():
        pass

    @is_plot_func(creator_name="i_do_not_exist")
    def pfdec_bad_name():
        pass

    assert declared_pf_by_attrs(pfdec_name)
    assert declared_pf_by_attrs(pfdec_type)
    assert declared_pf_by_attrs(pfdec_type_and_name)
    assert not declared_pf_by_attrs(pfdec_subtype)
    assert not declared_pf_by_attrs(pfdec_subtype_name)
    assert not declared_pf_by_attrs(pfdec_bad_type)
    assert not declared_pf_by_attrs(pfdec_bad_name)

    # Also test for a derived class
    mpc2 = MockPlotCreator2("can_plot", **init_kwargs)

    assert not declared_pf_by_attrs(pfdec_name, mpc2, "base")
    assert declared_pf_by_attrs(pfdec_type, mpc2, "base")
    assert declared_pf_by_attrs(pfdec_type_and_name, mpc2, "base")
    assert declared_pf_by_attrs(pfdec_subtype, mpc2, "base")
    assert declared_pf_by_attrs(pfdec_subtype_name, mpc2, "base")
    assert not declared_pf_by_attrs(pfdec_bad_type, mpc2, "base")
    assert not declared_pf_by_attrs(pfdec_bad_name, mpc2, "base")
