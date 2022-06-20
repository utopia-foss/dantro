"""Tests features of the base class.

NOTE Some of the tests are easier to carry out with PyPlotCreator; see there.
"""

import logging
import os

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


def write_something(dm, *, out_path, **kwargs):
    """Writes the kwargs to the given path"""
    with open(out_path, "w") as f:
        f.write(str(kwargs))
    return 42


@pytest.fixture
def init_kwargs(tmpdir) -> dict:
    """Default initialisation kwargs"""
    return dict(
        dm=DataManager(data_dir=tmpdir),
        plot_func=write_something,
        default_ext="ext",
    )


# Tests -----------------------------------------------------------------------


def test_init(init_kwargs):
    """Tests initialisation"""
    # Basic case
    BasePlotCreator("init", **init_kwargs)

    # Mocked class also works
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
    assert callable(mpc.plot_func)
    assert mpc.plot_func_name == "write_something"

    # Assert setting of default extension checks work
    mpc.default_ext = "one"
    assert mpc.default_ext == "one"

    with pytest.raises(ValueError, match="Extension 'three' not supported"):
        mpc.default_ext = "three"
    assert mpc.default_ext == "one"

    # Assert the get_ext method works (does nothing here)
    assert mpc.get_ext() == mpc.default_ext


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


def test_BasePlotCreator_plot(init_kwargs, tmpdir):
    """Tests the BasePlotCreator's plot method, which also works on its own"""
    pc = BasePlotCreator("test", **init_kwargs)

    p1 = tmpdir.join("plot1")
    pc(
        out_path=p1,
        some_kwargs="foobar",
    )
    assert os.path.isfile(p1)
    with open(p1) as f:
        "some_kwargs" in f.read()

    # Can also explicitly pass a plot function
    @is_plot_func(use_dag=True)
    def my_func(*, out_path: str, data: dict, some_kwarg: int):
        assert isinstance(out_path, str)
        assert data == dict(foo="bar")
        assert some_kwarg == 123

    pc._plot_func = my_func
    pc(
        out_path=str(tmpdir.join("plot2")),
        some_kwarg=123,
        use_dag=True,
        select=dict(foo=dict(path=".", transform=[dict(define="bar")])),
    )


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
        invocation_options=dict(pass_dag_object_along=True),
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
    assert "dag" not in ds1
    assert ds1["data"] == dict()
    assert ds1["foo"] == "bar"
    assert ds1["baz"] == 123

    # ... but now!
    assert mpc.dag is not None

    # Now with some actual transformations, results are generated
    _, ds2 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params2)
    assert "use_dag" not in ds2
    assert "transform" not in ds2
    assert ds2["data"] == dict(sum=3)

    # It's possible to pass `compute_only`
    _, ds3 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params3)
    assert ds3["data"] == dict(sub=1)
    assert "transform" not in ds3
    assert "compute_only" not in ds3

    # It's possible to pass file cache default values via DAG options
    _, ds4 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params4)
    assert ds4["data"] == dict(sub=1)
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

    # --- With BasePlotCreator, without mocking
    bpc = BasePlotCreator("test", **init_kwargs)

    # Need a testing plot function and change some class variables for that
    def check_kwargs(*args, _has_dm: bool, _has_kwargs: list, **kwargs):
        if _has_dm:
            assert len(args) == 1
            dm = args[0]
            assert isinstance(dm, DataManager)
        else:
            assert not args

        for kw in _has_kwargs:
            print("Checking existence of kwarg:  ", kw)
            assert kw in kwargs

    bpc._plot_func = check_kwargs

    # Perform data selection via __call__ to test it is carried through
    assert not bpc.DAG_USE_DEFAULT

    bpc(
        out_path=tmpdir.join("foo"),
        use_dag=True,
        **params2,
        _has_dm=False,
        _has_kwargs=("data",),
    )

    bpc(
        out_path=tmpdir.join("bar"),
        use_dag=False,
        **params2,
        _has_dm=True,
        _has_kwargs=(),
    )

    bpc(out_path=tmpdir.join("spam"), **params2, _has_dm=True, _has_kwargs=())

    # And with default enabled
    bpc.DAG_USE_DEFAULT = True

    bpc(
        out_path=tmpdir.join("baz"),
        **params2,
        _has_dm=False,
        _has_kwargs=("data",),
    )

    with pytest.raises(AssertionError):
        bpc(
            out_path=tmpdir.join("spam"),
            use_dag=False,
            **params2,
            _has_dm=False,
            _has_kwargs=("data",),
        )


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
        invocation_options=dict(pass_dag_object_along=True),
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
