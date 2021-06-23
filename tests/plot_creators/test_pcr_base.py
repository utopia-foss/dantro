"""Tests features of the base class."""

import pytest

from dantro.dag import TransformationDAG, _ResultPlaceholder
from dantro.data_mngr import DataManager
from dantro.exceptions import *
from dantro.plot_creators import BasePlotCreator

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
    """Tests the data selection interface"""
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
    assert not mpc.DAG_SUPPORTED
    mpc.DAG_SUPPORTED = True
    mpc(out_path=tmpdir.join("foo"), use_dag=True, **params0)
