"""Tests features of the base class."""

import pytest

from dantro.data_mngr import DataManager
from dantro.plot_creators import BasePlotCreator
from dantro.dag import TransformationDAG

# Test classes ----------------------------------------------------------------

class MockPlotCreator(BasePlotCreator):
    """Test class to test the base class implementation"""

    EXTENSIONS = ("one", "two", "ext")

    def plot(self, out_path: str=None, **cfg):
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


def test_call(init_kwargs, tmpdir):
    """Test the call to the plot creator"""
    mpc = MockPlotCreator("test", **init_kwargs)

    mpc(out_path=tmpdir.join("call0"), foo="bar")
    mpc(out_path=tmpdir.join("call1"), foo="bar")

    # Same output path should fail:
    with pytest.raises(FileExistsError, match="There already exists a"):
        mpc(out_path=tmpdir.join("call1"), foo="bar")

def test_data_selection_interface(init_kwargs, tmpdir):
    """Tests the data selection interface"""
    mpc = MockPlotCreator("test", **init_kwargs)

    # Some test parameters
    params0 = dict(foo="bar", baz=123)  # mock parameters for some plot config
    params1 = dict(**params0)
    params2 = dict(**params1,
                   transform=[dict(operation='add', args=[1,2], tag='sum')])
    params3 = dict(transform=[dict(operation='add', args=[1,2], tag='sum'),
                              dict(operation='sub', args=[3,2], tag='sub')],
                   compute_only=['sub'])
    params4 = dict(**params3,
                   dag_options=dict(file_cache_defaults=dict(write=False,
                                                             read=True)))
    params5 = dict(**params3,
                   dag_options=dict(file_cache_defaults=dict(write=False,
                                                             read=True),
                                    select_base='nonexisting'))

    # Disabled DAG usage -> parameters should be passed through
    ds0 = mpc._perform_data_selection(use_dag=False, plot_kwargs=params0)
    assert 'use_dag' not in ds0
    assert ds0 == params0
    assert ds0 is not params0

    # Enabled DAG usage -> DAG should be created, but of course without result
    ds1 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params1)
    assert 'use_dag' not in ds1
    assert isinstance(ds1['dag'], TransformationDAG)
    assert ds1['dag_results'] == dict()
    assert ds1['foo'] == "bar"
    assert ds1['baz'] == 123

    # Now with some actual transformations, results are generated
    ds2 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params2)
    assert 'use_dag' not in ds2
    assert 'transform' not in ds2
    assert ds2['dag_results'] == dict(sum=3)

    # It's possible to pass `compute_only`
    ds3 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params3)
    assert ds3['dag_results'] == dict(sub=1)
    assert 'transform' not in ds3
    assert 'compute_only' not in ds3

    # It's possible to pass file cache default values via DAG options
    ds4 = mpc._perform_data_selection(use_dag=True, plot_kwargs=params4)
    assert ds4['dag_results'] == dict(sub=1)
    assert 'transform' not in ds4
    assert 'compute_only' not in ds4
    assert 'file_cache_defaults' not in ds4
    assert ds4['dag']._fc_opts == dict(write=False, read=True)

    # It's possible to pass parameters through to TransformationDAG. If they
    # are bad, it will fail
    with pytest.raises(KeyError,
                       match="cannot be the basis of future select operation"):
        mpc._perform_data_selection(use_dag=True, plot_kwargs=params5)
