"""Tests the dantro.plot.utils module"""

import pytest

from dantro.plot import is_plot_func
from dantro.plot.utils.plot_func import PlotFuncResolver

from ..creators.test_base import MockPlotCreator

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def tmp_module_file(tmpdir) -> str:
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


# -----------------------------------------------------------------------------


def test_PlotFuncResolver(tmpdir):
    """Tests the PlotFuncResolver class"""
    pfr = PlotFuncResolver()
    assert pfr.base_pkg is PlotFuncResolver.BASE_PKG
    assert pfr.base_module_file_dir is None

    # Check with invalid directories
    with pytest.raises(ValueError, match="needs to be an absolute path"):
        PlotFuncResolver(base_module_file_dir="foo/bar/baz")

    with pytest.raises(
        FileNotFoundError, match="does not exists or does not point"
    ):
        PlotFuncResolver(base_module_file_dir=tmpdir.join("foo.bar"))


def test_PlotFuncResolver_resolve(tmpdir, tmp_module_file):
    """Tests whether the _resolve_plot_func works as expected"""
    pfr = PlotFuncResolver()

    # Test with valid arguments
    # Directly passing a callable should just return it
    func = lambda foo: "bar"
    assert pfr.resolve(plot_func=func) is func

    # Giving a module file should load that module. Test by calling function
    wfunc = pfr.resolve(
        module_file=tmp_module_file, plot_func="write_something"
    )
    wfunc("foo", out_path=tmpdir.join("wfunc_output")) == 42

    # ...but only for absolute paths
    with pytest.raises(ValueError, match="Need to specify `base_module_file_"):
        pfr.resolve(module_file="some/relative/path", plot_func="foobar")

    # Giving a module name works also
    assert callable(pfr.resolve(module=".basic", plot_func="lineplot"))

    # Not giving enough arguments will fail
    with pytest.raises(TypeError, match="neither argument"):
        pfr.resolve(plot_func="foo")

    # So will a plot_func of wrong type
    with pytest.raises(TypeError, match="needs to be a string or a callable"):
        pfr.resolve(plot_func=666, module="foo")

    # Can have longer plot_func modstr as well, resolved recursively
    assert callable(pfr.resolve(module=".basic", plot_func="plt.plot"))
    # NOTE that this would not work as a plot function; just for testing here


# -----------------------------------------------------------------------------


def test_is_plot_func():
    """Test the is_plot_func decorator"""
    # Needs no arguments
    @is_plot_func()
    def my_func():
        pass

    assert my_func.is_plot_func
    assert my_func.use_dag is None
    assert my_func.creator is None

    # Can take some specific ones, though, without checks
    @is_plot_func(
        creator=MockPlotCreator,
        use_dag=True,
        add_attributes=dict(foo="bar"),
    )
    def my_func2():
        pass

    assert my_func2.is_plot_func
    assert my_func2.use_dag
    assert my_func2.creator is MockPlotCreator
    assert my_func2.foo == "bar"


def test_is_plot_func_deprecations():

    with pytest.warns(DeprecationWarning, match="`creator_type`"):

        @is_plot_func(creator_type=MockPlotCreator)
        def my_func():
            pass

    with pytest.warns(DeprecationWarning, match="`creator_name`"):

        @is_plot_func(creator_name="foo")
        def my_func():
            pass

    with pytest.raises(ValueError, match="Cannot pass both .* deprecated"):

        @is_plot_func(creator_name="foo", creator_type=MockPlotCreator)
        def my_func():
            pass

    with pytest.raises(ValueError, match="Cannot pass both `creator`"):

        @is_plot_func(creator="foo", creator_type=MockPlotCreator)
        def my_func():
            pass


def test_is_plot_func_PyPlotCreator_features(tmpdir):
    """Test the is_plot_func decorator's PyPlotCreator-related features"""

    # Expanded interface
    @is_plot_func(
        helper_defaults=dict(foo="bar"),
        use_helper=True,
        supports_animation=False,
    )
    def my_func():
        pass

    assert my_func.helper_defaults == dict(foo="bar")
    assert my_func.use_helper
    assert my_func.supports_animation is False

    # Helper_defaults can be an absolute path to a file that is then loaded
    with open(tmpdir.join("foo.yml"), "a+") as f:
        f.write("{foo: spam}")

    @is_plot_func(helper_defaults=tmpdir.join("foo.yml"))
    def my_func2():
        pass

    assert isinstance(my_func2.helper_defaults, dict)
    assert my_func2.helper_defaults == dict(foo="spam")

    # User is expaned (but file is missing)
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        is_plot_func(helper_defaults="~/something/something.yml")

    # Relative path not allowed
    with pytest.raises(ValueError, match="was a relative path: some/rel"):
        is_plot_func(helper_defaults="some/relative/path")
