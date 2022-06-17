"""Tests the dantro.plot.utils module"""

import pytest

from dantro.plot import is_plot_func

from .creators.test_base import MockPlotCreator

# -----------------------------------------------------------------------------


def test_is_plot_func():
    """Test the is_plot_func decorator"""
    # Needs no arguments
    @is_plot_func()
    def my_func():
        pass

    assert my_func.is_plot_func
    assert my_func.use_dag is None
    assert my_func.creator_type is None
    assert my_func.creator_name is None

    # Can take some specific ones, though, without checks
    @is_plot_func(
        creator_type=MockPlotCreator,
        creator_name="base",
        use_dag=True,
        add_attributes=dict(foo="bar"),
    )
    def my_func2():
        pass

    assert my_func2.is_plot_func
    assert my_func2.use_dag
    assert my_func2.creator_type is MockPlotCreator
    assert my_func2.creator_name == "base"
    assert my_func2.foo == "bar"


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
