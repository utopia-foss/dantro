"""Tests the internal :py:mod:`dantro._import_tools` module"""

import copy
import importlib
import os
import sys

import pytest

import dantro._import_tools as it

# Fixtures --------------------------------------------------------------------


@pytest.fixture
def with_tmp_sys_path():
    """Work on a temporary sys.path .. of course this mirrors the functionality
    of _import_tools.temporary_sys_path, but is much simpler to do here in the
    fixture.
    """
    initial_sys_path = copy.copy(sys.path)
    yield

    sys.path = initial_sys_path


# -----------------------------------------------------------------------------


def test_remove_from_sys_modules():

    import email
    import email.charset
    import email.encoders

    assert "email" in sys.modules
    assert "email.encoders" in sys.modules
    assert "email.charset" in sys.modules

    it.remove_from_sys_modules(lambda m: m.startswith("email.encoders"))
    assert "email.encoders" not in sys.modules
    assert "email" in sys.modules
    assert "email.charset" in sys.modules

    it.remove_from_sys_modules(lambda m: m.startswith("email"))
    assert "email.encoders" not in sys.modules
    assert "email" not in sys.modules
    assert "email.charset" not in sys.modules


def test_added_sys_path(with_tmp_sys_path):
    initial_sys_path = copy.deepcopy(sys.path)

    p = "/foo"
    with it.added_sys_path(p):
        assert p in sys.path
        assert sys.path != initial_sys_path

    assert p not in sys.path
    assert sys.path == initial_sys_path

    # Can also add something within the context that will not be removed
    with it.added_sys_path(p):
        assert p in sys.path

        sys.path.append(p * 2)
        sys.path.insert(0, p * 3)
    assert p not in sys.path
    assert p * 2 in sys.path
    assert p * 3 in sys.path

    # If the path already exists, nothing changes and it remains in there
    sys.path.insert(0, p)
    assert p in sys.path

    with it.added_sys_path(p):
        assert p in sys.path

        # Can still add something here that is not removed afterwards
        sys.path.insert(0, p * 4)
    assert p in sys.path
    assert p * 4 in sys.path


def test_temporary_modules_cache():
    # For testing, use the math module

    import math

    assert "math" in sys.modules

    # Need a module that is not cached; make sure of it
    it.remove_from_sys_modules(lambda m: m.startswith("math"))
    assert "math" not in sys.modules
    initial_sys_modules = copy.copy(sys.modules)

    with it.temporary_sys_modules():
        assert "math" not in sys.modules
        sys.modules["math"] = importlib.import_module("math")
        assert "math" in sys.modules

    # Should not be there after leaving the context
    assert "math" not in sys.modules

    # Doing nothin in the context
    with it.temporary_sys_modules():
        pass

    # Unless configured in a way that the cache is only reset if there was an
    # error in the context
    with it.temporary_sys_modules(reset_only_on_fail=True):
        assert "math" not in sys.modules
        sys.modules["math"] = importlib.import_module("math")
        assert "math" in sys.modules
    assert "math" in sys.modules

    it.remove_from_sys_modules(lambda m: m.startswith("math"))
    with pytest.raises(ImportError, match="imagine a failure"):
        with it.temporary_sys_modules(reset_only_on_fail=True):
            assert "math" not in sys.modules
            sys.modules["math"] = importlib.import_module("math")
            assert "math" in sys.modules
            raise ImportError("imagine a failure")

    assert "math" not in sys.modules


# -----------------------------------------------------------------------------


def test_get_from_module():
    """Test the get_from_module function"""
    import dantro
    import dantro.plot

    assert it.get_from_module(dantro, name="plot") is dantro.plot
    assert (
        it.get_from_module(dantro, name="plot.plot_helper")
        is dantro.plot.plot_helper
    )

    with pytest.raises(AttributeError, match="Failed to retrieve"):
        it.get_from_module(dantro, name="plot.plot_helper.foo_bar")


def test_import_module_or_object():
    """Tests the import_module_or_object function"""
    _import = it.import_module_or_object

    # No module given: import from builtins
    assert _import(name="str") is str

    # No name given: import module
    import math

    assert _import(module="math").sqrt(9) == 3

    # Relative module given: imports (by default) from dantro; but can also
    # adjust the default
    import dantro
    import dantro.plot.plot_helper as dph

    assert _import(".plot.plot_helper") is dph
    assert _import(".plot_helper", package="dantro.plot") is dph

    # Can also retrieve a name
    assert _import(".plot.plot_helper", "PlotHelper") is dph.PlotHelper
    assert _import(".plot.plot_helper", "PlotHelper.fig") is dph.PlotHelper.fig


def test_import_name():
    """Can also import directly by name"""
    import dantro.plot.plot_helper as dph

    _import = it.import_name

    assert _import("dantro.plot.plot_helper") is dph
    assert _import(".plot.plot_helper") is dph
    assert _import(".plot.plot_helper.PlotHelper") is dph.PlotHelper

    # Fails for nested names
    with pytest.raises(ModuleNotFoundError, match="No module"):
        _import("dantro.plot.plot_helper.PlotHelper.fig")


def test_import_module_from_path(tmpdir):
    """Test importing a module from a directory"""
    # Write a test module
    test_mod = (
        "foo = 'bar'\n"
        "spam = 123\n"
        "\n"
        "def my_func() -> str:\n"
        "    return foo\n"
        "\n"
        "class SomeClass:\n"
        "    def __init__(self, *args):\n"
        "        self.args = args\n"
        "        self.foo = foo\n"
        "        self.spam = spam\n"
    )
    test_mod_dir = tmpdir.mkdir("test_mod")
    test_module_file = str(test_mod_dir.join("__init__.py"))
    with open(test_module_file, "w") as f:
        f.write(test_mod)

    # Now load that module
    _import = it.import_module_from_path

    mod = _import(mod_path=test_mod_dir, mod_str="test_mod")
    assert mod.foo == "bar"
    assert mod.my_func() == mod.foo
    assert mod.SomeClass().spam == 123

    # Missing directory
    with pytest.raises(FileNotFoundError, match="path to.*existing directory"):
        _import(mod_path="/i/do/not/exist", mod_str="foo")

    # Module not importable, e.g. due to a syntax error
    with open(test_module_file, "a+") as f:
        f.write("\n:= some very ( bad (( syntax !\n")

    with pytest.raises(ImportError, match="Failed importing"):
        _import(mod_path=test_mod_dir, mod_str="bad_mod_str")

    # Can suppress raising
    assert _import(mod_path=test_mod_dir, mod_str="_bad_", debug=False) is None


def test_import_module_from_file(tmpdir):
    """Test importing a module from a file"""
    # Write a test module
    test_mod = (
        "foo = 'bar'\n"
        "spam = 123\n"
        "\n"
        "def my_func() -> str:\n"
        "    return foo\n"
        "\n"
        "class SomeClass:\n"
        "    def __init__(self, *args):\n"
        "        self.args = args\n"
        "        self.foo = foo\n"
        "        self.spam = spam\n"
    )
    test_mod_dir = tmpdir.mkdir("test_mod")
    test_module_file = str(test_mod_dir.join("my_file.py"))
    with open(test_module_file, "w") as f:
        f.write(test_mod)

    # Now load that module
    _import = it.import_module_from_file

    mod = _import(test_module_file)
    assert mod.foo == "bar"
    assert mod.my_func() == mod.foo
    assert mod.SomeClass().spam == 123
    assert mod.__name__ == "from_file.my_file"

    # Can also be a relative path if a base pkg was given
    mod2 = _import(
        os.path.join(*test_module_file.split("/")[-3:]),
        base_dir=os.path.join("/", *test_module_file.split("/")[:-3]),
    )
    assert mod2.foo == mod.foo

    # But not otherwise
    with pytest.raises(ValueError, match="Cannot import from a relative"):
        _import("some/file/path")


def test_resolve_types():
    import math

    import numpy as np

    types = it.resolve_types(
        ["numpy.array", "numpy.random.randint", "math.sqrt"]
    )

    assert (types[0]([1, 2, 3]) > 0).all()
    assert types[1](10) >= 0
    assert types[2](16) == 4

    assert types[0] is np.array
    assert types[1] is np.random.randint
    assert types[2] is math.sqrt


# -----------------------------------------------------------------------------


def test_LazyLoader():
    LazyLoader = it.LazyLoader

    np = LazyLoader("numpy")

    # Can use it normally
    assert (np.array([1, 2, 3]) > 0).all()
    assert np.random.randint(10) >= 0

    # Can also import submodules
    nprand = LazyLoader("numpy.random")
    assert nprand.randint(10) >= 0

    # _depth argument delays resolution upon attribute call
    np = LazyLoader("numpy", _depth=1)
    assert isinstance(np.some_attribute, LazyLoader)
    assert np.some_attribute._depth == 0

    assert np.random.randint(10) >= 0


def test_resolve_lazy_imports():
    import math

    import numpy as np

    d = dict(
        math=it.LazyLoader("math"),
        numpy=dict(random=it.LazyLoader("numpy.random"), foo="bar"),
    )
    resolved = it.resolve_lazy_imports(d)
    d["math"] is math
    d["numpy"]["random"] is np.random
    d["numpy"]["foo"] == "bar"
