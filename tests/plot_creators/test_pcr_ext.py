"""Tests the ExternalPlotCreator class."""

import pytest

from dantro.plot_creators import ExternalPlotCreator


# Fixtures --------------------------------------------------------------------
# Import some from other tests
from ..test_plot_mngr import dm

@pytest.fixture
def init_kwargs(dm) -> dict:
    """Default initialisation kwargs"""
    return dict(dm=dm, default_ext="pdf")

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

def test_init(init_kwargs, tmpdir):
    """Tests initialisation"""
    ExternalPlotCreator("init", **init_kwargs)

    # Test passing a base_module_file_dir
    ExternalPlotCreator("init", **init_kwargs,
                        base_module_file_dir=tmpdir)
    
    # Check with invalid directories
    with pytest.raises(ValueError, match="needs to be an absolute path"):
        ExternalPlotCreator("init", **init_kwargs,
                            base_module_file_dir="foo/bar/baz")

    with pytest.raises(ValueError, match="does not exists or does not point"):
        ExternalPlotCreator("init", **init_kwargs,
                            base_module_file_dir=tmpdir.join("foo.bar"))


def test_resolve_plot_func(init_kwargs, tmpdir, tmp_module):
    """Tests whether the _resolve_plot_func"""
    epc = ExternalPlotCreator("init", **init_kwargs)

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
