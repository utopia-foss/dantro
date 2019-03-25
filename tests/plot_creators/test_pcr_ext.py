"""Tests the ExternalPlotCreator class."""

import inspect
from pkg_resources import resource_filename

import pytest
import copy
import matplotlib.pyplot as plt

from dantro.tools import load_yml, recursive_update
from dantro.plot_creators import ExternalPlotCreator, UniversePlotCreator
from dantro.plot_creators import is_plot_func

# Load configuration files
PLOTS_AUTO_DETECT = load_yml(resource_filename("tests", "cfg/auto_detect.yml"))


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

@pytest.fixture
def tmp_rc_file(tmpdir) -> str:
    """Creates a temporary yaml file with matplotlib rcParams"""
    rc_paramaters = (
        "figure.dpi: 10 \n"
        "axes.grid: True\n"
        )

    path = tmpdir.join("test_rc_file.yml")
    path.write(rc_paramaters)

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

def test_style_context(init_kwargs, tmp_rc_file):
    """Tests if the style context has been set"""
    # .. Test _prepare_style_context directly .................................
    epc = ExternalPlotCreator("direct", **init_kwargs)
    psc = epc._prepare_style_context

    # String for base_style
    assert psc(base_style="classic")["_internal.classic_mode"]

    # Update order for bsae style
    base_style = ["classic", "dark_background"]
    assert psc(base_style=base_style)["figure.facecolor"] == "black"
    assert psc(base_style=base_style[::-1])["figure.facecolor"] == "0.75"

    # Invalid base_style value
    with pytest.raises(ValueError, match="Style 'foo' is not a valid"):
        psc(base_style="foo")

    # Invalid base_style type
    with pytest.raises(TypeError, match="Argument `base_style` need be"):
        psc(base_style=123)

    # Bad RC path
    with pytest.raises(ValueError, match="needs to be an absolute path"):
        psc(rc_file="foo.yml")

    with pytest.raises(ValueError, match="No file was found at path"):
        psc(rc_file="/foo.yml")

    # .. Integration tests ....................................................
    # Style dict for use in the test
    style = {"base_style" : ["classic", "dark_background"], 
             "rc_file" : tmp_rc_file, "font.size" : 2.0}

    # Test plot function to check wether the given style context is entered
    def test_plot_func(*args, expected_rc_params: dict=None, **_):
        """Compares the entries of a dictionary with rc_params and the
        currently used rcParams oft matplotlib"""
        if expected_rc_params is None:
            # Get the defaults
            expected_rc_params = plt.rcParamsDefault

        # Compare the used rcParams with the expected value
        for key, expected_val in expected_rc_params.items():
            print("Testing rc parameter '{}' ...".format(key))
            assert plt.rcParams[key] == expected_val

        print("All RC parameters matched.", end="\n\n")
    
    # .. Without style given to init ..........................................
    epc = ExternalPlotCreator("without_defaults", **init_kwargs)

    # Without defaults, the cache attribute should be empty
    assert epc._default_rc_params is None

    # Call the plot method of the creator which internally calls the plot_func
    # in a given rc_context to test if the style is set correctly:
    # No style given, should use defaults
    epc.plot(out_path="test_path", plot_func=test_plot_func)

    # Style given
    epc.plot(out_path="test_path", plot_func=test_plot_func, style=style,
             expected_rc_params=epc._prepare_style_context(**style))

    # Custom style
    test_style = recursive_update(epc._prepare_style_context(**style), 
                                  {"font.size" : 20.0})
    epc.plot(out_path="test_path", plot_func=test_plot_func, style=test_style,
             expected_rc_params=epc._prepare_style_context(**test_style))

    # Ignoring defaults should also work (but have no effect)
    epc.plot(out_path="test_path", plot_func=test_plot_func,
             style=dict(**test_style, ignore_defaults=True),
             expected_rc_params=epc._prepare_style_context(**test_style))

    # .. With style given to init .............................................
    # Initialize plot creators with and without a default style 
    epc = ExternalPlotCreator("with_defaults", **init_kwargs, style=style)

    # Check wether the default style contains the correct parameters, this
    # should serve as a test for the _prepare_style_context method 
    assert epc._default_rc_params["axes.facecolor"] == "black"
    assert epc._default_rc_params["figure.dpi"] == 10
    assert epc._default_rc_params["axes.grid"]
    assert epc._default_rc_params["font.size"] == 2.0

    # No style given, should use the `style` passed to init
    epc.plot(out_path="test_path", plot_func=test_plot_func,
             expected_rc_params=epc._prepare_style_context(**style))

    # Style given, should update the style given at init
    update_style = recursive_update(epc._prepare_style_context(**style),
                                    {"font.size" : 20.0})
    epc.plot(out_path="test_path", plot_func=test_plot_func,
             style=update_style,
             expected_rc_params=epc._prepare_style_context(**update_style))

    # Ignore the defaults and nothing passed; should use matplotlib defaults
    epc.plot(out_path="test_path", plot_func=test_plot_func,
             style=dict(ignore_defaults=True))


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

def test_can_plot(init_kwargs, tmp_module):
    """Tests the can_plot and _valid_plot_func_signature methods"""
    epc = ExternalPlotCreator("can_plot", **init_kwargs)

    # Should work for the .basic lineplot
    assert epc.can_plot("ext",
                        module=".basic", plot_func="lineplot")
    # This one is also decorated, thus the function signature is not checked

    # ... and for the function given in the module file
    assert epc.can_plot("ext",
                        module_file=tmp_module, plot_func="write_something")
    # This one is NOT decorated, thus the function signature IS checked

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
    
    @is_plot_func(creator_type=ExternalPlotCreator)
    def pfdec_type():
        pass
    
    @is_plot_func(creator_type=ExternalPlotCreator, creator_name="foo")
    def pfdec_type_and_name():
        pass
    
    @is_plot_func(creator_type=UniversePlotCreator)
    def pfdec_subtype():
        pass
    
    @is_plot_func(creator_name="universe")
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
    upc = UniversePlotCreator("can_plot", **init_kwargs)

    assert not declared_pf_by_attrs(pfdec_name, upc, "universe")
    assert declared_pf_by_attrs(pfdec_type, upc, "universe")
    assert declared_pf_by_attrs(pfdec_type_and_name, upc, "universe")
    assert declared_pf_by_attrs(pfdec_subtype, upc, "universe")
    assert declared_pf_by_attrs(pfdec_subtype_name, upc, "universe")
    assert not declared_pf_by_attrs(pfdec_bad_type, upc, "universe")
    assert not declared_pf_by_attrs(pfdec_bad_name, upc, "universe")


    # Test the _valid_plot_func_signature method . . . . . . . . . . . . . . .
    def valid_sig(func, throw: bool=False) -> bool:
        return epc._valid_plot_func_signature(inspect.signature(func),
                                              raise_if_invalid=throw)

    # Create a few functions to test with
    def valid_func_1(dm, *, out_path: str, **kwargs):
        pass

    def valid_func_2(arg1, *, kwarg1, out_path: str, kwarg2=None, **kwargs):
        pass
    
    def valid_func_3(arg1, *, out_path: str=None):
        pass

    def bad_func_1(arg1, arg2, *, out_path: str, **kwargs):
        pass

    def bad_func_2(arg1, **kwargs):
        pass
    
    def bad_func_3(*args, out_path: str, **kwargs):
        pass
    
    def bad_func_4(dm, out_path: str, **kwargs):
        pass
    
    def bad_func_5(dm, *, kwarg1, **kwargs):
        pass

    assert valid_sig(valid_func_1)
    assert valid_sig(valid_func_2)
    assert valid_sig(valid_func_3)

    with pytest.raises(ValueError,
                       match=("Expected 1 POSITIONAL_OR_KEYWORD argument\(s\) "
                              "but the plot function allowed 2")):
        valid_sig(bad_func_1, True)

    with pytest.raises(ValueError,
                       match=("Did not find all of the expected KEYWORD_ONLY "
                              "arguments \(out_path\) in the plot function")):
        valid_sig(bad_func_2, True)

    with pytest.raises(ValueError,
                       match=("Expected 1 POSITIONAL_OR_KEYWORD argument\(s\) "
                              "but the plot function allowed 0")):
        valid_sig(bad_func_3, True)

    with pytest.raises(ValueError,
                       match=("Expected 1 POSITIONAL_OR_KEYWORD argument\(s\) "
                              "but the plot function allowed 2: dm, out_pat")):
        valid_sig(bad_func_4, True)

    with pytest.raises(ValueError,
                       match=("Did not find all of the expected KEYWORD_ONLY "
                              "arguments \(out_path\) in the plot function")):
        valid_sig(bad_func_5, True)

    # Disallow *args, and **kwargs
    epc._AD_ALLOW_VAR_POSITIONAL = False
    epc._AD_ALLOW_VAR_KEYWORD = False

    # Some error messages should include more text now
    with pytest.raises(ValueError,
                       match="VAR_POSITIONAL arguments are not allowed, but"):
        valid_sig(bad_func_3, True)

    with pytest.raises(ValueError,
                       match="VAR_KEYWORD arguments are not allowed, but the"):
        valid_sig(bad_func_4, True)

    with pytest.raises(ValueError,
                       match="VAR_KEYWORD arguments are not allowed, but the"):
        valid_sig(bad_func_5, True)


    # To provoke a POSITIONAL_ONLY error, expect more than one of them
    epc._AD_NUM_POSITIONAL_ONLY = 42
    with pytest.raises(ValueError,
                       match="Expected 42 POSITIONAL_ONLY argument\(s\) but"):
        valid_sig(valid_func_2, True)
