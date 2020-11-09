"""Tests the ExternalPlotCreator class."""

from pkg_resources import resource_filename

import pytest
import matplotlib.pyplot as plt

from dantro.data_mngr import DataManager
from dantro.tools import load_yml, recursive_update
from dantro.plot_creators import ExternalPlotCreator, UniversePlotCreator
from dantro.plot_creators import is_plot_func, PlotHelper
from dantro.plot_creators.pcr_ext import figure_leak_prevention
from dantro.dag import TransformationDAG

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
    rc_paramaters = "figure.dpi: 10 \naxes.grid: True\n"

    path = tmpdir.join("test_rc_file.yml")
    path.write(rc_paramaters)

    return path


# Tests -----------------------------------------------------------------------


def test_init(init_kwargs, tmpdir):
    """Tests initialisation"""
    ExternalPlotCreator("init", **init_kwargs)

    # Test passing a base_module_file_dir
    ExternalPlotCreator("init", **init_kwargs, base_module_file_dir=tmpdir)

    # Check with invalid directories
    with pytest.raises(ValueError, match="needs to be an absolute path"):
        ExternalPlotCreator(
            "init", **init_kwargs, base_module_file_dir="foo/bar/baz"
        )

    with pytest.raises(ValueError, match="does not exists or does not point"):
        ExternalPlotCreator(
            "init", **init_kwargs, base_module_file_dir=tmpdir.join("foo.bar")
        )


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
            # Need to skip over some keys which are not very robust to check
            if key in ('backend_fallback',):
                print(f"Not testing rc parameter '{key}' ...")
                continue

            print(f"Testing rc parameter '{key}' ...")
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


# -----------------------------------------------------------------------------


def test_use_dag(tmpdir, init_kwargs):
    """Tests whether DAG parameters are passed through properly to the plot
    function ...
    """
    epc = ExternalPlotCreator("dag_tests", **init_kwargs)

    # Some temporary output path
    out_path = str(tmpdir.join("foo"))

    # Some plotting callables for testing arguments that are passed
    def pf_with_dag(*, data, out_path: str):
        assert isinstance(data, dict)
        assert isinstance(out_path, str)

    def pf_without_dag(dm, *, out_path: str):
        assert isinstance(dm, DataManager)
        assert isinstance(out_path, str)

    # Invoke plots with different callable vs. dag-usage combinations
    epc.plot(out_path=out_path, plot_func=pf_with_dag, use_dag=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'data'"):
        epc.plot(out_path=out_path, plot_func=pf_without_dag, use_dag=True)

    epc.plot(out_path=out_path, plot_func=pf_without_dag, use_dag=False)

    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        epc.plot(out_path=out_path, plot_func=pf_with_dag, use_dag=False)

    # Passing the DAG object along to plot function, using function attributes
    def pf_with_dag_object(*, data, dag, out_path):
        assert isinstance(data, dict)
        assert isinstance(dag, TransformationDAG)
        assert isinstance(out_path, str)

    pf_with_dag_object.use_dag = True
    pf_with_dag_object.pass_dag_object_along = True

    epc.plot(out_path=out_path, plot_func=pf_with_dag_object)

    # With helper enabled
    def pf_with_dag_and_helper(*, data, hlpr):
        assert isinstance(data, dict)
        assert isinstance(hlpr, PlotHelper)

    pf_with_dag_and_helper.use_dag = True
    pf_with_dag_and_helper.use_helper = True

    epc.plot(out_path=out_path, plot_func=pf_with_dag_and_helper)

    def pf_with_dag_and_helper_and_dag_object(*, data, dag, hlpr):
        assert isinstance(data, dict)
        assert isinstance(dag, TransformationDAG)
        assert isinstance(hlpr, PlotHelper)

    pf_with_dag_and_helper_and_dag_object.use_dag = True
    pf_with_dag_and_helper_and_dag_object.use_helper = True
    pf_with_dag_and_helper_and_dag_object.pass_dag_object_along = True

    epc.plot(
        out_path=out_path, plot_func=pf_with_dag_and_helper_and_dag_object
    )

    # Overwriting DAG usage enabled in attribute but disabled via plot config
    def pf_with_dag_disabled_via_cfg(dm, *, out_path):
        assert isinstance(dm, DataManager)

    pf_with_dag_disabled_via_cfg.use_dag = True
    pf_with_dag_disabled_via_cfg.pass_dag_object_along = True

    epc.plot(
        out_path=out_path,
        plot_func=pf_with_dag_disabled_via_cfg,
        use_dag=False,
    )

    # Unpacking dag results
    def pf_with_dag_results_unpacked(*, out_path, foo, bar):
        pass

    pf_with_dag_results_unpacked.use_dag = True
    pf_with_dag_results_unpacked.unpack_dag_results = True
    pf_with_dag_results_unpacked.compute_only_required_dag_tags = True

    epc.plot(
        out_path=out_path,
        plot_func=pf_with_dag_results_unpacked,
        transform=[dict(define=1, tag="foo"), dict(define=2, tag="bar")],
    )

    with pytest.raises(TypeError, match="Failed unpacking DAG results!"):
        epc.plot(
            out_path=out_path,
            plot_func=pf_with_dag_results_unpacked,
            transform=[dict(define=1, tag="foo"), dict(define=2, tag="bar")],
            foo="bar",
        )


def test_dag_required_tags(tmpdir, init_kwargs):
    """Tests the requirements for certain tags expected by the plot function"""
    epc = ExternalPlotCreator("dag_required_tags", **init_kwargs)

    # Some temporary output path
    out_path = str(tmpdir.join("foo"))

    # Some basic transformations for testing
    sum_and_sub = [dict(add=[1, 2], tag="sum"), dict(sub=[3, 2], tag="sub")]

    # Define a plot function for testing expected tags
    def pf(*, data: dict, dag, out_path, expected_tags: set):
        assert isinstance(dag, TransformationDAG)
        if expected_tags is not None:
            assert set(data.keys()) == expected_tags
            assert all([t in dag.tags for t in expected_tags])

    pf.use_dag = True
    pf.pass_dag_object_along = True

    # Without required tags given, there should be no checks
    epc.plot(
        out_path=out_path,
        plot_func=pf,
        transform=sum_and_sub,
        expected_tags=None,
    )

    # Now, require some tags. These should be the only ones set now.
    pf.required_dag_tags = ("sum", "sub")
    epc.plot(
        out_path=out_path,
        plot_func=pf,
        transform=sum_and_sub,
        expected_tags={"sum", "sub"},
    )

    # Modify required tags and check that this leads to an error
    pf.required_dag_tags = ("sum", "sub", "mul")

    with pytest.raises(
        ValueError,
        match=(
            "required tags that were not specified in the "
            "DAG: mul. Available tags: .*sum, sub"
        ),
    ):
        epc.plot(out_path=out_path, plot_func=pf, transform=sum_and_sub)
        # expected_tags not checked here

    # ... unless there actually is another transformation
    epc.plot(
        out_path=out_path,
        plot_func=pf,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        expected_tags={"sum", "sub", "mul"},
    )

    # What about if there are more transformations?
    # By default, all are computed, leading to a ZeroDivisionError here
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        epc.plot(
            out_path=out_path,
            plot_func=pf,
            transform=sum_and_sub
            + [dict(mul=[1, 1], tag="mul"), dict(div=[1, 0], tag="div")],
            expected_tags={"sum", "sub", "mul"},
        )  # not checked

    # When setting the compute_only_required_dag_tags, this is not an issue
    pf.required_dag_tags = ("sum", "sub")
    pf.compute_only_required_dag_tags = True
    epc.plot(
        out_path=out_path,
        plot_func=pf,
        transform=sum_and_sub
        + [dict(mul=[1, 1], tag="mul"), dict(div=[1, 0], tag="div")],
        expected_tags={"sum", "sub"},
    )

    # ... but the compute_only argument is stronger:
    with pytest.raises(
        ValueError,
        match=(
            "required tags that were not set to be computed by the DAG: sub."
        ),
    ):
        epc.plot(
            out_path=out_path,
            plot_func=pf,
            compute_only=["sum"],
            transform=sum_and_sub,
        )

    # Adjust computed tags via config to provoke an error message
    pf.compute_only_required_dag_tags = False
    with pytest.raises(
        ValueError,
        match=(
            "required tags that were not set to be computed "
            "by the DAG: sum, sub. Make sure to set the "
            "`compute_only` argument such that results"
        ),
    ):
        epc.plot(
            out_path=out_path,
            plot_func=pf,
            transform=sum_and_sub,
            compute_only=[],
        )

    # Disabled DAG usage should also raise an error
    with pytest.raises(ValueError, match="requires DAG tags to be computed"):
        epc.plot(out_path=out_path, plot_func=pf, use_dag=False)

    # For completeness, also test via the is_plot_func decorator
    @is_plot_func(use_dag=True)
    def pf_dec(*, data: dict, hlpr, expected_tags: set):
        assert isinstance(hlpr, PlotHelper)
        if expected_tags is not None:
            assert set(data.keys()) == expected_tags

    # The default settings
    assert pf_dec.use_dag is True
    assert pf_dec.required_dag_tags is None
    assert pf_dec.compute_only_required_dag_tags is True
    assert pf_dec.pass_dag_object_along is False

    # ... without required DAG tags
    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub,
        expected_tags={"sum", "sub"},
    )

    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        expected_tags={"sum", "sub", "mul"},
    )

    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        compute_only=["sum"],
        expected_tags={"sum"},
    )

    # ... with required DAG tags (and compute_only_required_dag_tags ENABLED)
    pf_dec.required_dag_tags = ("sum", "sub")

    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        expected_tags={"sum", "sub"},
    )

    with pytest.raises(
        ValueError,
        match=(
            "required tags that were not set to be computed by the DAG: sub."
        ),
    ):
        epc.plot(
            out_path=out_path,
            plot_func=pf_dec,
            transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
            compute_only=["sum"],
            expected_tags={"sum", "sub"},
        )

    # ... with required DAG tags (and compute_only_required_dag_tags DISABLED)
    pf_dec.compute_only_required_dag_tags = False

    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        expected_tags={"sum", "sub", "mul"},
    )

    epc.plot(
        out_path=out_path,
        plot_func=pf_dec,
        transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
        compute_only=["sum", "sub"],
        expected_tags={"sum", "sub"},
    )

    with pytest.raises(
        ValueError,
        match=(
            "required tags that were not set to be computed "
            "by the DAG: sum, sub."
        ),
    ):
        epc.plot(
            out_path=out_path,
            plot_func=pf_dec,
            transform=sum_and_sub + [dict(mul=[1, 1], tag="mul")],
            compute_only=[],
            expected_tags={},
        )


# -----------------------------------------------------------------------------


def test_can_plot(init_kwargs, tmp_module):
    """Tests the can_plot and _valid_plot_func_signature methods"""
    epc = ExternalPlotCreator("can_plot", **init_kwargs)

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


def test_decorator(tmpdir):
    """Test the is_plot_func decorator"""
    # Needs no arguments
    is_plot_func()

    # Can take some specific ones, though, without checks
    is_plot_func(
        creator_type=ExternalPlotCreator,
        creator_name="foo",
        use_helper=True,
        helper_defaults=dict(),
        supports_animation=True,
        add_attributes=dict(),
    )

    # Helper_defaults can be an absolute path
    is_plot_func(helper_defaults=tmpdir.join("foo.yml"))

    # User is expaned (but file is missing)
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        is_plot_func(helper_defaults="~/something/something.yml")

    # Relative path not allowed
    with pytest.raises(ValueError, match="was a relative path: some/rel"):
        is_plot_func(helper_defaults="some/relative/path")


def test_figure_leak_prevention():
    """Tests the figure_leak_prevention context manager"""
    # After a fresh start, open some figures
    plt.close("all")
    figs = [plt.figure() for i in range(3)]
    assert plt.get_fignums() == [1, 2, 3]

    with figure_leak_prevention():
        # Open some more. These should be closed when exiting.
        figs = [plt.figure() for i in range(3)]
        assert plt.get_fignums() == [1, 2, 3, 4, 5, 6]
        assert plt.gcf().number == 6

    assert plt.get_fignums() == [1, 2, 3, 6]

    # Once more, now with an exception, which should lead to the current fig
    # not surviving beyond the context
    with pytest.raises(Exception):
        with figure_leak_prevention(close_current_fig_on_raise=True):
            figs = [plt.figure() for i in range(2)]
            assert plt.get_fignums() == [1, 2, 3, 6, 7, 8]
            assert plt.gcf().number == 8

            raise Exception()

    assert plt.get_fignums() == [1, 2, 3, 6]
