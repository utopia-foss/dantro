"""Tests the PlotManager class"""

import os
from pkg_resources import resource_filename

import numpy as np

import pytest

import paramspace as psp

from dantro.tools import load_yml
from dantro.data_mngr import DataManager
from dantro.containers import NumpyDataContainer as NumpyDC
from dantro.plot_mngr import (
    PlotManager,
    PlottingError,
    PlotConfigError,
    InvalidCreator,
    PlotCreatorError,
    SkipPlot,
)

# Local constants .............................................................
# Paths
PLOTS_EXT_PATH = resource_filename("tests", "cfg/plots_ext.yml")
PLOTS_EXT2_PATH = resource_filename("tests", "cfg/plots_ext2.yml")
BASE_EXT_PATH = resource_filename("tests", "cfg/base_ext.yml")
UPDATE_BASE_EXT_PATH = resource_filename("tests", "cfg/update_base_ext.yml")
BASED_ON_EXT_PATH = resource_filename("tests", "cfg/based_on_ext.yml")
AUTO_DETECT_PATH = resource_filename("tests", "cfg/auto_detect.yml")

# Configurations
PLOTS_EXT = load_yml(PLOTS_EXT_PATH)
PLOTS_EXT2 = load_yml(PLOTS_EXT2_PATH)
BASE_EXT = load_yml(BASE_EXT_PATH)
UPDATE_BASE_EXT = load_yml(UPDATE_BASE_EXT_PATH)
BASED_ON_EXT = load_yml(BASED_ON_EXT_PATH)
PLOTS_AUTO_DETECT = load_yml(AUTO_DETECT_PATH)
# PLOTS_DECL = load_yml(resource_filename("tests", "cfg/plots_decl.yml"))
# PLOTS_VEGA = load_yml(resource_filename("tests", "cfg/plots_vega.yml"))


# Test classes ----------------------------------------------------------------


# Fixtures --------------------------------------------------------------------


@pytest.fixture
def dm(tmpdir) -> DataManager:
    """Returns a DataManager with some test data for plotting."""
    # Initialize it to a temporary direcotry and without load config
    dm = DataManager(tmpdir)

    # Now add data to it
    # Groups
    vectors = dm.new_group("vectors")
    _ = dm.new_group("ndarrays")

    # Vectorial datasets
    vals = 100
    vectors.add(NumpyDC(name="times", data=np.linspace(0, 1, vals)))
    vectors.add(NumpyDC(name="values", data=np.random.rand(vals)))
    vectors.add(NumpyDC(name="more_values", data=np.random.rand(vals)))

    # Multidimensional datasets
    # TODO

    return dm


@pytest.fixture
def pm_kwargs(tmpdir) -> dict:
    """Common plot manager kwargs to use; uses the ExternalPlotCreator for all
    the tests."""
    # Create a test module that just writes a file to the given path
    write_something_funcdef = (
        "def write_something(dm, *, out_path, **kwargs):\n"
        "    '''Writes the kwargs to the given path'''\n"
        "    with open(out_path, 'w') as f:\n"
        "        f.write(str(kwargs))\n"
    )

    tmpdir.join("test_module.py").write(write_something_funcdef)

    # Pass the tmpdir to the ExternalPlotCreator __init__
    cik = dict(
        external=dict(default_ext="pdf", base_module_file_dir=str(tmpdir))
    )

    return dict(
        raise_exc=True, default_creator="external", creator_init_kwargs=cik
    )


@pytest.fixture
def pcr_ext_kwargs() -> dict:
    """Returns valid kwargs to make a ExternalPlotCreator plot"""
    return dict(module=".basic", plot_func="lineplot", y="vectors/values")


@pytest.fixture
def pspace_plots() -> dict:
    """Returns a plot configuration (external creator) with parameter sweeps"""

    # Create a sweep over the y-keys for the lineplot
    y_pdim = psp.ParamDim(
        default="vectors/values",
        values=["vectors/values", "vectors/more_values"],
    )

    # Assemble the dict
    return dict(
        sweep=psp.ParamSpace(
            dict(
                module=".basic",
                plot_func="lineplot",
                # kwargs to the plot function
                y=y_pdim,
            )
        )
    )


# Tests -----------------------------------------------------------------------


def test_init(dm, tmpdir):
    """Tests initialisation"""
    # Test different ways to initialize
    # Only with DataManager; will then later have to pass configuration
    PlotManager(dm=dm)

    # With a configuration dict
    PlotManager(dm=dm, plots_cfg={})

    # With a path to a configuration file
    PlotManager(dm=dm, plots_cfg=PLOTS_EXT_PATH)

    # Based on a configuration dict
    PlotManager(dm=dm, base_cfg={}, plots_cfg={})
    PlotManager(dm=dm, base_cfg={}, update_base_cfg={}, plots_cfg={})

    # Based on a configuration file
    PlotManager(dm=dm, base_cfg=BASE_EXT_PATH, plots_cfg=BASED_ON_EXT_PATH)

    # Based on a updated configuration file
    PlotManager(
        dm=dm,
        base_cfg=BASE_EXT_PATH,
        update_base_cfg=UPDATE_BASE_EXT_PATH,
        plots_cfg=BASED_ON_EXT_PATH,
    )

    # With a separate output directory
    PlotManager(dm=dm, out_dir=tmpdir.mkdir("out"))

    # With updating out_fstrs
    pm = PlotManager(dm=dm, out_fstrs=dict(state="foo"))
    assert pm._out_fstrs.get("state") == "foo"

    # Giving an invalid default creator
    with pytest.raises(ValueError, match="No such creator 'invalid'"):
        PlotManager(dm=dm, default_creator="invalid")


def test_plotting(dm, pm_kwargs, pcr_ext_kwargs):
    """Test the plotting functionality of the PlotManager with a plots_cfg
    setup
    """

    def assert_num_plots(pm: PlotManager, num: int):
        """Helper function to check if the plot info is ok"""
        assert len(pm.plot_info) == num

    # The plot manager to test everything with
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_EXT, **pm_kwargs)

    # Plot all from the given plots config file
    pm.plot_from_cfg()
    assert_num_plots(pm, 4)  # 3 configured, one is pspace with volume 2

    # Assert that plot files were created
    for pi in pm.plot_info:
        print("Checking plot info: ", pi)
        assert pi["out_path"]
        assert os.path.exists(pi["out_path"])

    # Check invalid specifications and that they create no plots
    # An invalid key should be propagated
    with pytest.raises(ValueError, match="Could not find a configuration"):
        pm.plot_from_cfg(plot_only=["invalid_key"])
    assert_num_plots(pm, 4)

    # Invalid plot specification
    with pytest.raises(PlotConfigError, match="invalid plots specifications"):
        pm.plot_from_cfg(invalid_entry=(1, 2, 3))
    assert_num_plots(pm, 4)

    # Now, directly using the plot function
    # If default values were given during init, this should work.
    # Also, additional initialisation kwargs for the creator should be passable
    pm.plot(
        "foo", **pcr_ext_kwargs, creator_init_kwargs=dict(default_ext="pdf")
    )
    assert_num_plots(pm, 4 + 1)

    # Otherwise, without out_dir or creator arguments, not:
    with pytest.raises(ValueError, match="No `out_dir` specified"):
        PlotManager(dm=dm, out_dir=None).plot("foo")

    with pytest.raises(ValueError, match="No `creator` argument given"):
        PlotManager(dm=dm).plot("foo")

    with pytest.raises(ValueError, match="nor auto-detection enabled."):
        PlotManager(dm=dm, auto_detect_creator=False).plot("foo")
        # Same as the above case

    # Assert that config files were created
    pm.plot("bar", **pcr_ext_kwargs)
    assert_num_plots(pm, 4 + 2)
    assert pm.plot_info[-1]["plot_cfg_path"]
    assert os.path.exists(pm.plot_info[-1]["plot_cfg_path"])

    pm.plot("baz", **pcr_ext_kwargs, save_plot_cfg=False)
    assert_num_plots(pm, 4 + 3)
    assert pm.plot_info[-1]["plot_cfg_path"] is None


def test_plot_names(dm):
    """Test that plot names cannot have bad names"""
    plot_names = ("some*bad*name!", "another[bad]:name", "a/good/name")
    noop = dict(plot_func=lambda *a, **k: print("Noop plot:", a, k))
    plots_cfg = {k: noop for k in plot_names}

    # Can initialize it with these names
    pm = PlotManager(
        dm=dm,
        plots_cfg=plots_cfg,
        default_creator="external",
        save_plot_cfg=False,
    )

    # But cannot plot with them
    with pytest.raises(ValueError, match="contains unsupported characters!"):
        pm.plot_from_cfg()

    # Unless only plotting a good name
    pm.plot_from_cfg(plot_only=["a/good/name"])

    # Also tested when invoking plot method directly
    with pytest.raises(ValueError, match="contains unsupported characters!"):
        pm.plot(name="not_a_good_name?")


def test_plot_only(dm):
    """Tests the plot_only feature"""
    plot_names = ("foo", "baz", "bar", "spam/foo", "spam/bar")
    noop = dict(plot_func=lambda *a, **k: print("Noop plot:", a, k))
    plots_cfg = {k: noop for k in plot_names}

    pm = PlotManager(
        dm=dm,
        plots_cfg=plots_cfg,
        default_creator="external",
        save_plot_cfg=False,
    )

    assert len(pm.plot_info) == 0
    pm.plot_from_cfg(plot_only=[])
    assert len(pm.plot_info) == 0

    # Specific name
    pm.plot_from_cfg(plot_only=["foo"])
    assert len(pm.plot_info) == 1
    assert pm.plot_info[-1]["name"] == "foo"

    # Multiple names
    pm.plot_from_cfg(plot_only=["foo", "bar"])
    assert len(pm.plot_info) == 1 + 2
    assert pm.plot_info[-2]["name"] == "foo"
    assert pm.plot_info[-1]["name"] == "bar"

    # Wildcard ... remaining in order of definition
    pm.plot_from_cfg(plot_only=["ba*", "spam*"])
    assert len(pm.plot_info) == 3 + 4
    assert pm.plot_info[-4]["name"] == "baz"
    assert pm.plot_info[-3]["name"] == "bar"
    assert pm.plot_info[-2]["name"] == "spam/foo"
    assert pm.plot_info[-1]["name"] == "spam/bar"

    # No match, but with wildcard
    pm.plot_from_cfg(plot_only=["fish*"])
    assert len(pm.plot_info) == 7 + 0

    # No match, but without wildcard: error
    with pytest.raises(ValueError, match="Could not find.*fish"):
        pm.plot_from_cfg(plot_only=["fish"])


def test_plot_locations(dm, pm_kwargs, pcr_ext_kwargs):
    """Tests the locations of plots and config files are as expected.

    This also makes sure that it is possible to have plot names with slashes
    in them, which should lead to directories being created.
    """
    pm = PlotManager(dm=dm, **pm_kwargs)

    # The plot files and the config file should always be side by side,
    # regardless of whether the file name has a slash in it or not and also
    # regardless of whether a sweep is configured or not
    for name in ("foo", "bar/baz/spam"):
        # Regular plot
        pm.plot(name, **pcr_ext_kwargs)

        info = pm.plot_info[-1]
        assert os.path.isfile(info["out_path"])
        assert os.path.isfile(info["plot_cfg_path"])
        assert os.path.dirname(info["out_path"]) == os.path.dirname(
            info["plot_cfg_path"]
        )

        # Sweep plot
        pm.plot(name, from_pspace=psp.ParamSpace(pcr_ext_kwargs))

        info = pm.plot_info[-1]  # only the last plot
        assert info["plot_cfg_path"] is None  # not saved for individual plots
        assert os.path.isfile(info["out_path"])
        assert os.path.isdir(info["target_dir"])
        assert os.path.dirname(info["out_path"]) == info["target_dir"]
        assert os.path.isfile(
            os.path.join(info["target_dir"], "sweep_cfg.yml")
        )


def test_plotting_from_file_path(dm, pm_kwargs):
    """Test plotting from file path works"""
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_EXT, **pm_kwargs)
    pm.plot_from_cfg(plots_cfg=PLOTS_EXT_PATH)


def test_plotting_overwrite(
    dm, pm_kwargs, pcr_ext_kwargs, tmpdir, pspace_plots
):
    """Tests that it is possible to specify a custom output path and overwrite
    existing plots.
    """
    pm = PlotManager(dm=dm, **pm_kwargs)
    custom_dir = tmpdir.join("custom_dir")

    pc = pm.plot("foo", out_dir=str(custom_dir), **pcr_ext_kwargs)

    # Have two files created now, plot and its config. Get the plots creation
    # date and time to make sure the later one overwrote this one
    assert len(custom_dir.listdir()) == 2
    plot_mtime = custom_dir.join("foo.pdf").mtime()

    # Should not be able to overwrite it
    with pytest.raises(PlotCreatorError, match="There already exists a file"):
        pm.plot("foo", out_dir=str(custom_dir), **pcr_ext_kwargs)
    assert plot_mtime == custom_dir.join("foo.pdf").mtime()

    # Now, with a plot creator that allows overwriting, it should work ... but
    # the PlotManager will throw an error because the backup of the plot config
    # already exists
    with pytest.raises(FileExistsError, match="File exists"):
        pm.plot(
            "foo", out_dir=str(custom_dir), exist_ok=True, **pcr_ext_kwargs
        )

    # Set the config_exists_action in the plot manager accordingly
    pm = PlotManager(dm=dm, **pm_kwargs, cfg_exists_action="overwrite")
    pc = pm.plot(
        "foo", out_dir=str(custom_dir), exist_ok=True, **pcr_ext_kwargs
    )
    assert len(custom_dir.listdir()) == 2
    assert plot_mtime < custom_dir.join("foo.pdf").mtime()

    # Also works with a sweep plot
    pc = pm.plot(
        "bar/baz",
        from_pspace=pspace_plots["sweep"]._dict,
        out_dir=str(custom_dir),
    )
    assert len(custom_dir.join("bar/baz").listdir()) == 2 + 1  # 2 plots, 1 cfg
    sweep_plot = custom_dir.join("bar/baz/1__y_vectors-values.pdf")
    sweep_mtime = sweep_plot.mtime()

    pc = pm.plot(
        "bar/baz",
        from_pspace=pspace_plots["sweep"]._dict,
        out_dir=str(custom_dir),
        exist_ok=True,
    )
    assert len(custom_dir.join("bar/baz").listdir()) == 2 + 1
    assert sweep_mtime < sweep_plot.mtime()


def test_plotting_based_on(dm, pm_kwargs):
    """Test plotting from plots_cfg using a base_cfg and a plots_cfg"""
    def assert_num_plots(pm: PlotManager, num: int):
        """Helper function to check if the plot info is ok"""
        assert len(pm.plot_info) == num

    pm = PlotManager(dm=dm, base_cfg=BASE_EXT, update_base_cfg=UPDATE_BASE_EXT,
                     **pm_kwargs)

    # Plot all from the given default config file
    pm.plot_from_cfg(plots_cfg=BASED_ON_EXT_PATH)
    assert_num_plots(pm, 5)  # 4 configured, one is pspace with volume 2


    # Assert that plot files were created
    for pi in pm.plot_info:
        print("Checking plot info: ", pi)
        assert pi['out_path']
        assert os.path.exists(pi['out_path'])

    # Check sequence of strings
    d = pm._resolve_based_on(cfg=dict(foo="test", something="something"),
                             based_on=('foo', 'bar'))
    assert d['foo'] == "test"
    assert d['bar'] == "bar"
    assert d['spam'] == "bar"
    assert d['something'] == "something"

    # Also during initialization
    pm2 = PlotManager(dm=dm, base_cfg=BASE_EXT,
                      update_base_cfg=dict(baz=dict(based_on=('foo', 'bar'),
                                                    foo="test",
                                                    something="something")),
                      **pm_kwargs)
    baz = pm2._base_cfg['baz']
    assert baz['foo'] == "test"
    assert baz['bar'] == "bar"
    assert baz['spam'] == "bar"
    assert baz['something'] == "something"


    # Check invalid specifications and that they create no plots
    with pytest.raises(KeyError, match="No base plot config(.*) 'invalid"):
        update_plots_cfg = {'invalid_based_on': {'based_on': 'invalid_key'}}
        pm.plot_from_cfg(plot_only=["invalid_based_on"],
                         **update_plots_cfg)
    assert_num_plots(pm, 5)  # No new plots

    # Bad based_on during resolution
    with pytest.raises(KeyError, match="No base plot config(.*) 'bad_"):
        pm.plot(name="foo", based_on="bad_based_on")
    assert_num_plots(pm, 5)  # No new plots

    # Should also be an error during initialization
    with pytest.raises(KeyError, match="No base plot configuration named 'ba"):
        PlotManager(dm=dm, base_cfg=BASE_EXT,
                    update_base_cfg=dict(bad_based_on=dict(based_on="baaad")),
                    **pm_kwargs)


def test_plots_enabled(dm, pm_kwargs, pcr_ext_kwargs):
    """Tests the handling of `enabled` key in plots configuration"""
    pm = PlotManager(
        dm=dm,
        **pm_kwargs,
        plots_cfg=dict(
            foo=dict(enabled=False, **pcr_ext_kwargs),
            bar=dict(enabled=True, **pcr_ext_kwargs),
        ),
        cfg_exists_action="skip",
    )

    # No plots should be created like this
    pm.plot_from_cfg(plot_only=[])
    assert len(pm.plot_info) == 0

    # Force plotting disabled foo plot
    pm.plot_from_cfg(plot_only=["foo"])
    assert len(pm.plot_info) == 1

    # This will have no effect; bar would be plotted anyway
    pm.plot_from_cfg(
        plot_only=["bar"], bar=dict(file_ext="png")
    )  # to avoid file name conflicts
    assert len(pm.plot_info) == 2

    # Without plot_only, should only plot bar
    pm.plot_from_cfg()
    assert len(pm.plot_info) == 3


def test_sweep(dm, pm_kwargs, pspace_plots):
    """Test that sweeps work"""
    pm = PlotManager(dm=dm, **pm_kwargs)

    # Plot the sweep
    pm.plot_from_cfg(**pspace_plots)

    # This should have created 2 plots
    assert len(pm.plot_info) == 2

    # By passing a config to `from_pspace` that is no ParamSpace (in this case
    # the internally stored dict) a ParamSpace should be created from that dict
    pm.plot("foo", from_pspace=pspace_plots["sweep"]._dict)

    # This should have created two more plots
    assert len(pm.plot_info) == 2 + 2

    # Assert that all plots were created
    for pi in pm.plot_info:
        assert os.path.exists(pi["out_path"])

        # None of them should have a plot config saved
        assert pi["plot_cfg_path"] is None


def test_file_ext(dm, pm_kwargs, pcr_ext_kwargs):
    """Check file extension handling"""
    # Without given default extension
    PlotManager(
        dm=dm, plots_cfg=PLOTS_EXT, out_dir="no1/", **pm_kwargs
    ).plot_from_cfg()

    # With extension (with dot)
    pm_kwargs["creator_init_kwargs"]["external"]["default_ext"] = "pdf"
    PlotManager(
        dm=dm, plots_cfg=PLOTS_EXT, out_dir="no2/", **pm_kwargs
    ).plot_from_cfg()

    # ...and without dot
    pm_kwargs["creator_init_kwargs"]["external"]["default_ext"] = ".pdf"
    PlotManager(
        dm=dm, plots_cfg=PLOTS_EXT, out_dir="no3/", **pm_kwargs
    ).plot_from_cfg()

    # ...and with None -> should result in ext == ""
    pm_kwargs["creator_init_kwargs"]["external"]["default_ext"] = None
    PlotManager(
        dm=dm, plots_cfg=PLOTS_EXT, out_dir="no4/", **pm_kwargs
    ).plot_from_cfg()

    # Test sweeping with file_ext parameter set (needs to be popped)
    pm = PlotManager(dm=dm, plots_cfg=PLOTS_EXT2_PATH, **pm_kwargs)
    pm.plot_from_cfg(plot_only=["with_file_ext"])


def test_raise_exc(dm, pm_kwargs):
    """Tests that the `raise_exc` argument behaves as desired"""
    # Empty plot config should either log and return None ...
    assert PlotManager(dm=dm, raise_exc=False).plot_from_cfg() is None

    # ... or raise an error
    with pytest.raises(PlotConfigError, match="Got empty `plots_cfg`"):
        PlotManager(dm=dm, raise_exc=True).plot_from_cfg()

    # Test calls to the plot creators with and without raise_exc
    pm_exc = PlotManager(dm=dm, **pm_kwargs)
    pm_log = PlotManager(dm=dm, **pm_kwargs)
    pm_log.raise_exc = False

    # This should only log
    pm_log.plot(name="logs", module=".basic", plot_func="lineplot")

    # While this one should raise
    with pytest.raises(
        PlottingError, match=r"An error occurred during plotting .* 'raises'"
    ):
        pm_exc.plot(name="raises", module=".basic", plot_func="lineplot")

    # ... unless silenced explicitly
    pm_exc.plot(
        name="raises", debug=False, module=".basic", plot_func="lineplot"
    )

    # Inversely, pm_log can also be made to raise an exception
    with pytest.raises(
        PlottingError, match=r"An error occurred during plotting .* 'logs'"
    ):
        pm_log.plot(
            name="logs", debug=True, module=".basic", plot_func="lineplot"
        )


def test_save_plot_cfg(tmpdir, dm, pm_kwargs):
    """Tests saving of the plot configuration"""
    pm_kwargs["raise_exc"] = True
    pm = PlotManager(dm=dm, **pm_kwargs)

    save_kwargs = dict(
        name="cfg_save_test",
        creator_name="testcreator",
        target_dir=str(tmpdir),
    )

    # First write
    path = pm._save_plot_cfg(dict(foo="bar"), **save_kwargs)
    assert os.path.isfile(path)
    fsize = os.path.getsize(path)

    # Should raise (default)
    with pytest.raises(FileExistsError, match="cfg_save_test"):
        pm._save_plot_cfg(dict(foo="bar"), **save_kwargs)

    # 'skip' and make sure that the modification time was not changed
    pm._save_plot_cfg(dict(foo="barz"), **save_kwargs, exists_action="skip")
    assert os.path.getsize(path) == fsize  # did not change, because skipped

    # 'overwrite'
    pm._save_plot_cfg(
        dict(foo="barzz"), **save_kwargs, exists_action="overwrite"
    )
    assert os.path.getsize(path) == fsize + 2  # changed, because overwritten

    # 'overwrite_nowarn'
    pm._save_plot_cfg(
        dict(foo="barzz"), **save_kwargs, exists_action="overwrite_nowarn"
    )
    assert os.path.getsize(path) == fsize + 2  # changed, because overwritten

    # 'append'
    pm._save_plot_cfg(
        dict(foo="barzzz"), **save_kwargs, exists_action="append"
    )
    assert os.path.getsize(path) >= 2 * fsize  # because appended

    # Test error messages
    with pytest.raises(ValueError, match="Invalid value 'invalid' for arg"):
        pm._save_plot_cfg(
            dict(foo="barzz"), **save_kwargs, exists_action="invalid"
        )


def test_auto_detect_creator(dm):
    """Tests the auto-detection feature"""
    pm = PlotManager(dm=dm, auto_detect_creator=True)

    # This should work
    pc = pm.plot(name="pcr_ext", **PLOTS_AUTO_DETECT["pcr_ext"])
    assert type(pc) is pm.CREATORS["external"]

    # No matching candidate
    with pytest.raises(InvalidCreator, match="declared itself a candidate!"):
        pm.plot(name="fail", **PLOTS_AUTO_DETECT["fail"])


def test_plot_skipping(dm, pm_kwargs):
    """Tests the SkipPlot exception leading to skipped plots"""

    def plot_func_that_skips(*_, n: int = 0, skip_mod: int = 1, **kwargs):
        """A do-nothing function that raises SkipPlot if n % skip_mod == 0"""
        if n % skip_mod == 0:
            raise SkipPlot(f"{n} % {skip_mod} == 0")
        return

    pm_kwargs["raise_exc"] = True
    pm_kwargs["save_plot_cfg"] = False
    pm = PlotManager(dm=dm, **pm_kwargs)
    assert not pm.plot_info

    pm.plot("skipped", plot_func=plot_func_that_skips)
    assert len(pm.plot_info) == 1
    assert pm.plot_info[-1]["creator_rv"] == "skipped"

    pm.plot("not_skipped", plot_func=plot_func_that_skips, n=1, skip_mod=10)
    assert len(pm.plot_info) == 2
    assert pm.plot_info[-1]["creator_rv"] is True

    # Again, now for a parameter space configuration where every second
    # combination of parameters will lead to skipping
    pm.plot(
        "skipping",
        plot_func=plot_func_that_skips,
        skip_mod=2,
        from_pspace=psp.ParamSpace(
            dict(n=psp.ParamDim(default=0, range=[10]))
        ),
    )
    assert len(pm.plot_info) == 2 + 10
    assert all(pi["creator_rv"] == "skipped" for pi in pm.plot_info[-10::2])
    assert all(pi["creator_rv"] is True for pi in pm.plot_info[-9::2])
