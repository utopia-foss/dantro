"""Tests the PlotManager class"""

import copy
import os
import time
from collections import OrderedDict

import numpy as np
import paramspace as psp
import pytest

from dantro._import_tools import get_resource_path
from dantro.containers import NumpyDataContainer as NumpyDC
from dantro.data_mngr import DataManager
from dantro.exceptions import ParallelPlottingError
from dantro.plot import PyPlotCreator
from dantro.plot_mngr import (
    InvalidCreator,
    PlotConfigError,
    PlotCreatorError,
    PlotManager,
    PlottingError,
    SkipPlot,
)
from dantro.tools import load_yml

from ._fixtures import *

# Local constants .............................................................
# TODO Find better names for files

# Paths
PLOTS_EXT_PATH = get_resource_path("tests", "cfg/plots_ext.yml")
PLOTS_EXT2_PATH = get_resource_path("tests", "cfg/plots_ext2.yml")
PLOTS_EMPTY_PATH = get_resource_path("tests", "cfg/plots_empty.yml")
PLOTS_PARALLEL_PATH = get_resource_path("tests", "cfg/plots_parallel.yml")
BASE_EXT_PATH = get_resource_path("tests", "cfg/base_ext.yml")
UPDATE_BASE_EXT_PATH = get_resource_path("tests", "cfg/update_base_ext.yml")
BASED_ON_EXT_PATH = get_resource_path("tests", "cfg/based_on_ext.yml")

# Configurations
PLOTS_EXT = load_yml(PLOTS_EXT_PATH)
PLOTS_EXT2 = load_yml(PLOTS_EXT2_PATH)
PLOTS_EMPTY = load_yml(PLOTS_EMPTY_PATH)
PLOTS_PARALLEL = load_yml(PLOTS_PARALLEL_PATH)
BASE_EXT = load_yml(BASE_EXT_PATH)
UPDATE_BASE_EXT = load_yml(UPDATE_BASE_EXT_PATH)
BASED_ON_EXT = load_yml(BASED_ON_EXT_PATH)


# Fixtures --------------------------------------------------------------------


@pytest.fixture
def dm(tmpdir_or_local_dir) -> DataManager:
    """Returns a DataManager with some test data for plotting."""
    # Initialize it to a temporary direcotry and without load config
    dm = DataManager(tmpdir_or_local_dir)

    # Now add data to it
    # Groups
    vectors = dm.new_group("vectors")
    _ = dm.new_group("ndarrays")

    # Vectorial datasets
    vals = 100
    vectors.add(NumpyDC(name="times", data=np.linspace(0, 1, vals)))
    vectors.add(NumpyDC(name="values", data=np.random.rand(vals)))
    vectors.add(NumpyDC(name="more_values", data=np.random.rand(vals)))
    vectors.add(NumpyDC(name="even_more_values", data=np.random.rand(vals)))

    # Multidimensional datasets
    # TODO

    return dm


@pytest.fixture
def pm_kwargs(tmpdir) -> dict:
    """Common plot manager kwargs to use; uses the PyPlotCreator for all
    the tests."""
    # Create a test module that just writes a file to the given path
    # This does not create an image file!
    write_something_funcdef = (
        "def write_something(dm, *, out_path, **kwargs):\n"
        "    '''Writes the kwargs to the given path'''\n"
        "    with open(out_path, 'w') as f:\n"
        "        f.write(str(kwargs))\n"
    )

    tmpdir.join("test_module.py").write(write_something_funcdef)

    # Pass the tmpdir to the PyPlotCreator.__init__
    cik = dict(pyplot=dict(default_ext="pdf"))

    return dict(
        raise_exc=True,
        default_creator="pyplot",
        creator_init_kwargs=cik,
        plot_func_resolver_init_kwargs=dict(base_module_file_dir=str(tmpdir)),
    )


@pytest.fixture
def pcr_pyplot_kwargs() -> dict:
    """Returns valid kwargs to make a PyPlotCreator plot"""
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


def test_init(dm, out_dir):
    """Tests initialisation"""
    # Test different ways to initialize
    # Only with DataManager; will then later have to pass configuration
    PlotManager(dm=dm)

    # With a configuration dict
    PlotManager(dm=dm, default_plots_cfg={})

    # With a path to a configuration file
    PlotManager(dm=dm, default_plots_cfg=PLOTS_EXT_PATH)

    # Different base configuration pools
    PlotManager(
        dm=dm,
        base_cfg_pools=(("base", BASE_EXT_PATH),),
        default_plots_cfg=BASED_ON_EXT_PATH,
    )

    # Based on a updated configuration file
    PlotManager(
        dm=dm,
        base_cfg_pools=OrderedDict([("base1", BASE_EXT_PATH)]).items(),
        default_plots_cfg=BASED_ON_EXT_PATH,
    )

    # With a separate output directory
    PlotManager(dm=dm, out_dir=out_dir)

    # With updating out_fstrs
    pm = PlotManager(dm=dm, out_fstrs=dict(state="foo"))
    assert pm.out_fstrs.get("state") == "foo"

    # Giving an invalid default creator
    with pytest.raises(ValueError, match="No such creator 'invalid'"):
        PlotManager(dm=dm, default_creator="invalid")


def test_plotting(dm, pm_kwargs, pcr_pyplot_kwargs):
    """Test the plotting functionality of the PlotManager with a plots_cfg
    setup
    """

    def assert_num_plots(pm: PlotManager, num: int):
        """Helper function to check if the plot info is ok"""
        assert len(pm.plot_info) == num

    # The plot manager to test everything with
    pm = PlotManager(dm=dm, default_plots_cfg=PLOTS_EXT, **pm_kwargs)

    # Plot all from the given plots config file
    pm.plot_from_cfg()
    assert_num_plots(pm, 7)
    # 4 plots configured: two regular ones, one pspace volume 2, one volume 3

    # Assert that plot files were created
    for pi in pm.plot_info:
        print("\nChecking plot info: ", pi)
        assert pi["out_path"]
        assert os.path.exists(pi["out_path"])

    # Check invalid specifications and that they create no plots
    # An invalid key should be propagated
    with pytest.raises(ValueError, match="Could not find a plot config"):
        pm.plot_from_cfg(plot_only=["invalid_key"])
    assert_num_plots(pm, 7)

    # Invalid plot specification
    with pytest.raises(PlotConfigError, match="invalid plots specifications"):
        pm.plot_from_cfg(invalid_entry=(1, 2, 3))
    assert_num_plots(pm, 7)

    # Now, directly using the plot function
    # If default values were given during init, this should work.
    # Also, additional initialisation kwargs for the creator should be passable
    pm.plot(
        "foo", **pcr_pyplot_kwargs, creator_init_kwargs=dict(default_ext="pdf")
    )
    assert_num_plots(pm, 7 + 1)

    # Otherwise, without out_dir or creator arguments, not:
    with pytest.raises(PlotConfigError, match="No `out_dir` specified"):
        PlotManager(dm=dm, out_dir=None).plot("foo", plot_func=123)

    with pytest.raises(
        InvalidCreator, match="Could not determine a plot creator"
    ):
        PlotManager(dm=dm).plot("foo", plot_func=lambda: 0)

    # With some error during config preparation
    with pytest.raises(PlotCreatorError, match="Missing required keyword-arg"):
        pm.plot("foo", creator="universe", plot_func=lambda: 0)  # missing args

    # Assert that config files were created
    pm.plot("bar", **pcr_pyplot_kwargs)
    assert_num_plots(pm, 7 + 2)
    assert pm.plot_info[-1]["plot_cfg_path"]
    assert os.path.exists(pm.plot_info[-1]["plot_cfg_path"])

    pm.plot("baz", **pcr_pyplot_kwargs, save_plot_cfg=False)
    assert_num_plots(pm, 7 + 3)
    assert pm.plot_info[-1]["plot_cfg_path"] is None

    # Can also pass a custom creator type or callable
    pm.plot(
        "custom_creator",
        creator=PyPlotCreator,
        **pcr_pyplot_kwargs,
        save_plot_cfg=False,
    )

    pm.plot(
        "custom_creator_factory",
        creator=lambda *a, **k: PyPlotCreator(*a, **k),
        **pcr_pyplot_kwargs,
        save_plot_cfg=False,
    )


def test_plot_names(dm):
    """Test that plot names cannot have bad names"""
    plot_names = ("some*bad*name!", "another[bad]:name", "a/good/name")
    noop = dict(plot_func=lambda *a, **k: print("Noop plot:", a, k))
    plots_cfg = {k: noop for k in plot_names}

    # Can initialize it with these names
    pm = PlotManager(
        dm=dm,
        default_plots_cfg=plots_cfg,
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
        default_plots_cfg=plots_cfg,
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


def test_plot_locations(dm, pm_kwargs, pcr_pyplot_kwargs):
    """Tests the locations of plots and config files are as expected.

    This also makes sure that it is possible to have plot names with slashes
    in them, which should lead to directories being created.
    """
    pm = PlotManager(dm=dm, **pm_kwargs)

    # The plot files and the config file should always be side by side,
    # regardless of whether the file name has a slash in it or not and also
    # regardless of whether a sweep is configured or not
    for name in ("foo", "foo/bar/baz"):
        # Regular plot
        pm.plot(name, **pcr_pyplot_kwargs)

        info = pm.plot_info[-1]
        assert os.path.isfile(info["out_path"])
        assert os.path.isfile(info["plot_cfg_path"])
        assert os.path.dirname(info["out_path"]) == os.path.dirname(
            info["plot_cfg_path"]
        )

        # Sweep plot with zero-volume: behaves basically like a regular plot
        zero_vol_pspace = psp.ParamSpace(pcr_pyplot_kwargs)
        assert zero_vol_pspace.volume == 0
        pm.plot(name + "_0vol", from_pspace=zero_vol_pspace)

        info = pm.plot_info[-1]
        assert os.path.isfile(info["plot_cfg_path"])
        assert os.path.isfile(info["out_path"])
        assert os.path.isdir(info["target_dir"])
        assert os.path.dirname(info["out_path"]) == info["target_dir"]
        assert os.path.dirname(info["out_path"]) == os.path.dirname(
            info["plot_cfg_path"]
        )

        # Sweep plot with nonzero-volume
        sweep_kwargs = copy.deepcopy(pcr_pyplot_kwargs)
        sweep_kwargs["lw"] = psp.ParamDim(default=1.2, values=[1.0, 2.0, 3.0])
        pm.plot(name + "_sweep", from_pspace=sweep_kwargs)

        for info in pm.plot_info[-3:]:
            assert not info["plot_cfg_path"]  # not saved for individual plots
            assert os.path.isfile(info["out_path"])
            assert os.path.isdir(info["target_dir"])
            assert os.path.dirname(info["out_path"]) == info["target_dir"]
            assert os.path.isfile(
                os.path.join(info["target_dir"], "sweep_cfg.yml")
            )


def test_plotting_from_file_path(dm, pm_kwargs):
    """Test plotting from file path works"""
    pm = PlotManager(dm=dm, default_plots_cfg=PLOTS_EXT, **pm_kwargs)
    pm.plot_from_cfg(plots_cfg=PLOTS_EXT_PATH)

    # Can also be an empty / none-like yaml file
    assert load_yml(PLOTS_EMPTY_PATH) is None
    pm.plot_from_cfg(plots_cfg=PLOTS_EMPTY_PATH)


def test_plotting_overwrite(
    dm, pm_kwargs, pcr_pyplot_kwargs, tmpdir, pspace_plots
):
    """Tests that it is possible to specify a custom output path and overwrite
    existing plots.
    """
    pm = PlotManager(dm=dm, **pm_kwargs)
    custom_dir = tmpdir.join("custom_dir")

    pc = pm.plot("foo", out_dir=str(custom_dir), **pcr_pyplot_kwargs)

    # Have two files created now, plot and its config. Get the plots creation
    # date and time to make sure the later one overwrote this one
    assert len(custom_dir.listdir()) == 2
    plot_mtime = custom_dir.join("foo.pdf").mtime()

    # Should not be able to overwrite it
    with pytest.raises(PlotCreatorError, match="There already exists a file"):
        pm.plot("foo", out_dir=str(custom_dir), **pcr_pyplot_kwargs)
    assert plot_mtime == custom_dir.join("foo.pdf").mtime()

    # Now, with a plot creator that allows overwriting, it should work ... but
    # the PlotManager will throw an error because the backup of the plot config
    # already exists
    with pytest.raises(FileExistsError, match="File exists"):
        pm.plot(
            "foo", out_dir=str(custom_dir), exist_ok=True, **pcr_pyplot_kwargs
        )

    # Set the config_exists_action in the plot manager accordingly
    pm = PlotManager(dm=dm, **pm_kwargs, cfg_exists_action="overwrite")
    pc = pm.plot(
        "foo", out_dir=str(custom_dir), exist_ok=True, **pcr_pyplot_kwargs
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


def test_base_cfg_pool(dm, pm_kwargs):
    """Tests the setup and interface of the base config pool"""
    pm_kwargs = dict(dm=dm, **pm_kwargs)
    pm = PlotManager(
        base_cfg_pools=(("base", BASE_EXT), ("update", UPDATE_BASE_EXT)),
        **pm_kwargs,
    )

    # Were added in the correct order
    assert list(pm.base_cfg_pools.keys()) == ["dantro_base", "base", "update"]

    # Can suppress the dantro-internal base config
    pm = PlotManager(
        base_cfg_pools=(("base", BASE_EXT), ("update", UPDATE_BASE_EXT)),
        use_dantro_base_cfg_pool=False,
        **pm_kwargs,
    )
    assert list(pm.base_cfg_pools.keys()) == ["base", "update"]

    # Errors when adding already existing or special entries
    with pytest.raises(ValueError, match="already exists"):
        pm.add_base_cfg_pool(label="base", plots_cfg={})

    with pytest.raises(ValueError, match="special labels"):
        pm.add_base_cfg_pool(label="plot", plots_cfg={})


def test_plotting_based_on(dm, pm_kwargs):
    """Test plotting from plots_cfg using a base_cfg and a plots_cfg"""

    def assert_num_plots(pm: PlotManager, num: int):
        """Helper function to check if the plot info is ok"""
        assert len(pm.plot_info) == num

    pm = PlotManager(
        dm=dm,
        base_cfg_pools=(
            ("base", BASE_EXT),
            ("update", UPDATE_BASE_EXT),
        ),
        **pm_kwargs,
    )

    # Plot all from the given default config file
    pm.plot_from_cfg(plots_cfg=BASED_ON_EXT_PATH)
    assert_num_plots(pm, 5)  # 4 configured, one is pspace with volume 2

    # Assert that plot files were created
    for pi in pm.plot_info:
        print("Checking plot info: ", pi)
        assert pi["out_path"]
        assert os.path.exists(pi["out_path"])

    # Check invalid specifications and that they create no plots
    with pytest.raises(
        PlotConfigError, match="Did not find a base plot config"
    ):
        update_plots_cfg = {"invalid_based_on": {"based_on": "invalid_key"}}
        pm.plot_from_cfg(plot_only=["invalid_based_on"], **update_plots_cfg)
    assert_num_plots(pm, 5)  # No new plots

    # Check close matches, "Did you mean"-feature
    with pytest.raises(PlotConfigError, match="Did you mean"):
        update_plots_cfg = {"invalid_based_on": {"based_on": "fooo"}}
        pm.plot_from_cfg(plot_only=["invalid_based_on"], **update_plots_cfg)
    assert_num_plots(pm, 5)  # No new plots

    # Check directly from plot
    with pytest.raises(
        PlotConfigError, match="Did not find a base plot config.*bad_based_on"
    ):
        pm.plot(name="foo", based_on="bad_based_on")
    assert_num_plots(pm, 5)  # No new plots

    # Inheritance shortcuts
    pm.plot_from_cfg(plots_cfg=dict(from_func=True))
    assert_num_plots(pm, 6)

    time.sleep(1)
    pm.plot_from_cfg(plots_cfg=dict(from_func="inherit"))
    assert_num_plots(pm, 7)

    pm.plot_from_cfg(plots_cfg=dict(from_func=False))
    assert_num_plots(pm, 7)  # No new plots

    with pytest.raises(TypeError, match="12345"):
        pm.plot_from_cfg(plots_cfg=dict(from_func=12345))
    assert_num_plots(pm, 7)  # No new plots


def test_plots_enabled(dm, pm_kwargs, pcr_pyplot_kwargs):
    """Tests the handling of `enabled` key in plots configuration"""
    pm = PlotManager(
        dm=dm,
        **pm_kwargs,
        default_plots_cfg=dict(
            foo=dict(enabled=False, **pcr_pyplot_kwargs),
            bar=dict(enabled=True, **pcr_pyplot_kwargs),
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


def test_file_ext(dm, pm_kwargs, pcr_pyplot_kwargs):
    """Check file extension handling"""
    # Without given default extension
    PlotManager(
        dm=dm, default_plots_cfg=PLOTS_EXT, out_dir="no1/", **pm_kwargs
    ).plot_from_cfg()

    # With extension (with dot)
    pm_kwargs["creator_init_kwargs"]["pyplot"]["default_ext"] = "pdf"
    PlotManager(
        dm=dm, default_plots_cfg=PLOTS_EXT, out_dir="no2/", **pm_kwargs
    ).plot_from_cfg()

    # ...and without dot
    pm_kwargs["creator_init_kwargs"]["pyplot"]["default_ext"] = ".pdf"
    PlotManager(
        dm=dm, default_plots_cfg=PLOTS_EXT, out_dir="no3/", **pm_kwargs
    ).plot_from_cfg()

    # ...and with None -> should result in ext == ""
    pm_kwargs["creator_init_kwargs"]["pyplot"]["default_ext"] = None
    PlotManager(
        dm=dm, default_plots_cfg=PLOTS_EXT, out_dir="no4/", **pm_kwargs
    ).plot_from_cfg()

    # Test sweeping with file_ext parameter set (needs to be popped)
    pm = PlotManager(dm=dm, default_plots_cfg=PLOTS_EXT2_PATH, **pm_kwargs)
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


def test_save_plot_cfg(tmpdir_or_local_dir, dm, pm_kwargs):
    """Tests saving of the plot configuration"""
    pm_kwargs["raise_exc"] = True
    pm = PlotManager(dm=dm, **pm_kwargs)

    save_kwargs = dict(
        name="cfg_save_test",
        creator_name="testcreator",
        target_dir=str(tmpdir_or_local_dir),
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


def test_plot_from_saved_plot_cfg(dm, pm_kwargs):
    """Tests whether creating a plot from a saved plot configuration works
    just as well.
    """
    pm_kwargs["raise_exc"] = True
    pm = PlotManager(dm=dm, default_plots_cfg=PLOTS_EXT, **pm_kwargs)

    # Perform some plots
    pm.plot_from_cfg(out_dir="run1/")

    # Check that the output files were generated and store information on the
    # plot configuration files
    plot_cfg_paths = set()
    file_sizes = dict()

    for pi in pm.plot_info:
        print("\nChecking plot info: ", pi)
        assert pi["out_path"]
        assert os.path.exists(pi["out_path"])

        if not pi["part_of_sweep"]:
            plot_cfg_path = os.path.join(
                os.path.dirname(pi["out_path"]),
                pi["name"] + "_cfg.yml",
            )
            assert plot_cfg_path == pi["plot_cfg_path"]

        else:
            plot_cfg_path = os.path.join(
                os.path.dirname(pi["out_path"]),
                "sweep_cfg.yml",
            )

        assert os.path.isfile(plot_cfg_path)
        plot_cfg_paths.add(plot_cfg_path)
        file_sizes[pi["out_path"]] = os.path.getsize(pi["out_path"])

    # Now plot again using those plot config files
    for plot_cfg_path in plot_cfg_paths:
        pm.plot_from_cfg(plots_cfg=plot_cfg_path, out_dir="run2/")

        # Check that files have approximately the same size as the previously
        # created ones.
        # For sweeps, will only check the last file, but that's good enough.
        pi = pm.plot_info[-1]
        assert os.path.isfile(pi["out_path"])

        fs_run1 = file_sizes[pi["out_path"].replace("run2", "run1")]
        fs_run2 = os.path.getsize(pi["out_path"])
        assert abs(fs_run2 - fs_run1) < 128


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


def test_parallel_pspace_plot(dm, pm_kwargs, caplog):
    caplog.set_level(10)
    pspace_vol = 3 * 4 * 6  # defined in plots_parallel.yml

    pm_kwargs["raise_exc"] = True
    pm = PlotManager(dm=dm, default_plots_cfg=PLOTS_PARALLEL, **pm_kwargs)
    assert pm.raise_exc

    pm.plot_from_cfg()

    # Check that parallel plotting was used by checking the log messages
    _logs = caplog.text
    assert "Adding 'thread' plot task" in _logs
    assert "Adding 'process' plot task" in _logs
    assert f"Submitting {pspace_vol} tasks to ProcessPoolExecutor" in _logs
    assert "Benchmarking executor spawning overhead using 5 tasks" in _logs
    assert "Executor overhead:" in _logs
    assert "Parallel plotting now commencing ..." in _logs
    assert f"Getting results for plot 1/{pspace_vol}" in _logs
    # NOTE somehow not everything is captured here ... not super reliable

    # Skipping is also handled
    def plot_func_that_skips(*_, n: int = 0, skip_mod: int = 1, **__):
        """A do-nothing function that raises SkipPlot if n % skip_mod == 0"""
        if n % skip_mod == 0:
            raise SkipPlot(f"{n} % {skip_mod} == 0")
        return

    pm.plot(
        "skipping",
        plot_func=plot_func_that_skips,
        skip_mod=2,
        from_pspace=psp.ParamSpace(
            dict(n=psp.ParamDim(default=0, range=[10]))
        ),
        parallel=dict(enabled=True, executor="thread"),
    )

    # -- Check exceptions
    with pytest.raises(ValueError, match="some_bad_eXeCuTor_name"):
        pm.plot_from_cfg(plot_only=("error_bad_executor_name",))

    # Errors propagate
    with pytest.raises(ParallelPlottingError, match="some_bad_argument"):
        pm.plot_from_cfg(plot_only=("error_propagates_from_thread",))

    with pytest.raises(ParallelPlottingError, match="another_bad_argument"):
        pm.plot_from_cfg(plot_only=("error_propagates_from_process",))

    # The fallback_with_fail will still fail if the error is not due to
    # parallel processing.
    with pytest.raises(ParallelPlottingError, match="some_bad_argument"):
        pm.plot_from_cfg(plot_only=("fallback_with_fail",))

    with pytest.raises(ParallelPlottingError, match="See error log"):
        pm.plot_from_cfg(plot_only=("without_exception_summary",))

    # Provoke an error only in part of the parameter space
    with pytest.raises(ParallelPlottingError, match="total of 2 exceptions"):
        pm.plot(
            "sometimes_failing",
            plot_func=plot_func_that_skips,
            skip_mod=2,
            from_pspace=psp.ParamSpace(
                dict(n=psp.ParamDim(default=0, values=[0, 1, "bad", 2, "X"]))
            ),
            parallel=dict(enabled=True, executor="thread"),
        )
