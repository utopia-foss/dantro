"""Test the DataManager class and the loader functions"""

import copy
import glob
import os

import dill as pkl
import h5py as h5
import numpy as np
import pytest
import xarray as xr

import dantro.base
import dantro.data_mngr
from dantro._import_tools import get_resource_path
from dantro.containers import (
    NumpyDataContainer,
    ObjectContainer,
    PassthroughContainer,
    StringContainer,
    XrDataContainer,
)
from dantro.data_loaders import (
    AllAvailableLoadersMixin,
    Hdf5LoaderMixin,
    NumpyLoaderMixin,
    PandasLoaderMixin,
    PickleLoaderMixin,
    TextLoaderMixin,
    XarrayLoaderMixin,
    YamlLoaderMixin,
)
from dantro.exceptions import *
from dantro.groups import OrderedDataGroup
from dantro.mixins import Hdf5ProxySupportMixin
from dantro.tools import total_bytesize, write_yml

from .test_base import pickle_roundtrip

# Local constants
LOAD_CFG_PATH = get_resource_path("tests", "cfg/load_cfg.yml")

# Test classes ----------------------------------------------------------------


class DataManager(YamlLoaderMixin, dantro.data_mngr.DataManager):
    """A DataManager-derived class for testing the implementation"""

    # Set the class variable to test group class lookup via name
    _DATA_GROUP_CLASSES = dict(ordered=OrderedDataGroup)

    # A (bad) load function for testing
    def _load_bad_loadfunc(self):
        pass


class TextDataManager(TextLoaderMixin, DataManager):
    """A data manager that is able to load text files"""


class PklDataManager(PickleLoaderMixin, DataManager):
    """A data manager that is able to load pickled files"""


class NumpyDataManager(NumpyLoaderMixin, DataManager):
    """A DataManager to load numpy data"""


class CSVDataManager(PandasLoaderMixin, NumpyLoaderMixin, DataManager):
    """A DataManager to load CSV data"""


class XarrayDataManager(XarrayLoaderMixin, DataManager):
    """A DataManager to load xarray data"""


class NumpyTestDC(Hdf5ProxySupportMixin, NumpyDataContainer):
    """A data container class that provides numpy proxy access"""


class XrTestDC(Hdf5ProxySupportMixin, XrDataContainer):
    """A data container class that provides xarray proxy access"""


class DummyDC(NumpyTestDC):
    """A data container class for testing the _HDF5_DSET_MAP"""


class DummyGroup(OrderedDataGroup):
    """A data container class for testing the _HDF5_GROUP_MAP"""


class Hdf5DataManager(Hdf5LoaderMixin, DataManager):
    """A DataManager-derived class to test the Hdf5LoaderMixin class"""

    # Define the class to use for loading the datasets and the mappings
    _HDF5_DSET_DEFAULT_CLS = NumpyTestDC
    _HDF5_DSET_MAP = dict(dummy=DummyDC, labelled=XrTestDC)
    _HDF5_GROUP_MAP = dict(dummy=DummyGroup)
    _HDF5_MAP_FROM_ATTR = "container_type"


class FullDataManager(AllAvailableLoadersMixin, DataManager):
    """A DataManager with all the loaders included"""


# Fixtures --------------------------------------------------------------------


@pytest.fixture
def data_dir(tmpdir) -> str:
    """Writes some dummy data to a temporary directory and returns the path to
    that directory"""
    # Create YAML dummy data and write it out
    foobar = dict(
        one=1, two=2, go_deeper=dict(eleven=11), a_list=list(range(10))
    )

    lamo = dict(nothing="to see here")

    write_yml(foobar, path=tmpdir.join("foobar.yml"))
    write_yml(lamo, path=tmpdir.join("lamo.yml"))
    write_yml(lamo, path=tmpdir.join("also_lamo.yml"))
    write_yml(lamo, path=tmpdir.join("looooooooooong_filename.yml"))

    subdir = tmpdir.mkdir("sub")
    write_yml(foobar, path=subdir.join("abc123.yml"))
    write_yml(foobar, path=subdir.join("abcdef.yml"))

    subsubdir = subdir.mkdir("sub")
    write_yml(foobar, path=subsubdir.join("abc234.yml"))

    merged = tmpdir.mkdir("merged")
    write_yml(foobar, path=merged.join("data0.yml"))
    write_yml(foobar, path=merged.join("data1.yml"))
    write_yml(foobar, path=merged.join("data2.yml"))
    write_yml(foobar, path=merged.join("cfg0.yml"))
    write_yml(foobar, path=merged.join("cfg1.yml"))
    write_yml(foobar, path=merged.join("cfg2.yml"))

    # Some files that are unreadable with the YAML loader
    with open(tmpdir.join("not_loadable.bad_yml"), "x+") as f:
        f.write("{bad syntax: , ]})!*\n")
    with open(tmpdir.join("bad_anchor.bad_yml"), "x+") as f:
        f.write("{foo: *bad_anchor}\n")

    return tmpdir


@pytest.fixture
def dm(data_dir) -> DataManager:
    """Returns a DataManager without load configuration"""
    return DataManager(data_dir, out_dir=None)


@pytest.fixture
def text_dm(data_dir) -> TextDataManager:
    """Manager with test data for TextLoaderMixin"""
    # Create a sundirectory for the text data
    text_dir = data_dir.mkdir("text_data")

    # Define a string to dump
    to_dump = "This is a test string \n with two lines\n"

    # save the file
    with open(text_dir.join("test_string.txt"), mode="w") as f:
        f.write(to_dump)

    return TextDataManager(data_dir, out_dir=None)


@pytest.fixture
def pkl_dm(data_dir) -> PklDataManager:
    """Pickles some objects to test the pickle loading function"""
    # Create a subdirectory for the pickles
    pkl_dir = data_dir.mkdir("pickles")

    # Define objects to dump
    to_dump = [
        ("a_dict", dict(foo="bar", baz=123)),
        ("a_list", [1, 2, 3]),
        ("int_arr", np.ones((2, 3, 4), dtype=int)),
    ]

    # Create the pickles
    for name, obj in to_dump:
        with open(pkl_dir.join(name + ".pkl"), mode="wb") as pkl_file:
            pkl.dump(obj, pkl_file)

    return PklDataManager(data_dir, out_dir=None)


@pytest.fixture
def np_dm(data_dir) -> NumpyDataManager:
    """Manager with test data for NumpyLoaderMixin"""
    # Create a subdirectory for the pickles
    npy_dir = data_dir.mkdir("np_data")

    # Define objects to dump
    to_dump = dict()
    to_dump["zeros_int"] = np.zeros((2, 3, 4), dtype=int)
    to_dump["zeros_float"] = np.zeros((2, 3, 4), dtype=float)
    # TODO add some more here

    # Dump the objects as binary
    for name, obj in to_dump.items():
        print(f"Dumping {type(obj)} '{name}' ...\n{obj}")
        np.save(str(npy_dir.join(name + ".npy")), obj)
        print("Dumped.\n")

    return NumpyDataManager(data_dir, out_dir=None)


@pytest.fixture
def csv_dm(data_dir) -> NumpyDataManager:
    """Manager with test CSV data"""
    # Create a subdirectory for the pickles
    csv_dir = data_dir.mkdir("csv")

    # Manually create some CSV data
    with open(csv_dir.join("simple_int.csv"), "x") as f:
        f.write("# some heading line\n")
        f.write("1 2 3\n")
        f.write("4 5 6\n")

    with open(csv_dir.join("simple_float.csv"), "x") as f:
        f.write("# some heading line\n")
        f.write("1.0 2.0 3.0\n")
        f.write("4.0 5.0 6.0\n")

    with open(csv_dir.join("sep_comma.csv"), "x") as f:
        f.write("# some heading line\n")
        f.write(" 1, 2  ,   3 \n")
        f.write("40, 5.0,   6  \n")
        f.write("# some footer line\n")

    # Now with column names, inferred by pandas
    # Subset of the penguins dataset
    with open(csv_dir.join("penguins.csv"), "x") as f:
        f.write(
            "species,island,bill_length_mm,bill_depth_mm,"
            "flipper_length_mm,body_mass_g,sex\n"
        )
        f.write("Adelie,Torgersen,39.1,18.7,181.0,3750.0,Male\n")
        f.write("Adelie,Torgersen,39.5,17.4,186.0,3800.0,Female\n")
        f.write("Adelie,Torgersen,40.3,18.0,195.0,3250.0,Female\n")
        f.write("Adelie,Torgersen,,,,,\n")
        f.write("Adelie,Torgersen,36.7,19.3,193.0,3450.0,Female\n")
        f.write("Adelie,Torgersen,39.3,20.6,190.0,3650.0,Male\n")
        f.write("Adelie,Biscoe,37.8,18.3,174.0,3400.0,Female\n")
        f.write("Adelie,Biscoe,37.7,18.7,180.0,3600.0,Male\n")
        f.write("Adelie,Biscoe,35.9,19.2,189.0,3800.0,Female\n")
        f.write("Adelie,Biscoe,38.2,18.1,185.0,3950.0,Male\n")
        f.write("Adelie,Dream,39.5,16.7,178.0,3250.0,Female\n")
        f.write("Adelie,Dream,37.2,18.1,178.0,3900.0,Male\n")
        f.write("Adelie,Dream,39.5,17.8,188.0,3300.0,Female\n")
        f.write("Adelie,Dream,40.9,18.9,184.0,3900.0,Male\n")
        f.write("Adelie,Dream,36.4,17.0,195.0,3325.0,Female\n")
        f.write("Adelie,Dream,39.2,21.1,196.0,4150.0,Male\n")
        f.write("Adelie,Dream,38.8,20.0,190.0,3950.0,Male\n")
        f.write("Adelie,Biscoe,39.6,17.7,186.0,3500.0,Female\n")
        f.write("Adelie,Biscoe,40.1,18.9,188.0,4300.0,Male\n")
        f.write("Adelie,Biscoe,35.0,17.9,190.0,3450.0,Female\n")
        f.write("Adelie,Biscoe,42.0,19.5,200.0,4050.0,Male\n")

    # Write out some seaborn test data, csv and space-separated
    import seaborn as sns

    for dset_name in ("iris", "planets", "taxis"):
        df = sns.load_dataset(dset_name)
        with open(csv_dir.join(f"{dset_name}.csv"), "x") as f:
            f.write(df.to_csv())

        with open(csv_dir.join(f"{dset_name}.tsv"), "x") as f:
            f.write(df.to_csv(sep="\t"))

    return CSVDataManager(data_dir, out_dir=None)


@pytest.fixture
def xr_dm(data_dir) -> XarrayDataManager:
    """Manager with test data for XarrayLoaderMixin"""
    # Create a subdirectory for the pickles
    xr_dir = data_dir.mkdir("xr_data")

    # Define da to dump
    das, dsets = dict(), dict()

    das["zeros"] = xr.DataArray(
        data=np.zeros((2, 3, 4)), name="zeros", attrs=dict(foo="bar")
    )
    # TODO add some more here

    dsets["zeros"] = xr.Dataset()
    dsets["zeros"]["int"] = (("x", "y", "z"), np.zeros((2, 3, 4), dtype=int))
    dsets["zeros"]["float"] = (("x", "y", "z"), np.zeros((2, 3, 4)))
    # TODO add some more here

    # Dump DataArrays and Datasets, separately
    for name, da in das.items():
        da.to_netcdf(str(xr_dir.join(name + ".nc_da")), engine="h5netcdf")

    for name, dset in dsets.items():
        dset.to_netcdf(str(xr_dir.join(name + ".nc_ds")), engine="h5netcdf")

    return XarrayDataManager(data_dir, out_dir=None)


@pytest.fixture
def hdf5_dm(data_dir) -> Hdf5DataManager:
    """Returns a Hdf5DataManager without load configuration.

    Additionally to the yaml files in the data_dir, some hdf5 files with dummy
    data are added.
    """
    # Create a subdirectory for that data
    h5dir = data_dir.mkdir("hdf5_data")

    # --- Create a file with basic structures: dataset, group, attribute ---
    basic = h5.File(h5dir.join("basic.h5"), "w")

    basic.create_dataset("float_dset", data=np.zeros((2, 3, 4), dtype=float))
    basic.create_dataset("int_dset", data=np.ones((1, 2, 3), dtype=int))
    basic.create_group("group")
    basic.create_group("UpperCaseGroup")
    basic.attrs["foo"] = "file level attribute"
    basic["group"].attrs["foo"] = "group level attribute"
    basic["int_dset"].attrs["foo"] = "dset level attribute"

    # Also write some encoded data
    basic["group"].attrs["encoded_arr"] = np.array(["foo", "bar"], dtype="S3")
    basic["group"].attrs["encoded_arra"] = np.array(["foo", "bar"], dtype="S")
    basic["group"].attrs["encoded_arrb"] = np.array([b"foo", b"bar"])
    basic["group"].attrs["encoded_utf8"] = bytes("🎉", "utf8")
    basic["group"].attrs["encoded_utf16"] = bytes("👻", "utf16")

    basic.close()

    # --- Create a file with nested groups ---
    nested = h5.File(h5dir.join("nested.h5"), "w")

    nested.create_group("group1")
    nested.create_group("group2")
    nested["group1"].create_group("group11")
    nested["group1"].create_group("group12")
    nested["group2"].create_group("group21")
    nested["group2"].create_group("group22")
    nested["group1"]["group11"].create_group("group111")
    nested["group1"]["group11"]["group111"].create_dataset(
        "dset", data=np.random.random(size=(3, 4, 5))
    )

    nested.close()

    # --- Create a file to test mapping ---
    mapping = h5.File(h5dir.join("mapping.h5"), "w")

    mapping.create_group("dummy_group")
    mapping["dummy_group"].attrs["container_type"] = "dummy"
    mapping.create_dataset("dummy_dset", data=np.zeros((1, 2)))
    mapping["dummy_dset"].attrs["container_type"] = "dummy"

    mapping.create_dataset("another_dummy", data=np.zeros((2, 3, 4)))
    mapping["another_dummy"].attrs["container_type"] = bytes("dummy", "utf8")

    mapping.create_dataset("one_more_dummy", data=np.zeros((2, 3, 4)))
    mapping["one_more_dummy"].attrs["container_type"] = np.array(
        "dummy", dtype="S5"
    )

    mapping.create_group("badmap_group")
    mapping["badmap_group"].attrs["container_type"] = "badmap"
    mapping.create_dataset("badmap_dset", data=np.zeros((1, 2)))
    mapping["badmap_dset"].attrs["container_type"] = "badmap"

    mapping.close()

    # --- Create a file with labelled data as test for a complex scenario ---
    labelled = h5.File(h5dir.join("labelled.h5"), "w")

    some_dset = labelled.create_dataset("some_dset", data=np.zeros((2, 3, 4)))
    for k, v in dict(
        container_type="labelled",
        dims=["x", "y", "z"],
        coords__x=[0, 1],
        coords__y=[0, 10, 20],
        coords_mode__z="linked",
        coords__z="./coords/z",
    ).items():
        some_dset.attrs[k] = v
    labelled.create_dataset("coords/z", data=np.array([1, 2, 3, 4]))

    labelled.close()

    # Instantiate a data manager for this directory
    return Hdf5DataManager(data_dir, out_dir=None)


# End of fixtures -------------------------------------------------------------

# =============================================================================
# General Tests ===============================================================


def test_init(data_dir):
    """Test the initialisation of a DataManager"""
    # Initialize via path to yaml file
    dm = DataManager(data_dir, out_dir=None, load_cfg=LOAD_CFG_PATH)

    # Assert that only the data directory is available
    assert dm.dirs["data"] == data_dir
    assert dm.dirs["out"] is False

    # Initialize via dict and with default out dir
    dm = DataManager(
        data_dir, load_cfg=dict(test=dict(loader="yaml", glob_str="*.yml"))
    )

    # Assert folders are existing and correctly linked
    assert dm.dirs["data"] == data_dir
    assert os.path.isdir(dm.dirs["data"])
    assert os.path.isdir(dm.dirs["out"])

    # The DataManager's path should start within root
    assert dm.parent is None
    assert dm.path == f"/{os.path.basename(data_dir)}_Manager"

    # It should create a hashstr
    assert len(dm.hashstr) == 32
    assert hash(dm.hashstr) == hash(dm)

    # There should be a default tree file path
    assert ".d3" in dm.tree_cache_path
    assert not dm.tree_cache_exists

    # It should be possible to set condensed tree parameters
    assert dm._COND_TREE_MAX_LEVEL == 10
    dm = DataManager(
        data_dir,
        out_dir=None,
        load_cfg=LOAD_CFG_PATH,
        condensed_tree_params=dict(max_level=42),
    )
    assert dm._COND_TREE_MAX_LEVEL == 42

    with pytest.raises(KeyError, match="Invalid condensed tree parameter"):
        DataManager(
            data_dir,
            out_dir=None,
            load_cfg=LOAD_CFG_PATH,
            condensed_tree_params=dict(foo=123),
        )


def test_init_with_create_groups(tmpdir):
    """Tests the create_groups argument to __init__"""
    # Check group creation from a list of names
    test_groups = ["abc", "def", "123"]
    dm = DataManager(tmpdir, out_dir=None, create_groups=test_groups)

    for grp_name in test_groups:
        assert grp_name in dm
        assert isinstance(dm[grp_name], dm._NEW_GROUP_CLS)

    # And from a list of mixed names and dicts
    test_groups2 = [
        "ghi",
        dict(path="grp1"),
        dict(path="grp2", Cls=dm._NEW_GROUP_CLS),
        dict(path="grp3", Cls="ordered"),
    ]

    dm2 = DataManager(tmpdir, out_dir=None, create_groups=test_groups2)

    assert "ghi" in dm2
    assert "grp1" in dm2
    assert "grp2" in dm2
    assert "grp3" in dm2
    assert isinstance(dm2["grp1"], dm2._NEW_GROUP_CLS)
    assert isinstance(dm2["grp2"], dm2._NEW_GROUP_CLS)
    assert isinstance(dm2["grp3"], OrderedDataGroup)

    # Without the class variable set, initialisation with a class fails
    with pytest.raises(AttributeError, match="No type registry available"):
        dantro.data_mngr.DataManager(
            tmpdir, out_dir=None, create_groups=[dict(path="foo", Cls="bar")]
        )


def test_available_loaders(data_dir):
    dm = FullDataManager(data_dir)
    assert "yaml" in dm.available_loaders
    assert "file" not in dm.available_loaders  # because _load_file


def test_loading(dm):
    """Tests whether loading works by using the default DataManager, i.e. that
    with the YamlLoaderMixin ...
    """
    # Check loading from config dict or file ..................................
    # No load config given
    dm.load_from_cfg()

    # Single entry
    dm.load_from_cfg(
        load_cfg=dict(barfoo=dict(loader="yaml", glob_str="foobar.yml")),
        print_tree=True,
    )

    # Assert that the top level entries are all available and content is right
    assert "barfoo" in dm

    barfoo = dm["barfoo"]
    assert barfoo["one"] == 1
    assert barfoo["two"] == 2
    assert barfoo["go_deeper"]["eleven"] == 11
    assert barfoo["a_list"] == list(range(10))

    # Check the `update_load_cfg` argument, this time printing condensed
    dm.load_from_cfg(
        update_load_cfg=dict(
            barfoo2=dict(loader="yaml", glob_str="foobar.yml")
        ),
        print_tree="condensed",
    )

    # Invalid load config
    with pytest.raises(TypeError):
        dm.load_from_cfg(update_load_cfg=dict(barfoo2=[1, 2, 3]))

    # Check single entry loading ..............................................
    # Load another single entry, this time forcing a group to be created
    dm.load("barbaz", loader="yaml", glob_str="foobar.yml", print_tree=True)
    assert "barbaz" in dm
    barbaz = dm["barbaz"]

    # Loading can also be disabled (would lead to a collision otherwise)
    dm.load("barbaz", loader="yaml", glob_str="foobar.yml", enabled=False)
    assert dm["barbaz"] is barbaz

    # Load with the yaml object loader
    dm.load(
        "barbaz_obj",
        loader="yaml_to_object",
        glob_str="foobar.yml",
        print_tree="condensed",
    )
    assert "barbaz_obj" in dm
    assert isinstance(dm["barbaz_obj"], ObjectContainer)

    # Load again, this time with more data
    dm.load("all_yaml", loader="yaml", glob_str="*.yml")

    assert "all_yaml" in dm
    assert "all_yaml/foobar" in dm
    assert "all_yaml/lamo" in dm
    assert "all_yaml/also_lamo" in dm
    assert "all_yaml/looooooooooong_filename" in dm

    # Now see what happens if loading into an existing target_path
    dm.load(
        "more_yaml",
        loader="yaml",
        glob_str="*.yml",
        target_path="all_yaml/more_yaml/{basename:}",
    )

    assert "all_yaml/more_yaml" in dm
    assert "all_yaml/more_yaml/foobar" in dm
    assert "all_yaml/more_yaml/lamo" in dm

    # ...and into a non-existing one
    dm.load(
        "more_yaml",
        loader="yaml",
        glob_str="*.yml",
        target_path="all_yaml2/more_yaml/{basename:}",
    )

    assert "all_yaml2/more_yaml" in dm
    assert "all_yaml2/more_yaml/foobar" in dm

    # Ignore some files and assert that they were not loaded
    dm.load(
        "some_more_yaml",
        loader="yaml",
        glob_str="**/*.yml",
        ignore=["*lamo*", "missing.yml"],
    )

    assert "some_more_yaml/foobar" in dm
    assert "some_more_yaml/missing" not in dm
    assert "some_more_yaml/lamo" not in dm
    assert "some_more_yaml/also_lamo" not in dm

    print(dm.tree)

    # If given a list of glob strings, possibly matching files more than once,
    # they should only be loaded once (internally using a set)
    dm.load("multiglob", loader="yaml", glob_str=["*.yml", "*yml"])
    assert len(dm["multiglob"]) == 4  # 8 files match, only 4 should be loaded

    print(dm.tree)

    # It is also possible to load by giving a custom base_path, which is in
    # this case just the same directory, but explicitly given
    dm.load(
        "custom_base_path",
        loader="yaml",
        base_path=dm.dirs["data"],
        glob_str=["*.yml", "*yml"],
        ignore=["*.bad_yml"],
        required=True,
    )
    assert len(dm["custom_base_path"]) == 4

    print(dm.tree)

    # Can also match directory paths
    dm.load(
        "local_dirs",
        loader="fspath",
        glob_str="*",
        recursive=False,
        include_directories=True,
        include_files=False,
        required=True,
    )
    print(dm["local_dirs"].tree)
    assert "local_dirs/sub" in dm
    assert "local_dirs/merged" in dm
    assert len(dm["local_dirs"]) == 2

    dm.load(
        "all_files",
        loader="fspath",
        glob_str="**/*",
        recursive=True,
        include_directories=False,
        include_files=True,
        required=True,
    )
    print(dm["all_files"].tree)
    assert "all_files/foobar" in dm
    assert "all_files/abc123" in dm
    assert "all_files/abc234" in dm
    assert "all_files/sub" not in dm
    assert "all_files/merged" not in dm
    assert len(dm["all_files"]) == 15


def test_loading_errors(dm):
    """Test the cases in which errors are raised or warnings are created."""
    # Load some data that will create collisions
    dm.load_from_cfg(
        load_cfg=dict(barfoo=dict(loader="yaml", glob_str="foobar.yml")),
        print_tree=True,
    )

    # With name collisions, an error should be raised
    with pytest.raises(ExistingDataError):
        dm.load("barfoo", loader="yaml", glob_str="foobar.yml")

    # Unless loading is disabled anyway
    dm.load("barfoo", loader="yaml", glob_str="foobar.yml", enabled=False)

    # Relative base paths should not work
    with pytest.raises(ValueError, match="needs be an absolute path"):
        dm.load(
            "barfoo",
            loader="yaml",
            base_path="some/rel/path",
            glob_str="foobar.yml",
        )

    # Check for missing data ..................................................
    # Check for data missing that was required
    with pytest.raises(RequiredDataMissingError):
        dm.load(
            "i_need_this", loader="yaml", glob_str="needed.yml", required=True
        )

    # Check for warning being given when data was missing but not required
    with pytest.warns(MissingDataWarning):
        dm.load("might_need_this", loader="yaml", glob_str="maybe_needed.yml")

    # Check for invalid loaders ...............................................
    with pytest.raises(LoaderError, match="Available loaders:.*text.*yaml"):
        dm.load("nopenopenope", loader="nope", glob_str="*")

    with pytest.raises(LoaderError, match="misses required attribute"):
        dm.load("nopenopenope", loader="bad_loadfunc", glob_str="*")

    # Loading itself may fail .................................................
    with pytest.raises(DataLoadingError, match="Failed loading file"):
        dm.load("failing", loader="yaml", glob_str="*.bad_yml", required=True)
    assert "failing" not in dm

    # ... but it will only warn if the data was not required
    dm.load("failing", loader="yaml", glob_str="*.bad_yml", required=False)
    assert "failing" not in dm


def test_load_func_name():
    """Makes sure that a load function named ``_load_file`` is not possible"""
    from dantro.data_loaders import add_loader

    with pytest.raises(AssertionError):

        class MyDataManager(DataManager):
            @add_loader(TargetCls=ObjectContainer)
            def _load_file(self):  # already exists, should not be overwritten!
                pass

    # This works
    class MyDataManager(DataManager):
        @add_loader(TargetCls=ObjectContainer)
        def _load_from_file(self):
            pass


def test_loading_exists_action(dm):
    """Tests whether behaviour upon existing data is as desired"""
    # Load the `barfoo` entry that will later create a collision
    dm.load_from_cfg(
        load_cfg=dict(barfoo=dict(loader="yaml", glob_str="foobar.yml")),
        print_tree=True,
    )

    # warn if loading is skipped; should still hold `barfoo` afterwards
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load(
            "barfoo",
            loader="yaml",
            glob_str="foobar.yml",
            exists_action="skip",
        )
    assert isinstance(dm["barfoo"], dantro.base.BaseDataContainer)
    assert "one" in dm["barfoo"]

    # same without warning
    dm.load(
        "barfoo",
        loader="yaml",
        glob_str="foobar.yml",
        exists_action="skip_nowarn",
    )
    assert isinstance(dm["barfoo"], dantro.base.BaseDataContainer)
    assert "one" in dm["barfoo"]

    # It should not be possible to change a container into a group
    with pytest.raises(
        dantro.data_mngr.ExistingDataError,
        match="The object at 'barfoo' in DataManager",
    ):
        dm.load(
            "barfoo",
            loader="yaml",
            glob_str="*.yml",
            target_path="barfoo/{basename:}",
            exists_action="overwrite",
        )

    # With barfoo/foobar being a container, this should also fail
    with pytest.raises(
        dantro.data_mngr.ExistingDataError,
        match="Tried to create a new group 'barfoo'",
    ):
        dm.load(
            "barfoo",
            loader="yaml",
            glob_str="*.yml",
            target_path="barfoo/foobar/{basename:}",
            exists_action="overwrite",
        )

    # Overwriting with a container should work
    dm.load(
        "barfoo",
        loader="yaml",
        glob_str="lamo.yml",
        exists_action="overwrite_nowarn",
    )
    assert isinstance(dm["barfoo"], dantro.base.BaseDataContainer)
    assert "one" not in dm["barfoo"]
    assert "nothing" in dm["barfoo"]

    # Check for invalid `exists_action` value
    with pytest.raises(ValueError):
        dm.load(
            "barfoo",
            loader="yaml",
            glob_str="foobar.yml",
            exists_action="very bad value, much illegal",
        )

    # Load a group
    dm.load("a_group", loader="yaml", glob_str="*lamo.yml")
    assert isinstance(dm["a_group"], dantro.base.BaseDataGroup)
    assert "lamo" in dm["a_group"]
    assert "also_lamo" in dm["a_group"]
    assert "foobar" not in dm["a_group"]
    assert "looooooooooong_filename" not in dm["a_group"]

    # Check that there is a warning for existing element in a group
    with pytest.warns(dantro.data_mngr.ExistingDataWarning) as record:
        dm.load(
            "more_yamls",
            loader="yaml",
            glob_str="*.yml",
            target_path="a_group/{basename:}",
            exists_action="skip",
        )
    assert len(record) == 2
    assert all(
        [
            issubclass(r.category, dantro.data_mngr.ExistingDataWarning)
            for r in record
        ]
    )

    # ...and that the elements were added
    assert "foobar" in dm["a_group"]
    assert "looooooooooong_filename" in dm["a_group"]

    # Check that a group _can_ be overwritten by a container
    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm.load("a_group", loader="yaml", glob_str="lamo.yml")

    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load(
            "a_group",
            loader="yaml",
            glob_str="lamo.yml",
            exists_action="overwrite",
        )
    assert not isinstance(dm["a_group"], dantro.base.BaseDataGroup)


def test_new_groups(dm):
    """Check that group creation from paths works"""
    # Simple creation
    dm.new_group("foobar")
    assert "foobar" in dm

    # Recursive
    dm.new_group("foo/bar/baz")
    assert "foo/bar/baz" in dm

    # A group in the path already exists
    dm.new_group("foo/bar/baz/foooo")
    assert "foo/bar/baz/foooo" in dm

    # With data existing at a path, there should be another error
    dm.load(
        "foobar",
        loader="yaml",
        glob_str="foobar.yml",
        target_path="foo/bar/baz/foobar",
    )

    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm.new_group("foo/bar/baz/foobar")


def test_loading_regex(dm):
    """Check whether regex name extraction works"""
    # Regex loading is optional
    dm.load(
        "no_regex",
        loader="yaml",
        glob_str="sub/*.yml",
        target_path="sub/{ext}/{basename}",
        print_tree=True,
    )

    assert "sub/yml/abc123" in dm
    assert "sub/yml/abcdef" in dm

    # This should work but raise a warning for the `abcdef` entry, falling
    # back to using the full basename
    with pytest.warns(dantro.data_mngr.NoMatchWarning):
        dm.load(
            "sub_foobar",
            loader="yaml",
            glob_str="sub/*.yml",
            path_regex=r"sub/abc(\d+)\.yml",
            target_path="sub_foobar/{match:}",
            print_tree=True,
        )

    assert "sub_foobar/123" in dm
    assert "sub_foobar/abcdef" in dm

    # Can also use named and unnamed groups
    dm.load(
        "named_and_unnamed_groups",
        loader="yaml",
        glob_str="sub/*.yml",
        path_regex=r"(?P<a>sub)/abc(?P<b>[\d\w]+)\.(?P<ext>.*)",
        target_path="{match}/{groups[0]}/{named[a]}/{named[ext]}/{named[b]}",
        print_tree=True,
    )

    assert "sub/sub/sub/yml/123" in dm
    assert "sub/sub/sub/yml/def" in dm

    # Bad group indices (or keys) will lead to errors
    with pytest.raises(ValueError, match="Failed evaluating"):
        dm.load(
            "fail",
            loader="yaml",
            glob_str="sub/*.yml",
            path_regex=r"sub/abc(?P<bar>[\d\w]+)\.yml",
            target_path="spam/{groups[1]}/{basename}}",
            print_tree=True,
        )

    with pytest.raises(ValueError, match="Failed evaluating"):
        dm.load(
            "fail",
            loader="yaml",
            glob_str="sub/*.yml",
            path_regex=r"sub/abc(?P<bar>[\d\w]+)\.yml",
            target_path="spam/{named[invalid]}/{basename}}",
            print_tree=True,
        )

    # There should be a warning for non-matching regex
    with pytest.warns(dantro.data_mngr.NoMatchWarning):
        dm.load(
            "more_foobar1",
            loader="yaml",
            glob_str="foobar.yml",
            path_regex="will_not_match",
            target_path="sub/{match:}",
        )

    # There should be an error if the regex is creating non-unique names
    with pytest.raises(
        dantro.data_mngr.ExistingDataError,
        match="Path 'sub_foobar/abc' already exists.",
    ):
        dm.load(
            "bad_sub_foobar",
            loader="yaml",
            glob_str="sub/*.yml",
            path_regex=r"([abc]*)\w+.yml",
            target_path="sub_foobar/{match:}",
        )


def test_load_as_attr(dm):
    """Check whether loading into attributes of existing objects works"""

    # Create a group to load into
    grp = dm.new_group("a_group")

    # Load and store as attribute of that group
    dm.load(
        "loaded_attr",
        loader="yaml",
        glob_str="foobar.yml",
        target_path="a_group",
        load_as_attr=True,
    )
    assert "loaded_attr" in grp.attrs

    # Load and unpack
    dm.load(
        "unpacked_attr",
        loader="yaml",
        glob_str="foobar.yml",
        target_path="a_group",
        load_as_attr=True,
        unpack_data=True,
    )
    assert "unpacked_attr" in grp.attrs
    assert isinstance(grp.attrs["unpacked_attr"], dict)

    # Use target_path with regex
    grp.new_group("data0")
    grp.new_group("data1")
    grp.new_group("data2")

    dm.load(
        "data",
        loader="yaml",
        glob_str="merged/data*.yml",
        path_regex=r"merged/data(\d+).yml",
        target_path="a_group/data{match:}",
        load_as_attr=True,
    )

    assert "data" in grp["data0"].attrs
    assert "data" in grp["data1"].attrs
    assert "data" in grp["data2"].attrs

    # Test that the correct exceptions are being raised
    # Attribute already existing
    with pytest.raises(
        dantro.data_mngr.ExistingDataError,
        match="attribute with the name 'loaded_attr' already",
    ):
        dm.load(
            "loaded_attr",
            loader="yaml",
            glob_str="lamo.yml",
            target_path="a_group",
            load_as_attr=True,
        )

    # Group not yet existing
    with pytest.raises(
        dantro.data_mngr.RequiredDataMissingError,
        match="a group or container already needs to exist",
    ):
        dm.load(
            "foo",
            loader="yaml",
            glob_str="lamo.yml",
            target_path="nonexisting_group",
            load_as_attr=True,
        )

    # No target path given
    with pytest.raises(
        ValueError, match="With `load_as_attr`, the `target_path`"
    ):
        dm.load(
            "foo",
            loader="yaml",
            glob_str="lamo.yml",
            target_path=None,
            load_as_attr=True,
        )


def test_target_path(dm):
    """Check whether the `target_path` argument works as desired"""

    # Bad format string give a useful error message
    with pytest.raises(ValueError, match="KeyError: 'bad_key'"):
        dm.load(
            "foo", loader="yaml", glob_str="*.yml", target_path="{bad_key:}"
        )

    # Check whether loading into a matched group will work
    dm.load(
        "merged_cfg",
        loader="yaml",
        glob_str="merged/cfg*.yml",
        path_regex=r"merged/cfg(\d+).yml",
        target_path="merged/foo{match:}/cfg",
    )

    dm.load(
        "merged_data",
        loader="yaml",
        glob_str="merged/data*.yml",
        path_regex=r"merged/data(\d+).yml",
        target_path="merged/foo{match:}/data",
    )

    # Assert that the loaded data has the desired form
    assert "merged" in dm

    assert "merged/foo0" in dm
    assert "merged/foo1" in dm
    assert "merged/foo2" in dm

    assert "merged/foo0/cfg" in dm
    assert "merged/foo1/cfg" in dm
    assert "merged/foo2/cfg" in dm

    assert "merged/foo0/data" in dm
    assert "merged/foo1/data" in dm
    assert "merged/foo2/data" in dm

    # Test the `target_group` argument
    # Giving both should fail
    with pytest.raises(ValueError, match="Received both arguments.*"):
        dm.load(
            "foo",
            loader="yaml",
            glob_str="*.yml",
            target_group="foo",
            target_path="foo",
        )

    # Giving a glob string that matches at most a single file
    dm.load(
        "foobar",
        loader="yaml",
        glob_str="foobar.yml",
        target_group="foo_group",
    )
    assert "foo_group/foobar" in dm

    # Giving a glob string that matches possibly more than one file
    dm.load(
        "barfoo", loader="yaml", glob_str="*.yml", target_group="barfoo_group"
    )
    assert "barfoo_group" in dm
    assert "barfoo_group/foobar" in dm
    assert "barfoo_group/lamo" in dm
    assert "barfoo_group/also_lamo" in dm


def test_parse_file_path(dm):
    """Tests file path parsing"""
    parse = dm._parse_file_path
    ddir = dm.dirs["data"]

    assert parse("foo/bar") == os.path.join(ddir, "foo/bar")
    assert parse("foo/b.ar") == os.path.join(ddir, "foo/b.ar")
    assert parse("foo/b", default_ext=".ar") == os.path.join(ddir, "foo/b.ar")

    assert parse("/foo/bar") == "/foo/bar"
    assert parse("/foo/b.ar") == "/foo/b.ar"
    assert parse("/foo/b", default_ext=".ar") == "/foo/b.ar"

    assert parse("~/foo") == os.path.expanduser("~/foo")
    assert parse("~/f.oo") == os.path.expanduser("~/f.oo")
    assert parse("~/f", default_ext=".oo") == os.path.expanduser("~/f.oo")


def test_parsing_parallel_loading_options(dm):
    """Tests the helper function that parses parallel loading options"""
    from dantro.data_mngr import _parse_parallel_opts

    # For the following tests, use an assumed CPU count (which makes handling
    # in the CI much more robust, where multiple cores may not be available).
    cpus = 8
    parse = lambda f, **k: _parse_parallel_opts(f, **k, cpu_count=cpus)

    # Need a file list with at least two files, otherwise will not be parallel
    # anyway. There should be plenty of files in the fixture's data directory:
    f = glob.glob(os.path.join(dm.dirs["data"], "**/*"), recursive=True)
    assert len(f) > 2

    fs = total_bytesize(f)
    assert fs > 1000

    # Test parsing function
    assert parse(f) == cpus
    assert parse(f, enabled=True) == cpus
    assert parse(f, enabled=False) == 1

    assert parse(f, processes=1) == 1
    assert parse(f, processes=3) == 3
    assert parse(f, processes=-2) == min(cpus - 2, len(f))
    assert parse(f, processes=cpus + 1) == min(cpus + 1, len(f))

    assert parse(f, min_files=0) == cpus
    assert parse(f, min_files=len(f) - 1) == cpus
    assert parse(f, min_files=len(f) + 1) == 1

    assert parse(f, min_total_size=1) == cpus
    assert parse(f, min_total_size=0.1) == cpus
    assert parse(f, min_total_size=fs - 1) == cpus
    assert parse(f, min_total_size=fs + 1) == 1

    # all combined
    assert (
        parse(f, processes=-1, min_files=len(f) - 1, min_total_size=fs - 1)
        == cpus - 1
    )

    # will never return more cpus than files
    # can mock files list here, because file size is not checked here
    assert parse(["foo", "bar"], processes=1000) == 2
    assert parse(["foo"] * cpus, processes=2 * cpus) == cpus
    assert parse(["foo"] * 10000, processes=2 * cpus) == 2 * cpus
    assert parse(["foo"] * 10000, processes=-2) == cpus - 2


def test_parallel(dm, hdf5_dm):
    """Tests basic parallel loading interface"""
    dm.load(
        "yml0", loader="yaml", glob_str="*.yml", parallel=True, required=True
    )

    assert "yml0" in dm
    assert "yml0/foobar" in dm
    assert "yml0/lamo" in dm
    assert "yml0/also_lamo" in dm
    assert "yml0/looooooooooong_filename" in dm

    # Can also use a shortcut for number of processes
    dm.load(
        "yml1",
        loader="yaml",
        glob_str="*.yml",
        parallel=-1,
        required=True,
    )

    # ... or provide more options via a dict
    dm.load(
        "yml2",
        loader="yaml",
        glob_str="*.yml",
        parallel=dict(processes=2, min_files=3),
        required=True,
    )

    # Force parallel loaders by fixing the available cpu count.
    # Then provoke an error during loading by provoding an unparseable file.
    with pytest.raises(
        DataLoadingError, match="There were 2 errors during parallel loading"
    ):
        dm.load(
            "fail",
            loader="yaml",
            glob_str="*.bad_yml",
            parallel=dict(cpu_count=4),
            required=True,
        )
    assert "fail" not in dm

    # ... which will only warn if its not required
    dm.load(
        "fail",
        loader="yaml",
        glob_str="*.bad_yml",
        parallel=True,
        required=False,
    )
    assert "fail" not in dm


# =============================================================================
# == Data Loaders =============================================================
# =============================================================================

# Registry --------------------------------------------------------------------


def test_data_loader_registry(dm):
    """Tests that even a DataManager without mixins has the full data loader
    registry available."""
    # By default, all loaders the DataManager can find should be equivalent
    # to those in the registry
    assert dm.available_loaders == sorted(dm._loader_registry.keys())

    # Can also get loaders that are not mixed in which will be read from the
    # loader registry
    assert not hasattr(dm, "_load_pickle")
    loader, _ = dm._resolve_loader("pickle")
    assert loader._orig is dm._loader_registry["pickle"]

    # Also works on aliases
    assert not hasattr(dm, "_load_pkl")
    loader, _ = dm._resolve_loader("pkl")
    assert loader._orig is dm._loader_registry["pkl"]


# FSPathLoaderMixin tests -----------------------------------------------------


def test_fspath_loader(dm):
    """Tests the Path-based loader"""
    dm.load(
        "local_dirs",
        loader="fspath",
        glob_str="*",
        recursive=False,
        include_directories=True,
        include_files=False,
        required=True,
    )
    print(dm["local_dirs"].tree)
    assert "local_dirs/sub" in dm
    assert "local_dirs/merged" in dm
    assert len(dm["local_dirs"]) == 2

    dm.load(
        "all_files",
        loader="fspath",
        glob_str="**/*",
        recursive=True,
        include_directories=False,
        include_files=True,
        required=True,
    )
    print(dm["all_files"].tree)
    assert "all_files/foobar" in dm
    assert "all_files/abc123" in dm
    assert "all_files/abc234" in dm
    assert "all_files/sub" not in dm
    assert "all_files/merged" not in dm
    assert len(dm["all_files"]) == 15

    # Loading a nested directory tree into a flat hierarchy will not work
    with pytest.raises(
        ExistingDataError, match="'nested_dirs_fail/sub' already exists"
    ):
        dm.load(
            "nested_dirs_fail",
            loader="fspath",
            glob_str="**/*",
            recursive=True,
            include_directories=True,
            include_files=False,
            required=True,
        )

    # … need to adjust target path to allow for that
    dm.load(
        "nested_to_flat",
        loader="fspath",
        glob_str="**/*",
        recursive=True,
        target_path="nested_to_flat/{relpath_cleaned:}",
        include_directories=True,
        include_files=False,
        required=True,
        target_path_kwargs=dict(join_char_replacement="_#_"),
    )

    print(dm["nested_to_flat"].tree)
    assert "nested_to_flat/merged" in dm
    assert "nested_to_flat/sub" in dm
    assert "nested_to_flat/sub_#_sub" in dm
    assert len(dm["nested_to_flat"]) == 3


def test_fstree_loader(dm):
    """Tests the tree-based loader"""
    dm.load(
        "simple",
        loader="fstree",
        glob_str=".",  # the data directory path
        required=True,
    )

    print(dm["simple"].tree)
    assert len(dm["simple"]) == 8
    assert "simple/merged" in dm
    assert "simple/sub" in dm
    assert "simple/sub/sub" in dm

    assert "simple/foobar.yml" in dm
    assert "simple/sub/sub/abc234.yml" in dm

    # Apply some filters
    dm.load(
        "filtered",
        loader="fstree",
        glob_str=".",  # the data directory path
        required=True,
        tree_glob=dict(
            glob_str=["merged/*", "*"],
            ignore=["*.bad_yml", "**/data*"],
            include_directories=False,
        ),
    )
    res = dm["filtered"]
    print(res.tree)

    assert "merged" in res
    assert "sub" not in res

    assert "foobar.yml" in res
    assert "merged/cfg0.yml" in res
    assert "not_loadable.bad_yml" not in res


# TextLoaderMixin tests -------------------------------------------------------


def test_text_loader(text_dm):
    """Test the plain text loader"""
    text_dm.load("text_data", loader="text", glob_str="text_data/*.txt")

    # Check that the plain text data is loaded and of expected type
    print(text_dm.tree)
    text_data = text_dm["text_data"]

    for name, cont in text_data.items():
        assert isinstance(cont, StringContainer)

        assert cont.data == "This is a test string \n with two lines\n"


# PickleLoaderMixin tests -----------------------------------------------------


def test_pkl_loader(pkl_dm):
    """Tests the pickle loader"""
    pkl_dm.load("pkls", loader="pickle", glob_str="pickles/*.pkl")
    pkls = pkl_dm["pkls"]

    assert len(pkls) == 3

    for name, cont in pkls.items():
        assert isinstance(cont, ObjectContainer)


# NumpyLoaderMixin tests ------------------------------------------------------


def test_numpy_loader_binary(np_dm):
    """Tests the numpy loader for binary data"""
    np_dm.load("np_data", loader="numpy", glob_str="np_data/*.npy")

    # Check that all files are loaded and of the expected type
    np_data = np_dm["np_data"]
    assert len(np_data) == 2

    for name, cont in np_data.items():
        assert isinstance(cont, NumpyDataContainer)

    # Specifically check content
    assert np_data["zeros_int"].dtype is np.dtype(int)
    assert np_data["zeros_int"].mean() == 0

    assert np_data["zeros_float"].dtype is np.dtype(float)
    assert np_data["zeros_float"].mean() == 0.0


def test_numpy_loader_txt(csv_dm):
    """Tests the numpy loader for text data"""
    dm = csv_dm
    kws = dict(loader="numpy_txt", required=True)
    dm.load("csv_data", **kws, glob_str="csv/simple*.csv")

    csv_data = dm["csv_data"]
    assert len(csv_data) == 2

    assert isinstance(csv_data["simple_int"], NumpyDataContainer)
    assert isinstance(csv_data["simple_float"], NumpyDataContainer)

    assert csv_data["simple_int"].shape == (2, 3)
    assert csv_data["simple_float"].shape == (2, 3)

    assert csv_data["simple_int"].dtype is np.dtype(float)
    assert csv_data["simple_float"].dtype is np.dtype(float)

    # Load again as ints, requiring a converter for float data
    dm.load(
        "int_data",
        **kws,
        glob_str="csv/simple*.csv",
        dtype=int,
        converters=float,
    )
    csv_data = dm["int_data"]
    assert csv_data["simple_int"].dtype is np.dtype(int)
    assert csv_data["simple_float"].dtype is np.dtype(int)

    # What about custom separators?
    dm.load(
        "custom_delim",
        **kws,
        glob_str="csv/sep*.csv",
        delimiter=",",
    )
    csv_data = dm["custom_delim"]
    assert csv_data["sep_comma"].shape == (2, 3)

    # And heterogeneous data? Needs a custom dtype
    dm.load(
        "mixed_dtypes",
        **kws,
        glob_str="csv/iris.csv",
        dtype="object",
        delimiter=",",
    )
    iris = dm["mixed_dtypes"]
    print(iris.data)
    assert iris.shape == (151, 6)
    assert iris.dtype is np.dtype(object)


# PandasLoaderMixin tests -----------------------------------------------------


def test_pandas_loader_csv(csv_dm):
    """Tests the pandas loader for CSV data"""
    dm = csv_dm
    kws = dict(loader="pandas_csv", required=True)

    # Can load everything, not requiring further arguments
    dm.load("csv_data", **kws, glob_str="csv/*.csv")

    data = dm["csv_data"]
    print(data.tree)
    assert len(data) == 7

    # Let's look at the loaded data
    penguins = data["penguins"]
    print(penguins.head())
    assert "species" in penguins.columns
    assert not np.isnan(penguins.loc[2]["bill_length_mm"])
    assert np.isnan(penguins.loc[3]["bill_length_mm"])

    # Compare loading of datasets with different separators
    dm.load("tsv_data", **kws, glob_str="csv/*.tsv", sep="\t")
    tsv_data = dm["tsv_data"]
    print(data["planets"].head())
    print(tsv_data["planets"].head())
    assert data["planets"].equals(tsv_data["planets"].data)


def test_pandas_loader_generic(csv_dm):
    """Tests the pandas loader for CSV data"""
    dm = csv_dm
    kws = dict(loader="pandas_generic", required=True)

    # Can load CSV data, just as before
    dm.load("csv_data", **kws, glob_str="csv/*.csv", reader="csv")
    data = dm["csv_data"]
    print(data.tree)
    assert len(data) == 7

    # How about excel data?
    # ... will fail because the file content is not excel and thus no engine
    #     can be determined
    with pytest.raises(DataLoadingError, match="file format cannot be det"):
        dm.load("excel_data", **kws, glob_str="**/*.csv", reader="excel")

    # Bad reader name
    with pytest.raises(DataLoadingError, match="Invalid") as exc_info:
        dm.load("fails", **kws, glob_str="*", reader="bad reader")

    # ...informs about available readers, excluding some that aren't file-based
    assert "excel, feather, fwf, hdf" in str(exc_info.value)
    assert "sql" not in str(exc_info.value)
    assert "clipboard" not in str(exc_info.value)


# XarrayLoaderMixin tests -----------------------------------------------------


def test_xarray_loader(xr_dm):
    """Tests the xarray loader"""
    xr_dm.load(
        "arrays",
        loader="xr_dataarray",
        glob_str="xr_data/*.nc_da",
        load_completely=True,
    )
    xr_dm.load(
        "dsets",
        loader="xr_dataset",
        glob_str="xr_data/*.nc_ds",
        load_completely=True,
    )

    # Check that all files are loaded and of the expected type
    das = xr_dm["arrays"]
    assert len(das) == 1
    for name, cont in das.items():
        assert isinstance(cont, XrDataContainer)

    dsets = xr_dm["dsets"]
    assert len(dsets) == 1
    for name, cont in dsets.items():
        assert isinstance(cont, PassthroughContainer)

    # Specifically check content
    assert das["zeros"].mean() == 0


# Hdf5LoaderMixin tests -------------------------------------------------------


def test_hdf5_loader_basics(hdf5_dm):
    """Test whether loading of hdf5 data works as desired"""
    hdf5_dm.load(
        "h5data", loader="hdf5", glob_str="**/*.h5", lower_case_keys=True
    )

    # Test that both files were loaded
    assert "h5data/basic" in hdf5_dm
    assert "h5data/nested" in hdf5_dm

    # Test that the basic datasets are there and their dtype & shape is correct
    assert hdf5_dm["h5data/basic/int_dset"].dtype == np.dtype(int)
    assert hdf5_dm["h5data/basic/float_dset"].dtype == np.dtype(float)

    assert hdf5_dm["h5data/basic/int_dset"].shape == (1, 2, 3)
    assert hdf5_dm["h5data/basic/float_dset"].shape == (2, 3, 4)

    # Test that attributes were loaded on file, group and dset level
    assert "foo" in hdf5_dm["h5data/basic"].attrs
    assert "foo" in hdf5_dm["h5data/basic/int_dset"].attrs
    assert "foo" in hdf5_dm["h5data/basic/group"].attrs

    # Test that keys were converted to lower case
    assert "uppercasegroup" in hdf5_dm["h5data/basic"]

    # Test that nested loading worked
    assert "h5data/nested/group1/group11/group111/dset" in hdf5_dm


def test_hdf5_proxy_loader(hdf5_dm):
    """Tests whether proxy loading of hdf5 data works"""
    hdf5_dm.load(
        "h5proxy",
        loader="hdf5_proxy",
        glob_str="**/*.h5",
        progress_params=dict(level=2),
    )

    h5data = hdf5_dm["h5proxy"]

    # Test whether the loaded datasets are proxies
    assert h5data["basic/int_dset"].data_is_proxy
    assert h5data["basic/float_dset"].data_is_proxy
    assert h5data["nested/group1/group11/group111/dset"].data_is_proxy

    # Test that dtype and shape access do not resolve
    assert h5data["basic/int_dset"].dtype == np.dtype(int)
    assert h5data["basic/float_dset"].dtype == np.dtype(float)

    assert h5data["basic/int_dset"].shape == (1, 2, 3)
    assert h5data["basic/float_dset"].shape == (2, 3, 4)

    assert h5data["basic/int_dset"].data_is_proxy
    assert h5data["basic/float_dset"].data_is_proxy

    str(h5data["basic/int_dset"].proxy)
    str(h5data["basic/float_dset"].proxy)

    # Test the resolve method, that will not change the proxy status of the
    # container the proxy is used in
    assert h5data["basic/int_dset"].data_is_proxy
    assert isinstance(h5data["basic/int_dset"].proxy.resolve(), np.ndarray)
    assert h5data["basic/int_dset"].data_is_proxy

    # Test that automatic resolution (by accessing data attribute) works
    assert isinstance(h5data["basic/float_dset"].data, np.ndarray)
    assert h5data["basic/float_dset"].data_is_proxy is False
    assert h5data["basic/float_dset"].proxy is None

    # However, when set to retain, the proxy object is retained
    h5data["nested/group1/group11/group111/dset"].PROXY_RETAIN = True
    op = h5data["nested/group1/group11/group111/dset"].proxy
    assert isinstance(
        h5data["nested/group1/group11/group111/dset"].data, np.ndarray
    )
    assert h5data["nested/group1/group11/group111/dset"].data_is_proxy is False
    assert h5data["nested/group1/group11/group111/dset"].proxy is op


def test_hdf5_mapping(hdf5_dm):
    """Tests whether container mapping works as desired, also with proxies"""
    hdf5_dm.load(
        "h5data",
        loader="hdf5_proxy",
        glob_str="**/*.h5",
        enable_mapping=True,
        required=True,
    )

    # Test that the mapping works
    assert "h5data/mapping" in hdf5_dm
    mp = hdf5_dm["h5data/mapping"]

    # Correct mapping should have yielded the custom types
    assert isinstance(mp["dummy_dset"], DummyDC)
    assert isinstance(mp["dummy_group"], DummyGroup)
    assert isinstance(mp["another_dummy"], DummyDC)
    assert isinstance(mp["one_more_dummy"], DummyDC)

    # Incorrect mapping should have loaded and yielded default types
    assert isinstance(mp["badmap_dset"], NumpyTestDC)
    assert isinstance(mp["badmap_group"], OrderedDataGroup)

    # Check again for labelled data (the more complex scenario)
    lab = hdf5_dm["h5data/labelled"]
    xrdc = lab["some_dset"]
    assert isinstance(xrdc, XrTestDC)
    assert xrdc.data_is_proxy
    xrdc.data
    assert not xrdc.data_is_proxy
    assert isinstance(xrdc.data, xr.DataArray)

    # ... linked coordinates were also loaded
    print(xrdc.coords)
    assert "z" in xrdc.coords
    assert (xrdc.coords["z"].values == [1, 2, 3, 4]).all()

    # With bad values for which attribute to use, mapping should fail
    hdf5_dm._HDF5_MAP_FROM_ATTR = None

    with pytest.raises(
        DataLoadingError, match="Could not determine from which attribute"
    ):
        hdf5_dm.load(
            "no_attr",
            loader="hdf5",
            glob_str="**/*.h5",
            ignore=["hdf5_data/labelled.h5"],
            enable_mapping=True,
            required=True,
        )

    # Explicitly passing an attribute name should work though
    hdf5_dm.load(
        "with_given_attr",
        loader="hdf5",
        glob_str="**/*.h5",
        ignore=["hdf5_data/labelled.h5"],  # would need proxy, see issue #273
        enable_mapping=True,
        map_from_attr="container_type",
        required=True,
    )


def test_hdf5_loader_parallel(hdf5_dm):
    """Test parallel loading"""
    # In a basic configuration
    hdf5_dm.load(
        "regular",
        loader="hdf5",
        glob_str="**/*.h5",
        required=True,
        parallel=True,
    )

    # With mapping and proxies features
    hdf5_dm.load(
        "as_proxy",
        loader="hdf5_proxy",
        glob_str="**/*.h5",
        enable_mapping=True,
        required=True,
        parallel=True,
    )


def test_hdf5_bytestring_conversion(hdf5_dm):
    """Tests whether bytestrings are decoded during loading"""
    hdf5_dm.load("decode_enabled", loader="hdf5", glob_str="**/*.h5")

    # Test that strings are loaded as strings
    grp = hdf5_dm["decode_enabled/basic/group"]
    assert isinstance(grp.attrs["encoded_utf8"], str)
    assert isinstance(grp.attrs["encoded_utf16"], str)
    assert grp.attrs["encoded_arr"].dtype.kind == "U"

    # Decoding works for utf8, but not for other encodings (here: utf16)
    assert grp.attrs["encoded_utf8"] == "🎉"
    assert grp.attrs["encoded_utf16"] != "👻"

    # When deactivated, the bytestring should be loaded
    # Enable the mapping to make sure that also works with decoding deactivated
    hdf5_dm._HDF5_DECODE_ATTR_BYTESTRINGS = False
    hdf5_dm.load(
        "decode_disabled",
        loader="hdf5",
        glob_str="**/*.h5",
        enable_mapping=True,
    )

    # Test that strings are NOT loaded as unicode strings now … inside arrays
    grp = hdf5_dm["decode_disabled/basic/group"]
    assert grp.attrs["encoded_arr"].dtype.kind == "S"

    # ... for other objects, h5py >= 3.0 already takes care of decoding ...
    assert isinstance(grp.attrs["encoded_utf8"], str)
    assert isinstance(grp.attrs["encoded_utf16"], str)
    assert grp.attrs["encoded_utf8"] == "🎉"

    # ... but assumes some other encoding. With decoding disabled, there's
    # nothing we can or should do about this.
    assert grp.attrs["encoded_utf16"] != "👻"


# Pickling, dumping, restoring ------------------------------------------------


def test_pickling(hdf5_dm):
    """Tests pickling"""
    dm = hdf5_dm
    dm.load("h5data", loader="hdf5", glob_str="**/*.h5", enable_mapping=True)

    # NOTE Can't easily compare dictionaries that contain numpy objects, thus
    #      use the string-based tree representation instead.
    assert pickle_roundtrip(dm).tree == dm.tree

    # Load more data (into different part of tree) but as proxy now
    dm.load("h5proxy", loader="hdf5_proxy", glob_str="**/*.h5")
    assert pickle_roundtrip(dm).tree == dm.tree

    # -- Spot tests on the round-tripped DataManager
    dmrt = pickle_roundtrip(dm)

    # Mapping should have been maintained
    assert isinstance(dmrt["h5data/mapping/dummy_dset"], DummyDC)
    assert isinstance(dmrt["h5data/mapping/dummy_group"], DummyGroup)
    assert isinstance(dmrt["h5data/mapping/another_dummy"], DummyDC)
    assert isinstance(dmrt["h5data/mapping/one_more_dummy"], DummyDC)
    assert isinstance(dmrt["h5data/mapping/badmap_dset"], NumpyTestDC)
    assert isinstance(dmrt["h5data/mapping/badmap_group"], OrderedDataGroup)

    # Proxies should not have been resolved ...
    assert dmrt["h5proxy/basic/int_dset"].data_is_proxy
    assert dmrt["h5proxy/basic/float_dset"].data_is_proxy
    assert dmrt["h5proxy/nested/group1/group11/group111/dset"].data_is_proxy

    # ... but are properly resolvable and compare as expected
    assert isinstance(dmrt["h5proxy/basic/int_dset"].data, np.ndarray)
    assert (
        dmrt["h5proxy/basic/int_dset"] == dmrt["h5data/basic/int_dset"]
    ).all()
    assert not dmrt["h5proxy/basic/int_dset"].data_is_proxy


def test_dump_and_restore(hdf5_dm):
    """Tests the DataManager dump and restore methods"""
    new_dm = lambda: Hdf5DataManager(hdf5_dm.dirs["data"], out_dir=None)

    # -- 0 -- Create and populate the reference object
    dm = hdf5_dm
    dm.load("h5data", loader="hdf5", glob_str="**/*.h5", enable_mapping=True)
    dm.load("h5proxy", loader="hdf5_proxy", glob_str="**/*.h5")

    print(dm.tree)

    # -- 1 -- Dump and restore (absolute path, empty tree)
    print("------ 1 ------")
    # Dump it to the default location
    p1 = dm.dump()

    # Check file extension and approximate file size
    assert os.path.isabs(p1)
    assert p1.endswith(".tree_cache.d3")
    d1_size = os.path.getsize(p1)
    assert 8 * 1024 < d1_size < 12 * 1024  # platform- and version-dependent!

    # Need a new DataManager in order to retain the one above for comparsion
    dm1 = new_dm()
    assert len(dm1) == 0

    # Restore it from the default location. The tree should match.
    dm1.restore()
    print("dm1", dm1.tree)
    assert dm1.tree == dm.tree

    # -- 2 -- Dump and restore (relative path, pre-populated tree)
    # Dump it
    print("------ 2 ------")
    p2 = dm.dump(path="some/local/dump2.foo")

    assert p2.endswith(".foo")
    d2_size = os.path.getsize(p2)
    assert d1_size == d2_size

    # Create a new DataManager and populate it with some data (outside the
    # to-be-merged hierarchy _and_ inside it ...)
    dm2 = new_dm()

    dm2.new_group("foo")
    dm2.new_group("foo/bar")
    dm2.new_container("foo/bar/baz", data="baz", Cls=StringContainer)
    dm2.new_container("foo/bar/spam", data="spam", Cls=StringContainer)
    dm2.new_container(
        "foo/bar/fish", data="i will persist", Cls=StringContainer
    )

    dm2.new_group("h5data")
    dm2.new_group("h5data/basic")
    dm2.new_container(
        "h5data/basic/int_dset", data="i will be replaced", Cls=StringContainer
    )

    print("dm2 (pre-populated)", dm2.tree)

    # Restore it now, merging with the existing data
    dm2.restore(from_path="some/local/dump2.foo", merge=True)
    print("dm2 (after restore)", dm2.tree)

    # The full trees cannot match, but the subtrees should
    assert dm.tree != dm2.tree
    assert dm["h5proxy"].tree == dm2["h5proxy"].tree

    # -- 3 -- Test proxy behaviour
    print("------ 3 ------")
    # Configure a container with proxy support to retain the proxy object
    proxy_path = "h5proxy/basic/int_dset"
    assert not dm[proxy_path].PROXY_RETAIN
    dm[proxy_path].PROXY_RETAIN = True

    # Resolve the proxy in the reference DataManager
    assert dm[proxy_path].data_is_proxy
    dm[proxy_path].data
    assert not dm[proxy_path].data_is_proxy
    print("dm", dm.tree)

    # Dump it
    p3 = dm.dump(path=os.path.join(dm.dirs["data"], "..", "dump1"))
    assert p3.endswith(".d3")
    d3_size = os.path.getsize(p3)
    assert d3_size > d1_size  # ... due to the proxy stuff

    # Dumping should NOT have reinstated the proxy in the reference DataManager
    print("dm", dm.tree)
    assert not dm[proxy_path].data_is_proxy

    # Restore it
    dm3 = new_dm()
    dm3.restore(from_path=p3)
    print("dm3", dm3.tree)

    # As the proxy status is different, the trees should NOT match
    assert dm3[proxy_path].data_is_proxy
    assert dm3.tree != dm.tree

    # Resolve the proxy by accessing it ...
    dm3[proxy_path].data
    assert not dm3[proxy_path].data_is_proxy

    # ... now they should be equal
    assert dm3.tree == dm.tree

    # -- 4 -- Resolved proxies w/o PROXY_REINSTATE_FOR_PICKLING store the data
    print("------ 4 ------")
    # Configure a container with proxy support to retain the proxy object
    proxy2_path = "h5proxy/basic/float_dset"
    assert not dm[proxy2_path].PROXY_RETAIN

    # Resolve the proxy in the reference DataManager
    assert dm[proxy2_path].data_is_proxy
    dm[proxy2_path].data
    assert not dm[proxy2_path].data_is_proxy
    print("dm", dm.tree)

    # Dump it
    p4 = dm.dump(path="dump4")
    assert p4.endswith(".d3")
    d4_size = os.path.getsize(p4)
    assert d4_size != d1_size
    assert d4_size != d3_size  # actually smaller than d3 here because there is
    # very little data inside proxy2_path ...

    # Restore it
    dm4 = new_dm()
    dm4.restore(from_path=p4)
    print("dm4", dm4.tree)

    # Dumping should NOT have reinstated the proxy in the reference DataManager
    # and the restored object should not have been loaded as proxy
    assert not dm[proxy2_path].data_is_proxy
    assert not dm4[proxy2_path].data_is_proxy

    # -- Errors --
    # Bad file path
    dm5 = new_dm()

    with pytest.raises(FileNotFoundError, match="no tree cache file at"):
        dm5.restore(from_path="i_dont_exist")
