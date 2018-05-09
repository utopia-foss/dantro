"""Test the DataManager"""

import os
import pkg_resources

import numpy as np
import h5py as h5
import pytest

import dantro.base
from dantro.container import NumpyDataContainer
from dantro.mixins import Hdf5ProxyMixin
import dantro.data_mngr
from dantro.data_loaders import YamlLoaderMixin, Hdf5LoaderMixin
from dantro.tools import write_yml

# Local constants
LOAD_CFG_PATH = pkg_resources.resource_filename('tests', 'cfg/load_cfg.yml')

# Test classes ----------------------------------------------------------------

class DataManager(YamlLoaderMixin, dantro.data_mngr.DataManager):
    """A DataManager-derived class for testing the implementation"""
    
    def _load_bad_loadfunc(self):
        pass

class NumpyTestDC(Hdf5ProxyMixin, NumpyDataContainer):
    """A data container class that provides numpy proxy access"""
    pass

class Hdf5DataManager(Hdf5LoaderMixin, DataManager):
    """A DataManager-derived class to test the Hdf5LoaderMixin class"""
    # Define the class to use for loading the datasets
    _HDF5_DSET_DEFAULT_CLS = NumpyTestDC

# Fixtures --------------------------------------------------------------------

@pytest.fixture
def data_dir(tmpdir) -> str:
    """Writes some dummy data to a temporary directory and returns the path to that directory"""
    # Create YAML dummy data and write it out
    foobar = dict(one=1, two=2,
                  go_deeper=dict(eleven=11),
                  a_list=list(range(10)))

    lamo = dict(nothing="to see here")

    write_yml(foobar, path=tmpdir.join("foobar.yml"))
    write_yml(lamo, path=tmpdir.join("lamo.yml"))
    write_yml(lamo, path=tmpdir.join("also_lamo.yml"))
    write_yml(lamo, path=tmpdir.join("looooooooooong_filename.yml"))

    subdir = tmpdir.mkdir("sub")
    write_yml(foobar, path=subdir.join("abc123.yml"))
    write_yml(foobar, path=subdir.join("abcdef.yml"))

    merged = tmpdir.mkdir("merged")
    write_yml(foobar, path=merged.join("data0.yml"))
    write_yml(foobar, path=merged.join("data1.yml"))
    write_yml(foobar, path=merged.join("data2.yml"))
    write_yml(foobar, path=merged.join("cfg0.yml"))
    write_yml(foobar, path=merged.join("cfg1.yml"))
    write_yml(foobar, path=merged.join("cfg2.yml"))

    return tmpdir

@pytest.fixture
def dm(data_dir) -> DataManager:
    """Returns a DataManager without load configuration"""
    return DataManager(data_dir, out_dir=None)

@pytest.fixture
def hdf5_dm(data_dir) -> Hdf5DataManager:
    """Returns a Hdf5DataManager without load configuration.

    Additionally to the yaml files in the data_dir, some hdf5 files with dummy
    data are added.
    """
    # Create a subdirectory for that data
    h5dir = data_dir.mkdir("hdf5_data")

    # Write some basics: a dataset, a group, an attribute
    basic = h5.File(h5dir.join("basic.h5"))
    basic.create_dataset("float_dset", data=np.zeros((2,3,4), dtype=float))
    basic.create_dataset("int_dset", data=np.ones((1,2,3), dtype=int))
    basic.create_group("group")
    basic.attrs['foo'] = "file level attribute"
    basic['group'].attrs['foo'] = "group level attribute"
    basic['int_dset'].attrs['foo'] = "dset level attribute"
    basic.close()

    # Write nested groups
    nested = h5.File(h5dir.join("nested.h5"))
    nested.create_group('group1')
    nested.create_group('group2')
    nested['group1'].create_group('group11')
    nested['group1'].create_group('group12')
    nested['group2'].create_group('group21')
    nested['group2'].create_group('group22')
    nested['group1']['group11'].create_group('group111')
    nested['group1']['group11']['group111'].create_dataset('dset', data=np.random.random(size=(3,4,5)))
    nested.close()

    return Hdf5DataManager(data_dir, out_dir=None)

# General tests ---------------------------------------------------------------

def test_init(data_dir):
    """Test the initialisation of a DataManager"""
    # Initialise via path to yaml file
    dm = DataManager(data_dir, out_dir=None, load_cfg=LOAD_CFG_PATH)

    # Assert that only the data directory is available
    assert dm.dirs['data'] == data_dir
    assert dm.dirs['out'] is False

    # Initialise via dict and with default out dir
    dm = DataManager(data_dir,
                     load_cfg=dict(test=dict(loader="yaml", glob_str="*.yml")))

    # Assert folders are existing and correctly linked
    assert dm.dirs['data'] == data_dir
    assert os.path.isdir(dm.dirs['data'])
    assert os.path.isdir(dm.dirs['out'])

def test_loading(dm):
    """Tests whether loading works by using the default DataManager, i.e. that
    with the YamlLoaderMixin ...
    """
    # Check loading from config dict or file ..................................
    # No load config given
    dm.load_from_cfg()

    # Single entry
    dm.load_from_cfg(load_cfg=dict(barfoo=dict(loader="yaml",
                                               glob_str="foobar.yml")),
                     print_tree=True)

    # Assert that the top level entries are all available and content is right
    assert 'barfoo' in dm

    barfoo = dm['barfoo']
    assert barfoo['one'] == 1
    assert barfoo['two'] == 2
    assert barfoo['go_deeper']['eleven'] == 11
    assert barfoo['a_list'] == list(range(10))

    # Check the `update_load_cfg` argument
    dm.load_from_cfg(update_load_cfg=dict(barfoo2=dict(loader="yaml",
                                                       glob_str="foobar.yml")),
                     print_tree=True)

    # Invalid load config
    with pytest.raises(TypeError):
        dm.load_from_cfg(update_load_cfg=dict(barfoo2=[1,2,3]))

    # Check single entry loading ..............................................
    # Load another single entry, this time forcing a group to be created
    dm.load('barbaz', loader='yaml', glob_str="foobar.yml",
            print_tree=True)

    assert 'barbaz' in dm

    # Load again, this time with more data
    dm.load('all_yaml', loader='yaml', glob_str="*.yml")

    assert 'all_yaml' in dm
    assert 'all_yaml/foobar' in dm
    assert 'all_yaml/lamo' in dm
    assert 'all_yaml/also_lamo' in dm
    assert 'all_yaml/looooooooooong_filename' in dm

    # Now see what happens if loading into an existing target_path
    dm.load('more_yaml', loader='yaml', glob_str="*.yml",
            target_path="all_yaml/more_yaml/{basename:}")

    assert 'all_yaml/more_yaml' in dm
    assert 'all_yaml/more_yaml/foobar' in dm
    assert 'all_yaml/more_yaml/lamo' in dm

    # ...and into a non-existing one
    dm.load('more_yaml', loader='yaml', glob_str="*.yml",
            target_path="all_yaml2/more_yaml/{basename:}")

    assert 'all_yaml2/more_yaml' in dm
    assert 'all_yaml2/more_yaml/foobar' in dm

    # Ignore some files and assert that they were not loaded
    dm.load('some_more_yaml', loader='yaml', glob_str="**/*.yml",
            ignore=["lamo.yml", "missing.yml"])

    assert 'some_more_yaml/foobar' in dm
    assert 'some_more_yaml/missing' not in dm
    assert 'some_more_yaml/lamo' not in dm

    print("{:tree}".format(dm))

    # If given a list of glob strings, possibly matching files more than once, they should only be loaded once
    dm.load('multiglob', loader='yaml', glob_str=['*.yml', '*yml'])
    assert len(dm['multiglob']) == 4  # 8 files match, only 4 should be loaded
    
    print("{:tree}".format(dm))


def test_loading_errors(dm):
    """Test the cases in which errors are raised or warnings are created."""
    # Load some data that will create collisions
    dm.load_from_cfg(load_cfg=dict(barfoo=dict(loader="yaml",
                                               glob_str="foobar.yml")),
                     print_tree=True)

    # With name collisions, an error should be raised
    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm.load('barfoo', loader='yaml', glob_str="foobar.yml")

    # Check for missing data ..................................................
    # Check for data missing that was required
    with pytest.raises(dantro.data_mngr.RequiredDataMissingError):
        dm.load('i_need_this', loader='yaml', glob_str="needed.yml",
                required=True)

    # Check for warning being given when data was missing but not required
    with pytest.warns(dantro.data_mngr.MissingDataWarning):
        dm.load('might_need_this', loader='yaml', glob_str="maybe_needed.yml")


    # Check for invalid loaders ...............................................
    with pytest.raises(dantro.data_mngr.LoaderError):
        dm.load('nopenopenope', loader='nope', glob_str="*.")
    
    with pytest.raises(dantro.data_mngr.LoaderError):
        dm.load('nopenopenope', loader='bad_loadfunc', glob_str="*")

    print("{:tree}".format(dm))


def test_loading_exists_action(dm):
    """Tests whether behaviour upon existing data is as desired"""
    # Load the `barfoo` entry that will later create a collision
    dm.load_from_cfg(load_cfg=dict(barfoo=dict(loader="yaml",
                                               glob_str="foobar.yml")),
                     print_tree=True)

    # warn if loading is skipped; should still hold `barfoo` afterwards
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load('barfoo', loader='yaml', glob_str="foobar.yml",
                exists_action='skip')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)
    assert 'one' in dm['barfoo']

    # same without warning
    dm.load('barfoo', loader='yaml', glob_str="foobar.yml",
            exists_action='skip_nowarn')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)
    assert 'one' in dm['barfoo']

    # It should not be possible to change a container into a group
    with pytest.raises(dantro.data_mngr.ExistingDataError,
                       match="The object at 'barfoo' in DataManager"):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                target_path='barfoo/{basename:}',
                exists_action='overwrite')

    # With barfoo/foobar being a container, this should also fail
    with pytest.raises(dantro.data_mngr.ExistingDataError,
                       match="Tried to create a group 'barfoo'"):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                target_path='barfoo/foobar/{basename:}',
                exists_action='overwrite')

    # Overwriting with a container should work
    dm.load('barfoo', loader='yaml', glob_str="lamo.yml",
            exists_action='overwrite_nowarn')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)
    assert 'one' not in dm['barfoo']
    assert 'nothing' in dm['barfoo']

    # Check for invalid `exists_action` value
    with pytest.raises(ValueError):
        dm.load('barfoo', loader='yaml', glob_str="foobar.yml",
                exists_action='very bad value, much illegal')

    # Load a group
    dm.load('a_group', loader='yaml', glob_str="*lamo.yml")
    assert isinstance(dm['a_group'], dantro.base.BaseDataGroup)
    assert 'lamo' in dm['a_group']
    assert 'also_lamo' in dm['a_group']
    assert 'foobar' not in dm['a_group']
    assert 'looooooooooong_filename' not in dm['a_group']

    # Check that there is a warning for existing element in a group
    with pytest.warns(None) as record:
        dm.load('more_yamls', loader='yaml', glob_str="*.yml",
                target_path='a_group/{basename:}',
                exists_action='skip')
    assert len(record) == 2
    assert all([issubclass(r.category, dantro.data_mngr.ExistingDataWarning)
                for r in record])

    # ...and that the elements were added
    assert 'foobar' in dm['a_group']
    assert 'looooooooooong_filename' in dm['a_group']

    # Check that a group _can_ be overwritten by a container
    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm.load('a_group', loader='yaml', glob_str="lamo.yml")

    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load('a_group', loader='yaml', glob_str="lamo.yml",
                exists_action='overwrite')
    assert not isinstance(dm['a_group'], dantro.base.BaseDataGroup)

def test_contains_group(dm):
    """Assert that the contains_group method works."""
    dm.load('group', loader='yaml', glob_str='*.yml')
    dm.load('subgroup', loader='yaml', glob_str='*.yml',
            target_path='group/subgroup/{basename:}')
    dm.load('subsubgroup', loader='yaml', glob_str='*.yml',
            target_path='group/subgroup/subsubgroup/{basename:}')

    assert dm._contains_group("group")
    assert dm._contains_group("group/subgroup")
    assert dm._contains_group("group/subgroup/subsubgroup")
    assert not dm._contains_group("group/foobar")
    assert not dm._contains_group("group/subgroup/foobar")
    assert not dm._contains_group("group/subgroup/subsubgroup/foobar")
    assert not dm._contains_group("i_dont_exist")
    assert not dm._contains_group("group/i_dont_exist")
    assert not dm._contains_group("group/i_dont_exist/i_dont_exist")

def test_create_groups(dm):
    """Check that group creation from paths works"""
    # Simple creation
    dm._create_groups("foobar")
    assert "foobar" in dm

    # Recursive
    dm._create_groups("foo/bar/baz")
    assert "foo/bar/baz" in dm

    # A group in the path already exists
    dm._create_groups("foo/bar/baz/foooo")
    assert "foo/bar/baz/foooo" in dm

    # Error with exist_ok=False
    with pytest.raises(dantro.data_mngr.ExistingGroupError):
        dm._create_groups("foo/bar", exist_ok=False)

    # With data existing at a path, there should be another error
    dm.load('foobar', loader='yaml', glob_str='foobar.yml',
            target_path='foo/bar/baz/foobar')

    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm._create_groups("foo/bar/baz/foobar")

    
def test_loading_regex(dm):
    """Check whether regex name extraction works"""
    # This should raise a warning for the `abcdef` entry
    with pytest.warns(dantro.data_mngr.NoMatchWarning):
        dm.load('sub_foobar', loader='yaml', glob_str="sub/*.yml",
                path_regex='sub/abc(\d+).yml',
                target_path='sub_foobar/{match:}',
                print_tree=True)

    assert 'sub_foobar/123' in dm
    assert 'sub_foobar/abcdef' in dm

    # There should be a warning for non-matching regex
    with pytest.warns(dantro.data_mngr.NoMatchWarning):
        dm.load('more_foobar1', loader='yaml', glob_str="foobar.yml",
                path_regex='will_not_match', target_path='sub/{match:}')

    # There should be an error if the `match` key is not used in target_path
    with pytest.raises(ValueError, match="Received the `path_regex` argument"):
        dm.load('more_foobar2', loader='yaml', glob_str="foobar.yml",
                path_regex='.*')

    # There should be an error if the regex is creating non-unique names
    with pytest.raises(dantro.data_mngr.ExistingDataError,
                       match="Path 'sub_foobar/abc' already exists."):
        dm.load('bad_sub_foobar', loader='yaml', glob_str="sub/*.yml",
                path_regex='([abc]*)\w+.yml',
                target_path='sub_foobar/{match:}')


def test_target_path(dm):
    """Check whether the `target_path` argument works as desired"""

    # Bad format string will fail
    with pytest.raises(ValueError, match="Invalid argument `target_path`."):
        dm.load('foo', loader='yaml', glob_str="*.yml",
                target_path="{bad_key:}")

    # Check whether loading into a matched group will work
    dm.load('merged_cfg', loader='yaml', glob_str="merged/cfg*.yml",
            path_regex='merged/cfg(\d+).yml',
            target_path='merged/foo{match:}/cfg')
    
    dm.load('merged_data', loader='yaml', glob_str="merged/data*.yml",
            path_regex='merged/data(\d+).yml',
            target_path='merged/foo{match:}/data')

    # Assert that the loaded data has the desired form
    assert 'merged' in dm

    assert 'merged/foo0' in dm
    assert 'merged/foo1' in dm
    assert 'merged/foo2' in dm

    assert 'merged/foo0/cfg' in dm
    assert 'merged/foo1/cfg' in dm
    assert 'merged/foo2/cfg' in dm

    assert 'merged/foo0/data' in dm
    assert 'merged/foo1/data' in dm
    assert 'merged/foo2/data' in dm

    # Test the `target_group` argument
    # Giving both should fail
    with pytest.raises(ValueError, match="Received both arguments.*"):
        dm.load('foo', loader='yaml', glob_str="*.yml",
                target_group='foo',
                target_path='foo')

    # Giving a glob string that matches at most a single file
    dm.load('foobar', loader='yaml', glob_str="foobar.yml",
            target_group='foo_group')
    assert 'foo_group/foobar' in dm

    # Giving a glob string that matches possibly more than one file
    dm.load('barfoo', loader='yaml', glob_str="*.yml",
            target_group='barfoo_group')
    assert 'barfoo_group' in dm
    assert 'barfoo_group/foobar' in dm
    assert 'barfoo_group/lamo' in dm
    assert 'barfoo_group/also_lamo' in dm


# Hdf5LoaderMixin tests -------------------------------------------------------

def test_hdf5_loader(hdf5_dm):
    """Test whether loading of hdf5 data works as desired"""
    hdf5_dm.load('h5data', loader='hdf5', glob_str="**/*.h5")

    # Test that both files were loaded
    assert 'h5data/basic' in hdf5_dm
    assert 'h5data/nested' in hdf5_dm

    # Test that the basic datasets are there and their dtype & shape is correct
    assert hdf5_dm['h5data/basic/int_dset'].dtype == np.dtype(int)
    assert hdf5_dm['h5data/basic/float_dset'].dtype == np.dtype(float)

    assert hdf5_dm['h5data/basic/int_dset'].shape == (1,2,3)
    assert hdf5_dm['h5data/basic/float_dset'].shape == (2,3,4)

    # Test that attributes were loaded on file, group and dset level
    assert 'foo' in hdf5_dm['h5data/basic'].attrs
    assert 'foo' in hdf5_dm['h5data/basic/int_dset'].attrs
    assert 'foo' in hdf5_dm['h5data/basic/group'].attrs

    # Test that nested loading worked
    assert 'h5data/nested/group1/group11/group111/dset' in hdf5_dm


def test_hdf5_proxy_loader(hdf5_dm):
    """Tests whether proxy loading of hdf5 data works"""
    hdf5_dm.load('h5proxy', loader='hdf5_proxy', glob_str="**/*.h5",
                 print_params=dict(level=2))

    h5data = hdf5_dm['h5proxy']

    # Test whether the loaded datasets are proxies
    assert h5data['basic/int_dset'].data_is_proxy
    assert h5data['basic/float_dset'].data_is_proxy
    assert h5data['nested/group1/group11/group111/dset'].data_is_proxy

    # Test that dtype and shape access do not resolve
    assert h5data['basic/int_dset'].dtype == np.dtype(int)
    assert h5data['basic/float_dset'].dtype == np.dtype(float)

    assert h5data['basic/int_dset'].shape == (1,2,3)
    assert h5data['basic/float_dset'].shape == (2,3,4)

    assert h5data['basic/int_dset'].data_is_proxy
    assert h5data['basic/float_dset'].data_is_proxy

    # Test the resolve method, that will not change the proxy status of the
    # container the proxy is used in
    assert h5data['basic/int_dset'].data_is_proxy
    assert isinstance(h5data['basic/int_dset'].proxy.resolve(), np.ndarray)
    assert h5data['basic/int_dset'].data_is_proxy

    # Test that automatic resolution (by accessing data attribute) works
    assert isinstance(h5data['basic/float_dset'].data, np.ndarray)
    assert h5data['basic/float_dset'].data_is_proxy is False
    
    assert isinstance(h5data['nested/group1/group11/group111/dset'].data,
                      np.ndarray)
    assert h5data['nested/group1/group11/group111/dset'].data_is_proxy is False
