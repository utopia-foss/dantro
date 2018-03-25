"""Test the DataManager"""

import os
import pkg_resources

import numpy as np
import h5py as h5
import pytest

import dantro.base
from dantro.container import NumpyDC
from dantro.mixins import NumpyProxyMixin
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

class NumpyTestDC(NumpyProxyMixin, NumpyDC):
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

    # Check the `update_load_cfg` argument
    dm.load_from_cfg(update_load_cfg=dict(barfoo2=dict(loader="yaml",
                                                       glob_str="foobar.yml")),
                     print_tree=True)

    # Invalid load config
    with pytest.raises(TypeError):
        dm.load_from_cfg(update_load_cfg=dict(barfoo2=[1,2,3]))

    # Assert that the top level entries are all available and content is right
    assert 'barfoo' in dm

    barfoo = dm['barfoo']
    assert barfoo['one'] == 1
    assert barfoo['two'] == 2
    assert barfoo['go_deeper']['eleven'] == 11
    assert barfoo['a_list'] == list(range(10))

    # Check single entry loading ..............................................
    # Load another single entry, this time forcing a group to be created
    dm.load('barbaz', loader='yaml', glob_str="foobar.yml",
            always_create_group=True, print_tree=True)

    assert 'barbaz' in dm
    assert 'barbaz/foobar' in dm

    # Load again, this time with more data
    dm.load('all_yaml', loader='yaml', glob_str="*.yml")

    assert 'all_yaml' in dm
    assert 'all_yaml/foobar' in dm
    assert 'all_yaml/lamo' in dm
    assert 'all_yaml/also_lamo' in dm
    assert 'all_yaml/looooooooooong_filename' in dm

    # Now see what happens if loading into an existing target_group is desired
    dm.load('more_yaml', loader='yaml', glob_str="*.yml",
            target_group="all_yaml")

    assert 'all_yaml/more_yaml' in dm
    assert 'all_yaml/more_yaml/foobar' in dm
    assert 'all_yaml/more_yaml/lamo' in dm

    # ...and into a non-existing one
    dm.load('more_yaml', loader='yaml', glob_str="*.yml",
            target_group="all_yaml2")

    assert 'all_yaml2/more_yaml' in dm
    assert 'all_yaml2/more_yaml/foobar' in dm

    # Ignore some files and assert that they were not loaded
    dm.load('some_more_yaml', loader='yaml', glob_str="**/*.yml",
            ignore=["lamo.yml", "missing.yml"])

    assert 'some_more_yaml/foobar' in dm
    assert 'some_more_yaml/missing' not in dm
    assert 'some_more_yaml/lamo' not in dm


    # This should fail if more than one group would need to be created
    with pytest.raises(NotImplementedError):
        dm.load('more_yaml', loader='yaml', glob_str="*.yml",
                target_group="all/yaml/goes/here")

    # With name collisions, an error should be raised
    with pytest.raises(dantro.data_mngr.ExistingDataError):
        dm.load('barfoo', loader='yaml', glob_str="*.yml")

    # Test different `exist_action` values ....................................
    # warn if loading is skipped; should still hold `barfoo` afterwards
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                exists_action='skip')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # same without warning
    dm.load('barfoo', loader='yaml', glob_str="*.yml",
            exists_action='skip_nowarn')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # with overwriting, the content should change
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                exists_action='overwrite',)
    assert isinstance(dm['barfoo'], dantro.base.BaseDataGroup)

    # overwrite again with the old one
    dm.load('barfoo', loader='yaml', glob_str="foobar.yml",
            exists_action='overwrite_nowarn')
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # Check for invalid `exists_action` value
    with pytest.raises(ValueError):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                exists_action='very bad value, much illegal')

    # Check that there is a warning for update
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        dm.load('barfoo', loader='yaml', glob_str="*.yml",
                exists_action='update')

    print("{:tree}".format(dm))

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

    # Check regex stuff .......................................................
    # Check whether regex name extraction works
    with pytest.warns(UserWarning):
        dm.load('sub_foobar', loader='yaml', glob_str="sub/*.yml",
                always_create_group=True, path_regex='([0-9]*).yml')

    assert 'sub_foobar/123' in dm
    assert 'sub_foobar/abcdef' in dm

    # There should be an error if the regex is creating non-unique names
    with pytest.raises(dantro.data_mngr.ExistingDataError,
                       match='.*resolves to unique names.*'):
        dm.load('bad_sub_foobar', loader='yaml', glob_str="sub/*.yml",
                always_create_group=True, path_regex='([abc]*)\w+.yml')

    # There should be a warning for a bad regex
    with pytest.warns(UserWarning):
        dm.load('more_foobar1', loader='yaml', glob_str="foobar.yml",
                path_regex='will_not_match')

    # ... or if trying to regex something that will not be loaded into a group
    with pytest.warns(UserWarning):
        dm.load('more_foobar2', loader='yaml', glob_str="foobar.yml",
                path_regex='(foo)*.yml')

# Hdf5LoaderMixin tests -------------------------------------------------------

def test_hdf5_loader(hdf5_dm):
    """Test whether loading of hdf5 data works as desired"""
    hdf5_dm.load('h5data', loader='hdf5', glob_str="**/*.h5")

    # Test that both files were loaded
    assert 'h5data/basic' in hdf5_dm
    assert 'h5data/nested' in hdf5_dm

    # Test that the basic datasets are there and their dtype is correct
    assert hdf5_dm['h5data/basic/int_dset']._data.dtype == np.dtype(int)
    assert hdf5_dm['h5data/basic/float_dset']._data.dtype == np.dtype(float)

    # Test that attributes were loaded on file, group and dset level
    assert 'foo' in hdf5_dm['h5data/basic'].attrs
    assert 'foo' in hdf5_dm['h5data/basic/int_dset'].attrs
    assert 'foo' in hdf5_dm['h5data/basic/group'].attrs

    # Test that nested loading worked
    assert 'h5data/nested/group1/group11/group111/dset' in hdf5_dm

def test_hdf5_proxy_loader(hdf5_dm):
    """Tests whether proxy loading of hdf5 data works"""
    hdf5_dm.load('h5proxy', loader='hdf5', glob_str="**/*.h5")
