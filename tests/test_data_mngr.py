"""Test the DataManager"""

import os
import pkg_resources

import pytest

import dantro.base
import dantro.data_mngr
from dantro.data_loaders import YamlLoaderMixin
from dantro.tools import write_yml

# Local constants
LOAD_CFG_PATH = pkg_resources.resource_filename('tests', 'cfg/load_cfg.yml')

# Test class ------------------------------------------------------------------

class DataManager(YamlLoaderMixin, dantro.data_mngr.DataManager):
    """A DataManager-derived class for testing the implementation"""
    
    def _load_bad_loadfunc(self):
        pass


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

# Tests -----------------------------------------------------------------------

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
    """Tests whether loading works"""
    def load_into_dm(*, dm=dm, update_load_cfg: dict=None, print_tree=True, exists_behaviour='raise', **load_cfg):
        """Shortcut for loading into the dm"""
        dm.load_data(load_cfg=load_cfg,
                     update_load_cfg=update_load_cfg,
                     exists_behaviour=exists_behaviour,
                     print_tree=print_tree)


    # No load config given
    load_into_dm()

    # Single entry
    load_into_dm(barfoo=dict(loader="yaml", glob_str="foobar.yml"))

    # Check the `update_load_cfg` argument
    load_into_dm(update_load_cfg=dict(barfoo2=dict(loader="yaml",
                                                  glob_str="foobar.yml")))

    # Invalid load config
    with pytest.raises(TypeError):
        load_into_dm(update_load_cfg=dict(barfoo2=[1,2,3]))


    # Assert that the top level entries are all available and content is right
    assert 'barfoo' in dm

    barfoo = dm['barfoo']
    assert barfoo['one'] == 1
    assert barfoo['two'] == 2
    assert barfoo['go_deeper']['eleven'] == 11
    assert barfoo['a_list'] == list(range(10))


    # Load another single entry, this time forcing a group to be created
    load_into_dm(barbaz=dict(loader='yaml', glob_str="foobar.yml",
                             always_create_group=True))

    assert 'barbaz' in dm
    assert 'barbaz/foobar' in dm

    # Load again, this time with more data
    load_into_dm(all_yaml=dict(loader='yaml', glob_str="*.yml"))

    assert 'all_yaml' in dm
    assert 'all_yaml/foobar' in dm
    assert 'all_yaml/lamo' in dm
    assert 'all_yaml/also_lamo' in dm
    assert 'all_yaml/looooooooooong_filename' in dm

    # Now see what happens if loading into an existing target_group is desired
    load_into_dm(more_yaml=dict(loader='yaml', glob_str="*.yml",
                                target_group="all_yaml"))

    assert 'all_yaml/more_yaml' in dm
    assert 'all_yaml/more_yaml/foobar' in dm
    assert 'all_yaml/more_yaml/lamo' in dm

    # ...and into a non-existing one
    load_into_dm(more_yaml=dict(loader='yaml', glob_str="*.yml",
                                target_group="all_yaml2"))

    assert 'all_yaml2/more_yaml' in dm
    assert 'all_yaml2/more_yaml/foobar' in dm

    # Ignore some files and assert that they were not loaded
    load_into_dm(some_more_yaml=dict(loader='yaml', glob_str="**/*.yml",
                                     ignore=["lamo.yml", "missing.yml"]))

    assert 'some_more_yaml/foobar' in dm
    assert 'some_more_yaml/missing' not in dm
    assert 'some_more_yaml/lamo' not in dm


    # This should fail if more than one group would need to be created
    with pytest.raises(NotImplementedError):
        load_into_dm(more_yaml=dict(loader='yaml', glob_str="*.yml",
                                    target_group="all/yaml/goes/here"))        

    # With name collisions, an error should be raised
    with pytest.raises(dantro.data_mngr.ExistingDataError):
        load_into_dm(barfoo=dict(loader='yaml', glob_str="*.yml"))

    # warn if loading is skipped; should still hold `barfoo` afterwards
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        load_into_dm(exists_behaviour='skip',
                     barfoo=dict(loader='yaml', glob_str="*.yml"))

    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # same without warning
    load_into_dm(exists_behaviour='skip_nowarn',
                 barfoo=dict(loader='yaml', glob_str="*.yml"))
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # with overwriting, the content should change
    with pytest.warns(dantro.data_mngr.ExistingDataWarning):
        load_into_dm(exists_behaviour='overwrite',
                     barfoo=dict(loader='yaml', glob_str="*.yml"))
    assert isinstance(dm['barfoo'], dantro.base.BaseDataGroup)

    # overwrite again with the old one
    load_into_dm(exists_behaviour='overwrite_nowarn',
                 barfoo=dict(loader='yaml', glob_str="foobar.yml"))
    assert isinstance(dm['barfoo'], dantro.base.BaseDataContainer)

    # Check for invalid `exists_behaviour` value
    with pytest.raises(ValueError):
        load_into_dm(exists_behaviour='very bad value, much illegal',
                     barfoo=dict(loader='yaml', glob_str="*.yml"))

    # Check for data missing that was required
    with pytest.raises(dantro.data_mngr.RequiredDataMissingError):
        load_into_dm(i_need_this=dict(loader='yaml', glob_str="needed.yml",
                                      required=True))

    # Check for warning being given when data was missing but not required
    with pytest.warns(dantro.data_mngr.MissingDataWarning):
        load_into_dm(might_need_this=dict(loader='yaml',
                                          glob_str="maybe_needed.yml",))

    # Check for invalid loaders
    with pytest.raises(dantro.data_mngr.LoaderError):
        load_into_dm(nopenopenope=dict(loader='nope', glob_str="*."))
    
    with pytest.raises(dantro.data_mngr.LoaderError):
        load_into_dm(nopenopenope=dict(loader='bad_loadfunc', glob_str="*"))

    # Check whether regex name extraction works
    with pytest.warns(UserWarning):
        load_into_dm(sub_foobar=dict(loader='yaml', glob_str="sub/*.yml",
                                     always_create_group=True,
                                     path_regex='([0-9]*).yml'))

    assert 'sub_foobar/123' in dm
    assert 'sub_foobar/abcdef' in dm

    # There should be an error if the regex is creating non-unique names
    with pytest.raises(dantro.data_mngr.ExistingDataError,
                       match='.*resolves to unique names.*'):
        load_into_dm(bad_sub_foobar=dict(loader='yaml', glob_str="sub/*.yml",
                                         always_create_group=True,
                                         path_regex='([abc]*)\w+.yml'))    

    # There should be a warning for a bad regex
    with pytest.warns(UserWarning):
        load_into_dm(more_foobar1=dict(loader='yaml', glob_str="foobar.yml",
                                       path_regex='will_not_match'))

    # ... or if trying to regex something that will not be loaded into a group
    with pytest.warns(UserWarning):
        load_into_dm(more_foobar2=dict(loader='yaml', glob_str="foobar.yml",
                                       path_regex='(foo)*.yml'))


