"""A test file that is used to implement usage examples which are then
literal-included into the documentation. This makes sure the usage examples
actually work.

Each of the includes has a start and end string:

    ### Start -- example_name
    # ... to be included code here
    ### End ---- example_name

It can be included into the sphinx documentation using:

    .. literalinclude:: ../tests/test_doc_examples.py
        :language: python
        :start-after: ### Start -- example_name
        :end-before:  ### End ---- example_name
        :dedent: 4

There is the option to remove indentation on sphinx-side, so no need to worry
about that.

Regarding the naming of the tests and example names:

    * Tests should be named ``test_<doc-file-name>_<example_name>``
    * Example names should be the same, unless there are multiple examples to
      be incldued from within one test, in which cases the separate examples
      should get a suffix, i.e. <doc-file-name>_<example_name>_<some-suffix>

In order to let the tests be independent, even for imports, there should NOT
be any imports on the global level of this test file!
"""

from pkg_resources import resource_filename

import pytest

import numpy as np
import h5py as h5

from dantro.tools import load_yml


# Local Variables -------------------------------------------------------------

DOC_EXAMPLES_CFG = resource_filename('tests', 'cfg/doc_examples.yml')


# Fixtures --------------------------------------------------------------------


@pytest.fixture
def data_dir(tmpdir):
    """Writes some dummy data to a tmpdir, returning the tmpdir object"""
    from dantro.tools import write_yml

    # Create YAML dummy data and write it out
    foobar = dict(one=1, two=2,
                  go_deeper=dict(eleven=11),
                  a_list=list(range(10)))
    barbaz = dict(nothing="to see here")

    write_yml(foobar, path=tmpdir.join("foobar.yml"))
    write_yml(barbaz, path=tmpdir.join("barbaz.yml"))
    write_yml(barbaz, path=tmpdir.join("also_barbaz.yml"))

    # Create a directory with some "config" files
    cfgdir = tmpdir.mkdir("config")

    for cfg_name in ("defaults", "user", "machine", "update", "combined"):
        write_yml(dict(),  # empty, but irrelevant
                  path=cfgdir.join(cfg_name+"_cfg.yml"))

    # Create some dummy HDF5 data
    h5dir = tmpdir.mkdir("measurements")

    # Create files of the same structure, containing groups and datasets
    for i in range(42):
        # Which day is this about?
        day = "day{:03d}".format(i)

        # Write some yaml file
        write_yml(dict(day=i),
                  path=h5dir.join(day + "_params.yml"))

        # Now the HDF5 data
        f = h5.File(h5dir.join(day + ".hdf5"), 'w')

        N = np.random.randint(100, 200)

        f.create_dataset("temperatures",
                         data=((np.random.random((N,)) - .4) * 70),
                         chunks=True)
        f.create_dataset("precipitation",
                         data=(np.random.random((N,)) * 1000.), dtype=int,
                         chunks=True)
        # TODO Consider adding coordinates here?!

        g = f.create_group("sensor_data")
        g.attrs['some_attribute'] = "this is some group level attribute"

        for j in range(23):
            _data = np.random.random((3, np.random.randint(80, 100)))
            g.create_dataset("sensor{:03d}".format(j), data=_data)

        f.close()

    return tmpdir


@pytest.fixture
def cfg() -> dict:
    """Loads the documentation examples config file"""
    return load_yml(DOC_EXAMPLES_CFG)

# -----------------------------------------------------------------------------
# examples.rst

def test_examples_all(data_dir):
    ### Start -- examples_setup_dantro
    from dantro import DataManager
    from dantro.data_loaders import Hdf5LoaderMixin, YamlLoaderMixin

    class MyDataManager(Hdf5LoaderMixin, YamlLoaderMixin, DataManager):
        """MyDataManager is a manager that can load HDF5 and YAML files"""
        pass  # Done here. Nothing else to do.
    ### End ---- examples_setup_dantro

    # .........................................................................
    data_dir_path = data_dir
    ### Start -- examples_loading_setup
    # Initialize the manager, associating it with a directory to load data from
    dm = MyDataManager(data_dir_path, name="happy_testing")
    ### End ---- examples_loading_setup

    ### Start -- examples_loading_empty_tree
    print(dm.tree)
    # Will print:
    #   Tree of MyDataManager 'happy_testing', 0 members, 0 attributes
    ### End ---- examples_loading_empty_tree

    ### Start -- examples_loading_yaml
    # Load YAML data from the data directory
    dm.load("my_cfg_files",    # the name of this entry
            loader="yaml",     # which loader to use
            glob_str="*.yml")  # which files to find and load from the data_dir

    # Have a look at what was loaded
    print(dm.tree)
    # Will print:
    #   Tree of MyDataManager 'happy_testing', 1 member, 0 attributes
    #    └─ my_cfg_files     <OrderedDataGroup, 3 members, 0 attributes>
    #       └┬ also_barbaz   <MutableMappingContainer, 1 attribute>
    #        ├ barbaz        <MutableMappingContainer, 1 attribute>
    #        └ foobar        <MutableMappingContainer, 1 attribute>
    ### End ---- examples_loading_yaml

    ### Start -- examples_loading_work_with_objects
    # Get the loaded objects
    foobar = dm['my_cfg_files']['foobar']
    barbaz = dm['my_cfg_files/barbaz']
    # ... can now work with these as if they were dicts
    ### End ---- examples_loading_work_with_objects

    ### Start -- examples_loading_iteration
    for container_name, container in dm['my_cfg_files'].items():
        print("Got container:", container_name, container)
        # ... do something with the containers also_barbaz, barbaz, and foobar
    ### End ---- examples_loading_iteration

    ### Start -- examples_loading_hdf5
    dm.load("measurements", loader="hdf5", glob_str="measurements/day*.hdf5")

    # Given the large amount of data, look only at a condensed tree
    print(dm.tree_condensed)
    # Will print something like:
    # Tree of MyDataManager 'happy_testing', 2 members, 0 attributes
    #  └┬ my_cfg_files           <OrderedDataGroup, 3 members, 0 attributes>
    #     └┬ also_barbaz         <MutableMappingContainer, 1 attribute>
    #      ├ barbaz              <MutableMappingContainer, 1 attribute>
    #      └ foobar              <MutableMappingContainer, 1 attribute>
    #   └ measurements           <OrderedDataGroup, 42 members, 0 attributes>
    #     └┬ day000              <OrderedDataGroup, 3 members, 0 attributes>
    #        └┬ precipitation    <NumpyDataContainer, int64, shape (148,), …
    #         ├ sensor_data      <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000     <NumpyDataContainer, float64, shape (3, 97), …
    #            ├ sensor001     <NumpyDataContainer, float64, shape (3, 92), …
    #            ├ ...                ... (19 more) ...
    #            ├ sensor021     <NumpyDataContainer, float64, shape (3, 91), …
    #            └ sensor022     <NumpyDataContainer, float64, shape (3, 97), …
    #         └ temperatures     <NumpyDataContainer, float64, shape (148,), …
    #      ├ day001              <OrderedDataGroup, 3 members, 0 attributes>
    #        └┬ precipitation    <NumpyDataContainer, int64, shape (169,), …
    #         ├ sensor_data      <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000     <NumpyDataContainer, float64, shape (3, 92), …
    #            ├ ...                ... (22 more) ...
    #         └ temperatures     <NumpyDataContainer, float64, shape (169,), …
    #      ├ ...                      ... (40 more) ...
    ### End ---- examples_loading_hdf5

    # .........................................................................
    ### Start -- examples_plotting_setup
    from dantro import PlotManager

    # Create a PlotManager and associate it with the existing DataManager
    pm = PlotManager(dm=dm)
    ### End ---- examples_plotting_setup
    pm.raise_exc = True

    ### Start -- examples_plotting_basic_lineplot
    pm.plot("my_example_lineplot",
            creator="external", module=".basic", plot_func="lineplot",
            y="measurements/day000/precipitation")
    ### End ---- examples_plotting_basic_lineplot



# -----------------------------------------------------------------------------
# philosophy.rst

def test_philosophy_all():
    ### Start -- philosophy_specializing
    from dantro import DataManager
    from dantro.data_loaders import YamlLoaderMixin

    class MyDataManager(YamlLoaderMixin, DataManager):
        """MyDataManager is a data manager that can load YAML files"""
        pass  # Done here. Nothing else to do.
    ### End ---- philosophy_specializing


# -----------------------------------------------------------------------------
# specializing.rst

def test_specializing_containers():
    ### Start -- specializing_mutable_sequence_container
    # Import the python abstract base class we want to adhere to
    from collections.abc import MutableSequence

    # Import base container class and the mixins we would like to use
    from dantro.base import BaseDataContainer
    from dantro.mixins import ItemAccessMixin, CollectionMixin, CheckDataMixin

    class MutableSequenceContainer(CheckDataMixin,
                                   ItemAccessMixin,
                                   CollectionMixin,
                                   BaseDataContainer,
                                   MutableSequence):
        """The MutableSequenceContainer stores sequence-like mutable data"""
    ### End ---- specializing_mutable_sequence_container

    ### Start -- specializing_msc_insert
        def insert(self, idx: int, val) -> None:
            """Insert an item at a given position.

            Args:
                idx (int): The index before which to insert
                val: The value to insert
            """
            self.data.insert(idx, val)
    ### End ---- specializing_msc_insert

    ### Start -- specializing_using_mutable_sequence
    dc = MutableSequenceContainer(name="my_mutable_sequence",
                                  data=[4, 8, 16])

    # Insert values
    dc.insert(0, 2)
    dc.insert(0, 1)

    # Item access and collection interface
    assert 16 in dc
    assert 32 not in dc
    assert dc[0] == 1

    for num in dc:
        print(num, end=", ")
    # prints:  1, 2, 4, 8, 16,
    ### End ---- specializing_using_mutable_sequence

    ### Start -- specializing_check_data_mixin
    class StrictlyListContainer(MutableSequenceContainer):
        """A MutableSequenceContainer that allows only a list as data"""
        DATA_EXPECTED_TYPES = (list,)     # as tuple or None (allow all)
        DATA_UNEXPECTED_ACTION = 'raise'  # can be: raise, warn, ignore

    # This will work
    some_list = StrictlyListContainer(name="some_list", data=["foo", "bar"])

    # The following will fail
    with pytest.raises(TypeError):
        StrictlyListContainer(name="some_tuple", data=("foo", "bar"))

    with pytest.raises(TypeError):
        StrictlyListContainer(name="some_tuple", data="just some string")
    ### End ---- specializing_check_data_mixin


def test_specializing_data_manager():
    ### Start -- specializing_data_manager
    import dantro
    from dantro.data_loaders import YamlLoaderMixin, PickleLoaderMixin

    class MyDataManager(PickleLoaderMixin,
                        YamlLoaderMixin,
                        dantro.DataManager):
        """A DataManager specialization that can load pickle and yaml data"""
    ### End ---- specializing_data_manager


# -----------------------------------------------------------------------------
# -- data_io ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# data_io/data_mngr.rst

def test_data_io_data_mngr(data_dir):
    my_data_dir = str(data_dir)

    ### Start -- data_io_data_mngr_example01
    import dantro
    from dantro.data_loaders import YamlLoaderMixin

    class MyDataManager(YamlLoaderMixin, dantro.DataManager):
        """A DataManager specialization that can load YAML data"""

    dm = MyDataManager(data_dir=my_data_dir)

    # Now, data can be loaded using the `load` command:
    dm.load("some_data",       # where to load the data to
            loader="yaml",     # which loader to use
            glob_str="*.yml")  # which files to find and load

    # Access it
    dm['some_data']
    # ...
    ### End ---- data_io_data_mngr_example01


def test_data_io_load_cfg(data_dir, cfg):
    my_data_dir = str(data_dir)
    cfg = cfg['data_io_load_cfg']
    my_load_cfg = {}  # dummy here

    ### Start -- data_io_load_cfg_setup
    import dantro
    from dantro.data_loaders import AllAvailableLoadersMixin

    class MyDataManager(AllAvailableLoadersMixin, dantro.DataManager):
        """A DataManager specialization that can load various kinds of data"""

    dm = MyDataManager(data_dir=my_data_dir, load_cfg=my_load_cfg)
    ### End ---- data_io_load_cfg_setup

    # Use a new DataManager, without output directory
    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=cfg['example01'])
    ### Start -- data_io_load_cfg_example01
    dm.load_from_cfg(print_tree=True)
    # Will print something like:
    # Tree of MyDataManager, 1 member, 0 attributes
    #  └─ cfg                         <OrderedDataGroup, 5 members, 0 attributes>
    #     └┬ combined                 <MutableMappingContainer, 1 attribute>
    #      ├ defaults                 <MutableMappingContainer, 1 attribute>
    #      ├ machine                  <MutableMappingContainer, 1 attribute>
    #      ├ update                   <MutableMappingContainer, 1 attribute>
    #      └ user                     <MutableMappingContainer, 1 attribute>
    ### End ---- data_io_load_cfg_example01

    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=cfg['example02'])
    ### Start -- data_io_load_cfg_example02
    dm.load_from_cfg(print_tree='condensed')
    # Will print something like:
    # Tree of MyDataManager, 1 member, 0 attributes
    #  └─ measurements                <OrderedDataGroup, 42 members, 0 attributes>
    #     └┬ 000                      <OrderedDataGroup, 2 members, 0 attributes>
    #        └┬ params                <MutableMappingContainer, 1 attribute>
    #         └ data                  <OrderedDataGroup, 3 members, 0 attributes>
    #           └┬ precipitation      <NumpyDataContainer, int64, shape (126,), 0 at…
    #            ├ sensor_data        <OrderedDataGroup, 23 members, 1 attribute>
    #              └┬ sensor000       <NumpyDataContainer, float64, shape (3, 89), 0 attributes>
    #               ├ sensor001       <NumpyDataContainer, float64, shape (3, 85), 0 attributes>
    #               ├ sensor002       <NumpyDataContainer, float64, shape (3, 94), 0 attributes>
    #               ├ ...             ... (18 more) ...
    #               ├ sensor021       <NumpyDataContainer, float64, shape (3, 80), 0 attributes>
    #               └ sensor022       <NumpyDataContainer, float64, shape (3, 99), 0 attributes>
    #            └ temperatures       <NumpyDataContainer, float64, shape (126,), 0 attributes>
    #      ├ 001                      <OrderedDataGroup, 2 members, 0 attributes>
    #        └┬ params                <MutableMappingContainer, 1 attribute>
    #         └ data                  <OrderedDataGroup, 3 members, 0 attributes>
    #           └┬ precipitation      <NumpyDataContainer, int64, shape (150,), 0 attributes>
    #            ├ sensor_data        <OrderedDataGroup, 23 members, 1 attribute>
    #              └┬ sensor000       <NumpyDataContainer, float64, shape (3, 99), 0 attributes>
    #               ├ sensor001       <NumpyDataContainer, float64, shape (3, 85), 0 attributes>
    #               ├ ...
    ### End ---- data_io_load_cfg_example02

    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=cfg['example03'])
    ### Start -- data_io_load_cfg_example03
    dm.load_from_cfg(print_tree='condensed')
    # Will print something like:
    # Tree of MyDataManager , 1 member, 0 attributes
    #  └─ measurements                <OrderedDataGroup, 42 members, 0 attributes>
    #     └┬ 000                      <OrderedDataGroup, 3 members, 1 attribute>
    #        └┬ precipitation         <NumpyDataContainer, int64, shape (165,), 0 attributes>
    #         ├ sensor_data           <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000          <NumpyDataContainer, float64, shape (3, 92), 0 attributes>
    #            ├ sensor001          <NumpyDataContainer, float64, shape (3, 91), 0 attributes>
    #            ├ sensor002          <NumpyDataContainer, float64, shape (3, 93), 0 attributes>
    #            ├ ...                ... (18 more) ...
    #            ├ sensor021          <NumpyDataContainer, float64, shape (3, 83), 0 attributes>
    #            └ sensor022          <NumpyDataContainer, float64, shape (3, 97), 0 attributes>
    #         └ temperatures          <NumpyDataContainer, float64, shape (165,), 0 attributes>
    #      ├ 001                      <OrderedDataGroup, 3 members, 1 attribute>
    #        └┬ precipitation         <NumpyDataContainer, int64, shape (181,), 0 attributes>
    #         ├ sensor_data           <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000          <NumpyDataContainer, float64, shape (3, 84), 0 attributes>
    #            ├ sensor001          <NumpyDataContainer, float64, shape (3, 85), 0 attributes>
    #            ├ ...

    # Check attribute access to the parameters
    for cont_name, data in dm['measurements'].items():
        params = data.attrs['params']
        assert params['day'] == int(cont_name)
    ### End ---- data_io_load_cfg_example03

    my_load_cfg = cfg['example04']
    ### Start -- data_io_load_cfg_example04
    from dantro.groups import TimeSeriesGroup

    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=my_load_cfg,
                       create_groups=[dict(path='measurements',
                                           Cls=TimeSeriesGroup)])

    dm.load_from_cfg(print_tree='condensed')
    # Will print something like:
    # Tree of MyDataManager , 1 member, 0 attributes
    #  └─ measurements                <TimeSeriesGroup, 42 members, 0 attributes>
    #     └┬ 000                      <OrderedDataGroup, 3 members, 0 attributes>
    #        └┬ precipitation         <NumpyDataContainer, int64, shape (165,), 0 attributes>
    #         ├ sensor_data           <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000          <NumpyDataContainer, float64, shape (3, 92), 0 attributes>
    #            ├ sensor001          <NumpyDataContainer, float64, shape (3, 91), 0 attributes>
    #            ├ sensor002          <NumpyDataContainer, float64, shape (3, 93), 0 attributes>
    #            ├ ...
    ### End ---- data_io_load_cfg_example04


    my_load_cfg = cfg['example05']
    ### Start -- data_io_load_cfg_example05
    from dantro.containers import XrDataContainer
    from dantro.mixins import Hdf5ProxySupportMixin

    class MyXrDataContainer(Hdf5ProxySupportMixin, XrDataContainer):
        """An xarray data container that allows proxy data"""

    class MyDataManager(AllAvailableLoadersMixin, dantro.DataManager):
        """A DataManager specialization that can load various kinds of data
        and uses containers that supply proxy support
        """
        # Configure the HDF5 loader to use the custom xarray container
        _HDF5_DSET_DEFAULT_CLS = MyXrDataContainer

    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=my_load_cfg)
    dm.load_from_cfg(print_tree='condensed')
    # Will print something like:
    # Tree of MyDataManager , 1 member, 0 attributes
    #  └─ measurements                <OrderedDataGroup, 42 members, 0 attributes>
    #     └┬ 000                      <OrderedDataGroup, 3 members, 0 attributes>
    #        └┬ precipitation         <MyXrDataContainer, proxy (hdf5, dask), int64, shape (165,), 0 attributes>
    #         ├ sensor_data           <OrderedDataGroup, 23 members, 1 attribute>
    #           └┬ sensor000          <MyXrDataContainer, proxy (hdf5, dask), float64, shape (3, 92), 0 attributes>
    #            ├ sensor001          <MyXrDataContainer, proxy (hdf5, dask), float64, shape (3, 91), 0 attributes>
    #            ├ sensor002          <MyXrDataContainer, proxy (hdf5, dask), float64, shape (3, 93), 0 attributes>
    #            ├ ...

    # Work with the data in the same way as before; it's loaded on the fly
    total_precipitation = 0.
    for day_data in dm['measurements'].values():
        total_precipitation += day_data['precipitation'].sum()
    ### End ---- data_io_load_cfg_example05


# -----------------------------------------------------------------------------
# data_io/faq.rst


def test_data_io_faq():
    ### Start -- data_io_faq_add_any_object
    from dantro.groups import OrderedDataGroup
    from dantro.containers import ObjectContainer, PassthroughContainer

    # The object we want to add to the tree
    some_object = ("foo", b"bar", 123, 4.56, None)

    # Use an ObjectContainer to store any object and provide simple item access
    cont1 = ObjectContainer(name="my_object_1", data=some_object)

    assert cont1.data is some_object
    assert cont1[0] == "foo"

    # For passing attribute calls through, use the PassthroughContainer:
    cont2 = PassthroughContainer(name="my_object_2", data=some_object)

    assert cont2.count("foo") == 1

    # Add them to a group
    grp = OrderedDataGroup(name="my_group")
    grp.add(cont1, cont2)
    ### End ---- data_io_faq_add_any_object


# -----------------------------------------------------------------------------
# -- plotting -----------------------------------------------------------------
# -----------------------------------------------------------------------------