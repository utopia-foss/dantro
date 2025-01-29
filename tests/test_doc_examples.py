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

import h5py as h5
import numpy as np
import pytest

from dantro._import_tools import get_resource_path
from dantro.tools import load_yml

# Local Variables -------------------------------------------------------------

DOC_EXAMPLES_CFG = get_resource_path("tests", "cfg/doc_examples.yml")


# Fixtures --------------------------------------------------------------------
from ._fixtures import *


@pytest.fixture
def data_dir(tmpdir):
    """Writes some dummy data to a tmpdir, returning the Path object"""
    from dantro.tools import write_yml

    out_dir = tmpdir

    # Create YAML dummy data and write it out
    foobar = dict(
        one=1, two=2, go_deeper=dict(eleven=11), a_list=list(range(10))
    )
    barbaz = dict(nothing="to see here")

    write_yml(foobar, path=out_dir.join("foobar.yml"))
    write_yml(barbaz, path=out_dir.join("barbaz.yml"))
    write_yml(barbaz, path=out_dir.join("also_barbaz.yml"))

    # Create a directory with some "config" files
    cfgdir = out_dir.mkdir("config")

    for cfg_name in ("defaults", "user", "machine", "update", "combined"):
        write_yml(
            dict(),  # empty, but irrelevant
            path=cfgdir.join(cfg_name + "_cfg.yml"),
        )

    # Create some dummy HDF5 data
    h5dir = out_dir.mkdir("measurements")

    # Create files of the same structure, containing groups and datasets
    for i in range(42):
        # Which day is this about?
        day = f"day{i:03d}"

        # Write some yaml file
        write_yml(dict(day=i), path=h5dir.join(day + "_params.yml"))

        # Now the HDF5 data
        f = h5.File(h5dir.join(day + ".hdf5"), "w")

        N = np.random.randint(100, 200)

        f.create_dataset(
            "temperatures",
            data=((np.random.random((N,)) - 0.4) * 70),
            chunks=True,
        )
        f.create_dataset(
            "precipitation",
            data=(np.random.random((N,)) * 1000.0),
            dtype=int,
            chunks=True,
        )
        # TODO Consider adding coordinates here?!

        g = f.create_group("sensor_data")
        g.attrs["some_attribute"] = "this is some group level attribute"

        for j in range(23):
            _data = np.random.random((3, np.random.randint(80, 100)))
            g.create_dataset(f"sensor{j:03d}", data=_data)

        f.close()

    return out_dir


@pytest.fixture
def cfg() -> dict:
    """Loads the documentation examples config file"""
    return load_yml(DOC_EXAMPLES_CFG)


# -----------------------------------------------------------------------------
# -- INCLUDES START BELOW -----------------------------------------------------
# -----------------------------------------------------------------------------
# NOTE Important! Turn off black formatting for everything that is included...
# fmt: off

# -----------------------------------------------------------------------------
# usage.rst

def test_usage_all(data_dir):
    ### Start -- usage_setup_dantro
    from dantro import DataManager
    from dantro.data_loaders import Hdf5LoaderMixin, YamlLoaderMixin

    class MyDataManager(Hdf5LoaderMixin, YamlLoaderMixin, DataManager):
        """MyDataManager is a manager that can load HDF5 and YAML files"""
        pass  # Done here. Nothing else to do.
    ### End ---- usage_setup_dantro

    # .........................................................................
    data_dir_path = data_dir

    ### Start -- usage_loading_setup
    # Initialize the manager, associating it with a directory to load data from
    dm = MyDataManager(data_dir_path, name="happy_testing")
    ### End ---- usage_loading_setup

    ### Start -- usage_loading_empty_tree
    print(dm.tree)
    # Will print:
    #   Tree of MyDataManager 'happy_testing', 0 members, 0 attributes
    ### End ---- usage_loading_empty_tree

    ### Start -- usage_loading_yaml
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
    ### End ---- usage_loading_yaml

    ### Start -- usage_loading_work_with_objects
    # Get the loaded objects
    foobar = dm["my_cfg_files"]["foobar"]
    barbaz = dm["my_cfg_files/barbaz"]
    # ... can now work with these as if they were dicts
    ### End ---- usage_loading_work_with_objects

    ### Start -- usage_loading_iteration
    for container_name, container in dm["my_cfg_files"].items():
        print("Got container:", container_name, container)
        # ... do something with the containers also_barbaz, barbaz, and foobar
    ### End ---- usage_loading_iteration

    # NOTE The hdf5 files that are loaded below are created by the data_dir
    #      fixture above for the purpose of this test. When using this code
    #      outside of examples, you typically want to replace it with your own
    #      data files.

    ### Start -- usage_loading_hdf5
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
    ### End ---- usage_loading_hdf5

    # .........................................................................
    ### Start -- usage_plotting_setup
    from dantro import PlotManager

    # Create a PlotManager and associate it with the existing DataManager
    pm = PlotManager(dm=dm)
    ### End ---- usage_plotting_setup
    pm.raise_exc = True

    ### Start -- usage_plotting_basic_lineplot
    pm.plot("my_example_lineplot",
            creator="external", module=".basic", plot_func="lineplot",
            y="measurements/day000/precipitation")
    ### End ---- usage_plotting_basic_lineplot



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
    from dantro.mixins import CheckDataMixin, CollectionMixin, ItemAccessMixin

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
    from dantro.data_loaders import PickleLoaderMixin, YamlLoaderMixin

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
    dm["some_data"]
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
                       load_cfg=cfg["example01"])
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
                       load_cfg=cfg["example02"])
    ### Start -- data_io_load_cfg_example02
    dm.load_from_cfg(print_tree="condensed")
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
                       load_cfg=cfg["example03"])
    ### Start -- data_io_load_cfg_example03
    dm.load_from_cfg(print_tree="condensed")
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
    for cont_name, data in dm["measurements"].items():
        params = data.attrs["params"]
        assert params["day"] == int(cont_name)
    ### End ---- data_io_load_cfg_example03

    my_load_cfg = cfg["example04"]
    ### Start -- data_io_load_cfg_example04
    from dantro.groups import TimeSeriesGroup

    dm = MyDataManager(data_dir=my_data_dir, out_dir=False,
                       load_cfg=my_load_cfg,
                       create_groups=[dict(path="measurements",
                                           Cls=TimeSeriesGroup)])

    dm.load_from_cfg(print_tree="condensed")
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


    my_load_cfg = cfg["example05"]
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
    dm.load_from_cfg(print_tree="condensed")
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
    for day_data in dm["measurements"].values():
        total_precipitation += day_data["precipitation"].sum()
    ### End ---- data_io_load_cfg_example05


# -----------------------------------------------------------------------------
# data_io/faq.rst


def test_data_io_faq():
    ### Start -- data_io_faq_add_any_object
    from dantro.containers import ObjectContainer, PassthroughContainer
    from dantro.groups import OrderedDataGroup

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
# -- data_structures/groups ---------------------------------------------------
# -----------------------------------------------------------------------------
# data_structures/groups/graph.rst

def test_groups_graphgroup():

    from dantro.containers import XrDataContainer
    from dantro.groups import GraphGroup, TimeSeriesGroup

    # Create some node data
    nodes               = XrDataContainer(name="nodes", data=np.arange(0, 10))
    node_prop           = XrDataContainer(name="some_node_prop",
                                          data=np.random.random(size=(2,10)),
                                          dims=('time','node_idx'))

    # Create some dynamic edge data at two different time steps
    edges_initial       = XrDataContainer(name="0",
                                    data=np.random.randint(0, 10, size=(9,2)),
                                    dims=('edge_idx','type'),
                                    coords=dict(edge_idx=range(9),
                                                type=['source','target']))
    edges_final         = XrDataContainer(name="10",
                                    data=np.random.randint(0, 10, size=(6,2)),
                                    dims=('edge_idx','type'),
                                    coords=dict(edge_idx=range(6),
                                                type=['source','target']))
    edge_prop_initial   = XrDataContainer(name="0",
                                          data=np.random.random(size=9),
                                          dims=('edge_idx',),
                                          coords=dict(edge_idx=range(9)))
    edge_prop_final     = XrDataContainer(name="10",
                                          data=np.random.random(size=6),
                                          dims=('edge_idx',),
                                          coords=dict(edge_idx=range(6)))
    other_edge_prop     = XrDataContainer(name="other_edge_prop",
                                          data=np.random.random(size=6),
                                          dims=('edge_idx',))

    # Create a GraphGroup and add the static data
    graph_group = GraphGroup(name="graph_group",
                             containers=[nodes, node_prop, other_edge_prop],
                             attrs={"directed": False, "parallel": False})

    # Add dynamic edge data as TimeSeriesGroup
    graph_group.new_group("edges", Cls=TimeSeriesGroup,
                          containers=[edges_initial, edges_final])
    graph_group.new_group("some_edge_prop", Cls=TimeSeriesGroup,
                          containers=[edge_prop_initial, edge_prop_final])

    # The resulting tree structure is the following:
    ### Start -- groups_graphgroup_datatree
    # graph_group                   <GraphGroup, 4 members, 2 attributes>
    # └┬ nodes                      <XrDataContainer, ..., shape (10,), 0 attributes>
    #  ├ some_node_prop             <XrDataContainer, ..., shape (2,10), 0 attributes>
    #  ├ edges                      <TimeSeriesGroup, 2 members, 0 attributes>
    #    └┬ 0                       <XrDataContainer, ..., shape (9,2), 0 attributes>
    #     └ 10                      <XrDataContainer, ..., shape (6,2), 0 attributes>
    #  ├ some_edge_prop             <TimeSeriesGroup, 2 members, 0 attributes>
    #    └┬ 0                       <XrDataContainer, ..., shape (9,), 0 attributes>
    #     └ 10                      <XrDataContainer, ..., shape (6,), 0 attributes>
    #  └ other_edge_prop            <XrDataContainer, ..., shape (6,), 0 attributes>
    ### End ---- groups_graphgroup_datatree

    ### Start -- groups_graphgroup_create_graph
    # Create the initial graph from the graph group without node/edge properties
    g = graph_group.create_graph(at_time=0) # time specified by value

    # Now, create the final graph with `some_node_prop` as node property and
    # `some_edge_prop` as edge property.
    g = graph_group.create_graph(at_time_idx=-1, # time specified via index
                                 node_props=["some_node_prop"],
                                 edge_props=["some_edge_prop"])
    ### End ---- groups_graphgroup_create_graph

    ### Start -- groups_graphgroup_set_properties
    # Set the edge property manually from the `other_edge_prop` data container
    # and select the data of the last time step
    graph_group.set_edge_property(g=g, name="other_edge_prop", at_time_idx=-1)
    ### End ---- groups_graphgroup_set_properties

    ext_data = XrDataContainer(name="ext_np",
                               data=np.random.random(size=(10,)),
                               dims=('node_idx',))

    ### Start -- groups_graphgroup_property_maps
    # Make the external data available in the graph group under the given key
    graph_group.register_property_map("my_ext_node_prop", data=ext_data)

    # Use the newly created key to set the external data as node property
    graph_group.set_node_property(g=g, name="my_ext_node_prop")

    # Alternatively, load the external data directly via the `data` argument
    graph_group.set_node_property(g=g, name="my_ext_node_prop", data=ext_data)
    ### End ---- groups_graphgroup_property_maps



# -----------------------------------------------------------------------------
# -- plotting -----------------------------------------------------------------
# -----------------------------------------------------------------------------
# plotting/plot_cfg_ref.rst
from .test_plot_mngr import dm as pm_dm
from .test_plot_mngr import pcr_pyplot_kwargs, pm_kwargs


def test_plot_cfg_ref(cfg, tmpdir, pm_dm, pm_kwargs, pcr_pyplot_kwargs):
    """Tests the examples for the plot configuration reference"""
    from dantro import PlotManager

    cfg = cfg["plot_cfg_ref"]
    dm = pm_dm

    # Run the examples
    pm = PlotManager(dm=dm, out_dir=str(tmpdir), raise_exc=True)
    pm.plot_from_cfg(plots_cfg=cfg["mngr_overview"])
