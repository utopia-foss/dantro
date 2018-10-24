"""Test BaseDataGroup-derived classes"""

from pkg_resources import resource_filename
import numpy as np
from typing import Union

import pytest

import numpy as np
import xarray as xr
import networkx as nx

from paramspace import ParamSpace, ParamDim


# Import the dantro objects to test here
from dantro.groups import OrderedDataGroup
from dantro.groups import ParamSpaceGroup, ParamSpaceStateGroup, NetworkGroup

from dantro.container import MutableSequenceContainer, MutableMappingContainer
from dantro.container import NumpyDataContainer

from dantro.tools import load_yml

# Local paths -----------------------------------------------------------------

SELECTOR_PATH = resource_filename('tests', 'cfg/selectors.yml')
NW_GRP_PATH = resource_filename('tests', 'cfg/nw_grps.yml')

from dantro.container import MutableSequenceContainer, NumpyDataContainer

# Fixtures --------------------------------------------------------------------

@pytest.fixture()
def pspace():
    """Used to setup a small pspace object to be tested on."""
    return ParamSpace(dict(foo="bar",
                           p0=ParamDim(default=0, values=[1, 2]),
                           p1=ParamDim(default=0, values=[1, 2, 3]),
                           p2=ParamDim(default=0, values=[1, 2, 3, 4, 5])))

@pytest.fixture()
def psp_grp(pspace):
    """Setup and populate a ParamSpaceGroup"""
    psp_grp = ParamSpaceGroup(name="mv", pspace=pspace)

    # Iterate over the parameter space and add groups to the ParamSpaceGroup
    for params, state_no_str in pspace.iterator(with_info='state_no_str'):
        grp = psp_grp.new_group(state_no_str)

        # Add the parameters as container to the group
        grp.add(MutableMappingContainer(name="cfg", data=params))

        # Create some paths that can be used for testing
        grp.new_group("testdata")
        farrs = grp.new_group("testdata/fixedsize")
        rarrs = grp.new_group("testdata/randsize")

        # Add two numpy dataset: symbolising state number and some random data
        state_no = int(state_no_str)
        state =  state_no * np.ones((3,4,5), dtype=int)
        randints = np.random.randint(10, size=(3,4,5))

        farrs.add(NumpyDataContainer(name="state", data=state,
                                     attrs=dict(state_no=state_no)))
        farrs.add(NumpyDataContainer(name="randints", data=randints,
                                     attrs=dict(foo="bar")))

        # Add some non-uniform data sets
        # 3d with last dimension differing in length
        randlen = np.ones((3, 4, np.random.randint(10, 30)), dtype="uint8")
        rarrs.add(NumpyDataContainer(name="randlen", data=randlen))

        # 3d but of different shape in all directions
        randshape = np.ones(np.random.randint(1, 10, size=3), dtype="uint8")
        rarrs.add(NumpyDataContainer(name="randshape", data=randshape))

    return psp_grp

@pytest.fixture()
def psp_grp_missing_data(psp_grp):
    """A ParamSpaceGroup with some states missing"""    
    for state_no in (12, 31, 38, 39, 52, 59, 66):
        if state_no in psp_grp:
            del psp_grp[state_no]

    return psp_grp

@pytest.fixture()
def selectors() -> dict:
    """Returns the dict of selectors, where each key is a selector
    specification
    """
    return load_yml(SELECTOR_PATH)

@pytest.fixture()
def nw_grp_cfgs() -> dict:
    """Returns the dict of NetworkGroup configurations"""
    return load_yml(NW_GRP_PATH)

@pytest.fixture()
def nw_grps(nw_grp_cfgs) -> Union[dict, dict]:
    """Creates a NetworkGroup to be tested below"""
    grps = dict()
    for name, cfg in nw_grp_cfgs.items():
        grps[name] = NetworkGroup(name=name, attrs=cfg["attrs"])

        # Add vertices and edges from config
        grps[name].new_container("vertices", Cls=NumpyDataContainer,
                          data=cfg["vertices"])

        grps[name].new_container("edges", Cls=NumpyDataContainer,
                          data=cfg["edges"])

        # Add vertex and edge properties to the NetworkGroup
        grps[name].new_container("vertex_prop", Cls=NumpyDataContainer,
                          data=cfg["test_vertex_prop"], attrs={"is_node_property": True})

        grps[name].new_container("edge_prop", Cls=NumpyDataContainer,
                          data=cfg["test_edge_prop"], attrs={"is_edge_property": True})

    return (grps, nw_grp_cfgs)

# Tests -----------------------------------------------------------------------

def test_ordered_data_group():
    """Tests whether the __init__ method behaves as desired"""
    # Basic initialisation, without containers
    dg1 = OrderedDataGroup(name="foo")

    # Names need be string
    with pytest.raises(TypeError, match="Name for OrderedDataGroup needs to"):
        OrderedDataGroup(name=123)

    # Passing some containers
    conts = [MutableSequenceContainer(name=str(i), data=list(range(i)))
             for i in range(10)]
    dg2 = OrderedDataGroup(name="bar", containers=conts)
    
    # Nest these together
    root = OrderedDataGroup(name="root", containers=[dg1, dg2])

    # If a non-container object is passed to a group, this should fail.
    with pytest.raises(TypeError):
        OrderedDataGroup(name="bar", containers=["foo", "bar"])

    # Try to access them
    assert 'foo' in root
    assert 'bar' in root
    assert 'baz' not in root

    # There should be a custom key error if accessing something not available
    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root['i_am_a_ghost']
    
    with pytest.raises(KeyError, match="No key or key sequence '.*' in .*!"):
        root['foo/is/a/ghost']

    # Test adding a new group in that group
    subgroup = root.new_group("subgroup")

    # Test it was added
    assert "subgroup" in root
    assert root["subgroup"] is subgroup

    # Adding it again should fail
    with pytest.raises(ValueError, match="has a member with name 'subgroup'"):
        root.new_group("subgroup")

    # Should also work when explicitly giving the class
    sg2 = root.new_group("sg2", Cls=OrderedDataGroup)
    assert isinstance(sg2, OrderedDataGroup)
    # TODO pass another class here

    # Should _not_ work with something that is not a class or not a group
    with pytest.raises(TypeError,
                       match="Argument `Cls` needs to be a class"):
        root.new_group("foobar", Cls="not_a_class")

    with pytest.raises(TypeError,
                       match="Argument `Cls` needs to be a subclass"):
        root.new_group("foobar", Cls=MutableSequenceContainer)

def test_group_creation():
    """Tests whether groups and containers can be created as desired."""
    root = OrderedDataGroup(name="root")

    # Add a group by name and check it was added
    foo = root.new_group("foo")
    assert foo in root

    # Add a container
    msc = root.new_container("spam", Cls=MutableSequenceContainer,
                             data=[1, 2, 3])
    assert msc in root

    # Now test adding groups by path
    bar = root.new_group("foo/bar")
    assert "foo/bar" in root
    assert "bar" in foo

    # Check that intermediate parts not existing leads to errors
    with pytest.raises(KeyError, match="Could not create OrderedDataGroup at"):
        root.new_group("some/longer/path")

    # Set the allowed container types of the bar group differently
    bar._ALLOWED_CONT_TYPES = (MutableSequenceContainer,)

    # ... this should now fail
    with pytest.raises(TypeError, match="Can only add objects derived from"):
        bar.new_group("baz")

    # While adding a MutableSequenceContainer should work
    bar.new_container("eggs", Cls=MutableSequenceContainer, data=[1, 2, 3])

def test_list_item_access():
    """Tests that passing lists with arbitrary content along __getitem__ works
    as desired ...
    """

    root = OrderedDataGroup(name="root")
    one = root.new_group("one")
    two = one.new_group("two")
    two.add(NumpyDataContainer(name="arr", data=np.zeros((2,3,4))))
    # Path created: root/one/two/arr

    # Test that regular item access is possible
    arr = root["one/two/arr"]

    # Test that access via a list-type path is possible
    sliced_arr = root[["one", "two", "arr", slice(None, 2)]]
    assert sliced_arr.shape == arr[slice(None, 2)].shape


# ParamSpaceGroup -------------------------------------------------------------

def test_pspace_group_basics(pspace):
    """Tests the ParamSpaceGroup"""
    psp_grp = ParamSpaceGroup(name="mv")

    # Check the _num_digs attribute
    assert psp_grp._num_digs == 0

    # Populate the group with some entries
    # These should work
    grp00 = psp_grp.new_group("00")
    grp01 = psp_grp.new_group("01")
    grp02 = psp_grp.new_group("02")
    grp99 = psp_grp.new_group("99")

    assert psp_grp._num_digs == 2

    # These should not, as is asserted by ParamSpaceStateGroup.__init__
    with pytest.raises(ValueError, match="need names that have a string"):
        psp_grp.new_group("123")

    with pytest.raises(ValueError, match="representible as integers"):
        psp_grp.new_group("foo")

    with pytest.raises(ValueError, match="be positive when converted to"):
        psp_grp.new_group("-1")


    # Check that only ParamSpaceStateGroups can be added and they are default
    with pytest.raises(TypeError, match="Can only add objects derived from"):
        psp_grp.new_group("foo", Cls=OrderedDataGroup)

    assert type(psp_grp.new_group("42")) is ParamSpaceStateGroup


    # Assert item access via integers works
    assert psp_grp[0] is grp00 is psp_grp["00"]
    assert psp_grp[1] is grp01 is psp_grp["01"]
    assert psp_grp[2] is grp02 is psp_grp["02"]
    
    assert 0 in psp_grp
    assert 1 in psp_grp
    assert 2 in psp_grp


    # Check the corresponding error messages
    with pytest.raises(KeyError, match="cannot be negative!"):
        psp_grp[-1]

    with pytest.raises(KeyError, match="cannot be larger than 99!"):
        psp_grp[100]


    # Check that a parameter space can be associated
    # ... which needs be of the correct type
    with pytest.raises(TypeError, match="needs to be a ParamSpace-derived"):
        psp_grp.pspace = "foo"

    psp_grp.pspace = pspace
    assert psp_grp.attrs['pspace'] == pspace

    # ... which cannot be changed
    with pytest.raises(RuntimeError, match="was already set, cannot set it"):
        psp_grp.pspace = "bar"


def test_pspace_group_select(psp_grp, selectors):
    """Tests the pspace-related behaviour of the ParamSpaceGroup"""
    pgrp = psp_grp
    psp = pgrp.pspace

    import warnings
    warnings.filterwarnings('error', category=FutureWarning)

    # They should match in size
    assert len(pgrp) == psp.volume

    # Test that loading on all scenarios works.
    dsets = dict()
    for name, sel in selectors.items():
        print("Now selecting data with selector '{}' ...".format(name))

        # Get the data
        dset = pgrp.select(**sel)
        print("  got data:", dset, "\n\n\n")

        # Save to dict of all datasets
        dsets[name] = dset

        # And make some general tests
        # Should be a Dataset
        assert isinstance(dset, xr.Dataset)

        # As the dataset has unordered dimensions, it is easier (& equivalent)
        # to use the array created from that to perform the remaining checks.
        arr = dset.to_array()
        assert 9 >= len(arr.dims) >= 4

        # The 0th dimension specifies the variable
        assert arr.dims[0] == "variable"

        # The following dimensions correspond to the parameter space's dims
        assert arr.dims[1] == "p0"
        assert arr.dims[2] == "p1"
        assert arr.dims[3] == "p2"

        # Can only check the actual shape in this for-loop without subspace
        if not sel.get('subspace'):
            assert arr.shape[1:1 + psp.num_dims] == psp.shape
            assert arr.dims[1:1 + psp.num_dims] == tuple(psp.dims.keys())

    # Now test specific cases more explicitly.
    state = dsets['single_field'].state
    mf = dsets['multi_field']
    wdt = dsets['with_dtype']
    cfg = dsets['non_numeric'].cfg
    sub = dsets['subspace'].state
    rs_merge = dsets['randshape_merge'].randshape
    rs_concat = dsets['randshape_concat'].randshape

    # TODO check for structured data?

    # Positions match
    states = state.mean(['dim_0', 'dim_1', 'dim_2'])
    assert states[0,0,0] == 31
    assert states[0,0,1] == 32
    assert states[0,1,0] == 37
    assert states[1,0,0] == 55

    # Access via loc
    assert states.loc[dict(p0=1, p1=1, p2=1)] == 31
    assert states.loc[dict(p0=1, p1=1, p2=2)] == 32

    # TODO Check what happened to attributes

    # Custom names work
    assert list(mf.data_vars) == ["state", "randints", "config"]

    # Dimensions match
    assert len(mf.state.dims) == 6
    assert len(mf.randints.dims) == 6
    assert len(mf.config.dims) == 3

    # dtype was converted (with default method, i.e.: concat)
    assert wdt.state.dtype == "uint8"
    assert wdt.randints.dtype == "float32"

    # for arrays that needed alignment, the dtype cannot be preserved
    assert rs_concat.dtype == "float64"
    assert rs_merge.dtype == "float64"

    # config accessible by converting to python scalar
    assert isinstance(cfg[0,0,0].item(), dict)
    assert cfg[0,0,0].item() == dict(foo="bar", p0=1, p1=1, p2=1)

    # check the subspace shape and coordinats
    assert sub.shape == (1,2,3,3,4,5)
    assert list(sub.coords['p0']) == [1]
    assert list(sub.coords['p1']) == [1,2]
    assert list(sub.coords['p2']) == [1,3,4]


    # Test the rest of the .select interface ..................................
    with pytest.raises(ValueError, match="Need to specify one of the arg"):
        pgrp.select()
    
    with pytest.raises(ValueError, match="Can only specify either of the arg"):
        pgrp.select(field="foo", fields="foo")

    with pytest.raises(TypeError, match="needs to be a dict, but was"):
        pgrp.select(fields="foo")

    with pytest.raises(ValueError, match="invalid key in the 'foo' entry"):
        pgrp.select(fields=dict(foo=dict(invalid_key="spam")))

    with pytest.raises(ValueError, match="without having a parameter space"):
        ParamSpaceGroup(name="without_pspace").select()

    with pytest.raises(ValueError, match="Make sure the data was fully"):
        ParamSpaceGroup(name="without_pspace", pspace=psp).select(field="cfg")

    # Non-uniformly sized datasets will require trivial index labels
    with pytest.raises(ValueError, match="Combination of datasets failed;"):
        pgrp.select(field="testdata/randsize/randlen", method="concat")
    
    with pytest.raises(ValueError, match="Combination of datasets failed;"):
        pgrp.select(field="testdata/randsize/randlen", method="merge")


def test_pspace_group_select_missing_data(selectors, psp_grp_missing_data):
    """Test the .select method with missing state data"""
    pgrp = psp_grp_missing_data

    # test all selectors and assert that concat is not working but merge is.
    dsets = dict()
    
    for name, sel in selectors.items():
        print("Now selecting data with selector '{}' ...".format(name))
        sel.pop('method', None)

        # With concat, it should fail
        with pytest.raises(ValueError, match="No state (\d+) available in"):
            pgrp.select(**sel, method='concat')

        # With merge, it should succeed
        dset = pgrp.select(**sel, method='merge')
        print("  got data:", dset, "\n\n\n")
        dsets[name] = dset

    # Get some to check explicitly
    state = dsets['single_field'].state
    mf = dsets['multi_field']
    wdt = dsets['with_dtype']
    cfg = dsets['non_numeric'].cfg
    sub = dsets['subspace'].state

    # dtype should always be float, even if explicitly specified
    assert wdt.state.dtype == "float64"  # instead of uint8
    assert wdt.randints.dtype == "float32"

    # Test exceptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    with pytest.raises(ValueError, match="Invalid value for argument `method"):
        pgrp.select(field="cfg", method="some_invalid_method")


# NetworkGroup ----------------------------------------------------------------

def basic_network_creation(nw, cfg):
    # Get the attributes
    attrs = cfg["attrs"]
    directed = attrs["directed"]
    parallel = attrs["parallel"]

    # Check that the network is not empty, (not) directed ...
    assert(nx.is_empty(nw) == False)
    assert(nx.is_directed(nw) == directed)

    # Check the data type of the network
    if (not directed and not parallel):
        assert(type(nw) == nx.Graph)
    elif (directed and not parallel):
        assert(type(nw) == nx.DiGraph)
    elif (not directed and parallel):
        assert(type(nw) == nx.MultiGraph)
    else:
        assert(type(nw) == nx.MultiDiGraph)

    # Check that the vertices and edges given in the config coincide with
    # the ones stored inside of the network
    for v in cfg["vertices"]:
        assert(v in nx.nodes(nw))
    for e in cfg["edges"]:
        assert(tuple(e) in nx.edges(nw))

def test_network_group_basics(nw_grps):
    """Test the NetworkGroup"""
    # Get the groups and their corresponding configurations
    (grps, cfgs) = nw_grps

    for name, grp in grps.items():
        # Get the config
        cfg = cfgs[name]

        # Get the attributes
        attrs = cfg["attrs"]
        directed = attrs["directed"]
        parallel = attrs["parallel"]

        ### Case: Graph without any vertex or edge properties
        # Create the graph without any vertex or edge properties
        nw = grp.create_graph(directed=directed, parallel_edges=parallel)

        basic_network_creation(nw, cfg)

        # Check that there are no vertex or edge properties
        for v in nw.nodes():
            with pytest.raises(KeyError):
                nw[v]["test_vertex_prop"]

        for e in nw.edges():
            with pytest.raises(KeyError):
                nw[e]["test_edge_prop"]


        # ### Case: Graph with vertex and edge properties
        # # Create a new graph with vertex and edge properties
        # nw_p = grp.create_graph(directed=directed, parallel_edges=parallel,
        #                         with_node_properties=True,
        #                         with_edge_properties=True)
        
        # TODO

        ### Case: Graph with vertex but no edge properties
        # nw_p = grp.create_graph(directed=directed, parallel_edges=parallel,
        #                         with_node_properties=True,
        #                         with_edge_properties=False)

        # basic_network_creation(nw_p, cfg)

        # TODO

        # Test that the vertex properties are set correctly
        # but that there are no edge properties

        # TODO

        for e in nw_p.edges():
            with pytest.raises(KeyError):
                nw_p[e]["test_edge_prop"]

        # ### Case: Graph with no vertex but edge properties
        # nw_p = grp.create_graph(directed=directed, parallel_edges=parallel,
        #                         with_node_properties=False,
        #                         with_edge_properties=True)

        # TODO
