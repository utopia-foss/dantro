"""Test BaseDataGroup-derived classes"""

from pkg_resources import resource_filename
import numpy as np
from typing import Union

import pytest

import numpy as np
import xarray as xr
import networkx as nx
import networkx.exception
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

from paramspace import ParamSpace, ParamDim


# Import the dantro objects to test here
from dantro.groups import OrderedDataGroup
from dantro.groups import ParamSpaceGroup, ParamSpaceStateGroup
from dantro.groups import NetworkGroup

from dantro.containers import MutableSequenceContainer, MutableMappingContainer
from dantro.containers import NumpyDataContainer

from dantro.tools import load_yml

# Local paths -----------------------------------------------------------------

SELECTOR_PATH = resource_filename('tests', 'cfg/selectors.yml')
NW_GRP_PATH = resource_filename('tests', 'cfg/nw_grps.yml')

from dantro.containers import MutableSequenceContainer, NumpyDataContainer

# Helper functions ------------------------------------------------------------

def create_test_data(psp_grp: ParamSpaceGroup, *, params: dict, state_no_str: str):
    """Given a ParamSpaceGroup, adds test data to it"""
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


# Fixtures --------------------------------------------------------------------

@pytest.fixture()
def pspace():
    """Used to setup a small pspace object to be tested on."""
    return ParamSpace(dict(foo="bar",
                           p0=ParamDim(default=0, values=[1, 2], order=0),
                           p1=ParamDim(default=0, values=[1, 2, 3]),
                           p2=ParamDim(default=0, values=[1, 2, 3, 4, 5])))

@pytest.fixture()
def psp_grp(pspace):
    """Setup and populate a ParamSpaceGroup"""
    psp_grp = ParamSpaceGroup(name="mv", pspace=pspace)

    # Iterate over the parameter space and add groups to the ParamSpaceGroup
    for params, state_no_str in pspace.iterator(with_info='state_no_str'):
        create_test_data(psp_grp, params=params, state_no_str=state_no_str)

    return psp_grp

@pytest.fixture()
def psp_grp_missing_data(psp_grp):
    """A ParamSpaceGroup with some states missing"""    
    for state_no in (12, 31, 38, 39, 52, 59, 66):
        if state_no in psp_grp:
            del psp_grp[state_no]

    return psp_grp

@pytest.fixture()
def psp_grp_default(pspace):
    """Setup and populate a ParamSpaceGroup with only the default"""
    psp_grp = ParamSpaceGroup(name="mv_default",
                              pspace=ParamSpace(pspace.default))

    create_test_data(psp_grp, params=pspace.default, state_no_str="0")

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
        
        # Add nodes and edges from config
        # ... if this is not one of the keys where no nodes should be added:
        # The wrong_* config entries have the nodes or edges missing.
        if name != "wrong_nodes":
            grps[name].new_container('nodes', Cls=NumpyDataContainer,
                                     data=cfg['nodes'])

        if name != "wrong_edges":
            grps[name].new_container('edges', Cls=NumpyDataContainer,
                                     data=cfg['edges'])

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

    # Should raise an error withou Cls given
    with pytest.raises(ValueError, match="Got neither argument `Cls` nor"):
        root.new_container("spam2", Cls=None, data=[1, 2, 3])

    # Set the class variable and try again
    root._NEW_CONTAINER_CLS = MutableSequenceContainer
    msc2 = root.new_container("spam2", data=[1, 2, 3])
    assert msc2 in root
    assert isinstance(msc2, MutableSequenceContainer)

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

    # Check padded_int_key_width property of PaddedIntegerItemAccessMixin
    assert psp_grp.padded_int_key_width == None

    # Populate the group with some entries
    # These should work
    grp00 = psp_grp.new_group("00")
    grp01 = psp_grp.new_group("01")
    grp02 = psp_grp.new_group("02")
    grp99 = psp_grp.new_group("99")

    assert psp_grp.padded_int_key_width == 2

    # These should not, as is asserted by ParamSpaceStateGroup.__init__
    with pytest.raises(ValueError, match="need names of the same length"):
        psp_grp.new_group("123")

    with pytest.raises(ValueError, match="representible as integers"):
        psp_grp.new_group("fo")

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
    with pytest.raises(IndexError, match="out of range \[0, 99\]"):
        psp_grp[-1]

    with pytest.raises(IndexError, match="out of range \[0, 99\]"):
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

    # Bad subspace dimension names
    with pytest.raises(KeyError, match="no parameter dimension with name"):
        pgrp.select(field="testdata/fixedsize/randints",
                    subspace=dict(invalid_dim_name=[1,2,3]))

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
        with pytest.raises(ValueError, match=r"No state (\d+) available in"):
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


def test_pspace_group_select_default(psp_grp_default, selectors):
    """Select should also work if only the default universe is present,
    i.e. without any parameter dimensions defined."""
    pgrp = psp_grp_default

    assert pgrp.pspace.volume == 0

    # Test that loading on all scenarios (without subspace) works.
    dsets = dict()
    for name, sel in selectors.items():            
        print("Now selecting data with selector '{}' ...".format(name))

        # Get the data. Distinguish depending on whether subspace selection
        # takes place or not; if yes, it will fail.
        if 'subspace' not in sel:
            # Can just select
            dset = pgrp.select(**sel)

        else:
            with pytest.raises(ValueError, match="has no dimensions defined"):
                dset = pgrp.select(**sel)

        # Save to dict of all datasets
        print("  got data:", dset, "\n\n\n")
        dsets[name] = dset


# NetworkGroup ----------------------------------------------------------------

def test_network_group_basics(nw_grps):
    """Test the NetworkGroup"""

    # Helper functions --------------------------------------------------------
    def basic_network_creation_test(nw, cfg, *, name: str):
        # Get the attributes
        attrs = cfg["attrs"]
        directed = attrs["directed"]
        parallel = attrs["parallel"]

        # Check that the network is not empty, (not) directed ...
        assert nx.is_empty(nw) == False
        assert nx.is_directed(nw) == directed

        # Check the data type of the network
        if not directed and not parallel:
            assert isinstance(nw, nx.Graph)

        elif directed and not parallel:
            assert isinstance(nw, nx.DiGraph)

        elif not directed and parallel:
            assert isinstance(nw, nx.MultiGraph)

        else:
            assert isinstance(nw, nx.MultiDiGraph)

        # Check that the nodes and edges given in the config coincide with
        # the ones stored inside of the network
        nodes = cfg["nodes"]
        edges = cfg["edges"]

        for v in nodes:
            assert v in nx.nodes(nw)
        
        # Need to preprocess the case with transposed edges
        if name == "transposed_edges":
            edges = [[edges[0][i], edges[1][i]] for i,_ in enumerate(edges[0])]

        for e in edges:
            assert tuple(e) in nx.edges(nw)

    # Actual test -------------------------------------------------------------
    # Get the groups and their corresponding configurations
    (grps, cfgs) = nw_grps

    for name, grp in grps.items():
        print("Testing configuration {} ...".format(name))

        # Get the config
        cfg = cfgs[name]

        # Get the attributes
        attrs = cfg['attrs']

        ### Case: Graph without any node or edge attributes
        # Create the graph without any node or edge attributes
        # Check the regular cases
        if name not in ['wrong_nodes', 'wrong_edges', 'bad_edges']:
            # This should work
            nw = grp.create_graph()

        # Also test the failing cases
        elif name == 'wrong_nodes':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_node_container"):
                grp.create_graph()

            # Nothing else to check
            continue

        elif name == 'wrong_edges':
            with pytest.raises(KeyError,
                               match=r"Check if .* _NWG_edge_container"):
                grp.create_graph()

            # Nothing else to check
            continue

        elif name == 'bad_edges':
            with pytest.raises(nx.exception.NetworkXError,
                               match="must be a 2-tuple, 3-tuple or 4-tuple."):
                grp.create_graph()

            # Nothing else to check
            continue

        # Check that the basic graph creation works
        basic_network_creation_test(nw, cfg, name=name)
