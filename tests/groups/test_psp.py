"""Test the pspgrp module"""

import copy

import numpy as np
import pytest
import xarray as xr

from dantro._import_tools import get_resource_path
from dantro.groups import (
    OrderedDataGroup,
    ParamSpaceGroup,
    ParamSpaceStateGroup,
)
from dantro.tools import load_yml

SELECTOR_PATH = get_resource_path("tests", "cfg/selectors.yml")

from .._fixtures import *


@pytest.fixture()
def selectors() -> dict:
    """Returns the dict of selectors, where each key is a selector
    specification
    """
    return load_yml(SELECTOR_PATH)


# -----------------------------------------------------------------------------


def test_ParamSpaceGroup(pspace):
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

    # Check the property that inspects members
    assert not psp_grp.only_default_data_present

    # Check the corresponding error messages
    with pytest.raises(IndexError, match=r"out of range \[0, 99\]"):
        psp_grp[-1]

    with pytest.raises(IndexError, match=r"out of range \[0, 99\]"):
        psp_grp[100]

    # Check that a parameter space can be associated
    # ... which needs be of the correct type
    with pytest.raises(TypeError, match="needs to be a ParamSpace-derived"):
        psp_grp.pspace = "foo"

    psp_grp.pspace = pspace
    assert psp_grp.attrs["pspace"] == pspace

    # ... which cannot be changed
    with pytest.raises(RuntimeError, match="was already set, cannot set it"):
        psp_grp.pspace = "bar"

    # With only the default data available, the property evaluates to True
    psp_grp = ParamSpaceGroup(name="mv")
    psp_grp.new_group("00")
    assert psp_grp.only_default_data_present
    psp_grp.new_group("01")
    assert not psp_grp.only_default_data_present


def test_ParamSpaceGroup_select(psp_grp, selectors):
    """Tests the pspace-related behaviour of the ParamSpaceGroup"""
    pgrp = psp_grp
    psp = pgrp.pspace

    import warnings

    warnings.filterwarnings("error", category=FutureWarning)

    # They should match in size
    assert len(pgrp) == psp.volume

    # Test that loading on all scenarios works.
    dsets = dict()
    for name, sel in selectors.items():
        print(f"Now selecting data with selector '{name}' ...")

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
        if not sel.get("subspace"):
            assert arr.shape[1 : 1 + psp.num_dims] == psp.shape
            assert arr.dims[1 : 1 + psp.num_dims] == tuple(psp.dims.keys())

    # Now test specific cases more explicitly.
    state = dsets["single_field"].state
    mf = dsets["multi_field"]
    wdt = dsets["with_dtype"]
    cfg = dsets["non_numeric"].cfg
    sub = dsets["subspace"].state
    rs_merge = dsets["randshape_merge"].randshape
    rs_concat = dsets["randshape_concat"].randshape

    # TODO check for structured data?

    # Positions match
    states = state.mean(["dim_0", "dim_1", "dim_2"])
    assert states[0, 0, 0] == 31
    assert states[0, 0, 1] == 32
    assert states[0, 1, 0] == 37
    assert states[1, 0, 0] == 55

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
    assert rs_concat.dtype in ("float32", "float64")
    assert rs_merge.dtype in ("float32", "float64")

    # config accessible by converting to python scalar
    assert isinstance(cfg[0, 0, 0].item(), dict)
    assert cfg[0, 0, 0].item() == dict(foo="bar", p0=1, p1=1, p2=1)

    # check the subspace shape and coordinats
    assert sub.shape == (1, 2, 3, 3, 4, 5)
    assert list(sub.coords["p0"]) == [1]
    assert list(sub.coords["p1"]) == [1, 2]
    assert list(sub.coords["p2"]) == [1, 3, 4]

    # Test the rest of the .select interface ..................................
    with pytest.raises(ValueError, match="Need to specify one of the arg"):
        pgrp.select()

    with pytest.raises(ValueError, match="Can only specify either of the arg"):
        pgrp.select(field="foo", fields="foo")

    with pytest.raises(TypeError, match="needs to be a dict, but was"):
        pgrp.select(fields="foo")

    with pytest.raises(TypeError, match="missing 1 required keyword-only arg"):
        pgrp.select(fields=dict(foo=dict(invalid_key="spam")))

    with pytest.raises(ValueError, match="without having a parameter space"):
        ParamSpaceGroup(name="without_pspace").select()

    with pytest.raises(ValueError, match="Make sure the data was fully"):
        ParamSpaceGroup(name="without_pspace", pspace=psp).select(field="cfg")

    # Bad subspace dimension names
    with pytest.raises(ValueError, match="A parameter dimension with name"):
        pgrp.select(
            field="testdata/fixedsize/randints",
            subspace=dict(invalid_dim_name=[1, 2, 3]),
        )

    # Non-uniformly sized datasets will require trivial index labels
    with pytest.raises(ValueError, match="Combination of datasets failed;"):
        pgrp.select(field="testdata/randsize/randlen", method="concat")

    with pytest.raises(ValueError, match="Combination of datasets failed;"):
        pgrp.select(field="testdata/randsize/randlen", method="merge")


def test_ParamSpaceGroup_select_missing_data(selectors, psp_grp_missing_data):
    """Test the .select method with missing state data"""
    pgrp = psp_grp_missing_data

    # test all selectors and assert that concat is not working but merge is.
    dsets = dict()

    for name, sel in selectors.items():
        print(f"Now selecting data with selector '{name}' ...")
        sel.pop("method", None)

        # With concat, it should fail
        with pytest.raises(ValueError, match=r"No state (\d+) available in"):
            pgrp.select(**sel, method="concat")

        # With merge, it should succeed
        dset = pgrp.select(**sel, method="merge")
        print("  got data:", dset, "\n\n\n")
        dsets[name] = dset

    # Get some to check explicitly
    state = dsets["single_field"].state
    mf = dsets["multi_field"]
    wdt = dsets["with_dtype"]
    cfg = dsets["non_numeric"].cfg
    sub = dsets["subspace"].state

    # dtype should always be float, even if explicitly specified
    assert wdt.state.dtype in ("float32", "float64")  # instead of uint8
    assert wdt.randints.dtype == "float32"

    # Test exceptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    with pytest.raises(ValueError, match="Invalid value for argument `method"):
        pgrp.select(field="cfg", method="some_invalid_method")


def test_ParamSpaceGroup_select_default(psp_grp_default, selectors):
    """Select should also work if only the default universe is present,
    i.e. without any parameter dimensions defined."""
    pgrp = psp_grp_default

    assert pgrp.pspace.volume == 0

    # Test that loading on all scenarios (without subspace) works.
    dsets = dict()
    for name, sel in selectors.items():
        print(f"Now selecting data with selector '{name}' ...")

        # Get the data. Distinguish depending on whether subspace selection
        # takes place or not; if yes, it will fail.
        if "subspace" not in sel:
            # Can just select
            dset = pgrp.select(**sel)

        else:
            with pytest.raises(ValueError, match="has no dimensions defined"):
                dset = pgrp.select(**sel)

        # Save to dict of all datasets
        print("  got data:", dset, "\n\n\n")
        dsets[name] = dset


def test_ParamSpaceGroup_EXPERIMENTAL_transformator(psp_grp):
    """Select the experimental features of the pspgrp"""
    pgrp = psp_grp

    # A dummy transformator function just to check that it's invoked
    def i_will_raise(self, *args, **kwargs):
        assert isinstance(self, OrderedDataGroup)
        raise RuntimeError("hi " + " ".join(args))

    # Assign a transformator
    pgrp._PSPGRP_TRANSFORMATOR = i_will_raise

    # See that it can be invoked
    with pytest.raises(RuntimeError, match="hi foo bar"):
        pgrp.select(field=dict(path="testdata", transform=["foo", "bar"]))


# -----------------------------------------------------------------------------


def test_ParamSpaceStateGroup(psp_grp):
    """Tests ParamSpaceStateGroup features when embedded in a ParamSpaceGroup

    As ParamSpaceStateGroup cannot really exist on its own, there's no need to
    test it separately.
    """
    # -- Check coordinates property
    pss43 = psp_grp[43]
    assert pss43.coords == dict(p0=1, p1=3, p2=1)

    # ... for all entries
    for pss in psp_grp.values():
        c = pss.coords
        assert isinstance(c, dict)
        assert psp_grp.pspace.state_map.sel(**c) == int(pss.name)
