"""Test utilities, fixtures, ..."""

import os
import pathlib

import numpy as np
import pytest
import xarray as xr
from paramspace import ParamDim, ParamSpace

from dantro.containers import (
    MutableMappingContainer,
    NumpyDataContainer,
    XrDataContainer,
)
from dantro.groups import (
    OrderedDataGroup,
    ParamSpaceGroup,
    ParamSpaceStateGroup,
)

from . import (
    ABBREVIATE_TEST_OUTPUT_DIR,
    TEST_OUTPUT_DIR,
    TEST_VERBOSITY,
    USE_TEST_OUTPUT_DIR,
)

# -----------------------------------------------------------------------------
# Output Directory


@pytest.fixture
def tmpdir_or_local_dir(tmpdir, request) -> pathlib.Path:
    """If ``USE_TEST_OUTPUT_DIR`` is False, returns a temporary directory;
    otherwise a test-specific local directory within ``TEST_OUTPUT_DIR`` is
    returned.
    """
    if not USE_TEST_OUTPUT_DIR:
        return tmpdir

    if not ABBREVIATE_TEST_OUTPUT_DIR:
        # include the module and don't do any string replacements
        test_dir = os.path.join(
            TEST_OUTPUT_DIR,
            request.node.module.__name__,
            request.node.originalname,
        )
    else:
        # generate a shorter version without the module and with the test
        # prefixes dropped
        test_dir = os.path.join(
            TEST_OUTPUT_DIR,
            request.node.originalname.replace("test_", ""),
        )

    print(f"Using local test output directory:\n  {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    return pathlib.Path(test_dir)


out_dir = tmpdir_or_local_dir
"""Alias for ``tmpdir_or_local_dir`` fixture"""


# -----------------------------------------------------------------------------
# -- Test data creation -------------------------------------------------------
# -----------------------------------------------------------------------------

# .. paramspace-related .......................................................


def create_psp_test_data(
    psp_grp: ParamSpaceGroup, *, params: dict, state_no_str: str
):
    """Given a ParamSpaceGroup, adds test data to it"""
    grp = psp_grp.new_group(state_no_str)

    # Add the parameters as container to the group
    grp.add(MutableMappingContainer(name="cfg", data=params))

    # Create some paths that can be used for testing
    grp.new_group("testdata")
    farrs = grp.new_group("testdata/fixedsize")
    rarrs = grp.new_group("testdata/randsize")
    labelled = grp.new_group("labelled")

    # Add two numpy dataset: symbolising state number and some random data
    state_no = int(state_no_str)
    state = state_no * np.ones((3, 4, 5), dtype=int)
    randints = np.random.randint(10, size=(3, 4, 5))

    farrs.add(
        NumpyDataContainer(
            name="state", data=state, attrs=dict(state_no=state_no)
        )
    )
    farrs.add(
        NumpyDataContainer(
            name="randints", data=randints, attrs=dict(foo="bar")
        )
    )

    # Some labelled data
    labelled.add(
        XrDataContainer(
            name="randints",
            data=randints,
            attrs=dict(
                foo="bar",
                dims=("x", "y", "z"),
                coords__x=[1, 2, 3],
                coords__y=[1, 2, 3, 4],
                coords__z=[1, 2, 3, 4, 5],
            ),
        )
    )
    assert (labelled["randints"].coords["x"] == [1, 2, 3]).all()
    assert (labelled["randints"].coords["y"] == [1, 2, 3, 4]).all()
    assert (labelled["randints"].coords["z"] == [1, 2, 3, 4, 5]).all()

    # Add some non-uniform data sets
    # 3d with last dimension differing in length
    randlen = np.ones((3, 4, np.random.randint(10, 30)), dtype="uint8")
    rarrs.add(NumpyDataContainer(name="randlen", data=randlen))

    # 3d but of different shape in all directions
    randshape = np.ones(np.random.randint(1, 10, size=3), dtype="uint8")
    rarrs.add(NumpyDataContainer(name="randshape", data=randshape))


@pytest.fixture()
def pspace():
    """Used to setup a small pspace object to be tested on."""
    return ParamSpace(
        dict(
            foo="bar",
            p0=ParamDim(default=0, values=[1, 2], order=0),
            p1=ParamDim(default=0, values=[1, 2, 3]),
            p2=ParamDim(default=0, values=[1, 2, 3, 4, 5]),
        )
    )


@pytest.fixture()
def psp_grp(pspace):
    """Setup and populate a ParamSpaceGroup"""
    psp_grp = ParamSpaceGroup(name="mv", pspace=pspace)

    # Iterate over the parameter space and add groups to the ParamSpaceGroup
    for params, state_no_str in pspace.iterator(with_info="state_no_str"):
        create_psp_test_data(psp_grp, params=params, state_no_str=state_no_str)

    return psp_grp


@pytest.fixture()
def psp_grp_missing_data(pspace):
    """A ParamSpaceGroup with some states missing"""
    psp_grp = ParamSpaceGroup(name="mv_missing", pspace=pspace)

    for params, state_no_str in pspace.iterator(with_info="state_no_str"):
        if state_no_str in ("12", "31", "38", "39", "52", "59", "66"):
            continue
        create_psp_test_data(psp_grp, params=params, state_no_str=state_no_str)

    return psp_grp


@pytest.fixture()
def psp_grp_default(pspace):
    """Setup and populate a ParamSpaceGroup with only the default"""
    psp_grp = ParamSpaceGroup(
        name="mv_default", pspace=ParamSpace(pspace.default)
    )

    create_psp_test_data(psp_grp, params=pspace.default, state_no_str="0")

    return psp_grp
