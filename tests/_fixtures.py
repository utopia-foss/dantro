"""Test utilities, fixtures, ..."""

import os
import pathlib

import pytest

from . import TEST_OUTPUT_DIR, USE_TEST_OUTPUT_DIR

# -----------------------------------------------------------------------------


@pytest.fixture
def tmpdir_or_local_dir(tmpdir, request) -> pathlib.Path:
    """If ``USE_TEST_OUTPUT_DIR`` is False, returns a temporary directory;
    otherwise a test-specific local directory within ``TEST_OUTPUT_DIR`` is
    returned.
    """
    if not USE_TEST_OUTPUT_DIR:
        return tmpdir

    test_dir = os.path.join(
        TEST_OUTPUT_DIR,
        request.node.module.__name__,
        request.node.originalname,
    )
    os.makedirs(test_dir, exist_ok=True)
    return pathlib.Path(test_dir)
