"""Tests the utils.data_ops module"""

import pytest

import dantro
from dantro.utils import OPERATIONS, register_operation, apply_operation

# -----------------------------------------------------------------------------

def test_OPERATIONS():
    """Test the hash function"""
    assert isinstance(OPERATIONS, dict)

def test_register_operations():
    """Test operation registration"""
    pass

def test_apply_operations():
    """Test operation application"""
    pass
