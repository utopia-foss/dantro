"""Tests the PlotCreator class"""

import pytest

import dantro.plt_creator as pcr

# Fixtures --------------------------------------------------------------------

# Tests -----------------------------------------------------------------------

def test_init():
    """Tests initialisation"""
    pcr.BasePlotCreator()
