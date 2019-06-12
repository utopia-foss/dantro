"""Test the logging module"""

import pytest

from dantro import log

# -----------------------------------------------------------------------------

def test_logging():
    """...using the root logger"""
    log.trace("Trace")
    log.debug("Debug")
    log.note("Note")
    log.info("Info")
    log.progress("Progress")
    log.hilight("Hilight")
    log.warning("Warning")
    log.success("Success")
    log.error("Error")
    log.critical("Critical")
