"""Test the logging module"""

import pytest

# -----------------------------------------------------------------------------


def test_logging():
    """...using the root logger"""
    from dantro import _log

    _log.trace("Trace")
    _log.debug("Debug")
    _log.note("Note")
    _log.info("Info")
    _log.ping("Ping")
    _log.progress("Progress")
    _log.hilight("Hilight")
    _log.warning("Warning")
    _log.success("Success")
    _log.error("Error")
    _log.critical("Critical")
