"""Configures the DantroLogger for the whole package"""

import logging
import os
from logging import getLogger

# Define the additional log levels
TRACE: int = 5
REMARK: int = 12
NOTE: int = 18
PROGRESS: int = 22
CAUTION: int = 23
HILIGHT: int = 25
SUCCESS: int = 35

_LOG_SETTINGS = dict(suppress_in_child_process=False, divert_to=None)
"""A mutable object allowing to dynamically change some log settings"""


class DantroLogger(logging.getLoggerClass()):
    """The custom dantro logging class with additional log levels"""

    @staticmethod
    def change_settings(**updates):
        """Changes the module-level log settings"""
        _LOG_SETTINGS.update(updates)

    @property
    def settings(self) -> dict:
        return _LOG_SETTINGS

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

        logging.addLevelName(TRACE, "TRACE")
        logging.addLevelName(REMARK, "REMARK")
        logging.addLevelName(NOTE, "NOTE")
        logging.addLevelName(PROGRESS, "PROGRESS")
        logging.addLevelName(CAUTION, "CAUTION")
        logging.addLevelName(HILIGHT, "HILIGHT")
        logging.addLevelName(SUCCESS, "SUCCESS")

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    def remark(self, msg, *args, **kwargs):
        if self.isEnabledFor(REMARK):
            self._log(REMARK, msg, args, **kwargs)

    def note(self, msg, *args, **kwargs):
        if self.isEnabledFor(NOTE):
            self._log(NOTE, msg, args, **kwargs)

    def progress(self, msg, *args, **kwargs):
        if self.isEnabledFor(PROGRESS):
            self._log(PROGRESS, msg, args, **kwargs)

    def caution(self, msg, *args, **kwargs):
        if self.isEnabledFor(CAUTION):
            self._log(CAUTION, msg, args, **kwargs)

    def hilight(self, msg, *args, **kwargs):
        if self.isEnabledFor(HILIGHT):
            self._log(HILIGHT, msg, args, **kwargs)

    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)

    def _log(self, lvl, msg, *args, **kwargs):
        if self.settings.get("suppress_in_child_process"):
            in_child_proc = os.getppid() == os.getpgid(0)
            if in_child_proc:
                return

        divert_to = self.settings.get("divert_to")
        if divert_to is not None:
            divert_to.write(msg % args[0] + "\n")  # FIXME proper formatting
            return

        super()._log(lvl, msg, *args, **kwargs)


# .............................................................................


# Determine the log format
_DEFAULT_LOG_FORMAT: str = "%(levelname)-8s %(module)-12s %(message)s"
_DEFAULT_LOG_LEVEL: int = logging.INFO

DANTRO_LOG_FORMAT = os.getenv("DANTRO_LOG_FORMAT", _DEFAULT_LOG_FORMAT)
DANTRO_LOG_LEVEL = int(os.getenv("DANTRO_LOG_LEVEL", _DEFAULT_LOG_LEVEL))

# Configure logging, valid for the whole module
logging.setLoggerClass(DantroLogger)
logging.basicConfig(
    format=DANTRO_LOG_FORMAT,
    level=DANTRO_LOG_LEVEL,
)
