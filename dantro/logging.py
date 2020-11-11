"""Configures the DantroLogger for the whole package"""

import logging
from logging import getLogger

# Define the additional log levels
TRACE = 5
REMARK = 12
NOTE = 18
PROGRESS = 22
CAUTION = 23
HILIGHT = 25
SUCCESS = 35


class DantroLogger(logging.getLoggerClass()):
    """The custom dantro logging class with additional log levels"""

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


# Configure logging, valid for the whole module
logging.setLoggerClass(DantroLogger)
logging.basicConfig(
    format="%(levelname)-8s %(module)-12s %(message)s",
    level=logging.INFO,
)
