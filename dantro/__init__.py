"""dantro let's you load and manipulate hierarchically organized data"""

# Package version
__version__ = '0.11.0a0'

# Configure the logging module for the whole package here by importing the
# dantro-specific logging module
from .logging import getLogger
log = getLogger(__name__)

# Make manager classes available
from .data_mngr import DataManager
from .plot_mngr import PlotManager
