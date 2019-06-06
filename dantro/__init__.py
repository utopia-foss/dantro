"""The dantro package supplies classes to load and manipulate hierarchically
organised data
"""

# Configure the logging module for the whole package here
import logging
logging.basicConfig(format="%(levelname)-7s %(module)-12s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)


# Package version
__version__ = '0.9.0rc0'


# Make manager classes available
from .data_mngr import DataManager
from .plot_mngr import PlotManager
