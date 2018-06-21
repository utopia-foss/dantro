"""The dantro package supplies classes to load and manipulate hierarchically
organised data
"""

# Configure the logging module for the whole package here
import logging
logging.basicConfig(format="%(levelname)-7s %(module)-12s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# Define version variable
__version__ = '0.1b'

# TODO consider making some classes available here
