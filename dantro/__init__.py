"""The dantro package supplies classes to load and manipulate hierarchically
organised data
"""

# Configure the logging module for the whole package here
import logging
logging.basicConfig(format="%(levelname)-7s %(module)-12s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# Define version variable
__version__ = '0.4.0-pre.5'
# NOTE This should always be the same as in setup.py

# TODO consider making some classes available here
