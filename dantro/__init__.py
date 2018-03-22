"""The dantro package supplies classes to load and manipulate hierarchically organised data."""

# Configure the logging module for the whole package here
import logging
logging.basicConfig(format="%(levelname)-7s %(module)-14s %(message)s",
                    level=logging.DEBUG)
log = logging.getLogger(__name__)
