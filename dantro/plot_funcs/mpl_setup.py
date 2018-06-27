"""This module sets up matplotlib to be used in the other modules."""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Can define helper functions here
