""":py:mod:`dantro` provides a uniform interface for hierarchically structured
and semantically heterogeneous data.
It is built around three main features:

- **data handling**: loading heterogeneous data into a tree-like data structure, providing a uniform interface to it
- **data transformation**: performing arbitrary operations on the data, if necessary using lazy evaluation
- **data visualization**: creating a visual representation of the processed data

Together, these stages constitute a **data processing pipeline**:
an automated sequence of predefined, configurable operations.

See :ref:`the user manual <welcome>` for more information.
"""

__version__ = "0.18.0b4"
"""Package version"""

from .logging import getLogger

log = getLogger(__name__)

# -- Most important dantro classes --------------------------------------------
from .data_mngr import DataManager
from .plot_mngr import PlotManager
