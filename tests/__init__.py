# Pass regular log output to devnull; pytest catches log calls by other means

import logging
import os

logging.basicConfig(level=logging.DEBUG, stream=open(os.devnull, "w"))

# Adjust log level for certain modules
logging.getLogger("paramspace").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# .. Test-related variables ...................................................

TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_output")
"""An output directory that *can* be used to locally store data, e.g. for
looking at plot output.
"""

USE_TEST_OUTPUT_DIR = os.environ.get("DANTRO_USE_TEST_OUTPUT_DIR", False)
"""Whether to use the test output directory.

NOTE It depends on the tests if they actually take this flag into account!
     If using the ``tmpdir_or_local_dir`` fixture, this will be used to decide
     whether to return a tmpdir or a path within ``TEST_OUTPUT_DIR``.
"""
