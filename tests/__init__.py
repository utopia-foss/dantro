# Pass regular log output to devnull; pytest catches log calls by other means

import logging
import os
import platform

logging.basicConfig(level=logging.DEBUG, stream=open(os.devnull, "w"))

# Adjust log level for certain modules
logging.getLogger("paramspace").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# .............................................................................


def _str2bool(val: str):
    """Copy of strtobool from deprecated distutils package"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value {repr(val)}!")


# .. Test-related variables ...................................................

TEST_VERBOSITY: int = int(os.environ.get("DANTRO_TEST_VERBOSITY", 2))
"""A verbosity-controlling value. This can be interpreted in various ways by
the individual tests, but 0 should mean very low verbosity and 3 should be the
maximum verbosity."""

USE_TEST_OUTPUT_DIR: bool = _str2bool(
    os.environ.get("DANTRO_USE_TEST_OUTPUT_DIR", "false")
)
"""Whether to use the test output directory. Can be set via the environment
variable ``DANTRO_USE_TEST_OUTPUT_DIR``.

NOTE It depends on the tests if they actually take this flag into account!
     If using the ``tmpdir_or_local_dir`` fixture, this will be used to decide
     whether to return a tmpdir or a path within ``TEST_OUTPUT_DIR``.
"""

ABBREVIATE_TEST_OUTPUT_DIR: bool = _str2bool(
    os.environ.get("DANTRO_ABBREVIATE_TEST_OUTPUT_DIR", "false")
)

TEST_OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "_output")
"""An output directory that *can* be used to locally store data, e.g. for
looking at plot output. By default, this will be in the ``tests`` directory
itself, but if the ``DANTRO_TEST_OUTPUT_DIR`` environment variable is set, will
use that path instead.
"""

if os.environ.get("DANTRO_TEST_OUTPUT_DIR"):
    TEST_OUTPUT_DIR = os.environ["DANTRO_TEST_OUTPUT_DIR"]
    print(
        "Using test output directory set from environment variable:\n"
        f"  {TEST_OUTPUT_DIR}\n"
    )


ON_WINDOWS = platform.system() == "Windows"
"""Check whether the current platform is Windows. If tests cannot be run on Windows they can be skipped (this is
controlled from the test configs)"""
