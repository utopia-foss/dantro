#!/usr/bin/env python3
# fmt: off

from setuptools import find_packages, setup

# Dependencies for dantro itself
install_deps = [
    "numpy>=1.21.5",
    "xarray>=0.16.2",
    "dask>=2.10",
    "toolz>=0.10",          # For dask.delayed
    "distributed>=2.10",    # For dask's distributed scheduler
    "scipy>=1.7.3",         # Used as a netcdf4 storage engine for xarray
    "sympy>=1.7",
    "h5py>=3.6",
    "matplotlib>=3.3",
    "seaborn>=0.11",
    "networkx>=2.6",
    "ruamel.yaml>=0.16.12",
    "dill>=0.3.3",          # For faster and more powerful pickling
    "paramspace>=2.5.6",
]
# NOTE When changing any of the dependencies, make sure to update the table of
#      dependencies in README.md.
#      When adding a NEW dependency, make sure to denote it in the isort
#      configuration, see pyproject.toml.


# Minimal versions of all of the above.
# Excluding numpy here because it would make dependency resolution very hard
# in some cases, e.g. because of non-compatible binaries ...
minimal_install_deps = [
    dep.replace(">=", "==")
    if not dep.startswith("numpy") else dep
    for dep in install_deps
]

# Dependencies for running tests and general development of dantro
test_deps = [
    "pytest>=3.4",
    "pytest-cov>=2.5",
    "tox>=3.1",
    "pre-commit>=2.15",
]

# Dependencies for building the dantro documentation
doc_deps = [
    "sphinx==4.*",
    "sphinx-book-theme>=0.3.*",
    "sphinx-togglebutton",
    "ipython>=7.0",
]

# .............................................................................

DESCRIPTION = "Handle, transform, and visualize hierarchically structured data"
LONG_DESCRIPTION = """
``dantro``: handle, transform, and visualize hierarchically structured data
===========================================================================

``dantro`` – from *data* and *dentro* (Greek for *tree*) – is a Python
package that provides a uniform interface for hierarchically structured
and semantically heterogeneous data. It is built around three main
features:

-  **data handling:** loading heterogeneous data into a tree-like data
   structure, providing a uniform interface to it
-  **data transformation:** performing arbitrary operations on the data,
   if necessary using lazy evaluation
-  **data visualization:** creating a visual representation of the
   processed data

Together, these stages constitute a **data processing pipeline**: an
automated sequence of predefined, configurable operations. Akin to a
Continuous Integration pipeline, a data processing pipeline provides a
uniform, consistent, and easily extensible infrastructure that
contributes to more efficient and reproducible workflows. This can be
beneficial especially in a scientific context, for instance when
handling data that was generated by computer simulations.

``dantro`` is meant to be *integrated* into projects and be used to set up
such a data processing pipeline, customized to the needs of the project.
It is designed to be **easily customizable** to the requirements of the project
it is integrated in, even if the involved data is hierachically structured or
semantically heterogeneous.
Furthermore, it allows a **configuration-based specification** of all
operations via `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration
files; the resulting pipeline can then be controlled entirely via these
configuration files and without requiring code changes.

The ``dantro`` package is **open source software** released under the
`LGPLv3+ <https://www.gnu.org/licenses/lgpl-3.0.html>`_ license.
It was developed alongside the `Utopia project <https://gitlab.com/utopia-project/utopia>`_
(a modelling framework for complex and adaptive systems), but is an
independent package.

Learn more
----------

* `Documentation <https://dantro.readthedocs.io/>`_
* `Project page <https://gitlab.com/utopia-project/dantro>`_
* `README and installation instructions <https://gitlab.com/utopia-project/dantro#installing-dantro>`_
* `Publication in the Journal of Open Source Software <https://doi.org/10.21105/joss.02316>`_
* `Utopia Project Website <https://utopia-project.org/>`_

"""


# .............................................................................
# fmt: on

# A function to extract version number from __init__.py
def find_version(*file_paths) -> str:
    """Tries to extract a version from the given path sequence"""
    import codecs
    import os
    import re

    def read(*parts):
        """Reads a file from the given path sequence, relative to this file"""
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), "r") as fp:
            return fp.read()

    # Read the file and match the __version__ string
    file = read(*file_paths)
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in " + str(file_paths))


# .............................................................................

setup(
    name="dantro",
    #
    # Set the version from dantro.__version__
    version=find_version("dantro", "__init__.py"),
    #
    # Project info
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="dantro developers",
    author_email="dantro-dev@iup.uni.heidelberg.de",
    license="LGPL-3.0-or-later",
    url="https://gitlab.com/utopia-project/dantro",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or"
        " later (LGPLv3+)",
    ],
    #
    # Distribution details, dependencies, ...
    packages=find_packages(exclude=["tests.*", "tests"]),
    data_files=[("", ["COPYING", "COPYING.LESSER", "README.md"])],
    python_requires=">=3.8",
    install_requires=install_deps,
    tests_require=test_deps,
    test_suite="py.test",
    extras_require=dict(
        minimal_deps=minimal_install_deps,
        test=test_deps,
        doc=doc_deps,
        dev=test_deps + doc_deps,
    ),
)
