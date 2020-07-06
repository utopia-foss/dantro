#!/usr/bin/env python3

from setuptools import setup, find_packages

# Dependencies for dantro itself
install_deps = [
    'numpy>=1.17.4',
    'xarray>=0.15.1',
    'dask>=2.10.1',
    'toolz>=0.10.0',        # Needed for dask.delayed
    'distributed>=2.10.0',  # Needed for dask's distributed scheduler
    'scipy>=1.4.1',         # Used as a netcdf4 storage engine for xarray
    'sympy>=1.5.1',
    'h5py>=2.10.0',
    'matplotlib>=3.1.3',
    'networkx>=2.2',
    'ruamel.yaml>=0.16.10',
    'paramspace>=2.5.0',
]
# NOTE When changing any of the dependencies, make sure to update the table of
#      dependencies in README.md

# Minimal versions of all of the above
minimal_install_deps = [dep.replace(">=", "==") for dep in install_deps]

# Dependencies for the tests
test_deps = [
    'pytest>=3.4.0',
    'pytest-cov>=2.5.1',
    'tox>=3.1.2',
]

# Dependencies for the documentation
doc_deps = [
    'sphinx>=2.4,<3.0',
    'sphinx_rtd_theme>=0.5',
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
   structure and providing a uniform interface for it
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
operations via YAML configuration files; the resulting pipeline can then be
controlled entirely via these configuration files and without requiring
code changes.

The ``dantro`` package is **open source software** released under the
`LGPLv3+ <(https://www.gnu.org/licenses/lgpl-3.0.html>`_.
It was developed alongside the `Utopia project <https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia>`_,
but is an independent package.

Learn more
----------

* `Documentation <https://dantro.readthedocs.io/>`_
* `Project page <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro>`_
* `Installation instructions <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro#installing-dantro>`_

"""


# .............................................................................

# A function to extract version number from __init__.py
def find_version(*file_paths) -> str:
    """Tries to extract a version from the given path sequence"""
    import os, re, codecs

    def read(*parts):
        """Reads a file from the given path sequence, relative to this file"""
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), 'r') as fp:
            return fp.read()

    # Read the file and match the __version__ string
    file = read(*file_paths)
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in " + str(file_paths))


# .............................................................................

setup(
    name='dantro',
    #
    # Set the version from dantro.__version__
    version=find_version('dantro', '__init__.py'),
    #
    # Project info
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="dantro developers",
    author_email="dantro-dev@iup.uni.heidelberg.de",
    licence='LGPL-3.0-or-later',
    url='https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)'
    ],
    #
    # Distribution details, dependencies, ...
    packages=find_packages(exclude=["tests.*", "tests"]),
    data_files=[("", ["COPYING", "COPYING.LESSER", "README.md"])],
    install_requires=install_deps,
    tests_require=test_deps,
    test_suite='py.test',
    extras_require=dict(minimal_deps=minimal_install_deps,
                        test=test_deps, doc=doc_deps,
                        dev=test_deps + doc_deps)
)
