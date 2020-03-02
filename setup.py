#!/usr/bin/env python3

from setuptools import setup, find_packages

# Dependencies for dantro itself
install_deps = [
    'numpy>=1.14',
    'xarray>=0.15.0',
    'dask>=2.10.1',
    'toolz>=0.10.0',        # Needed for dask.delayed
    'distributed>=2.10.0',  # Needed for dask's distributed scheduler
    'scipy>=1.4.1',         # Used as a netcdf4 storage engine for xarray
    'h5py>=2.7.0',
    'networkx>=2.2',
    'ruamel.yaml>=0.16.5',
    'matplotlib>=3.1.3',
    'paramspace>=2.2.3'
    ]

# Dependencies for the tests
test_deps = ['pytest>=3.4.0', 'pytest-cov>=2.5.1', 'tox>=3.1.2']

# Dependencies for the documentation
doc_deps = ['sphinx>=2.4,<3.0', 'sphinx_rtd_theme']

# .............................................................................

DESCRIPTION = "Loads, handles, and plots hierarchically structured data"
LONG_DESCRIPTION = """
With dantro, hierarchically structured heterogeneous data can be loaded into a
uniform data structure, a so-called data tree. Furthermore, dantro provides
capabilities to generically work on and create plots from that data. The whole
package aims at implementing an abstract and homogeneous interface that allows
specializing the provided classes to the needs to the data that is to be worked
with. At the same time, it already provides useful functionality, which makes
specialization easier.

Learn More
----------

For more information, visit the project page and have a look at the README:
https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro

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
    #
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
    install_requires=install_deps,
    tests_require=test_deps,
    test_suite='py.test',
    extras_require=dict(test_deps=test_deps, doc_deps=doc_deps)
)
