#!/usr/bin/env python3

from setuptools import setup, find_packages

# Dependency lists
install_deps = [
    'h5py>=2.7.0',
    'numpy>=1.14',
    'matplotlib>=2.2.3',
    'xarray>=0.10.9',
    'networkx>=2.2',
    'paramspace>=2.1.0'
    ]
test_deps = ['pytest>=3.4.0', 'pytest-cov>=2.5.1', 'tox>=3.1.2']


setup(name='dantro',
      #
      # Set the version
      version='0.6.0-pre.1',
      # NOTE do not forget to set dantro.__init__.__version__!
      #
      # Project info
      description="Loading and handling hierarchical data",
      long_description=("With dantro, hierarchical data can be loaded and "
                        "conveniently handled, preserving the tree-like "
                        "structure. It aims at being abstract enough to be "
                        "used as a basis for more specialized data container "
                        "classes."),
      author="Yunus Sevinchan, Benjamin Herdeanu",
      author_email=("Yunus Sevinchan "
                    "<Yunus.Sevinchan@iup.uni.heidelberg.de>, "
                    "Benjamin Herdeanu "
                    "<Benjamin.Herdeanu@iup.uni.heidelberg.de>"),
      licence='MIT',
      url='https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro',
      classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities'
      ],
      #
      # Distribution details, dependencies, ...
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=install_deps,
      tests_require=test_deps,
      test_suite='py.test',
      extras_require=dict(test_deps=test_deps)
)
