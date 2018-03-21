#!/usr/bin/env python3

from setuptools import setup

# Dependency lists
install_deps = [
    'h5py>=2.7.0',
    'numpy>=1.14',
    'pandas>=0.21',
    'PyYAML>=3.12'
    ]
test_deps = ['pytest>=3.4.0', 'pytest-cov>=2.5.1']

setup(name='dantro',
      version='0.1b',
      description="Loading and handling hierarchical data",
      long_description=("With dantro, hierarchical data can be loaded and "
                        "conveniently handled, preserving the tree-like "
                        "structure. It aims at being abstract enough to be "
                        "used as a basis for more specialised data container "
                        "classes."),
      author="Yunus Sevinchan",
      author_email='Yunus.Sevinchan@iup.uni.heidelberg.de',
      licence='MIT',
      url='https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro',
      classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities'
      ],
      packages=['dantro'],
      install_requires=install_deps,
      tests_require=test_deps,
      test_suite='py.test',
      extras_require=dict(test_deps=test_deps)
)