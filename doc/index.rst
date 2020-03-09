Welcome to dantro's documentation!
==================================

The :py:mod:`dantro` package — from *data* and *dentro* (gr., for tree) – is a Python package to work with hierarchically organized data using a uniform interface.
It allows loading possibly heterogeneous data into a tree-like structure that can be conveniently handled for data manipulation, analysis, and plotting.


.. note::

    If you find any errors in this documentation or would like to contribute to the project, we are happy about your visit to the `project page <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro>`_.


.. toctree::
    :maxdepth: 2

    examples
    philosophy
    specializing
    integrating

.. toctree::
    :caption: Data Handling
    :maxdepth: 2
    :glob:

    data_io/data_mngr
    data_io/data_ops
    data_io/transform
    data_io/large_data
    data_io/multidim
    data_io/*


.. toctree::
    :caption: Plotting
    :maxdepth: 2
    :glob:

    plotting/plot_manager
    plotting/plot_creators
    plotting/plot_data_selection
    plotting/*

.. toctree::
    :caption: API Reference
    :maxdepth: 1

    dantro API Reference <api/dantro>

* :ref:`genindex`
* :ref:`modindex`
