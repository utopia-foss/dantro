Welcome to dantro's documentation!
==================================

The :py:mod:`dantro` package — from *data* and *dentro* (gr., for tree) – is a Python package to work with hierarchically organized data.
It allows loading possibly heterogeneous data into a tree-like structure that can be conveniently handled for data manipulation, analysis, and plotting.


.. note::

    This documentation is work in progress.
    If you find any errors or would like to contribute to it, please visit the `project page <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro>`_ and open an issue there.


.. toctree::
    :maxdepth: 1

    philosophy
    specializing

.. toctree::
    :caption: Data Handling
    :maxdepth: 1
    :glob:
   
    data_io/data_mngr
    data_io/*


.. toctree::
    :caption: Plotting
    :maxdepth: 1
    :glob:

    plotting/plot_manager
    plotting/plot_creators   
    plotting/*

.. toctree::
    :caption: API Reference
    :maxdepth: 1

    API Reference <api/dantro>

* :ref:`genindex`
* :ref:`modindex`
