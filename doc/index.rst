Welcome to dantro's documentation!
==================================

:py:mod:`dantro` – from *data* and *dentro* (Greek for *tree*) – is a Python package that provides a uniform interface for hierarchically structured and semantically heterogeneous data.
It allows loading possibly heterogeneous data into a tree-like structure that can be conveniently handled for data manipulation, analysis, and plotting.

.. note::

    If you find any errors in this documentation or would like to contribute to the project, we are happy about your visit to the `project page <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro>`_.

.. hint::

    An **introductory text about the motivation and scope of dantro** can be found in `this publication in the Journal of Open Source Software <https://doi.org/10.21105/joss.02316>`_.


.. toctree::
    :maxdepth: 2

    usage
    integrating
    specializing
    philosophy
    how-to-cite

.. toctree::
    :caption: Data Handling and Transformation
    :maxdepth: 2
    :glob:

    data_io/data_mngr
    data_io/data_ops
    data_io/transform
    data_io/large_data
    data_io/*

.. toctree::
    :caption: Data Structures
    :maxdepth: 2

    data_structures/index

.. toctree::
    :caption: Plotting
    :maxdepth: 2
    :glob:

    plotting/plot_manager
    plotting/plot_creators
    plotting/plot_data_selection
    plotting/plot_functions
    plotting/plot_cfg_ref
    plotting/*

.. toctree::
    :caption: API Reference
    :maxdepth: 1

    dantro API Reference <api/dantro>

* :ref:`genindex`
* :ref:`modindex`
