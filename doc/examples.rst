.. _examples:

Usage Examples
==============

This page will guide you through the basic usage of dantro.
The only prerequisite for running these examples is that dantro is installed.
For installation instructions, have a look at the README.

.. note::

    These examples do **not** go into depth about all dantro features but aim to give an overview.
    To get a deeper look, follow the links provided on this page and in the :doc:`rest of the documentation <index>`.

    Specifically, these examples do **not** show how dantro can be specialized for your use case and tightly integrated into your workflow.
    For that, see :doc:`specializing` and :doc:`integrating`, respectively.

.. contents::
    :local:
    :depth: 2

----

Setting up dantro
-----------------
To get started with dantro, the first thing to do is specializing it for your use case.
For the purpose of this example, let's say we are working on a project where we need to handle data stored in the HDF5 format and some YAML data.

The first step is to the :py:class:`~dantro.data_mngr.DataManager` to be able to load HDF5 data:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_setup_dantro
    :end-before:  ### End ---- examples_setup_dantro
    :dedent: 4

We now have the ``MyDataManager`` defined, which has all the data-loading capabilities we need.
There is no further setup necessary at this point.

To read more about specializing dantro, have a look at :doc:`specializing`.


.. _examples_loading_data:

Loading data
------------
Having defined a specialization of the :py:class:`~dantro.data_mngr.DataManager`, ``MyDataManager``, we now want to load some data with it.
To do so, we initialize such an object, specifying the directory we want to load data from.

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_setup
    :end-before:  ### End ---- examples_loading_setup
    :dedent: 4

The name can (optionally) be given to distinguish this manager from others.
Because we have not loaded any data yet, the data tree should be empty.
Let's check:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_empty_tree
    :end-before:  ### End ---- examples_loading_empty_tree
    :dedent: 4

Now, let's load some YAML data!
In the associated data directory, let's say we have some YAML files like ``foobar.yml``, which are some configuration files we want to have available.
To load these YAML files, we simply need to invoke the :py:meth:`~dantro.data_mngr.DataManager.load` method and specify the ``yaml`` loader which we made available by mixing in the :py:class:`~dantro.data_loaders.load_yaml.YamlLoaderMixin`.
Also, we need to specify the name of the data entry

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_yaml
    :end-before:  ### End ---- examples_loading_yaml
    :dedent: 4

.. note::

    The target path need not necessarily match the entry name, but more sophisticated ways of placing the loaded data inside the tree are also available.
    See :py:meth:`~dantro.data_mngr.DataManager.load` for more info.

With the configuration files loaded, let's work with them.
Access within the tree can happen simply via item access.
Item access *within* the tree also allows specifying paths, i.e. using ``/`` to traverse hierarchical levels:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_work_with_objects
    :end-before:  ### End ---- examples_loading_work_with_objects
    :dedent: 4

As you see, groups within the data tree behave like dictionaries.
Accordingly, we can also iterate over them as we would with dictionaries:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_iteration
    :end-before:  ### End ---- examples_loading_iteration
    :dedent: 4

Now, how about adding some numerical data to the tree, e.g. as stored in a hierarchically organized HDF5 file.
To do so, the ``hdf5`` loader can be used:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_loading_hdf5
    :end-before:  ### End ---- examples_loading_hdf5
    :dedent: 4

As can be seen in the tree, for each HDF5 file, a corresponding dantro group was created, e.g.: for ``measurements/day000.h5``, a ``measurements/day000`` group is available, which contains the hierarchically organized data from the HDF5 file.
For each HDF5 dataset, a corresponding :py:class:`~dantro.containers.numeric.NumpyDataContainer` was created.

.. note::

    The :py:class:`~dantro.data_mngr.DataManager` becomes especially powerful when groups and containers are specialized such that they can make use of knowledge about the structure of the data.

    For example, the ``measurements`` group semantically represents a time series.
    Ideally, the group it is loaded into should be able to combine the measurements for each day into a higher-dimensional object, thus making it easier to work with the data.
    This is possible by :doc:`specializing <specializing>` these groups.

To learn more about the :py:class:`~dantro.data_mngr.DataManager` and how data can be loaded, see :doc:`data_io/data_mngr`.


Plotting
--------

Plotting is orchestrated by the :py:class:`~dantro.plot_mngr.PlotManager`.
Let's create one and associate it with the existing :py:class:`~dantro.data_mngr.DataManager`:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_plotting_setup
    :end-before:  ### End ---- examples_plotting_setup
    :dedent: 4

To plot, we invoke the :py:meth:`~dantro.plot_mngr.PlotManager.plot` method:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- examples_plotting_basic_lineplot
    :end-before:  ### End ---- examples_plotting_basic_lineplot
    :dedent: 4

At this point, the arguments given to :py:meth:`~dantro.plot_mngr.PlotManager.plot` have not been explained.
Furthermore, the example seems not particularly useful, e.g. because of the manually specified path to the data.
So... what is this about?!

The full power of the plotting framework comes to shine only when it is specialized for the data you are evaluating and integrated into your workflow.
Once that is done, it allows:

* Generically specifying plots in configuration files, without the need to touch code
* Automatically generating plots for parts of the data tree
* Using declarative data preprocessing
* Defining plotting functions that can be re-used for different kinds of data
* Consistently specifying the aesthetics of one or multiple plots
* Conveniently creating animations
* ... and much more.

To learn more about the structure and the capabilities of the plotting framework, see :doc:`here <plotting/plot_manager>`.
