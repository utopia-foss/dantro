The :py:class:`~dantro.data_mngr.DataManager`
=============================================

The :py:class:`~dantro.data_mngr.DataManager` is at the core of dantro: it stores data in a hierarchical way, thus forming the root of a data tree, and enables the loading of data into the tree.

.. contents::
   :local:
   :depth: 3

----

Overview
--------
Essentially, the :py:class:`~dantro.data_mngr.DataManager` is a specialization of a :py:class:`~dantro.groups.ordered.OrderedDataGroup` that is extended with data loading capabilities.

It is attached to a so-called "data directory" which is the base directory where data can be loaded from.


Data Loaders
------------
To provide certain loading capabilities to the :py:class:`~dantro.data_mngr.DataManager`, the :py:mod:`~dantro.data_loaders` mixin classes can be used.
To learn more about specializing the data manager to have the desired loading capabilities, see :ref:`here <spec_data_mngr>`.

By default, the following mixins are available via the :py:class:`~dantro.data_loaders.AllAvailableLoadersMixin`:

.. autoclass:: dantro.data_loaders.AllAvailableLoadersMixin
    :noindex:
    :show-inheritance:
    :no-private-members:
    :no-members:


Loading Data
------------
To load data into the data tree, there are two main methods:

* The :py:meth:`~dantro.data_mngr.DataManager.load` method loads a single so-called *data entry*.
* The :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` method loads multiple such entries; the ``cfg`` refers to a set of configuration entries.

For example, having :ref:`specialized <spec_data_mngr>` a data manager, data can be loaded in the following way:

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_data_mngr_example01
    :end-before:  ### End ---- data_io_data_mngr_example01
    :dedent: 4



The Load Configuration
^^^^^^^^^^^^^^^^^^^^^^
A :doc:`core concept <../philosophy>` of dantro is to make a lot of functionality available via configuration hierarchies, which are well-representable using YAML configuration files.
This is also true for the :py:class:`~dantro.data_mngr.DataManager`, which can be initialized with a certain default load configuration, specifying multiple data entries to load.

When integrating dantro into your project, you will likely be in a situation where the structure of the data you are working with is known and more or less fixed.
In such scenarios, it makes sense to pre-define which data you would like to load, how it should be loaded, and where it should be placed in the data tree.

This load configuration can be passed to the :py:class:`~dantro.data_mngr.DataManager` during initialization using the ``load_cfg`` argument, either as a path to a YAML file or as a dictionary.
When then invoking :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg`, these default entries are loaded.
Alternatively, :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` also accepts a new load config or allows updating the default load config.


Example Load Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, some advanced examples for specific load configurations are shown.
These illustrate the various ways in which data can be loaded into the data tree.
While most examples use only one single data entry, these can be readily combined into a common load configuration.

The basic setup for all the examples is as follows:

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_setup
    :end-before:  ### End ---- data_io_load_cfg_setup
    :dedent: 4

The examples below are all structured in the following way:

* First, they show the configuration that is passed as the ``my_load_cfg`` parameter, represented as yaml.
* Then, they show the python invocation of the :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` method, including the resulting data tree.
* Finally, they make a few remarks on what happened.

For specific information on argument syntax, refer to the docstring of the :py:meth:`~dantro.data_mngr.DataManager.load` method.

Defining a target path within the data tree
"""""""""""""""""""""""""""""""""""""""""""
The ``target_path`` option allows more control over where data is loaded to.

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- data_io_load_cfg_example01
    :end-before:  ### End ---- data_io_load_cfg_example01
    :dedent: 4

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_example01
    :end-before:  ### End ---- data_io_load_cfg_example01
    :dedent: 4

Remarks:

* With the ``required`` argument, an error is raised when no files were matched by ``glob_str``.
* With the ``path_regex`` argument, information from the path of the files can be used to generate a ``target_path`` within the tree, using the ``{match:}`` format string.
  In this example, this is used to drop the ``_cfg`` suffix, which would otherwise appear in the data tree.
  The regular expression is currently limited to a single match.
* With a ``target_path`` given, the name of the data entry (here: ``my_config_files``) is decoupled from the position where the data is loaded to.
  Without that argument and the regex, the config files would have been loaded as ``my_config_files/combined_cfg``, for example.


Combining data entries
""""""""""""""""""""""
The ``target_path`` option also allows combining data from different data entries, e.g. when they belong to the same measurement time:

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- data_io_load_cfg_example02
    :end-before:  ### End ---- data_io_load_cfg_example02
    :dedent: 4

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_example02
    :end-before:  ### End ---- data_io_load_cfg_example02
    :dedent: 4


Loading data as container attributes
""""""""""""""""""""""""""""""""""""
In some scenarios, it is desirable to load some data not as a regular entry into the data tree, but as a container attribute.
Continuing with the example from above, we might want to load the parameters directly into the container for each day.

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- data_io_load_cfg_example03
    :end-before:  ### End ---- data_io_load_cfg_example03
    :dedent: 4

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_example03
    :end-before:  ### End ---- data_io_load_cfg_example03
    :dedent: 4

Note the ``000`` group showing one more attribute than in previous examples; this is the ``params`` attribute.

Remarks:

* By using ``load_as_attr``, the measurement parameters are made available as *container* attribute and become accessible via its :py:attr:`~dantro.base.BaseDataGroup.attrs` property.
  (This is not to be confused with regular python object attributes.)
* When using ``load_as_attr``, the entry name is used as the attribute name.
* The ``unpack_data`` option makes the stored object a dictionary, rather than a ``MutableMappingContainer``, reducing one level of indirection.


Prescribing tree structure and specializations
""""""""""""""""""""""""""""""""""""""""""""""
Sometimes, load configurations become easier to handle when an empty tree structure is created prior to loading.
This can be done using the :py:class:`~dantro.data_mngr.DataManager`\ 's ``create_groups`` argument, also allowing to specify custom group classes, e.g. to denote a time series.

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- data_io_load_cfg_example04
    :end-before:  ### End ---- data_io_load_cfg_example04
    :dedent: 4

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_example04
    :end-before:  ### End ---- data_io_load_cfg_example04
    :dedent: 4

Remarks:

* Multiple paths can be specified in ``create_groups``.
* Paths can also have multiple segments, like ``my/custom/group/path``.
* The ``dm['measurements']`` entry is now a :py:class:`~dantro.groups.time_series.TimeSeriesGroup`, and thus represents one dimension of the stored data, e.g. the ``precipitation`` data.


.. _load_cfg_example_dask:

Loading data as proxy
"""""""""""""""""""""
Sometimes, data is too large to be loaded into memory completely.
For example, if we are only interested in the precipitation data, the sensor data should not be loaded into memory.

Dantro :ref:`provides a mechanism <handling_large_data>` to build the data tree using placeholder objects, so-called *proxies*.
The following example illustrates that, and furthermore uses the `dask <https://dask.org>`_ framework to allow delayed computations.

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- data_io_load_cfg_example05
    :end-before:  ### End ---- data_io_load_cfg_example05
    :dedent: 4

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_load_cfg_example05
    :end-before:  ### End ---- data_io_load_cfg_example05
    :dedent: 4

Remarks:

* By default, the :py:class:`~dantro.containers.numeric.NumpyDataContainer` and :py:class:`~dantro.containers.xrdatactr.XrDataContainer` classes do not provide proxy support.
  This is why a custom class needs to be :doc:`specialized <../specializing>` to allow loading the data as proxy.
* Furthermore, the :py:class:`~dantro.data_mngr.DataManager`\ 's :py:class:`~dantro.data_loaders.load_hdf5.Hdf5LoaderMixin` needs to be told to use the custom data container class.

For details about loading large data using proxies and dask, see :ref:`handling_large_data`.
