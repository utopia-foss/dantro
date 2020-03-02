The :py:class:`~dantro.data_mngr.DataManager`
=============================================

The :py:class:`~dantro.data_mngr.DataManager` is at the core of dantro: it stores data in a hierarchical way, thus forming the root of a data tree, and enables the loading of data into the tree.

.. contents::
   :local:
   :depth: 2

----

Overview
--------
Essentially, the :py:class:`~dantro.data_mngr.DataManager` is a specialization of a :py:class:`~dantro.groups.ordered.OrderedDataGroup` that is extended with data loading capabilities.

It is attached to a so-called "data directory" which is the base directory where data can be loaded from.


Data Loaders
------------
To provide certain loading capabilities to the :py:class:`~dantro.data_mngr.DataManager`, the :py:mod:`~dantro.data_loaders` mixin classes can be used.

.. autoclass:: dantro.data_loaders.AllAvailableLoadersMixin
    :noindex:
    :show-inheritance:
    :no-private-members:
    :no-members:

To learn more about the specialization, see :ref:`here <spec_data_mngr>`.


Loading Data
------------
To load data into the data tree, there are two methods:

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
A core concept of dantro is to make a lot of functionality available via YAML-based configuration files.
This is also true for the :py:class:`~dantro.data_mngr.DataManager`, which can be initialized with a certain load configuration which specifies the data entries to load.

For a known structure of the output data, it makes sense to pre-define the configuration somewhere and use that configuration to load all required data.
This configuration can be passed to the :py:class:`~dantro.data_mngr.DataManager` during initialization using the ``load_cfg`` argument.

An example for a rather complex load configuration is from the Utopia project:

.. code-block:: yaml

    # Supply a default load configuration for the DataManager
    load_cfg:
      # Load the frontend configuration files from the config/ directory
      cfg:
        loader: yaml
        glob_str: 'config/*.yml'
        required: true
        path_regex: config/(\w+)_cfg.yml
        target_path: cfg/{match:}

      # Load the configuration files that are generated for _each_ simulation
      # These hold all information that is available to a single simulation and
      # are in an explicit, human-readable form.
      uni_cfg:
        loader: yaml
        glob_str: universes/uni*/config.yml
        required: true
        path_regex: universes/uni(\d+)/config.yml
        target_path: uni/{match:}/cfg

      # Load the binary output data from each simulation.
      data:
        loader: hdf5_proxy
        glob_str: universes/uni*/data.h5
        required: true
        path_regex: universes/uni(\d+)/data.h5
        target_path: uni/{match:}/data

Once the :py:class:`~dantro.data_mngr.DataManager` is configured this way, it becomes very easy to load all configured data entries via :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg`:

.. code-block:: python

    dm = MyDataManager(data_dir="~/my_data", load_cfg=load_cfg_dict)
    dm.load_from_cfg()

The resulting data tree is:

.. code-block::

    DataManager
     └┬ cfg
        └┬ base
         ├ meta
         ├ model
         ├ run
         └ update
      └ uni
        └┬ 0
           └┬ cfg
            └ data
              └─ ...
         ├ 1
         ...

…thus allowing access in the following way:

.. code-block:: python

    # Access the data
    meta_cfg = dm['cfg/meta']
    some_param = cfg['some']['parameter']

    # Do something with the universes
    for uni_name, uni in dm['uni'].items():
        print("Current universe: ", uni_name)
        do_something_with(data=uni['data'], cfg=uni['cfg'])
