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

It is attached to a data directory which is seen as the directory to load data from.

.. todo::

    Write more here.


Loading Data
------------
To load data into the data tree, there are two methods:

* The :py:meth:`~dantro.data_mngr.DataManager.load` method loads a single so-called *data entry*.
* The :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` method loads multiple such entries; the ``cfg`` refers to a set of configuration entries.


The Load Configuration
^^^^^^^^^^^^^^^^^^^^^^
A core concept of dantro is to make a lot of functionality available via YAML-based configuration files.
This is also true for the :py:class:`~dantro.data_mngr.DataManager`, which can be initialized with a certain load configuration which specifies the data entries to load.

.. todo::

    Add examples here.


Data Loaders
^^^^^^^^^^^^
To provide certain loading capabilities to the :py:class:`~dantro.data_mngr.DataManager`, the :py:mod:`~dantro.data_loaders` mixin classes can be used.


.. autoclass:: dantro.data_loaders.AllAvailableLoadersMixin
    :noindex:
    :show-inheritance:
    :no-private-members:
    :no-members:
