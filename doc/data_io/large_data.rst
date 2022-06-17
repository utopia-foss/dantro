.. _handling_large_data:

Handling Large Amounts of Data
==============================

When working with large amounts of data, memory is often a limitation.
Dantro provides capabilities to work with data even when it cannot be loaded into memory at the same time.

It does so, by providing a general proxy mechanism, that allows postponing the actual loading of data up to the point where the data is actually needed.
Furthermore, for numerical data, it integrates with the `dask <https://dask.org>`_ framework, which allows `delayed <https://docs.dask.org/en/latest/delayed.html>`_ computations.

.. contents::
   :local:
   :depth: 2


----


Data Proxies
------------

To handle large amounts of data, dantro provides so-called *proxy support* for :py:class:`~dantro.abc.AbstractDataContainer`-derived objects.
They work in the following way:

* Data is not loaded directly into the container, but a proxy object is created
* The proxy object stores instructions on how the data can be loaded *at a later point*
* This allows building a data tree without loading any actual data
* Once the actual data becomes necessary, *proxy resolution* takes place: the data is loaded and the placeholder object is replaced by the actual data

Objects that were loaded as proxy are marked with ``(proxy)`` in the tree representation.

To make containers capable of proxy support, the :py:class:`~dantro.mixins.proxy_support.ProxySupportMixin` (or a derived specialization) can be used.
Additionally, the :py:mod:`~dantro.data_loaders` need to be able to create proxy objects during the loading procedure.


Proxy resolution
^^^^^^^^^^^^^^^^
A few more words about proxy resolution:
Proxies are meant to be **drop-in replacements**, not changing the workflow or the interface in *any* way.

The :py:class:`~dantro.mixins.proxy_support.ProxySupportMixin` takes care of this capability by specializing the :py:attr:`~dantro.mixins.proxy_support.ProxySupportMixin.data` property of containers:

* Instead of directly returning the underlying data, it checks if a placeholder object is present.
* If so, it resolves that proxy object by invoking its :py:meth:`~dantro.abc.AbstractDataProxy.resolve` method
* The proxy resolves the data
* The resolved data is stored in the container and the proxy object is either discarded or retained, depending on the configuration of :py:class:`~dantro.mixins.proxy_support.ProxySupportMixin`

In this minimal setting, proxies only get resolved upon explicit calls to the ``data`` property.
Some containers, e.g. the :py:class:`~dantro.containers.numeric.NumpyDataContainer`, use the :py:class:`~dantro.mixins.general.ForwardAttrsToDataMixin` which leads to all attribute calls (that are not explicitly defined in the container) being forwarded to the data.
In effect, the dantro container becomes a very thin wrapper around the actual interface, and the proxy gets resolved whenever that underlying interface is accessed.
This is an important aspect of making the data proxies drop-in replacements.



Example: :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin` provides proxy loading capabilities for HDF5 data.
Instead of loading the datasets directly into memory, the structure and metadata of the HDF5 file are used to generate the tree, but for data containers, :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy` objects are placed.

Additionally, it stores metadata about the dataset, e.g. its shape, data type, dimensionality, dataset attributes.
Accessing those metadata attributes of the resulting container does **not** result in proxy resolution; they are resolved only when the *actual* data is needed.

To load HDF5 data as proxy:

* Customize a container using the :py:class:`~dantro.mixins.proxy_support.Hdf5ProxySupportMixin`
* Customize a :py:class:`~dantro.data_mngr.DataManager` with the :py:class:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin`
* Pass the ``load_as_proxy`` argument to the regular ``hdf5`` loader or, as a shortcut to achieving the same: use the ``hdf5_proxy`` loader



Dask Support
------------
There will be scenarios in which the data that is to be analyzed exceeds the limits of the physical memory of the machine.
Here, proxy objects don't help, as they only *postpone* the loading.

This is often the case for numerical data, typically represented in dantro by the :py:class:`~dantro.containers.xrdatactr.XrDataContainer`, which are based on `xarray <http://xarray.pydata.org/en/stable/>`_ data structures.
As xarray provides an interface to the `dask <https://dask.org>`_ framework and its delayed computation capabilities, dantro can make use of that interface as well.

The dask package allows working on chunked data, e.g. HDF5 data, and only load those parts that are necessary for a calculation, afterward freeing up the memory again.
Additionally, it does clever things by first building a tree of operations that are to be performed, then optimizing that tree, and only when the actual numerical result is needed, does the data need to be loaded.
Furthermore, as the data is chunked, it can potentially profit from parallel computation.
More info on that can be found `in the corresponding section of the xarray documentation <https://xarray.pydata.org/en/stable/dask.html>`_.

Dask can be used in dantro when the following requirements are fulfilled:

* The data that is to be loaded is representable by xarray data structures
* The data is stored in a *chunked* fashion, allowing to read it in parts
* There is a dantro data loader that allows creating proxy objects
* There is a dantro data proxy type that supports resolving objects as dask objects

The following example shows how this works with HDF5-based data.


Using dask with HDF5 data
^^^^^^^^^^^^^^^^^^^^^^^^^
To use dask when loading HDF5 data, arguments need to be passed to the :py:class:`~dantro.proxy.hdf5.Hdf5DataProxy` that it should not be resolved as the actual data, but as a dask representation of it.
This is controlled by the ``resolve_as_dask`` argument.

HDF5-data is loaded using the :py:class:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin`, which allows passing arguments to the proxy via the ``proxy_kwargs`` argument.
In other words, the following :py:meth:`~dantro.data_mngr.DataManager.load` command will lead to HDF5 data being loaded as proxies that will later be resolved as dask objects:

.. code-block:: python

    dm = DataManager("~/my_data")
    dm.load("some_data", loader="hdf5_proxy", glob_str="*.hdf5",
            proxy_kwargs=dict(resolve_as_dask=True))

In the tree representation of the loaded data, you will see dask-supporting proxies marked as ``proxy (hdf5, dask)``.

Importantly, the dask-supporting proxies also are drop-in replacements for regular proxies; hence, behavior and interfaces do not change, but there is the added capability of working with huge amounts of data when necessary.

For a more extensive example, have a look at :ref:`this load configuration example <load_cfg_example_dask>`.
