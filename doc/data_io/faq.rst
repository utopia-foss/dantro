
.. _faq_dataio:

Data Handling FAQs
==================

This page gathers frequently asked questions regarding the dantro data handling interface.

.. contents::
   :local:
   :depth: 2

Aside from these FAQs, make sure to have a look :doc:`at other documentation pages related to data handling <../index>`.

.. note::

    If you would like to add a question here, we are happy about contributions!
    Please visit the `project page <https://gitlab.com/utopia-project/dantro>`_ to open an issue or get more information about contributing.

----

The :py:class:`~dantro.data_mngr.DataManager`
---------------------------------------------

*No FAQs yet. Feel free to ask the first one!*


Data :py:mod:`~dantro.groups` and :py:mod:`~dantro.containers`
--------------------------------------------------------------

Can I add any object to the data tree?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In principle, yes. But the object needs to be wrapped to concur with the required interface.

The easiest way to achieve this for leaves of the data tree is by using the :py:class:`~dantro.containers.general.ObjectContainer` or the :py:class:`~dantro.containers.general.PassthroughContainer`:

.. literalinclude:: ../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- data_io_faq_add_any_object
    :end-before:  ### End ---- data_io_faq_add_any_object
    :dedent: 4

As demonstrated above, these container types provide a thin wrapping around the stored object.

**Background:** Objects that make up the data tree need to concur to the :py:class:`~dantro.abc.AbstractDataContainer` or :py:class:`~dantro.abc.AbstractDataGroup` interface.
While such a type can also be constructed fully manually (see :doc:`../specializing`), many use cases can be covered by combining an already existing type from the :py:mod:`~dantro.containers` or :py:mod:`~dantro.groups` modules with some :py:mod:`~dantro.mixins`.



The Data Transformation Framework
---------------------------------

These are questions revolving around the :py:class:`~dantro.dag.TransformationDAG`.
For an in-depth look, see :doc:`transform`.


I get HDF5 or NetCDF4 errors when using the cache. How can I resolve this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When writing xarray data to the cache, you *might* encounter the following error message:

.. code-block:: none

    RuntimeError: Failed saving transformation cache file for result of type
    dantro.containers.xrdatactr.XrDataContainer using storage function ...

This error should trace back to the ``to_netcdf4`` method of xarray ``Dataset`` or xarray ``DataArray`` objects.
That method inspects whether the netcdf4 package is available, and if so: uses it to write the cache file.
If it is not available, it uses the scipy interface to achieve the same.

As far as we know (as of February 2020), the error seems to occur when both the h5py package (needed by dantro) and the netcdf4 package (*not* required by dantro, but maybe by some other package you are using) are installed in your currently used Python environment.
To check this, you can call ``pip freeze`` and inspect the list of installed packages.
One further indication for this being the reason is when you find HDF5-related errors in the traceback, e.g. ``RuntimeError: NetCDF: HDF error``.

There are two known **solutions** to this issue:

1.  **Uninstall netcdf4** from the Python environment.
    This is of course only possible if no other package depends on it.
2.  **Explicitly specify the netcdf4 engine**, such that the scipy package performs the write operation, not the netcdf4 package.
    To achieve this, pass the ``engine`` argument to the write function by extending the arguments passed to the corresponding :py:class:`~dantro.dag.Transformation`:

    .. code-block:: yaml

        file_cache:
          storage_options:
            engine: scipy

    More information:

    * :ref:`Full syntax specification <dag_transform_full_syntax_spec>`
    * :ref:`Passing storage options as defaults <dag_file_cache_defaults>`. Note that the defaults may cause issues if cache files for non-xarray objects need to be created.
    * `xarray documentation <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html>`_ of the ``to_netcdf4`` method.
