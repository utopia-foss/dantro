Specializing :py:mod:`dantro` Classes
=====================================

This page shows a few examples of how to specialize :py:mod:`dantro` classes to your liking.

.. contents::
    :local:
    :depth: 2

----

.. _spec_data_container:

Specializing a data container
-----------------------------
As an example, let's look at the implementation of the :py:class:`~dantro.containers.general.MutableSequenceContainer`, a container that is meant to store mutable sequences:

.. code-block:: python

    # Import the python abstract base class we want to adhere to
    from collections.abc import MutableSequence

    # Import base mixin classes (others can be found in the mixin module)
    from dantro.base import (BaseDataContainer, ItemAccessMixin,
                             CollectionMixin, CheckDataMixin)


    class MutableSequenceContainer(CheckDataMixin,
                                   ItemAccessMixin,
                                   CollectionMixin,
                                   BaseDataContainer,
                                   MutableSequence):
        """The MutableSequenceContainer stores data that is sequence-like"""
        # ...

The steps to arrive at this point are as follows:

The ``collections.abc`` python module is also used by python to specify the interfaces for python-internal classes.
`In the documentation it says <https://docs.python.org/3/library/collections.abc.html>`_ that the ``MutableSequence`` inherits from ``Sequence`` and has the following abstract methods: ``__getitem__``, ``__setitem__``, ``__delitem__``, ``__len__``, and ``insert``.

As we want the resulting container to adhere to this interface, we set ``MutableSequence`` as the first class to inherit from.
The :py:class:`~dantro.base.BaseDataContainer` is what makes this object a dantro data container.
It implements some of the required methods to concur to the ``MutableSequence`` interface, but leaves others abstract.

Now, we need to supply implementations of these abstract methods.
That is the job of the following two (reading from right to left) mixin classes.  
In this case, the ``Sequence`` interface has to be fulfilled.
As a ``Sequence`` is nothing more than a ``Collection`` with item access, we can fulfill this by inheriting from the :py:class:`~dantro.mixins.base.CollectionMixin` and the :py:class:`~dantro.mixins.base.ItemAccessMixin`.

The :py:class:`~dantro.mixins.base.CheckDataMixin` is an example of how functionality can be added to the container while still adhering to the interface.
This mixin checks the provided data before storing it and allows specifying whether unexpected data should lead to warnings or exceptions.

Some methods will still remain abstract, in this case: ``insert``.
These need to be manually defined; the :py:class:`~dantro.containers.general.MutableSequenceContainer`\ 's :py:meth:`~dantro.containers.general.MutableSequenceContainer.insert` method does exactly that, thus becoming a fully non-abstract class.

Using a specialized data container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once defined, instantiation of a custom container works the same way as for other data containers:

.. code-block:: python

    dc = MutableSequenceContainer(name="my_mutable_sequence",
                                  data=[4, 8, 16])

    # Insert values
    dc.insert(0, 2)
    dc.insert(0, 1)

    # Item access and collection interface
    assert 16 in dc
    assert 32 not in dc
    assert dc[0] == 1

    for num in dc:
        print(num, end=", ")
    # prints:  1, 2, 4, 8, 16,

.. _spec_data_mngr:

Specializing the :py:class:`~dantro.data_mngr.DataManager`
----------------------------------------------------------
This works in essentially the same way: A :py:class:`~dantro.data_mngr.DataManager` is specialized by adding :py:mod:`~dantro.data_loaders` mixin classes.

.. code-block:: python

    import dantro as dtr
    import dantro.data_mngr
    from dantro.data_loaders import YamlLoaderMixin, PickleLoaderMixin


    class MyDataManager(PickleLoaderMixin,
                        YamlLoaderMixin,
                        dtr.data_mngr.DataManager):
        """This is a dantro data manager specialization that can load pickle
        and yaml data.
        """

That's all.

For more information, see :doc:`data_io/data_mngr`.

.. note::
    
    As an example, you can have a look at `data manager used in utopya <https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/blob/master/python/utopya/utopya/datamanager.py>`_.

