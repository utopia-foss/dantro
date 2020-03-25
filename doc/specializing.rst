Specializing :py:mod:`dantro` Classes
=====================================

This page shows a few examples of how to specialize :py:mod:`dantro` classes to your liking.
This step is an important aspect of adapting dantro to work with the data structures that you are frequently using, which is beneficial for good :doc:`integration <integrating>` into your workflow.

.. contents::
    :local:
    :depth: 2

----

.. _spec_data_container:

Specializing a data container
-----------------------------
As an example, let's look at the implementation of the :py:class:`~dantro.containers.general.MutableSequenceContainer`, a container that is meant to store mutable sequences:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- specializing_mutable_sequence_container
    :end-before:  ### End ---- specializing_mutable_sequence_container
    :dedent: 4

The steps to arrive at this point are as follows:

The ``collections.abc`` python module is also used by python to specify the interfaces for python-internal classes.
`In the documentation it says <https://docs.python.org/3/library/collections.abc.html>`_ that the ``MutableSequence`` inherits from ``Sequence`` and has the following abstract methods: ``__getitem__``, ``__setitem__``, ``__delitem__``, ``__len__``, and ``insert``.

As we want the resulting container to adhere to this interface, we set ``MutableSequence`` as the first class to inherit from.
The :py:class:`~dantro.base.BaseDataContainer` is what makes this object a dantro data container.
It implements some of the required methods to concur with the ``MutableSequence`` interface but leaves others abstract.

Now, we need to supply implementations of these abstract methods.
That is the job of the following two (reading from right to left) mixin classes.
In this case, the ``Sequence`` interface has to be fulfilled.
As a ``Sequence`` is nothing more than a ``Collection`` with item access, we can fulfill this by inheriting from the :py:class:`~dantro.mixins.base.CollectionMixin` and the :py:class:`~dantro.mixins.base.ItemAccessMixin`.

The :py:class:`~dantro.mixins.base.CheckDataMixin` is an example of how functionality can be added to the container while still adhering to the interface.
This mixin checks the provided data before storing it and allows specifying whether unexpected data should lead to warnings or exceptions; for an example, see :ref:`below <spec_configuring_mixins>`

Some methods will remain abstract, in this case: ``insert``.
These need to be manually defined; the :py:class:`~dantro.containers.general.MutableSequenceContainer`\ 's :py:meth:`~dantro.containers.general.MutableSequenceContainer.insert` method does exactly that, thus becoming a fully non-abstract class:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- specializing_msc_insert
    :end-before:  ### End ---- specializing_msc_insert
    :dedent: 8


Using a specialized data container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once defined, instantiation of a custom container works the same way as for other data containers:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- specializing_using_mutable_sequence
    :end-before:  ### End ---- specializing_using_mutable_sequence
    :dedent: 4


.. _spec_configuring_mixins:

Configuring mixins
^^^^^^^^^^^^^^^^^^
Many mixins allow some form of configuration.
This typically happens via class variables.

Let's define a new container that strictly requires its stored data to be a ``list``, i.e. an often-used mutable sequence type.
We can use the already-included :py:class:`~dantro.mixins.base.CheckDataMixin` such that it checks a type.
To do so, we set the :py:const:`~dantro.mixins.base.CheckDataMixin.DATA_EXPECTED_TYPES` to only allow ``list`` and we set :py:const:`~dantro.mixins.base.CheckDataMixin.DATA_UNEXPECTED_ACTION` to raise an exception if this is not the case.

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- specializing_check_data_mixin
    :end-before:  ### End ---- specializing_check_data_mixin
    :dedent: 4

Other mixins provide other class variables for specializing behavior.
Consult the documentation or the source code to find out which ones.

.. note::

    The class variables typically define the *default* behavior for a certain specialized type.
    However, depending on the mixin, its behavior might also depend on runtime information, e.g. specified in ``__init__``.

.. warning::

    We advise *against* overwriting class variables during the lifetime of an object.


.. _spec_data_mngr:

Specializing the :py:class:`~dantro.data_mngr.DataManager`
----------------------------------------------------------
This works in essentially the same way: A :py:class:`~dantro.data_mngr.DataManager` is specialized by adding :py:mod:`~dantro.data_loaders` mixin classes.

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- specializing_data_manager
    :end-before:  ### End ---- specializing_data_manager
    :dedent: 4

That's all.

For more information, see :doc:`data_io/data_mngr`.

.. note::

    As an example, you can have a look at `data manager used in utopya <https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/blob/master/python/utopya/utopya/datamanager.py>`_.

