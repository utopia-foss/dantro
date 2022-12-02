Specializing :py:mod:`dantro` Classes
=====================================

This page shows a few examples of how to specialize :py:mod:`dantro` classes to your liking.
This step is an important aspect of adapting dantro to work with the data structures that you are frequently using, which is beneficial for good :doc:`integration <integrating>` into your workflow.

.. note::

    The code snippets shown on this page are implemented as test cases to assert that they function as intended.
    To have a look at the full source code used in the examples below, you can :download:`download the relevant file <../tests/test_doc_examples.py>` or `view it online <https://gitlab.com/utopia-project/dantro/-/blob/master/tests/test_doc_examples.py>`_.

    Note that the integration into the test framework requires some additional code in those files, e.g. to generate dummy data.

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

.. hint::

    It's not strictly required to define a new ``DataManager`` class to use loader mixins:
    The :py:class:`~dantro.data_mngr.DataManager` is aware of all registered data loaders and can access them via the :py:data:`~dantro.data_loaders._registry.DATA_LOADERS` registry.

    However, if you plan on extending it, it may be the more convenient approach to define this custom class and include mixins.

.. note::

    When using :ref:`specialized container classes <spec_data_container>` such a custom :py:class:`~dantro.data_mngr.DataManager` is also the place to configure data loaders to use those classes.
    For example, when using the :py:class:`~dantro.data_loaders.hdf5.Hdf5LoaderMixin`, the ``_HDF5``\ -prefixed class variables can be set to use the specialized container classes rather than the defaults.

.. note::

    For an integration example, you can have a look at `the data manager used in utopya <https://gitlab.com/utopia-project/utopya/-/blob/main/utopya/eval/datamanager.py>`_.


.. _spec_data_loader:

Adding data loaders
^^^^^^^^^^^^^^^^^^^
Adding a custom data loader is simple.
As an example, let's look at how a data loader mixin for plain text files (:py:class:`~dantro.data_loaders.text.TextLoaderMixin`) is implemented in dantro:

.. literalinclude:: ../dantro/data_loaders/text.py
    :language: python

So basically:

#. Import the :py:class:`~dantro.data_loaders._registry.add_loader` decorator from ``dantro.data_loaders``
#. Define your mixin class
#. Add a method named ``_load_<name>`` and decorate it with ``@add_loader(TargetCls=SomeClass)``.

    .. note::

        Here, you have to decide for a target type for the return value of the loader.
        This can be any dantro container or group type, see :py:mod:`dantro.containers` or :py:mod:`dantro.groups`.

        If there is no suitable container type, you can either :ref:`specialize one yourself <spec_data_container>`.
        Alternatively, the :py:class:`~dantro.containers.general.PassthroughContainer` always works.

#. Fill in the method's body to implement the loading of your data.
#. Initialize and return the ``TargetCls`` object, passing the loaded ``data`` to it.



.. _spec_plot_mngr:

Specializing the :py:class:`~dantro.plot_mngr.PlotManager`
----------------------------------------------------------
The plot manager can be specialized to support further functionality simply by overloading methods that may or may not invoke the parent methods.
However, given the complexity of the plot manager, there is no guide on how to do this exactly: It depends a lot on what you want to achieve.

In a simple situation, a specialized :py:class:`~dantro.plot_mngr.PlotManager` may simply overwrite some default values via the class variables.
This could, for instance, be the plot function resolver, which defaults to :py:class:`~dantro.plot.utils.plot_func.PlotFuncResolver`:

.. testcode::

    import dantro

    class MyPlotFuncResolver(dantro.plot.utils.PlotFuncResolver):
        """A custom plot function resolver class"""

        BASE_PKG = "my_custom_package.plot_functions"
        """For relative module imports, regard this as the base package.
        A plot configuration ``module`` argument starting with a ``.`` is
        looked up in that module.

        Note that this needs to be an importable module.
        """

    class MyPyPlotManager(dantro.PlotManager):
        """My custom plot manager"""

        PLOT_FUNC_RESOLVER = MyPlotFuncResolver
        """Use a custom plot function resolver"""

.. note::

    For an operational example in a more complex framework setting, see `the specialization used in the utopya project <https://gitlab.com/utopia-project/utopya/-/blob/main/utopya/eval/plotmanager.py>`_.
    There, the :py:class:`~dantro.plot_mngr.PlotManager` is extended such that a number of custom module paths are made available for import.



.. _spec_plot_creators:

Specializing :py:class:`~dantro.plot.creators.base.BasePlotCreator`
-------------------------------------------------------------------
As described in :ref:`plot_creators`, dantro already supplies a range of plot creators.
Furthermore, dantro provides the :py:class:`~dantro.plot.creators.base.BasePlotCreator`, which provides an interface and a lot of the commonly used functionality.

Specialization thus can be of two kinds:

1. Using an existing plot creator and configuring it to your needs.
2. Implementing a whole *new* plot creator, e.g. because you desire to use a different plotting backend.

In general, we recommend to refer to the implementation of existing :py:mod:`dantro.plot.creators` as examples for how this can be achieved.
We are happy to support the implementation of new plot creators, so feel free to post an issue to `the project page <https://gitlab.com/utopia-project/dantro>`_.


.. note::

    After specializing a plot creator, make sure to let the :py:class:`~dantro.plot_mngr.PlotManager` (or your specialization of it) know about your new creator class.
    You can do so by extending its :py:attr:`~dantro.plot_mngr.PlotManager.CREATORS` mapping.

    Also see :doc:`the integration guide <integrating>` for an overview.
