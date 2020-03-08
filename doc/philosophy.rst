Philosophy and Design Concepts
==============================

:py:mod:`dantro` aims to be a general package for handling, transforming, and visualizing hierarchically organized data using a uniform interface.
This page gives an overview of key design concepts and use cases of dantro.

.. contents::
    :local:
    :depth: 1

----

.. _phil_data_tree:

The Data Tree
-------------
At the center of dantro is the data tree which is used to represent hierarchically structured data.
Sticking with the tree analogy, there are the following counterparts in dantro:

* The *root* of the tree is given by the :py:class:`~dantro.data_mngr.DataManager`
* The *branching points* correspond to :py:mod:`~dantro.groups`
* The *leaves* are represented by :py:mod:`~dantro.containers`

By organizing all the data that is to be worked with into this structure, it is accessible in a uniform way.



.. _phil_customization:

Customization
-------------
With the wide range of possibilities that data can be represented in and the many different ways to arrive at a visualization of such data, one key design goal of dantro is to be **easily customizable**, such that it can be adapted to the required use cases.

Subsequently, one central aim of dantro is to **not impose restrictions** on:

* types of data to work with,
* ways to load data,
* ways to process data,
* or ways to plot data.

Typically, when desiring to achieve this, a package may provide an interface that the projects that use it have to adhere to.
However, any specializations going beyond the provided interface are then typically outsourced to the projects that require these specializations.
This may result in needing to write (and re-write) a lot of code, which might not only be redundant but can also become hard to maintain.

In order to avoid this, :py:mod:`dantro` not only supplies an interface, but also provides the means to easily specialize the data structures in order to fit the different needs of certain use cases.
It already provides a set of specializations and implements them in an efficient way.
Extensions that are potentially valuable to a wider audience can be integrated into the package to avoid redundant reimplementations.


Enter: Mixin Classes.
^^^^^^^^^^^^^^^^^^^^^
This refers to the idea that functionality can be added to classes using `multiple inheritance <https://docs.python.org/3/tutorial/classes.html#multiple-inheritance>`_.
This concept is widely used in dantro to allow easy customization.

For example, if a :py:class:`~dantro.data_mngr.DataManager` is desired that needs a certain load functionality, this can be specified simply by *additionally* inheriting a certain mixin class, e.g. the :py:class:`~dantro.data_loaders.load_yaml.YamlLoaderMixin` for loading YAML files:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- philosophy_specializing
    :end-before:  ### End ---- philosophy_specializing
    :dedent: 4

In dantro, this is used by :py:mod:`~dantro.containers`, :py:mod:`~dantro.groups`, :py:mod:`~dantro.data_loaders` and even :py:mod:`~dantro.plot_creators`.
It allows adding functionality to classes in a granular fashion, customizing them for a particular use case.
At the same time, this approach makes it easy to retain a shared interface that allows storing all these heterogeneous objects in the same :ref:`data tree <phil_data_tree>`.

For more information on how to specialize dantro for use in your project, see :doc:`specializing`.


.. _phil_configurability:

Configurability
---------------
Another core concept of dantro is to make as many parameters as possible accessible via a consistent, hierarchical, dict-based interface.
Ideally, relevant parameters can be passed through from the highest modularization level down to the lowest one.

This approach is used throughout dantro but is most apparent in the :py:class:`~dantro.data_mngr.DataManager`, the :py:class:`~dantro.plot_mngr.PlotManager`, and the whole plotting interface.


Default Parameters
^^^^^^^^^^^^^^^^^^
High configurability may also come with a burden, namely the *need* to specify parameters, which can be difficult if their function is unclear.
To overcome this, dantro specifies sensible default parameters wherever possible, such that in the easiest case no additional configuration needs to be given.

Furthermore, it is often possible to specify default parameters, like a set of default load configurations or default plots, which can be used to appropriately customize the involved objects to the needs of a project.
The default parameters can then be updated with new values wherever necessary.

The main idea of having different sets of default parameters is, that **everything can be specified, but nothing need be.**
