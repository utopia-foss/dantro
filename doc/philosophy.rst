Philosophy and Design Concepts
==============================

:py:mod:`dantro` aims to be a general package for handling, processing, and visualizing hierarchically organized data using a uniform interface.
With the wide range of possibilities that data can be represented in and the many different ways to arrive at a visualization of such data, one key design goal of dantro is to be **easily customizable** to whatever a certain project needs.

Subsequently, one central aim of dantro is to **not impose restrictions** on:

* types of data to work with,
* ways to load data,
* ways to process data,
* or ways to plot data.

Typically, when desiring to achieve this, a package may provide an interface that the projects that use it have to adhere to.
However, any specializations going beyond the provided interface are then typically outsourced to the projects that require these specializations.
This may result in needing to write (and re-write!) a lot of code outside of this package, which might not only be redundant, but can also become hard to maintain.

In order to avoid this, :py:mod:`dantro` not only supplies an interface, but also provides the means to easily specialize the data structures in order to fit the needs of a certain use case.
It already provides a set of common specializations and implements them in an efficient and tested way.
Extensions that are potentially valuable to a wider audience can be integrated into the package to avoid redundant reimplementations.


Enter: Mixin Classes.
---------------------
This refers to the idea that functionality can be added to classes using `multiple inheritance <https://docs.python.org/3/tutorial/classes.html#multiple-inheritance>`_.

For example, if a :py:class:`~dantro.data_mngr.DataManager` is desired that needs a certain load functionality, this can be specified simply by *additionally* inheriting a certain mixin class, e.g. the :py:class:`~dantro.data_loaders.load_yaml.YamlLoaderMixin` for loading YAML files:

.. literalinclude:: ../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- philosophy_specializing
    :end-before:  ### End ---- philosophy_specializing
    :dedent: 4

This concept is widely used in dantro, namely by :py:mod:`~dantro.containers`, :py:mod:`~dantro.groups`, :py:mod:`~dantro.data_loaders` and even :py:mod:`~dantro.plot_creators`.
For more information on how to specialize dantro for use in your project, see :doc:`specializing`.
