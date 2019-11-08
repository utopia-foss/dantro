Philosophy and Design Concepts
==============================

:py:mod:`dantro` aims to be a rather general package, trying to not impose restrictions on:
* types of data to work with,
* the ways to load that data,
* the ways to process data,
* or the ways to plot data.

A common approach is to just use the package to strictly define a common interface and outsource all specializations to the projects that need to use it.
However, this can result in needing to write (and re-write!) a lot of code outside of this package, which can become hard to maintain.

In order to avoid this, :py:mod:`dantro` not only supplies an interface, but also provides classes that can be easily customized to fit the needs of a certain use case.

Enter: Mixin Classes.
---------------------
This refers to the idea that functionality can be added to classes using `multiple inheritance <https://docs.python.org/3/tutorial/classes.html#multiple-inheritance>`_.

For example, if a :py:class`~dantro.data_mngr.DataManager` is desired that needs a certain load functionality, this is specified simply by *additionally* inheriting a certain mixin class:

.. code-block:: python
    from dantro import DataManager
    from dantro.data_loaders import YamlLoaderMixin

    class MyDataManager(YamlLoaderMixin, DataManager):
        """My data manager can load YAML files."""
        pass  # Done here. Nothing else to do.

This concept is widely used in dantro; namely by :py:mod:`~dantro.containers`, :py:mod:`~dantro.groups`, :py:mod:`~dantro.data_loaders` and even :py:mod:`~dantro.plot_creators`.

