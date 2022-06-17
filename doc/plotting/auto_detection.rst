.. _plot_creator_auto_detection:

Plot Creator Auto Detection
===========================

.. note::

    This page describes an advanced feature.
    Make sure to have read all about :doc:`plot_manager` before diving into this.

Basic Idea
----------
This feature aims to reduce the arguments that need to be defined in a plot configuration or when defining a plot function.

The :py:class:`~dantro.plot_mngr.PlotManager` has as class variable a dictionary of :py:const:`~dantro.plot_mngr.PlotManager.CREATORS`, which is a mapping of common name strings to plot creator types, i.e. :py:class:`~dantro.abc.AbstractPlotCreator`-derived classes.
Usually, the ``creator`` argument to the :py:class:`~dantro.plot_mngr.PlotManager`\'s :py:meth:`~dantro.plot_mngr.PlotManager.plot` function is used to extract the plot creator type from that dictionary and then initialize the object.

However, the plot manager also has a ``auto_detect_creator`` feature.
This boolean argument can be given both to :py:meth:`~dantro.plot_mngr.PlotManager.__init__` as well as to :py:meth:`~dantro.plot_mngr.PlotManager.plot` and it can also be part of the plot configuration passed to :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg`.

If set, the ``creator`` argument needs no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is found out if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method as they see fit.
In unambiguous cases, the manager then uses the *single* candidate creator and continues plotting with that creator.


Specializations
---------------

:py:class:`~dantro.plot.creators.ext.ExternalPlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot.creators.ext.ExternalPlotCreator` and derived classes try auto-detection by checking if it can use the given plot configuration to resolve a plotting function.
Furthermore, it checks whether the plot function is marked with attributes that may specify which creator to use. The attributes that are looked at are, in this order:

* ``creator_type``: The type of the plot creator to use (or a parent type)
* ``creator_name``: The name of the plot creator *as registered* in the manager's :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` dict

To conveniently add these attributes to the plot function, the :py:func:`~dantro.plot.utils.is_plot_func.is_plot_func` decorator can be used:

.. code-block:: python

  from dantro.plot import is_plot_func, MultiversePlotCreator

  @is_plot_func(creator_type=MultiversePlotCreator)
  def my_mv_plot_func(dm: DataManager, *, out_path: str, mv_data, **kwargs):
      # ...

  # Alternatively: Specify only via the _name_ known to the PlotManager
  @is_plot_func(creator_name="universe")
  def my_uni_plot_func(dm: DataManager, *, out_path: str, uni, **kwargs):
      # ...

.. hint::

    When using the :doc:`data transformation framework <plot_data_selection>`, the signature of the plot functions is averse to the choice of a creator.
    This makes it possible to implement *generic* plotting functions, which can be used for all :py:class:`~dantro.plot.creators.ext.ExternalPlotCreator`\ -derived plot creators.

    In such cases, simply omit the ``creator_*`` argument to the decorator and specify the creator via the plot configuration.

.. note::

    Setting only the ``creator_name`` is recommended for scenarios where the import of the creator type is not desired.
    In other scenarios, it's best to use ``creator_type``
