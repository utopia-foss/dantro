The :py:class:`~dantro.plot_mngr.PlotManager`
=============================================

The :py:class:`~dantro.plot_mngr.PlotManager` orchestrates the whole plotting framework.
This document describes what it is and how it works together with the :doc:`plot_creators` to generate plots.


.. contents::
   :local:
   :depth: 2

----

Overview
--------
The :py:class:`~dantro.plot_mngr.PlotManager` manages the creation of plots.
So far, so obvious.

The idea of the :py:class:`~dantro.plot_mngr.PlotManager` is that it is aware of all available data and then gets instructed to create a set of plots from this data.
To create the actual plots, a set of plot configurations gets passed to it.
The :py:class:`~dantro.plot_mngr.PlotManager` then determines which plot creator it will need to instantiate and passes the plot configuration on to that plot creator, which takes care of all the actual plotting work.

The main methods to interact with are the following:

* :py:meth:`~dantro.plot_mngr.PlotManager.plot` expects the configuration for a single plot.
* :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg` expects a set of plot configurations and, for each configuration, creates the specified plots using :py:meth:`~dantro.plot_mngr.PlotManager.plot`.

Such a set of plot configurations may look like this:

.. code-block:: yaml

    # A set of plot configurations
    ---
    my_first_plot:
      creator: my_creator

      # ... plotting parameters

    my_second_plot:
      creator: another_creator
      # ... plotting parameters

This configuration-based approach makes the :py:class:`~dantro.plot_mngr.PlotManager` quite versatile and provides a set of features that the individual plot creators need not be aware of.

For example, it becomes possible to use **parameter sweeps** in the plot specification; the manager detects that it will need to create multiple plots and does so by repeatedly invoking the instantiated plot creator using the respective arguments for one point in that parameter space.

.. code-block:: yaml

    # A plot configuration using parameter sweeps
    ---
    my_plot: !pspace
      creator: my_creator

      some_parameter: !pdim
        default: 0
        values: [1, 2, 3]

      another_parameter: !pdim
        default: 42
        values: [23, 42]

The above configuration will create a directory ``my_plot`` and in there, it will create six plots for all possible parameter combinations.


Auto-Detection of a Plot Creator
--------------------------------
The :py:class:`~dantro.plot_mngr.PlotManager` has as class variable a dictionary of :py:const:`~dantro.plot_mngr.PlotManager.CREATORS`, which is a mapping of common name strings to plot creator types, i.e. :py:class:`~dantro.abc.AbstractPlotCreator`-derived classes.
Usually, the ``creator`` argument to the :py:class:`~dantro.plot_mngr.PlotManager`\'s :py:meth:`~dantro.plot_mngr.PlotManager.plot` function is used to extract the plot creator type from that dictionary and then initialize the object.

However, the plot manager also has a ``auto_detect_creator`` feature.
This boolean argument can be given both to :py:meth:`~dantro.plot_mngr.PlotManager.__init__` as well as to :py:meth:`~dantro.plot_mngr.PlotManager.plot` and it can also be part of the plot configuration passed to :py:meth:`~dantro.plot_mngr.PlotManager.plots_from_cfg`.

If set, the ``creator`` argument need no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is found out if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method as they see fit.
In unambiguous cases, the manager than uses the *single* candidate creator and continues plotting with that creator.


Auto-detection for :py:class:`~dantro.plot_creators.ExternalPlotCreator`\ s
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived classes try auto-detection by checking if it can use the given plot configuration to resolve a plotting function.
Furthermore, it checks whether the plot function is marked with attributes that may specify which creator to use. The attributes that are looked at are, in this order:

* ``creator_type``: The type of the plot creator to use (or a parent type)
* ``creator_name``: The name of the plot creator *as registered* in the manager's :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` dict

To conveniently add these attributes to the plot function, the :py:func:`~dantro.plot_creators.is_plot_func` decorator can be used:

.. code-block:: python

  from dantro.plot_creators import is_plot_func

  # Specify directly with the plot creator type
  from dantro.plot_creators import MultiversePlotCreator

  @is_plot_func(creator_type=MultiversePlotCreator)
  def my_mv_plot_func(dm: DataManager, *, out_path: str, mv_data, **kwargs):
      # ...

  # Alternatively: Specify only via the _name_ known to the PlotManager
  @is_plot_func(creator_name="universe")
  def my_uni_plot_func(dm: DataManager, *, out_path: str, uni, **kwargs):
      # ...

.. hint::

    When using the :doc:`data transformation framework <plot_data_selection>`, the signature of the plot functions is averse to the choice of a creator.
    This makes it possible to implement *generic* plotting functions, which can be used for all :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`\ -derived plot creators.

    In such cases, simply omit the ``creator_*`` argument to the decorator and specify the creator via the plot configuration.

.. note::
    
    Setting only the ``creator_name`` is recommended for scenarios where the import of the creator type is not desired.
    In other scenarios, it's best to use ``creator_type``

.. deprecated:: 0.10

    If no plot function attributes are given, there is still another way to auto-detect the desired plot creator: inspecting the plot function signature.

    This works, because derived creators *might* require a different plot function signature.
    For example, :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`) additionally passes ``mv_data`` as keyword-only argument.
    However, this approach can lead to ambiguous results and thus failing auto-detection. For those cases, it makes sense to specify plot function attributes via the decorator.

    Due to the ambiguity and the many different ways in which a plot function can be defined, this feature will be removed in v0.11.
