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
The :py:class`~dantro.plot_mngr.PlotManager` does not actually carry out any plots. Its purpose is to handle the configuration of some :doc:`plot creator <plot_creators>` classes; those implement the actual plotting functionality.
This way, the plots can be configured in a consistent way, profiting from the shared interface and the already implemented functions, while keeping the flexibility of having multiple ways to create plots.

To create the plots, a set of plot configurations gets passed to the :py:class:`~dantro.plot_mngr.PlotManager` which then determines which plot creator it will need to instantiate.
It then passes the plot configuration on to the respective plot creator, which takes care of all the actual plotting work.

The main methods to interact with the :py:class:`~dantro.plot_mngr.PlotManager` are the following:

* :py:meth:`~dantro.plot_mngr.PlotManager.plot` expects the configuration for a single plot.
* :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg` expects a set of plot configurations and, for each configuration, creates the specified plots using :py:meth:`~dantro.plot_mngr.PlotManager.plot`.

This configuration-based approach makes the :py:class:`~dantro.plot_mngr.PlotManager` quite versatile and provides a set of features that the individual plot creators need not be aware of.


The Plot Configuration
^^^^^^^^^^^^^^^^^^^^^^
A set of plot configurations may look like this:

.. code-block:: yaml

    values_over_time:  # this will also be the final name of the plot (without extension)
      # Select the creator to use
      creator: external
      # NOTE: This has to be known to `PlotManager` under this name.
      #       It can also be set as default during `PlotManager` initialisation.

      # Specify the module to find the plot_function in
      module: .basic  # Uses the dantro-internal plot functions

      # Specify the name of the plot function to load from that module
      plot_func: lineplot

      # The data manager is passed to that function as first positional argument.
      # Also, the generated output path is passed as `out_path` keyword argument.

      # All further kwargs on this level are passed on to that function.
      # Specify how to get to the data in the data manager
      x: vectors/times
      y: vectors/values

      # Specify styling
      fmt: go-
      # ...

    my_fancy_plot:
      # Select the creator to use
      creator: external

      # This time, get the module from a file
      module_file: /path/to/my/fancy/plotting/script.py
      # NOTE Can also be a relative path, if `base_module_file_dir` was set

      # Get the plot function from that module
      plot_func: my_plot_func

      # All further kwargs on this level are passed on to that function.
      # ...

This will create two plots: ``values_over_time`` and ``my_fancy_plot``.
Both are using :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` (known to :py:class:`~dantro.plot_mngr.PlotManager` by its name, ``external``) and are loading certain functions to use for plotting.


Parameter sweeps in plot configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the configuration-based approach, it becomes possible to use **parameter sweeps** in the plot specification; the manager detects that it will need to create multiple plots and does so by repeatedly invoking the instantiated plot creator using the respective arguments for the respective point in the parameter space.

.. code-block:: yaml

    multiple_plots: !pspace
      creator: external
      module: .basic
      plot_func: lineplot

      # All further kwargs on this level are passed on to that function.
      x: vectors/times

      # Create multiple plots with different y-values
      y: !pdim
        default: vectors/values
        values:
          - vectors/values
          - vectors/more_values

This will create two *files*, one with ``values`` over ``times``, one with ``more_values`` over ``times``.
By defining further ``!pdim``\ s, the combination of those parameters are each leading to a plot.



Auto-Detection of a Plot Creator
--------------------------------
The :py:class:`~dantro.plot_mngr.PlotManager` has as class variable a dictionary of :py:const:`~dantro.plot_mngr.PlotManager.CREATORS`, which is a mapping of common name strings to plot creator types, i.e. :py:class:`~dantro.abc.AbstractPlotCreator`-derived classes.
Usually, the ``creator`` argument to the :py:class:`~dantro.plot_mngr.PlotManager`\'s :py:meth:`~dantro.plot_mngr.PlotManager.plot` function is used to extract the plot creator type from that dictionary and then initialize the object.

However, the plot manager also has a ``auto_detect_creator`` feature.
This boolean argument can be given both to :py:meth:`~dantro.plot_mngr.PlotManager.__init__` as well as to :py:meth:`~dantro.plot_mngr.PlotManager.plot` and it can also be part of the plot configuration passed to :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg`.

If set, the ``creator`` argument need no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is found out if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method as they see fit.
In unambiguous cases, the manager than uses the *single* candidate creator and continues plotting with that creator.


Auto-detection for :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`\ s
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived classes try auto-detection by checking if it can use the given plot configuration to resolve a plotting function.
Furthermore, it checks whether the plot function is marked with attributes that may specify which creator to use. The attributes that are looked at are, in this order:

* ``creator_type``: The type of the plot creator to use (or a parent type)
* ``creator_name``: The name of the plot creator *as registered* in the manager's :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` dict

To conveniently add these attributes to the plot function, the :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator can be used:

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
