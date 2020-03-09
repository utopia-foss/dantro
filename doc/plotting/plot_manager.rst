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
The :py:class`~dantro.plot_mngr.PlotManager` does not carry out any plots. Its purpose is to handle the *configuration* of some :doc:`plot creator <plot_creators>` classes; those implement the actual plotting functionality.
This way, the plots can be configured consistently, profiting from the shared interface and the already implemented functions, while keeping the flexibility of having multiple ways to create plots.

To create the plots, a set of plot configurations gets passed to the :py:class:`~dantro.plot_mngr.PlotManager` which then determines which plot creator it will need to instantiate.
It then passes the plot configuration on to the respective plot creator, which takes care of all the actual plotting work.

The main methods to interact with the :py:class:`~dantro.plot_mngr.PlotManager` are the following:

* :py:meth:`~dantro.plot_mngr.PlotManager.plot` expects the configuration for a single plot.
* :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg` expects a set of plot configurations and, for each configuration, creates the specified plots using :py:meth:`~dantro.plot_mngr.PlotManager.plot`.

This configuration-based approach makes the :py:class:`~dantro.plot_mngr.PlotManager` quite versatile and provides a set of features that the individual plot creators need *not* be aware of.


Nomenclature
^^^^^^^^^^^^
To repeat, this is the basic vocabulary to understand the plotting framework and its structure:

* The :ref:`plot configuration <plot_cfg_overview>` contains all the parameters required to make one or multiple plots.
* The :ref:`plot creators <plot_creators>` create the actual plots. Given some plot configuration, they produce the plots as output.
* The :py:class:`~dantro.plot_mngr.PlotManager` orchestrates the plotting procedure by feeding the relevant plot configuration to a specific plot creator.

This page focusses on the capabilities of the :py:class:`~dantro.plot_mngr.PlotManager` itself.
For creator-specific capabilities, follow the corresponding links.


.. _plot_cfg_overview:

The Plot Configuration
----------------------
A set of plot configurations may look like this:

.. code-block:: yaml

    values_over_time:  # this will also be the final name of the plot (without extension)
      # Select the creator to use
      creator: external
      # NOTE: This has to be known to PlotManager under this name.
      #       It can also be set as default during PlotManager initialization.

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
      # NOTE Can also be a relative path if `base_module_file_dir` was set

      # Get the plot function from that module
      plot_func: my_plot_func

      # All further kwargs on this level are passed on to that function.
      # ...

This will create two plots: ``values_over_time`` and ``my_fancy_plot``.
Both are using :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` (known to :py:class:`~dantro.plot_mngr.PlotManager` by its name, ``external``) and are loading certain functions to use for plotting.

.. hint::

    Plot configuration entries starting with an underscore are ignored:

    .. code-block:: yaml

        ---
        _foobar:        # This entry is ignored
          # ...

        my_plot:        # -> creates my_plot
          # ...

        my_other_plot:  # -> creates my_other_plot
          # ...

    This can be useful when desiring to define YAML anchors that are used in the actual plot configuration entries.



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


.. _plot_cfg_inheritance:

Plot Configuration Inheritance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
New plot configurations can be based on existing ones.
This makes it very easy to define various plot functions without copy-pasting the plot configurations.

To do so, add the ``based_on`` key to your plot configuration.
As arguments, you can provide either a string or a sequence of strings, where the strings have to refer to names of so-called *base plot configuration entries*, or short *base configurations* (in contrast to regular plot configurations).
These are configuration entries that were passed to the :py:class:`~dantro.plot_mngr.PlotManager` during initialization using the ``base_cfg`` and ``update_base_cfg`` arguments.

For example, let's say there is the following base plot configuration, where you supply the parameters for a lineplot with a certain style:

.. code-block:: yaml

    my_gg_lineplot:
      creator: external
      module: basic
      plot_func: lineplot

      style:
        base_style: ggplot

To avoid repetition in the actual definition of a plot, the ``based_on`` key can then be used:

.. code-block:: yaml

  values_over_time:
    based_on: my_gg_lineplot

    x: vectors/times
    y: vectors/values

When ``based_on: my_gg_lineplot`` is given, first the configuration for ``my_gg_lineplot`` is loaded.
It is then recursively updated with the other keys, here ``x`` and ``y``.

.. note::

    **Reminder:** *Recursively* updating means that all levels of the configuration hierarchy can be updated.
    This happens by traversing along with all mapping-like parts of the configuration and updating their keys.

When providing a sequence, e.g. ``based_on: [foo, bar, baz]``, the first configuration is used as the base and is subsequently recursively updated with those that follow.
This can be used to subsequently build a configuration from several parts.
With the example above, we could also do the following:

.. code-block:: yaml

    # Base plot configuration, specifying
    .plot.line:
      creator: external
      module: basic
      plot_func: lineplot

    .style.default:
      style:
        base_style: ggplot

    ---
    # Actual plot configuration

    values_over_time:
      based_on: [.style.default, .plot.line]

      x: vectors/times
      y: vectors/values

This multiple inheritance approach has the following advantages:

* Allows defining defaults in a central place, using it later on
* Allows modularization of different aspects of the plot configuration
* Reduces repetition, e.g. of style configurations
* Retains full flexibility, as all parameters can be overwritten in the plot configuration

.. hint::

    When several base plot configurations are specified, we propose to use a naming scheme that describes the purpose of the base configuration entries and broadly categorizes the entry.
    In the example above, the `.plot` and `.style` prefixes denote the effect of the configuration.



Auto-Detection of a Plot Creator
--------------------------------
The :py:class:`~dantro.plot_mngr.PlotManager` has as class variable a dictionary of :py:const:`~dantro.plot_mngr.PlotManager.CREATORS`, which is a mapping of common name strings to plot creator types, i.e. :py:class:`~dantro.abc.AbstractPlotCreator`-derived classes.
Usually, the ``creator`` argument to the :py:class:`~dantro.plot_mngr.PlotManager`\'s :py:meth:`~dantro.plot_mngr.PlotManager.plot` function is used to extract the plot creator type from that dictionary and then initialize the object.

However, the plot manager also has a ``auto_detect_creator`` feature.
This boolean argument can be given both to :py:meth:`~dantro.plot_mngr.PlotManager.__init__` as well as to :py:meth:`~dantro.plot_mngr.PlotManager.plot` and it can also be part of the plot configuration passed to :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg`.

If set, the ``creator`` argument needs no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is found out if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method as they see fit.
In unambiguous cases, the manager then uses the *single* candidate creator and continues plotting with that creator.


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
