.. _plot_manager:

The :py:class:`~dantro.plot_mngr.PlotManager`
=============================================

The :py:class:`~dantro.plot_mngr.PlotManager` orchestrates the whole plotting framework.
This document describes what it is and how it works together with the :doc:`plot_creators` to generate plots.

.. contents::
   :local:
   :depth: 3

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
      # Also, the generated output path is passed as ``out_path`` keyword argument.

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
      # NOTE Can also be a relative path if ``base_module_file_dir`` was set

      # Get the plot function from that module
      plot_func: my_plot_func

      # All further kwargs on this level are passed on to that function.
      # ...

This will create two plots: ``values_over_time`` and ``my_fancy_plot``.
Both are using :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` (known to :py:class:`~dantro.plot_mngr.PlotManager` by its name, ``external``) and are loading certain functions to use for plotting.

.. hint::

    Plot configuration entries starting with an underscore or dot are ignored:

    .. code-block:: yaml

        ---
        _foobar:        # This entry is ignored
          some_defaults: &defaults
            foo: bar

        .barbaz:        # This entry is also ignored
          more_defaults: &more_defaults
            spam: fish

        my_plot:        # -> creates my_plot
          <<: [*defaults, *more_defaults]
          # ...

        my_other_plot:  # -> creates my_other_plot
          # ...

    This can be useful when desiring to define YAML anchors that are used in the actual plot configuration entries, e.g. for specifying defaults.

.. _psweep_plot_cfg:

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
Instead, a plot configuration can be successively assembled from separate parts.

To use this feature, add the ``based_on`` key to your plot configuration and specify the name or names of other plot configurations you want to let this plot be based on.
We call those plot configurations *base configurations* to distinguish them from the configuration the ``based_on`` key is used in.

These base configurations are then looked up in previously specified plot configurations, so-called *base plot configuration pools*.
They are passed to :py:class:`~dantro.plot_mngr.PlotManager` during initialization using the ``base_cfg_pools`` argument.

For example, let's say we have a base configuration pool that specifies a lineplot with a certain style:

.. code-block:: yaml

    # Base configuration pool, registered with PlotManager
    ---
    my_gg_lineplot:
      creator: external
      module: basic
      plot_func: lineplot

      style:
        base_style: ggplot

To avoid repetition in the actual definition of a plot, the ``based_on`` key can then be used:

.. code-block:: yaml

    # Plot configuration, e.g. as passed to PlotManager.plot()
    ---
    values_over_time:
      based_on: my_gg_lineplot

      x: vectors/times
      y: vectors/values

When ``based_on: my_gg_lineplot`` is given, *first* the configuration for ``my_gg_lineplot`` is loaded.
It is then recursively updated with the other keys, here ``x`` and ``y``, resulting in:

.. code-block:: yaml

    # Plot configuration with ``based_on`` entries fully resolved
    ---
    values_over_time:
      creator: external
      module: basic
      plot_func: lineplot

      style:
        base_style: ggplot

      x: vectors/times
      y: vectors/values

.. note::

    **Reminder:** *Recursively* updating means that all levels of the configuration hierarchy can be updated.
    This happens by traversing along with all mapping-like parts of the configuration and updating their keys.


Multiple inheritance
""""""""""""""""""""
When providing a sequence, e.g. ``based_on: [foo, bar, baz]``, the first configuration is used as the base and is subsequently recursively updated with those that follow, finally applying the updates from the plot configuration where ``based_on`` was defined in.
If there are conflicting keys, those from a *later* update take precedence over those from a previous base configuration.

This can be used to subsequently build a configuration from several parts.
With the example above, we could also do the following:

.. code-block:: yaml

    ---
    # Base plot configuration, specifying importable configuration chunks
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

    The names used in the examples for the plot configurations can be chosen arbitrarily (as long as they are valid plot names).

    However, we propose to **use a consistent naming scheme** that describes the purpose of the respective entries and broadly categorizes them.
    In the example above, the ``.plot`` and ``.style`` prefixes denote the effect of the configuration.
    This not only makes the plot definition more readable, but also helps to avoid conflicts with duplicate base configuration names — something that becomes more relevant with rising size of configuration pools.


Lookup rules
""""""""""""
In the examples above, only a single base configuration pool was defined.
However, lookups of base configurations are not restricted to a single pool.
This section provides more details on how it is determined which base configurations is used to assemble a plot configuration.

First of all: *what would multiple pools be good for*?
The answer is simple: it allows to include plot configurations into the pool that are spread out over multiple files, e.g. because they are part of different projects or in cases one has no control over them.
Instead of copying the content into one place, it is safest to make them available as they are.

Let's assume we have the following two base configuration pools registered, with ``---`` seperating the different pools.

.. code-block:: yaml

    ---
    # Style configuration
    .style.default:
      style:
        base_style: ggplot

    .style.poster:
      based_on: .style.default
      style:
        base_style: seaborn-poster
        lines.linewidth: 3
        lines.markersize: 10

    ---
    # Plot function definitions
    .plot.defaults:
      based_on: .style.default
      creator: external
      module: generic

    .plot.errorbars:
      based_on: .plot.defaults
      plot_func: errorbars

    .plot.facet_grid:
      based_on: .plot.defaults
      plot_func: facet_grid

Let's give this a closer look: Already *within* the pool, it is possible to use ``based_on``:

* In ``.style.poster``, the ``.style.default`` from the *same* pool is used.
* In ``.plot.defaults``, the ``.style.default`` is specified as well.
* The other ``.plot…`` entries base themselves on ``.plot.defaults``.

In the last case, looking up ``.plot.defaults`` will lead to its *own* ``based_on`` entry needing to be evaluated — and this is exactly what happens:
the resolver recursively inspects the looked up configurations and, if there are any ``based_on`` entries there, looks them up as well.

.. note::

    Lookups are only possible **within the same or a previous pool**.

    In the example above, the ``.plot…`` entries may look up the ``.style…`` entries but **not the other way around**.
    For more details on the lookup rules, see :py:func:`~dantro.plot._cfg.resolve_based_on`.

.. hint::

    **Wait, does this not allow to create loops?!**

    Yes, it might! However, the resolver will keep track of the base configurations it already visited and can thus detect when a dependency loop is created.
    In such a case, it will inform you about it and avoid running into an infinite recursion.

Ok, how would we assemble such a plot configuration now?
That's easiest to see with an example:

.. code-block:: yaml

    ---
    # Actual plot configuration

    my_default_plot:
      based_on: .plot.facet_grid

      select: # ... select some data for plotting ...

      transform: # ... and transform it ...

      # Visualize as heatmap
      kind: pcolormesh
      x: time
      y: temperature

    my_poster_plot:
      based_on:
        - my_default_plot
        - .style.advanced

      # Use a lineplot instead of the heatmap
      kind: line
      y: ~
      hue: temperature

To conclude, this feature allows to assemble plot configurations from different files or configuration hierarchies, always allowing to update recursively (unlike YAML inheritance).
This reduces the need for copying configurations into multiple places.


Features
--------

.. _plot_mngr_skipping_plots:

Skipping Plots
^^^^^^^^^^^^^^
To skip a plot, raise a :py:class:`dantro.exceptions.SkipPlot` exception anywhere in your plot function or the plot creator.

.. hint::

    When :ref:`using the data transformation framework for plot data selection <plot_creator_dag>`, you can invoke the ``raise_SkipPlot`` data operation to conditionally skip a plot with whatever logic you desire.
    See :py:func:`~dantro.data_ops.ctrl_ops.raise_SkipPlot` for more information.

    The easiest implementation is via the ``fallback`` of a failing operation, see :ref:`dag_error_handling`:

    .. code-block:: yaml

        my_plot:
          # ...
          dag_options:
            # Define a tag which includes a call to the raise_SkipPlot operation
            # (Use a private tag, such that it is not automatically evaluated)
            define:
              _skip_plot:
                - raise_SkipPlot

          transform:
            # ...
            # If the following operation fails, want to skip the current plot
            - some_operation: [foo, bar]
              allow_failure: silent
              fallback: !dag_tag _skip_plot

Additionally, plot creators can supply built-in plot configuration arguments that allow to skip a plot under certain conditions.
Currently, this is only done by the :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`, see :ref:`mv_plot_skipping`.

.. note::

    *For developers:*
    The :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator` provides the :py:meth:`~dantro.plot_creators.pcr_base.BasePlotCreator._check_skipping` method, which can be overwritten by plot creators to implement this behaviour.


What happens when a plot is skipped?
""""""""""""""""""""""""""""""""""""
Plotting stops immediately and returns control to the plot manager, which then informs the user about this via a log message.
For :ref:`parameter sweep plot configurations <psweep_plot_cfg>`, skipping is evaluated individually for each point in the plot configuration parameter space.

A few remarks regarding side effects (e.g., directories being created for plots that are later on decided to be skipped):

* Skipping will have fewer side effects if it is triggered as early as possible.
* If skipping is triggered by a built-in plot creator method, it is taken care that this happens *before* directory creation.
* If :py:class:`dantro.exceptions.SkipPlot` is raised at a later point, this *might* lead to intermediate directories having been created.

.. note::

    The plot configuration will **not** be saved for skipped plots.

    There is one exception though: if a :ref:`parameter sweep plot configuration <psweep_plot_cfg>` is being used and at least one of the plots of that sweep is *not* skipped, the corresponding plot configuration metadata will be stored alongside the plot output.


Further Reading
---------------

* :doc:`plot_creators`
* :doc:`faq`
* :doc:`plot_cfg_ref`
* :doc:`plot_data_selection`
* :doc:`auto_detection`
