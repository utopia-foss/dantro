.. default-domain:: dantro.plot_mngr

.. _plot_manager:

The :py:class:`.PlotManager`
============================

The :py:class:`.PlotManager` orchestrates the whole plotting framework.
This document describes what it is and how it works together with the :doc:`plot_creators` to generate plots.

.. contents::
   :local:
   :depth: 3


**Further reading:**

* :doc:`plot_creators`
* :doc:`plot_data_selection`
* :doc:`plot_cfg_ref`
* :doc:`faq`

----

Overview
--------
The :py:class:`.PlotManager` manages the creation of plots.
So far, so obvious.

The idea of the :py:class:`.PlotManager` is that it is aware of all available data and then gets instructed to create a set of plots from this data.
The :py:class`.PlotManager` does not carry out any plots.
Its purpose is to handle the *configuration* of some :doc:`plot creator <plot_creators>` classes; those implement the actual plotting functionality.
This way, the plots can be configured consistently, profiting from the shared interface and the already implemented functions, while keeping the flexibility of having multiple ways to create plots.

To create a plots, a so-called plot configuration gets passed to the :py:class:`.PlotManager`.
From the plot configuration, the manager determines which so-called plot function is desired and which plot creator is to be used.
After retrieving the plot function and instantiating the creator instance, the remaining plot configuration is passed to the plot creator, which is then responsible to create the actual plot output.

The main methods to interact with the :py:class:`.PlotManager` are the following:

* :py:meth:`.PlotManager.plot` expects the configuration for a single plot.
* :py:meth:`.PlotManager.plot_from_cfg` expects a set of plot configurations and, for each configuration, creates the specified plots using :py:meth:`.PlotManager.plot`.

This configuration-based approach makes the :py:class:`.PlotManager` quite versatile and provides a set of features that the individual plot creators need *not* be aware of.


Nomenclature
^^^^^^^^^^^^
To repeat, this is the basic vocabulary to understand the plotting framework and its structure:

* The :ref:`plot configuration <plot_cfg_overview>` contains all the parameters required to make one or multiple plots.
* The :ref:`plot creators <plot_creators>` create the actual plots. Given some plot configuration, they produce the plots as output.
* The :ref:`plot function <plot_func>` (or plotting function) is a callable that receives the plot data and generates the output; it is retrieved by the plot manager but invoked by the creator.
* The :py:class:`.PlotManager` orchestrates the plotting procedure by feeding the relevant plot configuration to a specific plot creator.

This page focusses on the capabilities of the :py:class:`.PlotManager` itself.
For creator-specific capabilities, follow the corresponding links.


.. _plot_cfg_overview:

The Plot Configuration
----------------------
A set of plot configurations may look like this:

.. code-block:: yaml

    values_over_time:  # this will also be the final name of the plot (without extension)
      # Select the creator to use
      creator: pyplot
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
      creator: pyplot

      # This time, get the module from a file
      module_file: /path/to/my/fancy/plotting/script.py
      # NOTE Can also be a relative path if ``base_module_file_dir`` was set

      # Get the plot function from that module
      plot_func: my_plot_func

      # All further kwargs on this level are passed on to that function.
      # ...

This will create two plots: ``values_over_time`` and ``my_fancy_plot``.
Both are using :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` (known to :py:class:`.PlotManager` by its name, ``pyplot``) and are loading certain functions to use for plotting.

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

        my/other/plot:  # -> creates my/other/plot
          # ...

    This can be useful when desiring to define YAML anchors that are used in the actual plot configuration entries, e.g. for specifying defaults.

.. _psweep_plot_cfg:

Parameter sweeps in plot configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the configuration-based approach, it becomes possible to use **parameter sweeps** in the plot specification; the manager detects that it will need to create multiple plots and does so by repeatedly invoking the instantiated plot creator using the respective arguments for the respective point in the parameter space.

.. code-block:: yaml

    multiple_plots: !pspace
      creator: pyplot
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
They are passed to :py:class:`.PlotManager` during initialization using the ``base_cfg_pools`` argument.

For example, let's say we have a base configuration pool that specifies a lineplot with a certain style:

.. code-block:: yaml

    # Base configuration pool, registered with PlotManager
    ---
    my_gg_lineplot:
      creator: pyplot
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
      creator: pyplot
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
      creator: pyplot
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
      creator: pyplot
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








.. _plot_func:

The Plot Function
-----------------
The plot function is the place where selected data and configuration arguments come together to generate the plot output.
The :py:class:`.PlotManager` takes care of retrieving the plotting function, and a :ref:`plot creator <plot_creators>` takes care of invoking it.
While these aspects are taken care of, the function itself still has to be implemented (and :ref:`communicated <plot_func_specification>`) to the plotting framework.

In short, a plot function can be something like this:

.. testcode::

    from dantro.plot import is_plot_func

    @is_plot_func(use_dag=True, required_dag_tags=("x", "y"))
    def my_plot(*, data: dict, out_path: str, **plot_kwargs):
        """A plot function using the data transformation framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            out_path: Where to save the plot output.
            **plot_kwargs: Further plotting arguments
        """
        x = data["x"]
        y = data["y"]

        # Do something with the data
        # ...

        # Save the plot at `out_path`
        # ...


For examples of how to then :ref:`specify <plot_func_specification>` that function via the plot configuration and details on how to :ref:`implement <plot_func_implement>` it, see the respective sections.



.. _plot_func_specification:

Plot Function Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's assume we have a plotting function defined somewhere and want to communicate to the :py:class:`.PlotManager` that this function is responsible for creating the plot output.

For the moment, the exact definition of the function is irrelevant.
You can read more about it :ref:`below <plot_func_implement>`.

.. _plot_func_import_from_module:

Importing a plotting function from a module
"""""""""""""""""""""""""""""""""""""""""""
To do this, the ``module`` and ``plot_func`` entries are required.
The following example shows a plot that uses a plot function from a package called ``utopya.eval.plots`` and another plot that uses some (importable) package from which the module and the plot function are imported:

.. code-block:: yaml

   ---
   my_plot:
     # Import some module from utopya.plot_funcs (note the leading dot)
     module: .distribution

     # Use the function with the following name from that module
     plot_func: my_plot_func

     # ... all other arguments

   my_other_plot:
     # Import a module from any installed package
     module: my_installed_plotting_package.some_module
     plot_func: my_plot_func

     # ... all other arguments



.. _plot_func_import_from_file:

Importing a plotting function from a file
"""""""""""""""""""""""""""""""""""""""""
There might be situations where you want or need to implement a plot function decoupled from all the existing code and without bothering about importability (which may require setting up a package, installation routine, etc).

This can be achieved by specifying the ``module_file`` key instead of the ``module`` key in the plot configuration.
That python module is then loaded from file and the ``plot_func`` key is used to retrieve the plotting function:

.. code-block:: yaml

   ---
   my_plot:
     # Load the following file as a python module
     module_file: ~/path/to/my/python/script.py

     # Use the function with the following name from that module
     plot_func: my_plot_func

     # ... all other arguments (as usual)

.. note::

    For those interested, the specification is interpreted by the :py:class:`~dantro.plot.utils.plot_func.PlotFuncResolver` class, which then takes care of resolving the correct plot function.
    This class can also be specialized; the :py:class:`.PlotManager` simply uses the class defined in its :py:attr:`.PLOT_FUNC_RESOLVER` class variable.




.. _plot_func_implement:

Implementing Plot Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Below, you will learn how to implement a plot function.

A plot function is basically any Python function that adheres to a compatible signature.

.. note::

    Depending on the chosen creator, the signature may vary.
    For instance, the :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` adds a number of additional features such that the plot function may need to accept additional arguments (like ``hlpr``); see :ref:`here <pyplot_plot_func>` for more information.


.. _is_plot_func_decorator:

The :py:class:`~dantro.plot.utils.plot_func.is_plot_func` decorator
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
When defining a plot function, we recommend using this decorator.
It takes care of providing essential information to the :py:class:`.PlotManager` and makes it easy to configure those parameters relevant for the plot function.

As an example, to specify which creator can be used for the plot function, the ``creator`` argument can be set right there aside the plot function definition.
To control the whether the plot creator should use the :ref:`data transformation framework <plot_creator_dag>`, the ``use_dag`` flag can be set and the ``required_dag_tags`` argument can specify which data tags the plot function expects.

For the above reasons, the :ref:`best way <plot_func_signature>` to implement a plot function is by using the :py:class:`~dantro.plot.utils.plot_func.is_plot_func` decorator.

The decorator also provides the following arguments that affect DAG usage:

- ``use_dag``: to enable or disable DAG usage. Disabled by default.
- ``required_dag_tags``: can be used to specify which tags are expected by the plot function; if these are not defined or not computed, an error will be raised.
- ``compute_only_required_dag_tags``: if the plot function defines required tags and ``compute_only is None``, the ``compute_only`` argument will be set such that only ``required_dag_tags`` are computed.
- ``pass_dag_object_along``: passes the :py:class:`~dantro.dag.TransformationDAG` object to the plot function as ``dag`` keyword argument.
- ``unpack_dag_results``: instead of passing the results as the ``data`` keyword argument, it unpacks the results dictionary, such that the tags can be specified directly in the plot function signature.
  Note that this puts some restrictions on tag names, prohibiting some characters as well as requiring that plot configuration parameters do not collide with the DAG results.
  This feature is best used in combination with ``required_dag_tags`` and ``compute_only_required_dag_tags`` enabled (which is the default).

Decorator usage puts all the relevant arguments for using the DAG framework into one place: the definition of the plot function.


.. _plot_func_signature:

Recommended plot function signature
"""""""""""""""""""""""""""""""""""
The **recommended way of implementing a plot function** sets the plot function up for use of the :ref:`data transformation framework <plot_creator_dag>` of the :py:class:`.BasePlotCreator` (and derived classes).
In such a case, the data selection is taken care of by the creator and then simply passed to the plot function, allowing to control data selection right from the plot configuration.

Let's say that we want to implement a plot function that requires some ``x`` and ``y`` data selected from the data tree.
In the definition of the plot function we can use the :ref:`decorator <is_plot_func_decorator>` to specify that these tags are required; the framework will then make sure that these results are computed.

An implementation then looks like this:

.. testcode::

    from dantro.plot import is_plot_func

    @is_plot_func(use_dag=True, required_dag_tags=("x", "y"))
    def my_plot(*, data: dict, out_path: str, **plot_kwargs):
        """A plot function using the data transformation framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            out_path: Where to save the plot output.
            **plot_kwargs: Further plotting arguments
        """
        x = data["x"]
        y = data["y"]

        # Do something with the data
        # ...

        # Save the plot at `out_path`
        # ...

The corresponding plot configuration could look like this:

.. code-block:: yaml

    my_plot:
      creator: base

      # Select the plot function
      # ...

      # Select data
      select:
        x: data/MyModel/some/path/foo
        y:
          path: data/MyModel/some/path/bar
          transform:
            - .mean
            - increment

      # ... further arguments

For more detail on the data selection syntax, see :ref:`plot_creator_dag`.

.. note::

    Derived plot creators may require a slightly different signature, possibly containing additional arguments depending on the enabled feature set.
    While this signature is mostly universal across creators, make sure to refer to your desired :ref:`creator <plot_creators>` for details.

    For instance, the :ref:`the PyPlotCreator <pyplot_func_recommended>` would require the plot function to accept an additional argument ``hlpr``.



.. _plot_func_without_dag:

Plot function without data transformation framework
"""""""""""""""""""""""""""""""""""""""""""""""""""
To not use the data transformation framework, simply omit the ``use_dag`` flag or set it to ``False`` in the decorator or the plot configuration.
When not using the transformation framework, the ``creator_type`` should be specified, thus making the plot function bound to one type of creator.

.. testcode::

    from dantro import DataManager
    from dantro.plot import is_plot_func, BasePlotCreator

    @is_plot_func(creator_type=BasePlotCreator)
    def my_plot(*, out_path: str, dm: DataManager, **additional_plot_kwargs):
        """A simple plot function.

        Args:
            out_path (str): The path to store the plot output at.
            dm (dantro.data_mngr.DataManager): The loaded data tree.
            **additional_kwargs: Anything else from the plot config.
        """
        # Select some data ...
        data = dm["foo/bar"]

        # Create the plot
        # ...

        # Save the plot
        # ...

.. note::

    The ``dm`` argument is only provided when *not* using the DAG framework.


.. _plot_func_bare_signature:

Plot function the bare basics
"""""""""""""""""""""""""""""
There is an even more basic way of defining a plot function, leaving out the :py:func:`~dantro.plot.utils.plot_func.is_plot_func` decorator altogether:

.. testcode::

    from dantro import DataManager

    def my_bare_basics_plot(
        dm: DataManager, *, out_path: str, **additional_kwargs
    ):
        """Bare-basics signature required by the BasePlotCreator.

        Args:
            dm: The DataManager object that contains all loaded data.
            out_path: The generated path at which this plot should be saved
            **additional_kwargs: Anything else from the plot config.
        """
        # Select the data
        data = dm["some/data/to/plot"]

        # Generate the plot
        # ...

        # Store the plot
        # ...

.. note::

    When using the bare basics version, you need to set the ``creator`` argument in the :ref:`plot configuration <plot_cfg_overview>` in order for the :py:class:`.PlotManager` to find the desired creator.

.. warning::

    This way of specifying plot functions is mainly retained for reasons of backwards-compatibility.
    If you can, avoid this form of plot function definition and use the :ref:`recommended signature instead <plot_func_signature>`.








.. _plot_mngr_features:

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
Currently, this is only done by the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`, see :ref:`mv_plot_skipping`.

.. note::

    *For developers:*
    The :py:class:`~dantro.plot.creators.base.BasePlotCreator` provides the :py:meth:`~dantro.plot.creators.base.BasePlotCreator._check_skipping` method, which can be overwritten by plot creators to implement this behaviour.


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
