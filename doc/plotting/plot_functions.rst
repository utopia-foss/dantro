.. default-domain:: dantro.plot.funcs.generic

.. _pcr_pyplot_plot_funcs:

Plot Functions
==============

This page gives an overview of plot functions that are implemented within :py:mod:`dantro` for the use with :ref:`pcr_pyplot` and derived plot creators.
These plot functions are meant to be as generic as possible, allowing to work with a wide variety of data.
They make use of the :ref:`dag_framework` for :ref:`plot_creator_dag`.

To use these plot functions, the following information needs to be specified in the plot configuration:

.. code-block:: yaml

    my_plot:
      creator: pyplot        # or: universe, multiverse, ...
      module: .generic       # absolute: dantro.plot.funcs.generic
      plot_func: facet_grid  # or: errorbar, errorbands, ...

      # ...

.. contents::
   :local:
   :depth: 2

----


.. _dag_generic_facet_grid:

:py:func:`~.facet_grid`: A Declarative Generic Plot Function
------------------------------------------------------------

Handling, transforming, and plotting high-dimensional data is difficult and often requires specialization to use-cases.
``dantro`` provides the generic :py:func:`~.facet_grid` plot function that - together with the other dantro features - allows for a declarative way of creating plots from high-dimensional data.

The idea is that high-dimensional raw data first is transformed using the :ref:`dag_framework`.
The :py:func:`~.facet_grid` function then gets the ready-to-plot data as input and visualizes it by automatically choosing an appropriate kind of plot – if possible and not explicitly given – in a declarative way through the specification of layout keywords such as ``col``\ ums, ``row``\ s, or ``hue``.
This approach is called `faceting <https://xarray.pydata.org/en/stable/user-guide/plotting.html#faceting>`_; dantro makes use of the `excellent plotting functionality of xarray <https://xarray.pydata.org/en/stable/plotting.html>`_ for this feature.
The :py:func:`~.facet_grid` plot function further extends the xarray plotting functionality by adding the possibility to create :ref:`animations <pcr_pyplot_animations>`, simply by using the ``frames`` argument to specify the data dimension to represent as individual frames of an animation.

The :py:class:`~dantro.plot.plot_helper.PlotHelper` interface then copes with the plot :ref:`style <pcr_pyplot_style>` and further layout.
All steps are fully configurable and optimized for the YAML-based plotting interface.
Thus, generating a plot of multidimensional data does not require touching any actual code but just specifying the desired representation in the plot configuration. 🎉

For more information, have a look at the :py:func:`~.facet_grid` docstring.


.. _dag_generic_facet_grid_auto_kind:

Automatically selecting plot kind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``kind`` keyword of the facet grid plot is quite important. It determines most of the aesthetics and the possible dimensionality that the to-be-visualized data may have.

However, in some scenarios, one would like to choose an *appropriate* plot kind.
While ``kind: None`` outsources the plot kind to xarray, this frequently leads to ``kind: hist`` being created, depending on which layout specifiers were given.

The :py:func:`~.determine_plot_kind` function used in :py:func:`~.facet_grid` uses the plot data's dimensionality to select a plotting ``kind``.
By default, the following mapping of data-dimensionality to plot kind is used:

.. literalinclude:: ../../dantro/plot/funcs/generic.py
    :language: python
    :start-after: _AUTO_PLOT_KINDS = {  # --- start literalinclude
    :end-before:  }   # --- end literalinclude
    :dedent: 4

Aside from the dimensionality as key, there are a few special cases that handle already-fixed layout encoding (``hue`` / ``x`` and ``y``); the case of  ``xr.Dataset``-like data; and a ``fallback`` option for all other dimensionalities or cases.
For details, see the docstring of :py:func:`~.determine_plot_kind`.

Setting ``kind: auto`` becomes especially powerful in conjunction with :ref:`dag_generic_auto_encoding`.


.. _dag_generic_auto_encoding:

Auto-encoding of plot layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dantro also adds the ``auto_encoding`` feature to the facet grid plot, which automatically associates data dimensions with certain layout encoding specifiers (``x``, ``y``, ``col``, and others).
With this functionality, the facet grid plot can be used to visualize high-dimensional data *regardless of the dimension names*; the only relevant information is the dimensionality of the data.

The available encodings for the :py:func:`~.facet_grid` plot are:

.. ipython::

    @suppress
    In [1]: from dantro.plot.funcs.generic import _FACET_GRID_KINDS

    @suppress
    In [2]: available_facet_grid_kinds = "\n".join([f"{kind:>15s} : {specs}" for kind, specs in _FACET_GRID_KINDS.items()])

    In [3]: print(available_facet_grid_kinds)

In combination with :ref:`dag_generic_facet_grid_auto_kind`, this further reduces the plot configuration arguments required to generate facet grid plots.

For further details, see :py:func:`~.determine_encoding`.


.. _dag_facet_grid_decorator:

Add custom plot ``kind``\ s that support faceting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While the already-available plot kinds of the facet grid cover many use cases, there is still room for extension.
As part of the :py:mod:`~dantro.plot.funcs.generic` plot functions module, dantro provides the :py:class:`~.make_facet_grid_plot` decorator that wraps the decorated function in such a way that it becomes facetable.

That means that after decoration:

- The function will support faceting in ``col``, ``row`` and ``frames`` in addition to those dimensions handled within the decorated function.
- It will be registered with the generic :py:func:`~.facet_grid` function, such that it is available as ``kind``.
- It will be integrated in such a way that it supports :ref:`auto encoding <dag_generic_auto_encoding>`.

The :py:class:`~.make_facet_grid_plot` decorator wraps the functionality of ``xarray.plot.FacetGrid`` and makes it easy to add faceting support to plot functions.
It can be used if the following requirements are fulfilled:

- Works with a single ``xr.Dataset`` or ``xr.DataArray`` object as input
- Will only plot to the *current* axis and not create a figure
- It is desired to have *the same kind of plot* repeated over multiple axes, the plots differing only in the slice of data passed to them.

As an example, have a look at the :py:func:`~.errorbars` plot function, which supercedes :ref:`the explicitly implemented <dag_generic_errorbar>` plot.



----

.. _dag_generic_errorbar:

:py:func:`~.errorbar` and :py:func:`~.errorbands`: Visualizing Confidence Intervals
----------------------------------------------------------------------------------------

.. deprecated:: 0.15

    **This function is deprecated and will be removed with version 1.0.**
    Instead, use :ref:`the generic facet grid function <dag_generic_facet_grid>` with ``kind: errorbars``, which has additional capabilities and almost the same interface (only difference being that it works with an ``xr.Dataset`` instead of two ``xr.DataArray``\ s).

The :py:func:`~.errorbar` and :py:func:`~.errorbands` plotting functions provide the ability to visualize data together with corresponding confidence intervals.
Similar to :py:func:`~.facet_grid`, these functions offer the ``hue`` and ``frames`` arguments, allowing to represent data with up to three dimensions.

.. hint::

    These plot functions also support the :ref:`auto-encoding feature <dag_generic_auto_encoding>`, similar to the facet grid plot.
    The available specifiers are: ``x``, ``hue`` and ``frames``.




----

.. _dag_multiplot:

:py:func:`~.multiplot`: Plot multiple functions on one axis
-----------------------------------------------------------
The :py:func:`~.multiplot` plotting function enables the consecutive application of multiple plot functions on the current axis generated and provided through the :py:class:`~dantro.plot.plot_helper.PlotHelper`.

Plot functions can be specified in three ways:

- as a string that is used to map to the corresponding function
- by importing a callable on the fly
- or by directly passing a callable function

For plot function lookup by string, the following `seaborn plot functions <https://seaborn.pydata.org/api.html>`_ and some matplotlib functions are available:

.. literalinclude:: ../../dantro/plot/funcs/_multiplot.py
    :language: python
    :start-after: MULTIPLOT_FUNC_KINDS = { # --- start literalinclude
    :end-before:  }   # --- end literalinclude
    :dedent: 4

To import a callable, specify a ``(module, name)`` tuple; this will use :py:func:`~dantro._import_tools.import_module_or_object` to carry out the import and traverse any modules.

You can also invoke any other function operating on a :py:class:`~matplotlib.axes.Axes` object by importing or constructing a callable via the :ref:`data transformation framework <plot_creator_dag>`.

Let us look at some example configurations to illustrate the above features:

.. code-block:: YAML

    # Minimal example
    sns_lineplot_example:
      plot_func: multiplot
      to_plot:
        # Plot a seaborn.lineplot
        # As data use the previously DAG-tagged 'seaborn_data'.
        # Note that it is important to specify the data to use
        # otherwise sns.lineplot plots and shows nothing!
        - function: sns.lineplot
          data: !dag_result seaborn_data
          # Add further sns.lineplot-specific kwargs below...
          markers: true

        # Can add more function specifications here to plot on the same axes

    # An advanced example
    sns_lineplot_and_more:
      plot_func: multiplot

      # Define some custom callable
      dag_options:
        define:
          my_custom_callable:
            - lambda: "lambda *, ratio, ax: ax.set_aspect(ratio)"

      to_plot:
        # Look up the callable from a dict
        - function: sns.lineplot
          data: !dag_result seaborn_data
          # Add further sns.lineplot-specific kwargs below...
          markers: true

        # Import a callable on the fly
        - function: [matplotlib, pyplot.plot]
          # plt.plot requires the x and y values to be passed as positional
          # arguments.
          args:
            - !dag_result plot_x
            - !dag_result plot_y
          # Add further plot-specific kwargs below...

        # Call the constructed plot function, passing the axis object along
        - function: !dag_result my_custom_callable
          pass_axis_object_as: ax
          ratio: 0.625

        # Can add more functions here, if desired

.. hint::

    As can be seen in the above example, it is possible to pass an axis object to the function, if needed.
    To do so, use the ``pass_axis_object_as`` argument to specify the name of the keyword argument the axis object should be passed on as.

.. hint::

    The actual implementation is part of the :py:mod:`~dantro.plot.plot_helper.PlotHelper` interface, which also gives access to arbitrary function invocations on the current axis.
    The corresponding helper function is named ``call`` (:py:meth:`~dantro.plot.plot_helper.PlotHelper._hlpr_call`).


Use ``multiplot`` with multiple subplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generating plots with multiple subplots is also possible via the :py:func:`~dantro.plot.funcs.multiplot.multiplot` function.
This is a two-step process:

- In the :py:class:`~dantro.plot.plot_helper.PlotHelper` configuration, specify the desired subplots of the figure using ``setup_figure``.
- In the :py:func:`~dantro.plot.funcs.multiplot.multiplot` configuration, address each axis separately and specify which function calls should be made on it.

Example:

.. literalinclude:: ../../tests/cfg/dag_plots.yml
    :language: yaml
    :start-after: ### Start -- multiplot_subplots
    :end-before:  ### End ---- multiplot_subplots
    :dedent: 6
