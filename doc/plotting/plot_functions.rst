.. default-domain:: dantro.plot_creators.ext_funcs.generic

.. _pcr_ext_plot_funcs:

Plot Functions
==============

This page gives an overview of plot functions that are implemented within :py:mod:`dantro` for the use with :ref:`pcr_ext` and derived plot creators.
These plot functions are meant to be as generic as possible, allowing to work with a wide variety of data.
They make use of the :ref:`dag_framework` for :ref:`plot_creator_dag`.

To use these plot functions, the following information needs to be specified in the plot configuration:

.. code-block:: yaml

    my_plot:
      creator: external      # or: universe, multiverse, ...
      module: .generic       # absolute: dantro.plot_creators.ext_funcs.generic
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
``dantro`` provides the generic :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` plot function that - together with the other dantro features - allows for a declarative way of creating plots from high-dimensional data.

The idea is that high-dimensional raw data first is transformed using the :ref:`dag_framework`.
The :py:func:`~.facet_grid` function then gets the ready-to-plot data as input and visualizes it by automatically choosing an appropriate kind of plot – if possible and not explicitely given – in a declarative way through specification of layout keywords such as ``col``\ ums, ``row``\ s, or ``hue``.
This approach is called `faceting <http://xarray.pydata.org/en/stable/plotting.html#faceting>`_; dantro makes use of the `excellent plotting functionality of xarray <http://xarray.pydata.org/en/stable/plotting.html>`_ for this feature.
The :py:func:`~.facet_grid` plot function further extends the xarray plotting functionality by adding the possibility to create :ref:`animations <pcr_ext_animations>`, simply by using the ``frames`` argument to specify the data dimension to represent as individual frames of an animation.

The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` interface then copes with the plot :ref:`style <pcr_ext_style>` and further layout.
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

.. literalinclude:: ../../dantro/plot_creators/ext_funcs/generic.py
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

In combination with :ref:`dag_generic_facet_grid_auto_kind`, this further reduces the plot configuration arguments required to generate facet grid plots.

For details, see :py:func:`~.determine_layout_encoding`.



----

.. _dag_generic_errorbar:

:py:func:`~.errorbar` and :py:func:`~.errorbands`: Visualizing Confidence Intervals
----------------------------------------------------------------------------------------
The :py:func:`~.errorbar` and :py:func:`~.errorbands` plotting functions provide the ability to visualize data together with corresponding confidence intervals.
Similar to :py:func:`~.facet_grid`, these functions offer the ``hue`` and ``frames`` arguments, allowing to represent data with up to three dimensions.

.. hint::

    These plot functions also support the :ref:`auto-encoding feature <dag_generic_auto_encoding>`, similar to the facet grid plot.
    The available specifiers are: ``x``, ``hue`` and ``frames``.
