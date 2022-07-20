.. _plot_examples:

Example Plots
=============

This page showcases plot configurations and the resulting output.

.. contents:: Table of Contents
    :local:
    :depth: 2

Further reading:

- :ref:`plot_manager`
- :ref:`plot_creators`
- :ref:`dantro_base_plots_ref`
- :ref:`plot_creator_dag`
- :ref:`data_ops_available`


.. admonition:: How to include the given examples into your plots
    :class: dropdown

    All plot configurations shown here are represented using YAML, which is the most convenient.
    Note that the examples *omit* the top-level of the :ref:`plots configuration file <plot_cfg_overview>`; if you want to use the plots in such a way, include the shown examples like this:

    .. code-block:: yaml

        my_plot:
          # ... content of the YAML examples you see here ...

    By the way: All plots on this page are *tested* as part of the dantro test suite and are generated dynamically.

.. todo::

    Describe how to generate the test data.


----


Errorbands plot using meta-operations
-------------------------------------
This is similar to an example in the `xarray documentation <https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html>`__, showing averaged air temperatures over time.

Using the :ref:`dantro base plots config <dantro_base_plots_ref>`, this example embeds pre-defined :ref:`meta-operations <dag_meta_ops>` which are then used to compute the mean and standard deviation of the example data.
In addition, a rolling mean is applied to smoothe out the data, and a coordinate dimension is transformed such that matplotlib can plot label ticks from it.

.. image:: ../_static/_gen/plots/doc_examples_errorbars.pdf
    :target: ../_static/_gen/plots/doc_examples_errorbars.pdf
    :width: 90%
    :alt: Errorbars plot example with air temperature data

.. admonition:: Plot configuration
    :class: dropdown

    .. literalinclude:: ../../tests/cfg/dag_plots.yml
        :language: yaml
        :dedent: 6
        :start-after: ### Start -- errorbars_example
        :end-before: ### End ---- errorbars_example


.. admonition:: DAG Visualization
    :class: dropdown

    The corresponding :ref:`DAG visualization <dag_graph_vis>` looks like this:

    .. image:: ../_static/_gen/plots/doc_examples_errorbars_dag_compute_success.pdf
        :target: ../_static/_gen/plots/doc_examples_errorbars_dag_compute_success.pdf
        :width: 90%
        :alt: DAG Visualization for errorbars example




3D scatter plot
---------------
Here, a 3D random walk is visualized using the :py:func:`~dantro.plot.funcs.generic.scatter3d` plot, accessible via the :ref:`facet grid interface <dag_generic_facet_grid>`.

.. image:: ../_static/_gen/plots/doc_examples_scatter3d.pdf
    :target: ../_static/_gen/plots/doc_examples_scatter3d.pdf
    :width: 90%
    :alt: 3D scatter plot example of a random walk

.. admonition:: Plot configuration
    :class: dropdown

    .. literalinclude:: ../../tests/cfg/dag_plots.yml
        :language: yaml
        :dedent: 6
        :start-after: ### Start -- scatter3d_example
        :end-before: ### End ---- scatter3d_example


Multiplot with subplots
-----------------------
This example showcases the :ref:`multiplot function <dag_multiplot>` and how to add different content on individual subplots.
Furthermore, it uses the import functionality of the plot function to call :py:func:`matplotlib.pyplot.ylabel` on the subplots.

.. image:: ../_static/_gen/plots/doc_examples_multiplot_subplots.pdf
    :target: ../_static/_gen/plots/doc_examples_multiplot_subplots.pdf
    :width: 90%
    :alt: Multiplot plot example with subplots and artificial time series data

.. admonition:: Plot configuration
    :class: dropdown

    .. literalinclude:: ../../tests/cfg/dag_plots.yml
        :language: yaml
        :dedent: 6
        :start-after: ### Start -- multiplot_subplots
        :end-before: ### End ---- multiplot_subplots
