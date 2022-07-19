
.. _dantro_base_plots_ref:

Base Plot Configuration Pool
============================
This page documents dantro's :ref:`base plot configuration pool <dantro_base_plots>`, sorted by segments and using the :ref:`naming convention <base_plots_naming_conventions>`.

.. hint::

    To quickly search for individual entries, the search functionality of your browser (``Cmd + F``) may be very helpful.
    Note that some entries (like those of the YAML anchors) may only be found if the :ref:`complete file reference <dantro_base_plots_ref_complete>` is expanded.

.. contents::
    :local:
    :depth: 2

----


``.defaults``: default entries
------------------------------

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .defaults
    :end-before: # ===



``.creator``: selecting a plot creator
--------------------------------------
More information: :ref:`plot_creators`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .creator
    :end-before: # ===


``.plot``: selecting a plot function
------------------------------------
More information:

- :ref:`plot_func`
- :ref:`pcr_pyplot_plot_funcs`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .plot
    :end-before: # ===


``.style``: choosing plot style
-------------------------------
More information: :ref:`pcr_pyplot_style`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .style
    :end-before: # ===


``.hlpr``: invoking individual plot helper functions
----------------------------------------------------
More information: :ref:`plot_helper`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .hlpr
    :end-before: # ===


``.animation``: controlling animation
-------------------------------------
More information: :ref:`pcr_pyplot_animations`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .animation
    :end-before: # ===


``.dag``: configuring the DAG framework
---------------------------------------
More information:

- :ref:`dag_framework`
- :ref:`plot_creator_dag`

.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .dag
    :end-before: # ===


``.dag.meta_ops``: meta operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following entries can be included into a plot configuration to make pre-defined :ref:`meta-operations <dag_meta_ops>` available for the data transformation framework.

.. admonition:: Example
    :class: dropdown

    As an example, let's include the ``.dag.meta_ops.compute_mean_and_stddev`` operation to calculate these values over the ``time`` and ``space`` dimensions and combine them into an :py:class:`xarray.Dataset`.

    .. code-block:: yaml

        my_mean_and_stddev_plot:
          based_on:
            - .creator.universe
            - .plot.facet_grid.errorbars
            - .dag.meta_ops.compute_mean_and_stddev

          select:
            some_data: path/to/some/data

          transform:
            - compute_mean_and_stddev: [!dag_tag some_data, [time, space]]
              tag: data

          y: mean
          yerr: stddev


.. literalinclude:: ../../dantro/cfg/base_plots.yml
    :language: yaml
    :start-after: # start: .dag.meta_ops
    :end-before: # ===


----

.. _dantro_base_plots_ref_complete:

Complete File Reference
-----------------------

.. toggle::

    .. literalinclude:: ../../dantro/cfg/base_plots.yml
        :language: yaml
        :end-before: # end of dantro base plots
