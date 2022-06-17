.. _plot_cfg_ref:

Plot Configuration Reference
============================
This page attempts to give an overview of the available configuration options in a plot configuration.
It distinguishes features that are handled by :doc:`plot_manager` and those handled by the plot creators.

In the following, all examples are given on the level of a single plot configuration, specifically:

.. code-block:: yaml

    my_plot:         # name of the plot
      # ...          # <-- everything in here is called "plot configuration"

.. contents::
   :local:
   :depth: 2

----


General Options
---------------

The options shown here are handled by :py:class:`~dantro.plot_mngr.PlotManager` or :py:class:`~dantro.plot.creators.base.BasePlotCreator` and are always available.

All options seen here basically act as **reserved keywords**.
Subsequently, they cannot be used downstream in the plot creator, because they are already handled and not passed on again.

.. literalinclude:: ../../tests/cfg/doc_examples.yml
    :language: yaml
    :start-after: ### Start -- plot_cfg_ref_mngr_overview
    :end-before:  ### End ---- plot_cfg_ref_mngr_overview
    :dedent: 2

.. note::

    The values given here are *not necessarily* also the default values.
    That depends on the used :py:class:`~dantro.plot_mngr.PlotManager` specialization and the involved plot creators.

    As always with configuration files, try to specify only those entries that *deviate* from the default setting.

.. hint::

    For more information regarding **overwriting of plot output and writing to a custom directory**, see :ref:`the FAQ <faq_plotting_overwrite>`.


Options handled by the :doc:`Plot Creators <plot_creators>`
-----------------------------------------------------------

The options made available by the individual plot creators, see their respective documentation entries:

* :doc:`plot_creators`
* :doc:`creators/pyplot`
* :doc:`creators/psp`
