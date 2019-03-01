The ``PlotManager``
===================

The :py:class:`~dantro.plot_mngr.PlotManager` orchestrates the whole plotting framework.
This document described what it is and how it works together with the :doc:`plot_creators` to generate plots.

.. contents::
   :local:
   :depth: 2

----

.. note::

  This document is very much work in progress


Auto-Detection of a Plot Creator
--------------------------------

The :py:class:`~dantro.plot_mngr.PlotManager` has as class variable a dictionary of :py:const:`~dantro.plot_mngr.PlotManager.CREATORS`, which is a mapping of common name strings to plot creator types, i.e. :py:class:`~dantro.abc.AbstractPlotCreator`-derived classes.
Usually, the ``creator`` argument to the :py:class:`~dantro.plot_mngr.PlotManager`\'s :py:meth:`~dantro.plot_mngr.PlotManager.plot` function is used to extract the plot creator type from that dictionary and then initialize the object.

However, the plot manager also has a ``auto_detect_creator`` feature.
This boolean argument can be given both to :py:meth:`~dantro.plot_mngr.PlotManager.__init__` as well as to :py:meth:`~dantro.plot_mngr.PlotManager.plot` and it can also be part of the plot configuration passed to :py:meth:`~dantro.plot_mngr.PlotManager.plots_from_cfg`.

If set, the ``creator`` argument need no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is tried if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method how they see fit. For example, the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` tries if it can use the given plot configuration to resolve a plotting function.
Furthermore, it checks the signature of the available plot functions; this is helpful in cases where derived creators (like :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`) are also candidates.
