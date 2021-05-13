.. _plot_creators:

Plot Creators
=============

Within the plotting framework, the plot creators are the classes that perform all the actual plotting work.
This document describes what they are and how they can be used.

For further reading on the individual plot creators, see:

.. toctree::
   :maxdepth: 2
   :glob:

   creators/*
   auto_detection

For specializing plot creators, see :ref:`here <spec_plot_creators>`.

----

A Family of Plot Creators
-------------------------

``AbstractPlotCreator`` - The plot creator interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As defined in :py:mod:`dantro.abc`, the :py:class:`~dantro.abc.AbstractPlotCreator` defines the interface for plot creators, i.e.: all the methods the :py:class:`~dantro.plot_mngr.PlotManager` expects and requires to create a plot.

By implementing the abstract methods, the behavior of the plot creators can be specified.

Part of the interface is that a plot creator will be initialized with the knowledge about a :py:class:`~dantro.data_mngr.DataManager`, that holds the data that should be used for plotting.


``BasePlotCreator`` - Implementing some default behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator` implements some of the abstract functions to make deriving new plot creators easier.


``ExternalPlotCreator`` - Creating plots from imported python modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` focusses on creating plots from a python plot function:

* The plot creator requires a so-called plot function that is executed to generate the plot. That plot function can be imported in various ways:

   * The included :py:mod:`~dantro.plot_creators.ext_funcs` subpackage supplies some plot functions
   * An already importable module, i.e. one that is installed or can be found in ``sys.path``
   * A plot function loaded from an external module file

* All remaining arguments of the plot configuration are passed on to the plot function
* The plot function can do whatever it wants, also meaning that it *has* to do everything by itself (getting data, saving plots, closing figures ...)

The plot function gets passed some data or the :py:class:`~dantro.data_mngr.DataManager` (to manually select data) and the rest of the plot configuration.
The required signature of the plot function depends on the chosen additional features of the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`, e.g., the :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` or :doc:`plot_data_selection`.

For more information, have a look at :ref:`the dedicated documentation page <pcr_ext>`.


``UniversePlotCreator`` & ``MultiversePlotCreator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented in :py:mod:`dantro.plot_creators.pcr_psp` are plot creators that work tightly with data stored in a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup`, i.e.: data that was created from a parameter sweep.
These are derived from :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and inherit all its abilities.

There are two different plot creators to work with this kind of data.
The :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator` allows selecting a certain subspace of the parameter space and creating a plot *for each* of these so-called "universes".

The :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator` on the other hand uses the capabilities of the :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup` to select and combine data from many universes, thus working on the "multiverse".
For the syntax needed to select the field and the subspace from the data, refer to :py:meth:`dantro.groups.pspgrp.ParamSpaceGroup.select`.

For more information, see :ref:`pcr_psp`.
