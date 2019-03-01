Plot Creators
=============

Within the plotting framework, the plot creators are the classes that perform all the actual plotting work.
This document described what they are and how they can be used.

.. contents::
   :local:
   :depth: 2

----

A Family of Plot Creators
-------------------------

``AbstractPlotCreator`` - The plot creator interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As defined in :py:mod:`dantro.abc`, the :py:class:`~dantro.abc.AbstractPlotCreator` defines the interface for plot creators, i.e.: all the methods the :py:class:`~dantro.plot_mngr.PlotManager` expects and requires in order to create a plot.

By implementing the abstract methods, the behaviour of the plot creators can be specified.

Part of the interface is that a plot creator will be initialized with the knowledge about a :py:class:`~dantro.data_mngr.DataManager`, that holds the data that should be used for plotting.


``BasePlotCreator`` - Implementing some default behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator` implements some of the abstract functions in order to make deriving new plot creators easier.


``ExternalPlotCreator`` - Creating plots from imported python modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` focusses on creating plots from any importable python module.
This can be both an installed module or a module loaded from a file.
Within the loaded module, a plotting function is expected, which gets passed the :py:class:`~dantro.data_mngr.DataManager` and the plot configuration.


``UniversePlotCreator`` & ``MultiversePlotCreator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented in :py:mod:`dantro.plot_creators.pcr_psp` are plot creators that work tightly with data stored in a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup`, i.e.: data that was created from a parameter sweep.

There are two different plot creators to work with this kind of data. The :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator` allows selecting a certain subspace of the parameter space and creating a plot *for each* of these so-called "universes".

The :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator` on the other hand uses the capabilities of the :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup` to select and combine data from many universes, thus working on the "multiverse".
For the syntax needed to select the field and the subspace from the data, refer to :py:meth:`dantro.groups.pspgrp.ParamSpaceGroup.select`.

