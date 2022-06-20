.. default-domain:: dantro.plot.creators

.. _plot_creators:

Plot Creators
=============

Within the plotting framework, the plot creators are the classes that perform all the actual plotting work.
This document describes what they are and how they can be used.

For further reading on the individual plot creators, see:

.. toctree::
   :maxdepth: 2
   :glob:

   creators/base
   creators/pyplot
   creators/psp

.. hint::

    For guidance about specializing plot creators, see :ref:`here <spec_plot_creators>`.

----

A Family of Plot Creators
-------------------------

:py:class:`~dantro.abc.AbstractPlotCreator` - The plot creator interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As defined in :py:mod:`dantro.abc`, the :py:class:`~dantro.abc.AbstractPlotCreator` defines the interface for plot creators, i.e.: all the methods the :py:class:`~dantro.plot_mngr.PlotManager` expects and requires to create a plot.

By implementing the abstract methods, the behavior of the plot creators can be specified.

Part of the interface is that a plot creator will be initialized with the knowledge about a :py:class:`~dantro.data_mngr.DataManager`, that holds the data that should be used for plotting.


:py:class:`.base.BasePlotCreator` - Implementing shared behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`.base.BasePlotCreator` implements the basic functionality that all derived plot creators profit from:
Parsing the plot configuration, selecting and transforming data, and retrieving and invoking a so-called plot function (where the actual visualization is implemented).

Continue reading about this :ref:`here <pcr_base>`.

.. hint::

    If you want to have full control of how your plot is generated, this is the creator to use:
    You can configure it to simply invoke an arbitrary plot function and pass it the available data – and you take care of all the rest within that function.

    In case you want dantro to automate more parts of the plotting routine, continue reading about more advanced creators below.



:py:class:`.PyPlotCreator` - Creating :py:mod:`matplotlib.pyplot`-based plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` specializes on creating plots using :py:mod:`matplotlib.pyplot`.
By making these assumptions about the used plotting backend, it allows to generalize plot setup, options, and style.
Furthermore, it allows to easily define animation update functions.

For more information, have a look at :ref:`the dedicated documentation page <pcr_pyplot>`.



:py:class:`~.psp.UniversePlotCreator` & :py:class:`.psp.MultiversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented in :py:mod:`dantro.plot.creators.psp` are plot creators that work tightly with data stored in a :py:class:`~dantro.groups.psp.ParamSpaceGroup`, i.e.: data that was created from a :py:class:`~paramspace.paramspace.ParamSpace` parameter sweep.
These are derived from :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` and inherit all its abilities.

There are two different plot creators to work with this kind of data.
The :py:class:`~dantro.plot.creators.psp.UniversePlotCreator` allows selecting a certain subspace of the parameter space and creating a plot *for each* of these so-called "universes".

The :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` on the other hand uses the capabilities of the :py:class:`~dantro.groups.psp.ParamSpaceGroup` and the :ref:`data transformation framework <dag_framework>` to select and combine data from many universes, thus working on the "multiverse".

See :ref:`pcr_psp` for more information.
