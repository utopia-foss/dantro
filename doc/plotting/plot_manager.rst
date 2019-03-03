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

If set, the ``creator`` argument need no longer be given in the plot configuration. By going through all registered :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` and instantiating them, it is found out if they declare that they :py:meth:`~dantro.abc.AbstractPlotCreator.can_plot` the given configuration.
Each creator can implement this method as they see fit.
In unambiguous cases, the manager than uses the *single* candidate creator and continues plotting with that creator.


Auto-detection of :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and derived classes try auto-detection by checking if it can use the given plot configuration to resolve a plotting function.
Furthermore, it checks whether the plot function is marked with attributes that may specify which creator to use. The attributes that are looked at are, in this order:

* ``creator_type``: The type of the plot creator to use (or a parent type)
* ``creator_name``: The name of the plot creator *as registered* in the manager's :py:const:`~dantro.plot_mngr.PlotManager.CREATORS` dict

To add these attributes, the :py:mod:`dantro.plot_creators` sub-package specifies the :py:class:`~dantro.plot_creators.is_plot_func` decorator, which takes care of setting attributes to the decorated plot function:

.. code-block:: python

  from dantro.plot_creators import is_plot_func

  # Specify directly with the plot creator type
  from dantro.plot_creators import MultiversePlotCreator

  @is_plot_func(creator_type=MultiversePlotCreator)
  def my_mv_plot_func(dm: DataManager, *, out_path: str, mv_data, **kwargs):
      # ...

  # Alternatively: Specify only via the _name_ known to the PlotManager
  @is_plot_func(creator_name="universe")
  def my_uni_plot_func(dm: DataManager, *, out_path: str, uni, **kwargs):
      # ...

Setting only the ``creator_name`` is recommended for scenarios where the import of the creator type is not desired.

If no plot function attributes are given, there is still another way to auto-detect the desired plot creator: inspecting the plot function signature.
This works, because derived creators *might* require a different plot function signature.
For example, :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`) additionally passes ``mv_data`` as keyword-only argument.
However, this approach can lead to ambiguous results and thus failing auto-detection. For those cases, it makes sense to specify plot function attributes via the decorator.
