.. default-domain:: dantro.plot.creators.base

.. _pcr_base:

The :py:class:`.BasePlotCreator`
================================
The :py:class:`.base.BasePlotCreator` implements the basic functionality that all derived plot creators profit from:

- Resolving a plot function, which can be a directly given callable, an importable module and name, or a path to a Python file that is to be imported.
- Parsing plot configuration arguments.
- Optionally, performing data selection from the associated :py:class:`~dantro.data_mngr.DataManager` using the :ref:`data transformation framework <dag_framework>`.
- Invoking the plot function with the selected and transformed data and other arguments from the plot configuration.

As such, the this base class is agnostic to the exact way of how plot output is generated; the plot function is responsible for that.


.. contents::
   :local:
   :depth: 2

----

.. _pcr_base_DAG_support:

Using the data transformation framework for plot data selection
---------------------------------------------------------------

The :ref:`data selection and transformation framework <dag_framework>` framework is a central part of dantro:
Using a directed, acyclic graph (DAG) of operations, it allows to work rather generically on the plot data held in a :ref:`data tree <data_mngr>`.
This is a powerful tool, especially when combined with the plotting framework.

The motivation for using the data transformation framework for plotting is the following:
Ideally, a plot function should focus *only* on the visualization of data: creating a meaningful representation of the data, be it a simple line plot, a heatmap or some other form of plot.
Everything else that happens before (data selection, pre-processing, transformation, etc.) and after (adjusting plot aesthetics, saving the plot, etc.) should ideally be decoupled from that process and, if possible, automated.

The :py:class:`.BasePlotCreator` aims to take care of what happens "before", data selection and transformation, and it uses the :ref:`data transformation framework <dag_framework>` for that.
(For what happens "after", more assumptions need to be made, which are only possible when having decided on a plot backend, like :ref:`pcr_pyplot` does.)

To that end, the :py:class:`.BasePlotCreator` uses a configuration-based syntax that can be passed alongside the plot configuration itself.
This config-based declaration is optimized for specification via YAML and looks something like this:

.. code-block:: yaml

    my_plot:
      creator: base

      select:
        mean_data:
          path: path/to/some_data
          transform:
            - .mean
        std_data:
          path: path/to/some_data
          transform:
            - .std

For more syntax examples, see :ref:`plot_creator_dag`.

Additionally, this approach allows to cache transformation results to a file.
This is very useful when the analysis of data takes a large amount of time compared to the plotting itself.



.. _pcr_base_specializing:

Specializing :py:class:`.BasePlotCreator`
-----------------------------------------
As common throughout dantro, the plot creators are specialized using class variables.
For :py:class:`.BasePlotCreator`, a specialization can look like this:

.. testcode::

    import dantro.plot.creators

    class MyPyPlotCreator(dantro.plot.creators.BasePlotCreator):
        """My custom plot creator"""

        EXTENSIONS = ("pdf", "png")
        """Allow only PDF or PNG extensions."""

        DAG_USE_BY_DEFAULT = True
        """Use the data transformation framework by default."""

.. hint::

    Make sure that the :py:class:`~dantro.plot_mngr.PlotManager` knows about your new creator by setting its :py:attr:`~dantro.plot_mngr.PlotManager.CREATORS` class variable accordingly.



Adjusting the data transformation routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the :py:class:`.BasePlotCreator`, :py:meth:`.BasePlotCreator._prepare_plot_func_args` is responsible of invoking data transformation, which is done right before invocation of the plot function.
Data selection and transformation itself happens in :py:meth:`.BasePlotCreator._perform_data_selection`.

If you plan to change the behavior of this aspect of the plot creator, ideally do so in :py:meth:`.BasePlotCreator._perform_data_selection` itself.
We recommend to only make minimal changes to :py:meth:`.BasePlotCreator._prepare_plot_func_args`.

.. hint::

    For implementation examples see :ref:`the parameter space plot creators <pcr_psp>`.
