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

.. _pcr_base_plot_func:

The plotting function
---------------------
The plot function is the place where selected data and configuration arguments come together to generate the plot output.
While the :py:class:`.base.BasePlotCreator` takes care of retrieving the plotting function, the function itself still has to be implemented.

In short, a plot function can be something like this:

.. testcode::

    from dantro.plot import is_plot_func

    @is_plot_func(use_dag=True, required_dag_tags=("x", "y"))
    def my_plot(*, data: dict, out_path: str, **plot_kwargs):
        """A plot function using the data transformation framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            out_path: Where to save the plot output.
            **plot_kwargs: Further plotting arguments
        """
        x = data["x"]
        y = data["y"]

        # Do something with the data
        # ...

        # Save the plot at `out_path`
        # ...

For examples of how to then :ref:`specify <pcr_base_specify_plot_func>` that function in the plot configuration and details on how to :ref:`implement <pcr_base_implement_plot_funcs>` it, see the respective sections linked here.



.. _pcr_base_specify_plot_func:

Specifying which plotting function to use
-----------------------------------------
Let's assume we have a plotting function defined somewhere and want to communicate to the :py:class:`.BasePlotCreator` that this function is responsible for creating the plot output.

For the moment, the exact definition of the function is irrelevant.
You can read more about it :ref:`below <pcr_base_implement_plot_funcs>`.

Importing a plotting function from a module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To do this, the ``module`` and ``plot_func`` entries are required.
The following example shows a plot that uses a plot function from a package called ``utopya.eval.plots`` and another plot that uses some (importable) package from which the module and the plot function are imported:

.. code-block:: yaml

   ---
   my_plot:
     # Import some module from utopya.plot_funcs (note the leading dot)
     module: .distribution

     # Use the function with the following name from that module
     plot_func: my_plot_func

     # ... all other arguments

   my_other_plot:
     # Import a module from any installed package
     module: my_installed_plotting_package.some_module
     plot_func: my_plot_func

     # ... all other arguments

.. note::

    For those interested, this is implemented in :py:meth:`.BasePlotCreator._resolve_plot_func`.


.. _pcr_base_import_plot_funcs:

Importing a plotting function from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There might be situations where you might want or need to implement a plot function decoupled from all the existing code and without bothering about importability (which may require setting up a package, installation routine, etc).

This can be achieved by specifying the ``module_file`` key instead of the ``module`` key in the plot configuration.
That python module is then loaded from file and the ``plot_func`` key is used to retrieve the plotting function:

.. code-block:: yaml

   ---
   my_plot:
     # Load the following file as a python module
     module_file: ~/path/to/my/python/script.py

     # Use the function with the following name from that module
     plot_func: my_plot_func

     # ... all other arguments (as usual)



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







.. _pcr_base_implement_plot_funcs:

Implementing plot functions
---------------------------
Below, you will learn how to implement a plot function that can be used with the :py:class:`.BasePlotCreator`.


.. _is_plot_func_decorator:

The :py:func:`~dantro.plot.utils.plot_func.is_plot_func` decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When defining a plot function, we recommend using this decorator.
It takes care of providing essential information to the :py:class:`.BasePlotCreator` and makes it easy to configure those parameters relevant for the plot function.

For example, to specify which creator should be used for the plot function, the ``creator_type`` can be given.
To control the :ref:`data transformation framework <plot_creator_dag>`, the ``use_dag`` flag can be set and the ``required_dag_tags`` argument can specify which data tags the plot function expects.


.. _pcr_base_recommended_sig:

Recommended plot function signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The **recommended way of implementing a plot function** for use with the :py:class:`.BasePlotCreator` makes use of the :ref:`data transformation framework <pcr_base_DAG_support>`.
In such a case, the data selection is taken care of by that framework, moving the data selection procedure to the plot configuration.

Let's say that we want to implement a plot function that requires some ``x`` and ``y`` data.
In the definition of the plot function we can specify that these tags are required; the framework will then make sure that these results are computed.
An implementation then looks like this:

.. testcode::

    from dantro.plot import is_plot_func

    @is_plot_func(use_dag=True, required_dag_tags=("x", "y"))
    def my_plot(*, data: dict, out_path: str, **plot_kwargs):
        """A plot function using the data transformation framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            out_path: Where to save the plot output.
            **plot_kwargs: Further plotting arguments
        """
        x = data["x"]
        y = data["y"]

        # Do something with the data
        # ...

        # Save the plot at `out_path`
        # ...


The corresponding plot configuration could look like this:

.. code-block:: yaml

    my_plot:
      creator: base

      # Select the plot function
      # ...

      # Select data
      select:
        x: data/MyModel/some/path/foo
        y:
          path: data/MyModel/some/path/bar
          transform:
            - .mean
            - increment

      # ... further arguments

For more detail on the syntax, see :ref:`above <pcr_base_DAG_support>`.


.. _pcr_base_other_sig:

Other possible plot function signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without data transformation framework
"""""""""""""""""""""""""""""""""""""
To not use the data transformation framework, simply omit the ``use_dag`` flag or set it to ``False`` in the decorator or the plot configuration.
When not using the transformation framework, the ``creator_type`` should be specified, thus making the plot function bound to one type of creator.

.. testcode::

    from dantro import DataManager
    from dantro.plot import is_plot_func, BasePlotCreator

    @is_plot_func(creator_type=BasePlotCreator)
    def my_plot(*, out_path: str, dm: DataManager, **additional_plot_kwargs):
        """A simple plot function.

        Args:
            out_path (str): The path to store the plot output at.
            dm (dantro.data_mngr.DataManager): The loaded data tree.
            **additional_kwargs: Anything else from the plot config.
        """
        # Select some data ...
        data = dm["foo/bar"]

        # Create the plot
        # ...

        # Save the plot
        # ...

.. note::

    The ``dm`` argument is only provided when *not* using the DAG framework.


.. _pcr_base_bare_sig:

Bare basics
"""""""""""
There is an even more basic way of defining a plot function, leaving out the :py:func:`~dantro.plot.utils.plot_func.is_plot_func` decorator altogether:

.. testcode::

    from dantro import DataManager

    def my_bare_basics_plot(
        dm: DataManager, *, out_path: str, **additional_kwargs
    ):
        """Bare-basics signature required by the BasePlotCreator.

        Args:
            dm: The DataManager object that contains all loaded data.
            out_path: The generated path at which this plot should be saved
            **additional_kwargs: Anything else from the plot config.
        """
        # Select the data
        data = dm["some/data/to/plot"]

        # Generate the plot
        # ...

        # Store the plot
        # ...

.. note::

    When using the bare basics version, you need to set the ``creator`` argument in the plot configuration in order for the plot manager to find the desired creator.

.. warning::

    This way of specifying plot functions is mainly retained for reasons of backwards-compatibility.
    If you can, avoid this form of plot function definition and use the :ref:`recommended signature instead <pcr_base_recommended_sig>`.



.. _pcr_base_specializing:

Specializing :py:class:`.BasePlotCreator`
-----------------------------------------
As common throughout dantro, the plot creators are specialized using class variables.
For :py:class:`.BasePlotCreator`, a specialization can look like this:

.. testcode::

    import dantro.plot.creators

    class MyPyPlotCreator(dantro.plot.creators.BasePlotCreator):
        """My custom plot creator"""

        BASE_PKG: str = "my_plot_funcs_package"
        """For relative module imports, regard this as the base package.
        A plot configuration ``module`` argument starting with a ``.`` os
        looked up in that module.

        Note that this needs to be an importable module.
        """

Furthermore, if the retrieval of the plot function needs to be adjusted, the private methods can be extended accordingly.
For example, the :py:meth:`.BasePlotCreator._get_module_via_import` method is responsible for importing a module.
By overwriting it, import behaviour can be customized:

.. testcode::

    def _get_module_via_import(self, module: str):
        """Extends the parent method by making a custom module available in
        case the regular import failed.
        """
        try:
            return super()._get_module_via_import(module)

        except ModuleNotFoundError as err:
            pass

        # Make some custom imports and return the resulting module
        # ...

.. note::

    For an operational example in a more complex framework setting, see `the specialization used in the utopya project <https://gitlab.com/utopia-project/utopya/-/blob/main/utopya/eval/plotcreators.py>`_.
    Here, the :py:class:`.PyPlotCreator` is extended such that a number of custom module paths are made available for import.
