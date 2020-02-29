.. _pcr_ext:

The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`
=================================================================

The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` focusses on creating plots from a Python function, the so-called *plot function*.
The term *external* refers to its ability to invoke plot functions from external modules, which can also be loaded from some file path.

.. note::

    There are specializations of the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` that make plotting of data originating from parameter sweeps easier.
    See :ref:`pcr_psp`.

.. contents::
   :local:
   :depth: 2

----


Specifying which plotting function to use
-----------------------------------------
Let's assume we have a plotting function defined somewhere and want to communicate to the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` that this function should be used for some plot.

For the moment, the exact definition of the function is irrelevant.
You can read more about it :ref:`below <pcr_ext_implement_plot_funcs>`.

Importing a plotting function from a module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To do this, the ``module`` and ``plot_func`` entries are required.
The following example shows a plot that uses a plot function from ``utopya.plot_funcs`` and another plot that uses some (importable) package from which the module and the plot function are imported:

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


.. _pcr_ext_external_plot_funcs:

Importing a plotting function from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There might be situations where you might want or need to implement a plot function decoupled from all the existing code.

This can be achieved by specifying the ``module_file`` key instead of the ``module`` key in the plot configuration.
The python module is then loaded from file and the ``plot_func`` key is used to retrieve the plotting function:

.. code-block:: yaml

   ---
   my_plot:
     # Load the following file as a python module
     module_file: ~/path/to/my/python/script.py

     # Use the function with the following name from that module
     plot_func: my_plot_func

     # ... all other arguments (as usual)




.. _pcr_ext_style:

Adjusting a plot's style
------------------------
All matplotlib-based plots can profit from this feature.

Using the ``style`` keyword, matplotlib parameters can be configured fully via the plot configuration; no need to touch the code.
Basically, this sets the ``matplotlib.rcParams`` and makes the matplotlib stylesheets available.

The following example illustrates the usage:

.. code-block:: yaml

    ---
    my_plot:
      # ...

      # Configure the plot style
      style:
        base_style: ~        # optional, name of a matplotlib style to use
        rc_file: ~           # optional, path to YAML file to load params from
        # ... all further parameters are interpreted directly as rcParams

In the following example, the ``ggplot`` style is used and subsequently adjusted by setting the linewidth, marker size and label sizes.

.. code-block:: yaml

    ---
    my_ggplot:
      # ...

      style:
        base_style: ggplot
        lines.linewidth : 3
        lines.markersize : 10
        xtick.labelsize : 16
        ytick.labelsize : 16


For the ``base_style`` entry, choose the name of a `matplotlib stylesheet <https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html>`_.
For valid RC parameters, see the `matplotlib customization documentation <https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html>`_.




.. _pcr_ext_plot_helper:

The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper`
-------------------------------------------------------------

The aim of the PlotHelper framework is to let the plot functions focus on what cannot easily be automated: being the bridge between some selected or :ref:`transformed <pcr_ext_DAG_support>` data and its visualization.
The plot function should not have to concern itself with things like plot aesthetics, as that can easily be automated.

The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` can make your life easier by quite a lot as it already takes care of setting up and saving a figure and makes large parts of the matplotlib interface accessible via the plot configuration.
That way, you don’t need to touch Python code for trivial tasks like changing the plot limits.
But even more advanced tasks, such as creating an animation, are conveniently done using this framework.

Most importantly, it will make your plots future-proof and let them profit from upcoming features.
A glimpse of that can be seen in how easy it is to implement an animated plot, see :ref:`below <pcr_ext_animations>`.

To learn, how you can enable the PlotHelper in your plot function, checkout the section on :ref:`implementing plot functions <pcr_ext_implement_plot_funcs>`.

As an example, the following plot configuration sets the title of the plot as well as the labels and limits of the axes:

.. code-block:: yaml

  my_plot:
    # ...

    # Configure the plot helpers
    helpers:
      set_title:
        title: This is My Fancy Plot
      set_labels:
        x: $A$
        y: Counts $N_A$
      set_limits:
        x: [0, max]
        y: [1.0, ~]

The enabled helpers are automatically invoked after the plot function has been called and before the plot is saved.
Aside from specifying values in the configuration, helpers can also be dynamically (re-)configured from within the plot function using :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.provide_defaults` or be invoked directly using :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.invoke_helper`.
To ensure that helpers stay disabled, regardless of configuration, you can call :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.mark_disabled` inside the plot function.

.. hint::

    The helper syntax is mostly equivalent to the matplotlib interface, but simplified in some cases, e.g. by using just ``x`` and ``y`` as arguments.
    If you get it wrong, the error message aims to be helpful: it provides the full signature and docstring of the invoked helper such that you can adjust the parameters to the required format.

    Thus, trial and error is a useful first try before digging into the :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` API reference.

Furthermore, notice how you can combine the capabilities of the plot helper framework with the ability to :ref:`set the plot style <pcr_ext_style>`.


Axis-specific helper configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` is not restricted to a single axis, but it can manage multiple axes aranged on a grid.
A possible plot configuration with axis-specific helpers could look as follows:

.. code-block:: yaml

  my_plot:
    # ...

    # Configure the plot helpers
    helpers:
      setup_figure:
        n_cols: 2
        sharey: True
      set_limits:
        x: [0, max]
        y: [1.0, ~]
      axis_specific:
        my_left_axis:
          axis: [0, 0]
          set_title:
            title: This is my left plot
        my_right_axis:
          axis: [1, 0]
          set_title:
            title: This is my right plot

Putting the above configuration into words:

* The :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.setup_figure` helper sets up a figure with with two subfigures that are accessible via the coordinate pairs ``[0, 0]`` and ``[1, 0]``.
* The ``set_limits`` helper is applied to all existing axes.
* Helpers for specific axes can be configured by passing an ``axis_specific`` dictionary.
  In the plot function, you can then switch axes using the :py:meth:`~dantro.plot_creators._plot_helper.PlotHelper.select_axis` method (the ``[0, 0]`` axis is selected initially).

.. note::

    The keys for the ``axis_specific`` configuration are arbitrary; the axes are defined solely by the internal ``axis`` entries.
    While this requires to specify a name for the axis, it also allows convenient recursive updating; thus, it is advisable to choose a somewhat meaningful name.


Specializing the helper
^^^^^^^^^^^^^^^^^^^^^^^

The dantro :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` already provides a default set of helpers that provide access to most of the matplotlib interface.
If you need any additional customized helpers, you can easily add new methods to a specialization of the helper:

.. code-block:: python

  import dantro.plot_creators

  class MyPlotHelper(dtr.plot_creators.PlotHelper):
      """A specialization of the dantro ``PlotHelper`` which can be used to add
      additional helper methods.
      """
      # You can add new helper methods here, prefixed with _hlpr_

Note that you will have to communicate this new plot helper type to the creator by setting :py:const:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator.PLOT_HELPER_CLS`.



.. _pcr_ext_DAG_support:

The data transformation framework
---------------------------------

As part of dantro, a :ref:`data selection and transformation framework <dag_framework>` based on a directed, acyclic graph (DAG) of operations is provided.
This is a powerful tool, especially when combined with the plotting framework.

The motivation of using this DAG framework for plotting is similar to the motivation of the :ref:`plot helper <pcr_ext_plot_helper>`:
Ideally, the plot function should focus on the visualization of some data; everything else before (data selection, transformation, etc.) and after (adjusting plot aesthetics, saving the plot, etc.) should be automated.

It uses a configuration-based syntax that is optimized for specification via YAML, right alongside the plot configuration.
Additionally, it allows to cache results to a file; this is very useful when the analysis of data takes a large amount of time compared to the plotting itself.

To learn more about this, :ref:`see here <dag_framework>`.

.. hint::

    If you are missing an operation, register it yourself:

    .. code-block:: python

        from dantro.utils import register_operation

        def my_operation(data, *, some_parameter, **more_parameters):
            """Some operation on the given data"""
            # Do something with data and the parameters
            # ...
            return new_data

        register_operation(name='my_operation', func=my_operation)

    Note that you are not allowed to override any existing operation.
    To avoid naming conflicts, it is advisable to use a unique name for the custom operation, e.g. if by prefixing the model name for some model-specific operation.





.. _pcr_ext_implement_plot_funcs:

Implementing plot functions
---------------------------
Below, you will learn how to implement a plot function that can be used with the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`.


.. _is_plot_func_decorator:

The :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When defining a plot function, we recommend using this decorator.
It takes care of providing essential information to the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and makes it easy to configure those parameters relevant for the plot function.

For example, to specify which creator should be used for the plot function, the ``creator_type`` can be given.
To control usage of the data transformation framework, the ``use_dag`` flag can be used and the ``required_dag_tags`` argument can specify which data tags the plot function expects.

.. _pcr_ext_recommended_sig:

Recommended plot function signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommended way of implementing a plot function makes use of both the :ref:`plot helper framework <pcr_ext_plot_helper>` and the :ref:`data transformation framework <pcr_ext_DAG_support>`.

When using the data transformation framework, the data selection is taken care of by that framework, moving the data selection procedure to the plot configuration.
In the plot function, one can specify which tags are required by the plot function; the framework will then make sure that these results are computed.
In this case, two tags called ``x`` and ``y`` are required which are then fed directly to the plot function.

Importantly, such a plot function can be **averse to any creator**, because it is compatible not only with the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` but also with all its specializations.
This makes it very flexible in its usage, serving solely as the bridge between data and visualization.

.. code-block:: python

    from dantro.plot_creators import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('x', 'y'))
    def my_plot(*, data: dict, hlpr: PlotHelper, **plot_kwargs):
        """A creator-averse plot function using the data transformation
        framework and the plot helper framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            hlpr: The associated plot helper.
            **plot_kwargs: Passed on to matplotlib.pyplot.plot
        """
        # Create a lineplot on the currently selected axis
        hlpr.ax.plot(data['x'], data['y'], **plot_kwargs)

        # Done! The plot helper saves the plot :tada:

Super simple, aye? :)

The corresponding plot configuration could look like this:

.. code-block:: yaml

    my_plot:
      creator: external

      # Select the plot function
      # ...

      # Select data
      select:
        x: data/MyModel/some/path/foo
        y:
          path: data/MyModel/some/path/bar
          transform:
            - mean: [!dag_prev ]
            - increment: [!dag_prev ]

      # Perform some transformation on the data
      transform: []

      # ... further arguments

For more detail on the syntax, see :ref:`above <pcr_ext_DAG_support>`.

While the plot *function* signature can remain as it is regardless of the chosen specialization of the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`, the plot *configuration* will differ for the specializations.
See :ref:`here <plot_data_selection_uni>` and :ref:`here <plot_data_selection_mv>` for more information.

.. note::

    This is the recommended way to define a plot function because it outsources a lot of the typical tasks (data selection and plot aesthetics) to dantro, allowing you to focus on implementing the bridge from data to visualization of the data.

    Using these features not only reduces the amount of code required in a plot function but also makes the plot function future-proof.
    We **highly** recommend to use *this* interface.




Other possible plot function signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without DAG framework
"""""""""""""""""""""
To not use the data transformation framework, simply omit the ``use_dag`` flag or set it to ``False`` in the decorator.
When not using the transformation framework, the ``creator_type`` should be specified, thus making the plot function bound to one type of creator.

.. code-block:: python

    from dantro import DataManager
    from dantro.plot_creators import is_plot_func, PlotHelper, ExternalPlotCreator

    @is_plot_func(creator_type=ExternalPlotCreator)
    def my_plot(dm: DataManager, *, hlpr: PlotHelper, **additional_kwargs):
        """A plot function using the plot helper framework.

        Args:
            dm: The DataManager object that contains all loaded data.
            hlpr: The associated plot helper
            **additional_kwargs: Anything else from the plot config.
        """
        # Select some data ...
        data = dm['foo/bar']

        # Create a lineplot on the currently selected axis
        hlpr.ax.plot(data)

        # When exiting here, the plot helper saves the plot.

.. note::

    The ``dm`` argument is only provided when *not* using the DAG framework.


Bare basics
"""""""""""
If you desire to do everything by yourself, you can disable the plot helper framework by passing ``use_helper=False`` to the decorator.
Subsequently, the ``hlpr`` argument is **not** passed to the plot function.

There is an even more basic version to do this, leaving out the :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator:

.. code-block:: python

    from dantro import DataManager

    def my_bare_basics_plot(dm: DataManager, *, out_path: str,
                            **additional_kwargs):
        """Bare-basics signature required by the ExternalPlotCreator.

        Args:
            dm: The DataManager object that contains all loaded data.
            out_path: The generated path at which this plot should be saved
            **additional_kwargs: Anything else from the plot config.
        """
        # Your code here ...

        # Save to the specified output path
        plt.savefig(out_path)

.. note::

    When using the bare basics version, you need to set the ``creator`` argument in the plot configuration in order for the plot manager to find the desired creator.



.. _pcr_ext_animations:

Animations
----------
With the :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` framework it is really simple to let your plot function support animation.

Say you have defined the following plot function:

.. code-block:: python

    from dantro.plot_creators import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('time_series',))
    def plot_some_data(*, data: dict,
                       hlpr: PlotHelper,
                       at_time: int,
                       **plot_kwargs):
        """Plots the data `time_series` for the selected time `at_time`."""
        # Via plot helper, perform a line plot of the data at the specified time
        hlpr.ax.plot(data['time_series'][at_time], **plot_kwargs)

        # Dynamically provide some information to the plot helper
        hlpr.provide_defaults('set_title',
                              title="My data at time {}".format(at_time))
        hlpr.provide_defaults('set_labels', y=dict(label="My data"))

To now make this function support animation, you only need to extend it by some
update function, register that function with the helper, and mark the plot function as supporting an animation:

.. code-block:: python

    from dantro.plot_creators import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('time_series',),
                  supports_animation=True)
    def plot_some_data(*, data: dict,
                       hlpr: PlotHelper,
                       at_time: int,
                       **plot_kwargs):
        """Plots the data `time_series` for the selected time `at_time`."""
        # Via plot helper, perform a line plot of the data at the specified time
        hlpr.ax.plot(data['time_series'][at_time], **plot_kwargs)

        # Dynamically provide some information to the plot helper
        hlpr.provide_defaults('set_title',
                              title="My data at time {}".format(at_time))
        hlpr.provide_defaults('set_labels', y=dict(label="My data"))

        # End of regular plot function
        # Define update function
        def update():
            """The animation update function: a python generator"""
            # Go over all available times
            for t, y_data in enumerate(data['time_series']):
                # Clear the plot and plot anew
                hlpr.ax.clear()
                hlpr.ax.plot(y_data, **plot_kwargs)

                # Set the title with current time step
                hlpr.invoke_helper('set_title',
                                   title="My data at time {}".format(t))
                # Set the y-label
                hlpr.provide_defaults('set_labels', y=dict(label="My data"))

                # Done with this frame. Yield control to the plot framework,
                # which will take care of grabbing the frame.
                yield

        # Register the animation update with the helper
        hlpr.register_animation_update(update)

Ok, so the following things happened:

    * The ``update`` function is defined
    * The ``update`` function is passed to helper via :py:meth:`dantro.plot_creators._plot_helper.PlotHelper.register_animation_update`
    * The plot function is marked ``supports_animation``

This is all that is needed to define an animation update for a plot.

There are a few things to look out for:

    * In order for the animation update actually being used, the feature needs to be enabled in the plot configuration.
      The behaviour of the animation is controlled via the ``animation`` key; in it, set the ``enabled`` flag.
    * The animation update function is expected to be a so-called Python Generator, thus using the yield keyword.
      For more information, have a look `here <https://wiki.python.org/moin/Generators>`_.
    * The file extension is taken care of by the ``PlotManager``, which is why it needs to be adjusted on the top level of the plot configuration, e.g.
      when storing the animation as a movie.
    * While whatever happens before the registration of the animation function is also executed, the animation update function should be build such as to also include the initial frame of the animation.
      This is to allow the plot function itself to be more flexible and the animation update not requiring to distinguish between initial frame and other frames.
    * In the example above, the ``set_labels`` helper has to be invoked for each frame as ``hlpr.ax.clear`` removes it.
      To avoid this, one could use the ``set_data`` method of the `Line2d <https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.lines.Line2D.html>`_ object, which is returned by `matplotlib.pyplot.plot <https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.plot.html>`_, to update the data.
      Depending on the objects used in your plot functions, there might exist a similar solution.

An example for an animation configuration is the following:

.. code-block:: yaml

  my_plot:
    # Regular plot configuration
    # ...

    # Specify file extension to use, with leading dot (handled by PlotManager)
    file_ext: .png        # change to .mp4 if using ffmpeg writer

    # Animation configuration
    animation:
      enabled: true       # false by default
      writer: frames      # which writer to use: frames, ffmpeg, ...
      writer_kwargs:      # additional configuration for each writer
        frames:           # passed to 'frames' writer
          saving:         # passed to Writer.saving method
            dpi: 254

        ffmpeg:
          init:           # passed to Writer.__init__ method
            fps: 15
          saving:
            dpi: 92
          grab_frame: {}  # passed to Writer.grab_frame and from there to savefig

      animation_update_kwargs:  {} # passed to the animation update function
