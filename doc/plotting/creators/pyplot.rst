.. default-domain:: dantro.plot.creators.pyplot

.. _pcr_pyplot:

The :py:class:`.PyPlotCreator`
==============================

The :py:class:`.PyPlotCreator` focusses on creating plots using :py:mod:`matplotlib.pyplot`.

Like the :py:class:`~dantro.plot.creators.base.BasePlotCreator`, it relies on the plots being defined in a so-called *plot function*, which can be retrieved from importable modules or even from some file path.
These plot functions are meant to provide a bridge between the selected and transformed data and their visualization.
The :py:class:`.PyPlotCreator` aims to make this process as smooth as possible by implementing a number of automations that reduce boilerplate code:

- The :ref:`plot helper interface <pcr_pyplot_helper>` provides an interface to :py:mod:`matplotlib.pyplot` and allows configuration-based manipulation of the axes limits, scales, and many other structural elements of a plot.
- With :ref:`style contexts <pcr_pyplot_style>`, plot aesthetics can be controlled right from the plot configuration, making consistent plotting styles more accessible.
- The integration of the :py:mod:`matplotlib.animation` framework allows to easily implement plot functions that generate animation output.


.. hint::

    There are further specializations of the :py:class:`.PyPlotCreator` that make plotting of data originating from parameter sweeps easier.
    See :ref:`pcr_psp` or the :ref:`creator overview <plot_creators>`.

.. note::

    Prior to dantro 0.18, this plot creator used to be called ``ExternalPlotCreator``, highlighting its ability to load external modules.

.. contents::
   :local:
   :depth: 3

----


.. _pcr_pyplot_helper:

The :py:class:`~dantro.plot.plot_helper.PlotHelper`
---------------------------------------------------
The :py:class:`.PyPlotCreator` allows to automate many of the :py:mod:`matplotlib.pyplot` function calls that would usually have to be part of the plot function itself.
Instead, the :py:class:`~dantro.plot.plot_helper.PlotHelper` takes up this task and provides a config-accessible bridge to the matplotlib interface.

See :ref:`here <plot_helper>` for more information on the plot helper framework.


.. _pcr_pyplot_style:

Adjusting a Plot's Style
------------------------
Using the ``style`` keyword, matplotlib RC parameters can be configured fully via the plot configuration; no need to touch the code.
Basically, this allows setting the :py:data:`matplotlib.rcParams` and makes the matplotlib stylesheets (:py:mod:`matplotlib.style`) available.

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


For the ``base_style`` entry, choose the name of a `matplotlib stylesheet <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
For valid RC parameters, see the `matplotlib customization documentation <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_.

.. hint::

    Even the `axes property cycle <https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html>`_, i.e. the ``axes.prop_cycle`` RC parameter, can be adjusted in this way.
    For example, to use a Tab20-based color cycle, specify:

    .. code-block:: yaml

        my_plot:
          # ...
          style:
            axes.prop_cycle: "cycler('color', ['1f77b4', 'aec7e8', 'ff7f0e', 'ffbb78', '2ca02c', '98df8a', 'd62728', 'ff9896', '9467bd', 'c5b0d5', '8c564b', 'c49c94', 'e377c2', 'f7b6d2', '7f7f7f', 'c7c7c7', 'bcbd22', 'dbdb8d', '17becf', '9edae5'])"

    The full syntax is supported here, including ``+`` and ``*`` operators between ``cycler(..)`` definitions.




.. _pyplot_plot_func:

Implementing Plot Functions
---------------------------
This section details how to implement plot functions for the :py:class:`.PyPlotCreator`, making use of its specializations.


.. _pyplot_func_recommended:

Recommended plot function signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommend plot function signature for this creator is not that different from the :ref:`general one <plot_func_signature>`:
It also makes use of the :ref:`data transformation framework <plot_creator_dag>` (implemented by :ref:`the parent class <pcr_base_DAG_support>`).

*Additionally*, however, it uses the :ref:`plot helper framework <plot_helper>` which requires that the plot function can handle an additional argument, ``hlpr``.
This :py:class:`~dantro.plot.plot_helper.PlotHelper` is the bridge to :py:mod:`matplotlib.pyplot` and thus also needs to be used to invoke any plot-related commands:

.. testcode::

    from dantro.plot import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=("x", "y"))
    def my_plot(*, data: dict, hlpr: PlotHelper, **plot_kwargs):
        """A creator-averse plot function using the data transformation
        framework and the plot helper framework.

        Args:
            data: The selected and transformed data, containing specified tags.
            hlpr: The associated plot helper.
            **plot_kwargs: Passed on to matplotlib.pyplot.plot
        """
        # Create a lineplot on the currently selected axis
        hlpr.ax.plot(data["x"], data["y"], **plot_kwargs)

        # Done! The plot helper saves the plot :tada:

Super simple, aye? :)

In the case of the :py:class:`.PyPlotCreator`, such a plot function can be **averse to any creator**, because it is compatible not only with the :py:class:`.PyPlotCreator` but also with :ref:`derived creators <pcr_psp>`.
This makes it very flexible in its usage, serving solely as the bridge between data and their visualization:
For that reason, the decorator does not specify a ``creator`` argument, but the plot configuration does.
The corresponding plot configuration could then look like this:

.. code-block:: yaml

    my_plot:
      creator: pyplot

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

.. note::

    While the plot *function* signature can remain as it is regardless of the chosen specialization of the :py:class:`.PyPlotCreator`, the plot *configuration* will differ for the specializations.
    See :ref:`here <plot_data_selection_uni>` and :ref:`here <plot_data_selection_mv>` for more information.

.. note::

    This is the **recommended way to define a plot function** because it outsources a lot of the typical tasks (data selection and plot aesthetics) to dantro, allowing you to focus on implementing the bridge from data to visualization of the data.

    Using these features not only reduces the amount of code required in a plot function but also makes the plot function future-proof.
    We **highly** recommend to use *this* interface.


Other possible plot function signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Without data transformation framework
"""""""""""""""""""""""""""""""""""""
There is the option to not using the transformation framework for data selection while still profiting from the plot helper.
Simply use the :ref:`plot function decorator <is_plot_func_decorator>` without passing ``use_dag``:

.. testcode::

    from dantro import DataManager
    from dantro.plot import is_plot_func, PyPlotCreator, PlotHelper

    @is_plot_func(creator=PyPlotCreator)
    def my_plot(
        *, dm: DataManager, hlpr: PlotHelper, **additional_plot_kwargs
    ):
        """A simple plot function using the plot helper framework.

        Args:
            dm: The loaded data tree.
            hlpr: The plot helper, taking care of setting up the figure and
                saving the plot.
            **additional_kwargs: Anything else from the plot config.
        """
        # Select some data ...
        data = dm["foo/bar"]

        # Create the plot
        hlpr.ax.plot(data)

        # Done. The helper will save the plot after the plot function returns.

.. note::

    The ``dm`` argument is only provided when *not* using the DAG framework.

.. hint::

    To omit the helper as well, pass ``use_helper=False`` to the decorator.
    In that case you will also have to take care of saving the plot to the ``out_path`` provided as argument to the plot function.


Bare basics
"""""""""""
If you do not want to use the decorator either, the signature is the same as :ref:`in the case of the base class <plot_func_bare_signature>`.





.. _pcr_pyplot_animations:

Animations
----------
With the :py:class:`~dantro.plot.plot_helper.PlotHelper` framework it is really simple to let your plot function support animation.

Say you have defined the following plot function:

.. testcode::

    from dantro.plot import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('time_series',))
    def plot_some_data(*, data: dict,
                       hlpr: PlotHelper,
                       at_time: int,
                       **plot_kwargs):
        """Plots the data ``time_series`` for the selected time ``at_time``."""
        # Via plot helper, perform a line plot of the data at the specified time
        hlpr.ax.plot(data['time_series'][at_time], **plot_kwargs)

        # Dynamically provide some information to the plot helper
        hlpr.provide_defaults('set_title',
                              title="My data at time {}".format(at_time))
        hlpr.provide_defaults('set_labels', y=dict(label="My data"))

To now make this function support animation, you only need to extend it by some
update function, register that function with the helper, and mark the plot function as supporting an animation:

.. testcode::

    from dantro.plot import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('time_series',),
                  supports_animation=True)
    def plot_some_data(*, data: dict,
                       hlpr: PlotHelper,
                       at_time: int,
                       **plot_kwargs):
        """Plots the data ``time_series`` for the selected time ``at_time``."""
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
    * The ``update`` function is passed to helper via :py:meth:`dantro.plot.plot_helper.PlotHelper.register_animation_update`
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
      To avoid this, one could use the ``set_data`` method of the `Line2d <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html>`_ object, which is returned by `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_, to update the data.
      Depending on the objects used in your plot functions, there might exist a similar solution.

.. warning::

    If it is not possible or too complicated to let the animation update function set the data directly, one typically has to redraw the axis or the whole figure.

    In such cases, two important steps need to be taken in order to ensure correct functioning of the :py:meth:`~dantro.plot.plot_helper.PlotHelper`:

        * Specifying the ``invoke_helpers_before_grab`` flag when calling :py:meth:`~dantro.plot.plot_helper.PlotHelper.register_animation_update`, such that the helpers are invoked before grabbing each frame.
        * If using a new figure object and/or axes grid, that needs to be communicated to the :py:meth:`~dantro.plot.plot_helper.PlotHelper` via :py:meth:`~dantro.plot.plot_helper.PlotHelper.attach_figure_and_axes`.

    For example implementations of such cases, refer to the plot functions specified in the :py:mod:`dantro.plot.funcs.generic` module.

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

      animation_update_kwargs: {}  # passed to the animation update function


.. _pcr_pyplot_animation_mode_switching:

Dynamically entering/exiting animation mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some situations, one might want to dynamically determine if an animation should be carried out or not.
For instance, this could be dependent on whether the dimensionality of the data requires another representation mode (the animation) or not.

For that purpose, the :py:class:`~dantro.plot.plot_helper.PlotHelper` supplies two methods to enter or exit animation mode, :py:meth:`~dantro.plot.plot_helper.PlotHelper.enable_animation` and :py:meth:`~dantro.plot.plot_helper.PlotHelper.disable_animation`.
When these are invoked, the plot function is *directly* left, the :py:class:`.PyPlotCreator` enables or disables the animation, and the plot function is invoked anew.

A few remarks:

    * The decision on entering or exiting animation mode should ideally occur as early as possible within a plot function.
    * Repeatedly switching between modes is *not* possible.
      You should implement the logic for entering or exiting animation mode in such a way, that flip-flopping between the two modes is not possible.
    * The ``animation`` parameters need to be given if *entering* into animation mode is desired.
      In such cases, ``animation.enabled`` key should be set to ``False``.
    * The :py:class:`~dantro.plot.plot_helper.PlotHelper` instance of the first plot function invocation will be discarded and a new instance will be created for the second invocation.

A plot function could then look like this:

.. testcode::

    from dantro.plot import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True, required_dag_tags=('nd_data',),
                  supports_animation=True)
    def plot_nd(*, data: dict, hlpr: PlotHelper,
                x: str, y: str, frames: str=None):
        """Performs an (animated) heatmap plot of 2D or 3D data.

        The ``x``, ``y``, and ``frames`` arguments specify which data dimension
        to associate with which representation.
        If the ``frames`` argument is not given, the data needs to be 2D.
        """
        d = data['nd_data']

        if frames and d.ndim == 3:
            hlpr.enable_animation()
        elif not frames and d.ndim == 2:
            hlpr.disable_animation()
        else:
            raise ValueError("Need either 2D data without the ``frames`` "
                             "argument, or 3D data with the ``frames`` "
                             "argument specified!")

        # Do the 2D plotting for x and y dimensions here
        # ...

        def update():
            """Update the heatmap using the ``frames`` argument"""
            # ...

        hlpr.register_animation_update(update)


.. _pcr_pyplot_specializing:

Specializing :py:class:`.PyPlotCreator`
---------------------------------------
This is basically the same as in :ref:`the base class <pcr_base_specializing>` with the additional ability to specialize the plot helper.

For specializing the :py:class:`~dantro.plot.plot_helper.PlotHelper`, see :ref:`here <plot_helper_spec>` and then set the :py:attr:`.PyPlotCreator.PLOT_HELPER_CLS` class variable accordingly.

.. note::

    For an operational example in a more complex framework setting, see `the specialization used in the utopya project <https://gitlab.com/utopia-project/utopya/-/blob/main/utopya/eval/plotcreators.py>`_.
