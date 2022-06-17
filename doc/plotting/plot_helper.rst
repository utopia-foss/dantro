.. default-domain:: dantro.plot.plot_helper

.. _plot_helper:

The :py:class:`.PlotHelper`
---------------------------

The aim of the :py:class:`.PlotHelper` is to let the plot functions focus on what cannot easily be automated: being the bridge between some selected or :ref:`transformed <pcr_pyplot_DAG_support>` data and its visualization.
The plot function should not have to concern itself with things like plot aesthetics, as that can easily be automated.

The :py:class:`.PlotHelper` can make your life easier by quite a lot as it already takes care of setting up and saving a figure and makes large parts of the :py:mod:`matplotlib.pyplot` interface accessible via the plot configuration.
That way, you don’t need to touch Python code for trivial tasks like changing the axis limits.
But even more advanced tasks, such as creating an animation, are conveniently done using this framework.

Due to the :py:class:`.PlotHelper` focussing on the :py:mod:`~matplotlib.pyplot` interface, it is only accessible via the :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`, see :ref:`pcr_pyplot_helper`.

Most importantly, it will make your plots future-proof and let them profit from upcoming features.
A glimpse of that can be seen in how easy it is to implement an animated plot, see :ref:`below <pcr_pyplot_animations>`.

To learn, how you can enable the PlotHelper in your plot function, checkout the section on :ref:`implementing plot functions <pcr_pyplot_implement_plot_funcs>`.

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
Aside from specifying values in the configuration, helpers can also be dynamically (re-)configured from within the plot function using :py:meth:`.PlotHelper.provide_defaults` or be invoked directly using :py:meth:`~.PlotHelper.invoke_helper`.
To ensure that helpers stay disabled, regardless of configuration, you can call :py:meth:`~.PlotHelper.mark_disabled` inside the plot function.

.. hint::

    The syntax for each individual helper is in large parts equivalent to matplotlib's `pyplot interface <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot>`_.
    It is however wrapped and simplified in some cases, e.g. by using just ``x`` and ``y`` as arguments and gathering such functionality under one helper.

    If you get it wrong, the error message aims to be helpful: it provides the full signature and docstring of the invoked helper such that you can adjust the parameters to the required format.

    Thus, trial and error is a useful initial approach before digging into the :py:class:`.PlotHelper` API reference.

Furthermore, notice how you can combine the capabilities of the plot helper framework with the ability to :ref:`set the plot style <pcr_pyplot_style>`.

Available helpers
^^^^^^^^^^^^^^^^^

The following helper methods are available:

.. ipython::

    In [1]: from dantro.plot import PlotHelper

    In [2]: hlpr = PlotHelper(out_path="~/my_output_directory")

    In [3]: print("\n".join(hlpr.available_helpers))


Additionally, there are "special" helpers that help with setting up and storing a figure:

- :py:meth:`.PlotHelper.setup_figure`
- :py:meth:`.PlotHelper.save_figure`

.. note::

    By default, helpers are regarded as **axis-level helpers**, as they operate on a single axis object.

    However, there are some helpers that may *only* be used on the whole figure, so-called **figure-level helpers** (e.g. ``set_suptitle`` and ``set_figlegend``).


Axis-specific helper configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`.PlotHelper` is not restricted to a single axis, but it can manage multiple axes aranged on a grid.
A possible plot configuration with axis-specific helpers could look as follows:

.. code-block:: yaml

  my_plot:
    # ...

    # Configure the plot helpers
    helpers:
      setup_figure:
        ncols: 2
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

* The :py:meth:`.PlotHelper.setup_figure` helper sets up a figure with with two subfigures that are accessible via the coordinate pairs ``[0, 0]`` and ``[1, 0]``.
* The ``set_limits`` helper is applied to all existing axes.
* Helpers for specific axes can be configured by passing an ``axis_specific`` dictionary.
  In the plot function, you can then switch axes using the :py:meth:`.PlotHelper.select_axis` method (the ``[0, 0]`` axis is selected initially).

.. note::

    The keys for the ``axis_specific`` configuration are arbitrary; the axes are defined solely by the internal ``axis`` entries.
    While this requires to specify a name for the axis, it also allows convenient recursive updating; thus, it is advisable to choose a somewhat meaningful name.

Alternatively, the axes match can be defined via the update key directly, for instance:

.. code-block:: yaml

  my_plot:
    # ...
    helpers:
      setup_figure:
        ncols: 2
        sharey: True
      axis_specific:
        [0, 0]:
          set_title:
            title: This is my left plot
        [1, 0]:
          axis: [1, 0]
          set_title:
            title: This is my right plot

.. hint::

    This syntax also supports simple pattern matching to apply axis-specific updates to plots from a whole row or column.
    To span over a row or column, simply replace the entry by ``None`` (in YAML: ``~``).

    For instance, ``[0, ~]`` matches all subplots in the *first* column and ``[~, -1]`` matches the whole bottom row.


Remarks
^^^^^^^

* Plot helpers can also be explicitly disabled via the configuration:

    .. code-block:: yaml

        helpers:
          set_labels:
            enabled: false
            # ...

* By default, an axis-level plot helper is not invoked on an axis that is empty, i.e. an axis that has no artists associated with it.
  This is to reduce errors that stem from e.g. attempting to extract limit values from an axis without data.
  If invocation is required nevertheless, it can be explicitly allowed via the ``skip_empty_axes`` configuration key:

    .. code-block:: yaml

        helpers:
          set_limits:
            skip_empty_axes: false
            # ...


.. _plot_helper_spec:

Specializing the helper
^^^^^^^^^^^^^^^^^^^^^^^
The dantro :py:class:`.PlotHelper` already provides a default set of helpers that provide access to most of the matplotlib interface.
If you need any additional customized helpers, you can easily add new methods to a specialization of the helper:

.. code-block:: python

  import dantro.plot.creators

  class MyPlotHelper(dtr.plot_creators.PlotHelper):
      """A specialization of the dantro ``PlotHelper`` which can be used to add
      additional helper methods.
      """
      # You can add new helper methods here, prefixed with _hlpr_

Note that you will have to communicate this new plot helper type to the creator by setting :py:const:`~dantro.plot.creators.pyplot.PyPlotCreator.PLOT_HELPER_CLS`.
