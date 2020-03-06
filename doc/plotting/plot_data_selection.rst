Automated Plot Data Selection
=============================

This page describes how the plot creators can make data for plotting available in a programmatic fashion.

Each plot creator is associated with a :py:class:`~dantro.data_mngr.DataManager` instance, which holds all the data that is currently available.
This data is usually made available to you such that you can select data which you can then pass on to whatever you use for plotting.

While manual selection directly from the data manager suffices for specific cases, automation is often desired.
The uniform interface of the :py:class:`~dantro.data_mngr.DataManager` paired with the :py:class:`~dantro.dag.TransformationDAG` framework makes automated data selection and transformation for plotting possible, while making available all the benefits of the :doc:`../data_io/transform` framework:

- Generic application of transformations on data
- Fully configuration-based interface
- Caching of computationally expensive results

This functionality is embedded at the level of the :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator`, making it available for all plot creators and allowing subclasses to taylor it to their needs.


.. contents::
   :local:
   :depth: 2

----


General remarks
---------------
This section holds information that is valid for *all* plot creators.

Enabling DAG usage
^^^^^^^^^^^^^^^^^^
To use the DAG for data selection, all you need to do is add the ``use_dag=True`` argument to a plot configuration.

.. code-block:: yaml

    # Some plot configuration file
    ---
    my_plot:
      use_dag: true

      # ... more arguments

.. _plot_creator_dag_args:

Arguments to control DAG behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You then have the following arguments available to control its behaviour:

    - ``select`` and ``transform``: select data and perform transformations on it, see :py:meth:`~dantro.dag.TransformationDAG.add_nodes`.
    - ``compute_only``: controls which tags are to be computed, see :py:meth:`~dantro.dag.TransformationDAG.compute`
    - ``dag_options``: passed to :py:class:`~dantro.dag.TransformationDAG` initialization, e.g. to control ``file_cache_defaults``.

.. note::

    If DAG usage is enabled, these arguments will be used *exclusively* for the DAG, i.e.: they are not available downstream in the plot creator.

The creation of the DAG and its computation is controlled by the chosen plot creator and can be specialized to suit that plot creator's needs.

.. _plot_creator_dag_usage:

Example
"""""""
Some example plot configuration to select some containers from the data manager, perform simple transformations on them and compute a ``result`` tag:

.. code-block:: yaml

    # Some plot configuration file
    ---
    my_plot:
      creator: my_creator

      # ... some plot arguments here ...

      # Data selection via DAG framework
      use_dag: true
      select:
        foo: some/path/foo
        bar:
          path: some/path/bar
          transform:
            - mean: [!dag_prev ]
            - increment: [!dag_prev ]
      transform:
        - add: [!dag_tag foo, !dag_tag bar]
          tag: result
      compute_only: [result]
      dag_options:
        file_cache_defaults:
          write: true
          read: true

        # ... other parameters here are passed on to TransformationDAG.__init__


DAG usage with :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`
----------------------------------------------------------------------------
The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` works exactly the same as in the general case.
After computation, the results are made available to the selected python plot function via the ``data`` keyword argument, which is a dictionary of the tags that were selected to be computed.

With this additional keyword argument being passed to the plot function, the plot function's signature also needs to support DAG usage, which makes it less comfortable to control DAG usage via the ``use_dag`` argument in the plot *configuration*.

Instead, the **best way** of implementing DAG support is via the :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator.
It provides the following arguments that have an effect on DAG usage:

- ``use_dag``: to enable or disable DAG usage. Disabled by default.
- ``required_dag_tags``: can be used to specify which tags are expected by the plot function; if these are not defined or not computed, an error will be raised.
- ``compute_only_required_dag_tags``: if the plot function defines required tags and ``compute_only is None``, the ``compute_only`` argument will be set such that only ``required_dag_tags`` are computed.
- ``pass_dag_object_along``: passes the :py:class:`~dantro.dag.TransformationDAG` object to the plot function as ``dag`` keyword argument.
- ``unpack_dag_results``: instead of passing the results as the ``data`` keyword argument, it unpacks the results dictionary, such that the tags can be specified directly in the plot function signature.
  Note that this puts some restrictions on tag names, prohibiting some characters as well as requiring that plot configuration parameters do not collide with the DAG results.
  This feature is best used in combination with ``required_dag_tags`` and ``compute_only_required_dag_tags`` enabled (which is the default).

Decorator usage puts all the relevant arguments for using the DAG framework into one place: the definition of the plot function.


.. _dag_generic_plot_func:

Defining a generic plot function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A plot function can then be defined via the following signature and the :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator:

.. code-block:: python

    @is_plot_func(use_dag=True)
    def my_plot_func(*, data: dict, hlpr: PlotHelper, **further_kwargs):
        """This is my custom plot function with preprocessed DAG data"""
        # ...

The only required arguments here are ``data`` and ``hlpr``.
The former contains all results from the DAG computation; the latter is the plot helper, which effectively is the interface to the visualization of the data.

**Importantly,** this makes the plot function averse to the specific choice of a creator: the plot function can be used with the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and from its specializations, :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator` and :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`.
In such cases, the ``creator_type`` should not be specified in the decorator, but it should be given in the plot configuration.


Specifying required tags
""""""""""""""""""""""""
If some specific tags are required, they can also be specified there:

.. code-block:: python

    @is_plot_func(use_dag=True, required_dag_tags=('x', 'y'))
    def simple_lineplot(*, data: dict, hlpr: PlotHelper, **plt_kwargs):
        """Creates a simple line plot for selected x and y data"""
        hlpr.ax.plot(data['x'], data['y'], **plt_kwargs)

The DAG can be configured in the same way as :ref:`in the general case <plot_creator_dag_usage>`.

.. hint::

    If you want the computed tags to be directly available in the plot function signature, use the ``unpack_dag_results`` flag in the decorator:

    .. code-block:: python

        @is_plot_func(use_dag=True, required_dag_tags=('x', 'y'),
                      unpack_dag_results=True)
        def simple_lineplot(*, x, y, hlpr: PlotHelper, **plt_kwargs):
            """Creates a simple line plot for selected x and y data"""
            hlpr.ax.plot(x, y, **plt_kwargs)


Accessing the :py:class:`~dantro.data_mngr.DataManager`
"""""""""""""""""""""""""""""""""""""""""""""""""""""""
As visible from the plot function above, the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` does **not** pass along the current :py:class:`~dantro.data_mngr.DataManager` instance as first positional argument (``dm``) when DAG usage is enabled.
This makes the plot function signature simpler and allows the creator-averse definition of plot functions while not restricting access to the data manager:

The data manager can still be accessed directly via the ``dm`` DAG tag.
Make sure to specify that it should be included, e.g. via ``compute_only`` or the ``required_dag_tags`` argument to the decorator.


.. _plot_data_selection_uni:

Special case: :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`, data selection and transformation has to occur based on data from the currently selected universe.
This is taken care of automatically by this creator: it dynamically sets the :py:meth:`~dantro.dag.TransformationDAG.select_base` property to the current universe, not requiring any further user action.
In effect, the ``select`` argument acts as if selections were to happen directly from the universe.

Except for the ``select_base`` and ``base_transform`` arguments, the full DAG interface is available via the :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`.

.. hint::

    To restore parts of the functionality of the already-in-use ``select_base`` and ``base_transform`` arguments, the ``select_path_prefix`` argument of :py:class:`~dantro.dag.TransformationDAG` can be used.
    It can be specified as part of ``dag_options`` and is prepended to all ``path`` arguments specified within ``select``.

Example
"""""""
The following suffices to define a :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`-based plot function:

.. code-block:: python

    @is_plot_func(creator_type=UniversePlotCreator, use_dag=True)
    def my_universe_plot(*, data: dict, hlpr: PlotHelper, **kwargs):
        """This is my custom universe plot function with DAG usage"""
        # ...

.. hint::

    To not restrict the plot function to a specific creator, using the :ref:`creator-averse plot function definition <dag_generic_plot_func>` is recommended, which omits the ``creator_type`` in the decorator and instead specifies it in the plot configuration.

The DAG can be configured in the same way as :ref:`in the general case <plot_creator_dag_usage>`.


.. _plot_data_selection_mv:

Special case: :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator` has a harder job: It has to select data from the whole multiverse subspace, apply transformations to it, and finally combine it, with optional further transformations following.

It does so fully within the DAG framework by building a separate DAG branch for each universe and bundling all of them into a transformation that combines the data.
This happens via the ``select_and_combine`` argument.

**Important:** The ``select_and_combine`` argument behaves differently to the ``select`` argument of the DAG interface!
This is because it has to accomodate various further configuration parameters that control the selection of universes and the multidimensional combination of the selected data.

The ``select_and_combine`` argument expects the following keys:

- ``fields``: all keys given here will appear as tags in the results dictionary.
  The values of these keys are dicts that contain the same parameters that can also be given to the ``select`` argument of the DAG interface.
  In other words: paths you would like to select form within each universe should be specified at ``select_and_combine.fields.<result_tag>.path`` rather than at ``select.<result_tag>.path``.
- ``base_path`` (optional): if given, this path is prepended to all paths given under ``fields``
- ``combination_method`` (optional, default: ``concat``): how to combine the selected and transformed data from the various universes. Available parameters:

    - ``concat``: attempts to preserve data types but is only possible if the universes fill a hypercube without holes
    - ``merge``: which is always possible, but leads to the data type falling back to float. Missing data will be ``np.nan`` in the results.

  The combination method can also be specified for each tag under ``select_and_combine.<result_tag>.combination_method``.
- ``subspace`` (optional): which multiverse subspace to work on. This is evaluated fully by the ``paramspace.ParamSpace.activate_subspace`` method.
  The subspace can also be specified for each tag under ``select_and_combine.<result_tag>.subspace``.

Remarks
"""""""
- The select operations on each universe set the ``omit_tag`` flag in order not to create a flood of only-internally-used tags. Setting tags manually here does not make sense, as the tag names would collide with tags from other universe branches.
- File caching is hard-coded to be disabled for the initial select operation and for the operation that attaches the parameter space coordinates to it. This behaviour cannot be influenced.
- The best place to cache is the result of the combination method.
- The regular ``select`` argument is still available, but it is applied only *after* the ``select_and_combine``-defined nodes were added and it does only act *globally*, i.e. not on *each* universe.
- The ``select_path_prefix`` argument to :py:class:`~dantro.dag.TransformationDAG` is not allowed for the :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`. Use the ``select_and_combine.base_path`` argument instead.

Example
"""""""
A :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`-based plot function can be implemented like this:

.. code-block:: python

    @is_plot_func(creator_type=MultiversePlotCreator, use_dag=True)
    def my_multiverse_plot(*, data: dict, hlpr: PlotHelper, **kwargs):
        """This is my custom multiverse plot function with DAG usage"""
        # ...

.. hint::

    To not restrict the plot function to a specific creator, using the :ref:`creator-averse plot function definition <dag_generic_plot_func>` is recommended, which omits the ``creator_type`` in the decorator and instead specifies it in the plot configuration.

An associated plot configuration might look like this:

.. code-block:: yaml

    ---
    my_plot:
      # ... some plot arguments here ...

      # Data selection via DAG framework
      select_and_combine:
        fields:
          foo: some/path/foo
          bar:
            path: some/path/bar
            transform:
              - mean: [!dag_prev ]
              - increment: [!dag_prev ]

        combination_method: concat  # can be `concat` (default) or `merge`
        subspace: ~                 # some subspace selection

      transform:
        - add: [!dag_tag foo, !dag_tag bar]
          tag: result


Full DAG configuration interface for multiverse selection
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example for all options available in the :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`.


.. code-block:: yaml

    # Full DAG specification for multiverse selection
    ---
    my_plot:
      # ... some plot arguments here ...

      # DAG parameters
      # Selection from multiple universes with subsequent combination
      select_and_combine:
        fields:
          # Define a tag 'foo' that will use the defaults defined directly on
          # the `select_and_combine` level
          foo: foo                       # `base_path` will be prepended here
                                         # resulting in: some/path/foo

          # Define a tag 'bar' that overwrites some of the defaults
          bar:
            path: bar
            subspace:                    # only use universes from a subspace
              seed: [0, 10]
              my_param: [-42., 42.]
            combination_method: merge    # overwriting default specified below
            combination_kwargs:          # passed to combine transformation
              file_cache:
                read: true
                write:
                  enabled: true
                  # Configure the file cache to only be written if this
                  # operation took a large amount of time.
                  min_cumulative_compute_time: 20.
            transform:
              - mean: !dag_prev
              - increment: [!dag_prev ]
              - some_op_with_kwargs:
                  data: !dag_prev
                  foo: bar
                  spam: 42
              - operation: my_operation
                args: [!dag_prev ]
                file_cache: {}      # can configure file cache here

        base_path: some_path        # if given, prepended to `path` in `fields`

        # Default arguments, can be overwritten in each `fields` entry
        combination_method: concat  # can be `concat` (default) or `merge`
        subspace: ~                 # some subspace selection

      # Additional selections, now based on `dm` tag
      select: {}

      # Additional transformations; all tags from above available here
      transform: []

      # Other DAG-related parameters: `compute_only`, `dag_options`
      # ...

.. note::

    This does not include *all* possible options for DAG configuration, but focusses on those options added by :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator` to work with multiverse data, e.g. ``subspace``, ``combination_kwargs``.

    For other arguments, see :ref:`dag_transform_full_syntax_spec`.
