.. _plot_creator_dag:

Plot Data Selection
===================

This page describes how the plot creators can make data for plotting available in a programmatic fashion.

Each plot creator is associated with a :py:class:`~dantro.data_mngr.DataManager` instance, which holds all the data that is currently available.
This data is usually made available to you such that you can select data which you can then pass on to whatever you use for plotting.

While manual selection directly from the data manager suffices for specific cases, automation is often desired.
The uniform interface of the :py:class:`~dantro.data_mngr.DataManager` paired with the :py:class:`~dantro.dag.TransformationDAG` framework makes automated data selection and transformation for plotting possible while making available all the benefits of the :doc:`../data_io/transform` framework:

- Generic application of transformations on data
- Fully configuration-based interface
- Caching of computationally expensive results

This functionality is embedded at the level of the :py:class:`~dantro.plot.creators.base.BasePlotCreator`, making it available for all plot creators and allowing subclasses to tailor it to their needs.

Additionally, :ref:`result placeholders <dag_result_placeholder>` can be specified inside the plot configuration, thus allowing to use transformation results not only for data selection, but also for programmatically determining other configuration parameters.


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
    - ``dag_options``: passed to :py:class:`~dantro.dag.TransformationDAG` initialization, e.g. to control ``file_cache_defaults``, ``verbosity``, or adding transformations via the ``define`` interface, see :ref:`dag_define`.

.. note::

    If DAG usage is enabled, these arguments will be used *exclusively* for the DAG, i.e.: they are not available downstream in the plot creator.

.. hint::

    To use :ref:`meta-operations <dag_meta_ops>` for plot data selection, define them under the ``dag_options.meta_operations`` key of a plot configuration.

    Same for adding nodes via the ``define`` interface (see :ref:`dag_define`), which is also only available via ``dag_options.define``.

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
        define:
          foo: bar
        verbosity: 3  # to show more profiling statistics (default: 1)
        file_cache_defaults:
          write: true
          read: true

        # ... other parameters here are passed on to TransformationDAG.__init__



.. _plot_creator_dag_object_cache:

DAG object caching
^^^^^^^^^^^^^^^^^^
For very complex data transformation sequences, DAGs can have many hundreds of thousands of nodes.
In those cases, parsing the DAG configuration and creating the corresponding objects can be time-consuming and begin to noticeably prolong the plotting procedure.

To remedy this, the plotting framework implements memory-caching of ``TransformationDAG`` objects such that they can be re-used across multiple plots or repeated invocation of the same plot.
The cache is used if the DAG-related configuration parameters (``transform``, ``select``, ...) are equal, i.e. have equal results when serialized using ``repr``.
In other words: if plots use the same data selection arguments, thus creating identical DAGs, the cache can be used.

Multiple aspects of caching can be controlled using the ``dag_object_cache`` parameter, passed via ``dag_options`` (see below):

* ``read``: whether to read from the cache (default: false)
* ``write``: whether to write from the cache (default: false)
* ``use_copy``: whether to read and write a deep copy of the ``TransformationDAG`` object to the cache (default: true).
* ``clear``: if set, will remove all objects from the cache (after reading from it) and trigger garbage collection (default: false)
* ``collect_garbage``: can be used to separately control garbage collection, e.g. to suppress it despite ``clear`` having been passed.

.. warning::

    Only use ``use_copy: false`` if you can be certain that plot functions do not change the object; this would create side effects that may be very hard to track down.

.. note::

    The ``clear`` option will also invoke general garbage collection (if not explicitly disabled).
    This will free up memory ... but it may also take some time.

Example
"""""""

.. code-block:: yaml

    # Some plot configuration file
    ---
    my_plot:
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

      # Enable DAG object caching
      dag_options:
        dag_object_cache:
          read: true
          write: true

          # Other parameters (and their default values)
          # use_copy: true       # true:  cache a deep copy of the object
          # clear: false         # true:  clears the object cache and invokes
                                 #        garbage collection
          # collect_garbage: ~   # true:  invokes garbage collection
                                 # false: suppresses garbage collection even
                                 #        if `clear` was set

    my_other_plot_using_the_cache:
      based_on: my_plot          # --> identical DAG arguments (if not overwritten below)

      # ... some plot arguments ...


DAG usage with :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`
----------------------------------------------------------------------------
The :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` works exactly the same as in the general case.
After computation, the results are made available to the selected python plot function via the ``data`` keyword argument, which is a dictionary of the tags that were selected to be computed.

With this additional keyword argument being passed to the plot function, the plot function's signature also needs to support DAG usage, which makes it less comfortable to control DAG usage via the ``use_dag`` argument in the plot *configuration*.

Instead, the **best way** of implementing DAG support is via the :py:func:`~dantro.plot.utils.is_plot_func.is_plot_func` decorator.
It provides the following arguments that affect DAG usage:

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
A plot function can then be defined via the following signature and the :py:func:`~dantro.plot.utils.is_plot_func.is_plot_func` decorator:

.. testcode::

    from dantro.plot import is_plot_func, PlotHelper

    @is_plot_func(use_dag=True)
    def my_plot_func(*, data: dict, hlpr: PlotHelper, **further_kwargs):
        """This is my custom plot function with preprocessed DAG data"""
        # ...
        pass

The only required arguments here are ``data`` and ``hlpr``.
The former contains all results from the DAG computation; the latter is the plot helper, which effectively is the interface to the visualization of the data.

**Importantly,** this makes the plot function averse to the specific choice of a creator: the plot function can be used with the :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` and from its specializations, :py:class:`~dantro.plot.creators.psp.UniversePlotCreator` and :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`.
In such cases, the ``creator_type`` should not be specified in the decorator, but it should be given in the plot configuration.


Specifying required tags
""""""""""""""""""""""""
If some specific tags are required, they can also be specified there:

.. testcode::

    @is_plot_func(use_dag=True, required_dag_tags=('x', 'y'))
    def simple_lineplot(*, data: dict, hlpr: "PlotHelper", **plt_kwargs):
        """Creates a simple line plot for selected x and y data"""
        hlpr.ax.plot(data['x'], data['y'], **plt_kwargs)

The DAG can be configured in the same way as :ref:`in the general case <plot_creator_dag_usage>`.

.. hint::

    If you want the computed tags to be directly available in the plot function signature, use the ``unpack_dag_results`` flag in the decorator:

    .. testcode::

        @is_plot_func(use_dag=True, required_dag_tags=('x', 'y'),
                      unpack_dag_results=True)
        def simple_lineplot(*, x, y, hlpr: "PlotHelper", **plt_kwargs):
            """Creates a simple line plot for selected x and y data"""
            hlpr.ax.plot(x, y, **plt_kwargs)


Accessing the :py:class:`~dantro.data_mngr.DataManager`
"""""""""""""""""""""""""""""""""""""""""""""""""""""""
As visible from the plot function above, the :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` does **not** pass along the current :py:class:`~dantro.data_mngr.DataManager` instance as first positional argument (``dm``) when DAG usage is enabled.
This makes the plot function signature simpler and allows the creator-averse definition of plot functions while not restricting access to the data manager:

The data manager can still be accessed directly via the ``dm`` DAG tag.
Make sure to specify that it should be included, e.g. via ``compute_only`` or the ``required_dag_tags`` argument to the decorator.


.. _plot_data_selection_uni:

Special case: :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`, data selection and transformation has to occur based on data from the currently selected universe.
This is taken care of automatically by this creator: it dynamically sets the :py:meth:`~dantro.dag.TransformationDAG.select_base` property to the current universe, not requiring any further user action.
In effect, the ``select`` argument acts as if selections were to happen directly from the universe.

Except for the ``select_base`` and ``base_transform`` arguments, the full DAG interface is available via the :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`.

.. hint::

    To restore parts of the functionality of the already-in-use ``select_base`` and ``base_transform`` arguments, the ``select_path_prefix`` argument of :py:class:`~dantro.dag.TransformationDAG` can be used.
    It can be specified as part of ``dag_options`` and is prepended to all ``path`` arguments specified within ``select``.

Example
"""""""
The following suffices to define a :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`-based plot function:

.. testcode::

    from dantro.plot import UniversePlotCreator

    @is_plot_func(creator_type=UniversePlotCreator, use_dag=True)
    def my_universe_plot(*, data: dict, hlpr: PlotHelper, **kwargs):
        """This is my custom universe plot function with DAG usage"""
        # ...
        pass

.. hint::

    To not restrict the plot function to a specific creator, using the :ref:`creator-averse plot function definition <dag_generic_plot_func>` is recommended, which omits the ``creator_type`` in the decorator and instead specifies it in the plot configuration.

The DAG can be configured in the same way as :ref:`in the general case <plot_creator_dag_usage>`.


.. _plot_data_selection_mv:

Special case: :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` has a harder job: It has to select data from the whole multiverse subspace, apply transformations to it, and finally combine it, with optional further transformations following.

It does so fully within the DAG framework by building a separate DAG branch for each universe and bundling all of them into a transformation that combines the data.
This happens via the ``select_and_combine`` argument.

**Important:** The ``select_and_combine`` argument behaves differently to the ``select`` argument of the DAG interface!
This is because it has to accommodate various further configuration parameters that control the selection of universes and the multidimensional combination of the selected data.

The ``select_and_combine`` argument expects the following keys:

- ``fields``: all keys given here will appear as tags in the results dictionary.
  The values of these keys are dictionaries that contain the same parameters that can also be given to the ``select`` argument of the DAG interface.
  In other words: paths you would like to select from within each universe should be specified at ``select_and_combine.fields.<result_tag>.path`` rather than at ``select.<result_tag>.path``.
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
- File caching is hard-coded to be disabled for the initial select operation and for the operation that attaches the parameter space coordinates to it. This behavior cannot be influenced.
- The best place to cache is the result of the combination method.
- The regular ``select`` argument is still available, but it is applied only *after* the ``select_and_combine``-defined nodes were added and it does only act *globally*, i.e. not on *each* universe.
- The ``select_path_prefix`` argument to :py:class:`~dantro.dag.TransformationDAG` is not allowed for the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`. Use the ``select_and_combine.base_path`` argument instead.

Example
"""""""
A :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`-based plot function can be implemented like this:

.. testcode::

    from dantro.plot import MultiversePlotCreator

    @is_plot_func(creator_type=MultiversePlotCreator, use_dag=True)
    def my_multiverse_plot(*, data: dict, hlpr: PlotHelper, **kwargs):
        """This is my custom multiverse plot function with DAG usage"""
        # ...
        pass

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

        combination_method: concat  # can be ``concat`` (default) or ``merge``
        subspace: ~                 # some subspace selection

      transform:
        - add: [!dag_tag foo, !dag_tag bar]
          tag: result


.. _plot_data_selection_mv_missing_data:

Handling missing data
"""""""""""""""""""""
In some cases, the :py:class:`~dantro.groups.psp.ParamSpaceGroup` associated with the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` might miss some states.
This can happen, for instance, if the to-be-plotted data is the result of a simulation for each point in parameter space and the simulation was stopped before visiting all these points.
In such a case, ``select_and_combine`` will typically fail.

Another reason for errors during this operation may be that the data structures between the different points in parameter space are different, such that a valid path within one :py:class:`~dantro.groups.psp.ParamSpaceStateGroup` (or: "universe") is *not* a valid path in another.

To be able to plot the partial data in both of these cases, this plot creator makes use of :ref:`the error handling feature in the data transformation framework <dag_error_handling>`.
It's as simple as adding the ``allow_missing_or_failing`` key to ``select_and_combine``:

.. literalinclude:: ../../tests/cfg/dag_plots.yml
    :language: yaml
    :start-after: ### Start -- mv_missing_data
    :end-before:  ### End ---- mv_missing_data
    :dedent: 6

This option kicks in when any of the following scenarios occur:

- A universe from the selected subspace is missing altogether
- The ``getitem`` operation for the given ``path`` within a universe fails
- Any operation within ``transform`` fails

In any of these cases, the data for the whole universe is discarded.
Instead, an empty ``xr.Dataset`` with the coordinates of that universe is used as fallback, with the following effect:
The corresponding coordinates will be present in the final ``xr.Dataset``, but they contain no data (or NaNs).
The latter is also the reason why the ``merge`` combination method is required here.

.. note::

    The rationale behind this behavior is that coordinate information is valuable, as it shows which data *would have been* available.
    If desired, null-like data can be dropped afterwards using the ``.dropna`` operation.

    In case of missing data, the error message will come from the ``dantro.expand_dims`` operation and contain information on the failure.

..warning::

    If *all* data is missing, ``select_and_combine`` will not be able to succeed, because there will be nothing to combine and insufficient information to create a null-like output instead.
    This feature is explicitly meant for data *partially* missing.

    The expected error message for such a case will be coming from ``dantro.merge``:

    ::

        The Dataset resulting from the xr.merge operation can only be reduced
        to a DataArray, if one and only one data variable is present in the
        Dataset! However, the merged Dataset contains 0 data variables.

.. hint::

    The ``allow_missing_or_failing`` argument accepts the same values as the ``allow_failure`` argument of the :ref:`error handling framework <dag_error_handling>`; in fact, it sets exactly that argument internally.

    Thus, the messaging behavior can be influenced as follows:

    .. code-block:: yaml

        select_and_combine:
          allow_missing_or_failing: silent        # other options: warn, log

.. hint::

    Same as ``combination_method`` and ``subspace``, the ``allow_missing_or_failing`` argument can also be specified separately for each field, overwriting the default value from the ``select_and_combine`` root level:

    .. code-block:: yaml

        select_and_combine:
          allow_missing_or_failing: silent
          fields:
            some_data:
              allow_missing_or_failing: warn   # overwrites default from above
              path: path/to/some/data


Applying transformations after combination of data
""""""""""""""""""""""""""""""""""""""""""""""""""
In some cases, it can be useful to define postprocessing transformations on the combined data.
For that purpose, there is the ``transform_after_combine`` option which can be added for each individual field or as a default on the ``select_and_combine`` level.
While this postprocessing can of course also be done alongside ``transform``, it is often easier to define this alongside the field.

Some example use cases:

* Perform some postprocessing on all fields, without having to repeat the definitions.
* Use ``print`` to see the result of the combination directly, without having to touch the ``transform`` definition.
* Call ``.squeeze`` to reduce the one-sized dimensions of a combination, which can simplify some plotting calls.


Custom combination method
"""""""""""""""""""""""""
Apart from the ``merge`` and ``concat`` combination methods, a custom combination method can also be used by specifying the name of an operation that is capable of combining the data in a desired way:

.. code-block:: yaml

    select_and_combine:
      # further kwargs are passed on to the chosen custom operation

      fields:
        some_data:
          path: path/to/some_data
          combination_method:
            operation: my_combination_operation
            pass_pspace: false  # default: false. If true, will pass additional
                                # keyword argument ``pspace``.
            # further kwargs passed to combination operation
          combination_kwargs:

Such a combination operation needs to have the following signature:

.. code-block:: python

    def my_combination_function(objs: list, **kwargs) -> xr.DataArray:
        # ...

Here, ``objs`` is a list of the data from each individual parameter space state ("universe"), ready with attached coordinates.

.. note::

    While the given ``objs`` already have coordinates assigned, you might be interested in some macroscopic information about the shape of the target data.
    To that end, an additional argument can be passed to the combination function by setting ``combination_method.pass_pspace: true``.

    The ``pspace`` argument is then a ``ParamSpace`` object (from the `paramspace package <https://pypi.org/project/paramspace/>`_) which contains information about the dimensionality of the data and the names and coordinates of the dimensions.
    The data in ``objs`` is ordered in the same way as the iteration over ``pspace``.



Full DAG configuration interface for multiverse selection
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of all options available in the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`.


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
          # the ``select_and_combine`` level, see below
          foo: foo                       # ``base_path`` will be prepended here
                                         # resulting in: some/path/foo

          # Define a tag 'bar' that overwrites some of the defaults
          bar:
            path: bar
            subspace:                    # only use universes from a subspace
              seed: [0, 10]
              my_param: [-42., 42.]
            combination_method: merge    # overwriting default specified below
            combination_kwargs:          # passed to Transformation.__init__
                                         # of the *tagged* output node
              file_cache:
                read: true
                write:
                  enabled: true
                  # Configure the file cache to only be written if this
                  # operation took a large amount of time.
                  min_cumulative_compute_time: 20.
            allow_missing_or_failing: silent  # transformations or path lookup
                                              # is allowed to fail
            transform:
              - mean: !dag_prev
              - increment: [!dag_prev ]
              - some_op_with_kwargs:
                  data: !dag_prev
                  foo: bar
                  spam: 42
              - operation: my_operation
                args: [!dag_prev ]
                file_cache: {}           # can configure file cache here

            transform_after_combine:     # applied after combination
              - increment
              - print

        base_path: some_path        # if given, prepended to ``path`` in ``fields``

        # Default arguments, can be overwritten in each ``fields`` entry
        combination_method: concat  # can be ``concat`` (default), ``merge``.
                                    # If a dict, may contain the key
                                    # ``operation`` which will then be used as
                                    # the operation to use for combination; any
                                    # further arguments are passed on to that
                                    # operation call.
        subspace: ~                 # some subspace selection
        allow_missing_or_failing: ~ # whether to allow missing universes or
                                    # failing transformations; can be: boolean,
                                    # ``log``, ``warn``, ``silent``
        transform_after_combine: ~

      # Additional selections, now based on ``dm`` tag
      select: {}

      # Additional transformations; all tags from above available here
      transform: []

      # Other DAG-related parameters: ``compute_only``, ``dag_options``
      # ...

.. note::

    This does not include *all* possible options for DAG configuration but focusses on those options added by :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` to work with multiverse data, e.g. ``subspace``, ``combination_kwargs``.

    For other arguments, see :ref:`dag_transform_full_syntax_spec`.


----

.. _dag_result_placeholder:

Using data transformation results in the plot configuration
-----------------------------------------------------------
The :ref:`data transformation framework <dag_framework>` can not only be used for the *selection* of plot data: using so-called "result placeholders", data transformation results can be used as part of the plot *configuration*.

One use case is to include a computation result, e.g. some mean value, into the title of the plot via the :ref:`plot helper <plot_helper>`.
In general, this feature allows to automate further parts of the plot configuration by giving access to the capabilities of the transformation framework.

Let's look at an example plot configuration:

.. literalinclude:: ../../tests/cfg/dag_plots.yml
    :language: yaml
    :start-after: ### Start -- placeholder_title
    :end-before:  ### End ---- placeholder_title
    :dedent: 6

As can be seen here, there are additional operations defined within ``transform``, which lead to the ``title_str`` tag.
In the helper configuration, that tag is referred to via the ``!dag_result`` YAML tag, thus creating a placeholder at the ``helpers.set_title.title`` key.

This illustrates the basic idea.
Of course, multiple placeholders can be used and they can be used *almost everywhere* inside the plot configuration; however, make sure to have a look at the :ref:`caveats <dag_result_placeholder_scope>` to learn about current limitations.

.. hint::

    When adding placeholders, you will notice additional log messages which inform about the placeholder names and their computation profile.


Caveats
^^^^^^^

.. _dag_result_placeholder_scope:

Where in the plot configuration can placeholders be used?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Placeholders can be used in *wide* parts of the plot configuration, but not everywhere.
If you encounter errors that refer to an ``unexpected ResultPlaceholder object``, this is probably because they were defined in a part of the plot configuration where they cannot be resolved.

**Where can (✅) placeholders always be used? Where can they never (❌) be used?**

* ✅ They *can* be used in *all* configuration entries that are passed through to the selected plot function of the :ref:`pcr_pyplot` and derived plot creators.
* ✅ They *can* be used within the ``helpers`` argument that controls the :ref:`plot_helper`.
* ❌ They can *not* be used for entries related to data transformation (``select``, ``transform``, ``dag_options``, ...) because these need to be evaluated in order to set up the :py:class:`~dantro.dag.TransformationDAG`.
* ❌ They can *not* be used for entries evaluated by the :ref:`plot_manager` (``out_path``, etc) or the plot creator *prior to data selection* (``animation``, ``style``, ``module``, etc).


Why is my placeholder not resolved?
"""""""""""""""""""""""""""""""""""
The identification and replacement of placeholders happens by recursively iterating through ``list``-like and ``dict``-like objects in the plot configuration ``dict``.
Typically, this reaches all places where these placeholders could be defined.
The only *exception* being if the placeholder is in some part of an *object* that does not behave like a ``list`` or a ``dict``.


Implementation details
^^^^^^^^^^^^^^^^^^^^^^
Under the hood, the ``!dag_result`` YAML tag is read as a :py:class:`~dantro._dag_utils.ResultPlaceholder` object, which simply stores the name of the tag that should come in its place.
After the plot data was computed, the :py:class:`~dantro.plot.creators.base.BasePlotCreator` inspects the plot configuration and recursively collects all these placeholder objects.
The :py:meth:`~dantro.dag.TransformationDAG.compute` method is then invoked to retrieve the specified results.
Subsequently, the placeholder entries in the plot configuration are replaced with the result from the computation.

For the above operations, functions from `the paramspace package <https://gitlab.com/blsqr/paramspace>`_ are used, specifically: ``paramspace.tools.recursive_collect`` and ``paramspace.tools.recursive_replace``.
