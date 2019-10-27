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

Arguments to control DAG behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You then have the following arguments available to control its behaviour:

    - ``select`` and ``transform``: select data and perform transformations on it, see :py:meth:`~dantro.dag.TransformationDAG.add_nodes`.
    - ``compute_only``: controls which tags are to be computed, see :py:meth:`~dantro.dag.TransformationDAG.compute`
    - ``dag_options``: passed to :py:class:`~dantro.dag.TransformationDAG` initialization, e.g. to control ``file_cache_defaults``.

.. note::

    If DAG usage is enabled, these arguments will be used *exclusively* for the DAG, i.e.: they are not available downstream in the plot creator.

The creation of the DAG and its computation is controlled by the chosen plot creator and can be specialized to suit that plot creator's needs.

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


DAG usage with :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator`
----------------------------------------------------------------------------
The :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` works exactly the same as in the general case.
After computation, the results are made available to the selected python plot function via the ``data`` keyword argument, which is a dictionary of the tags that were selected to be computed.

With this additional keyword argument being passed to the plot function, the plot function's signature also needs to support DAG usage, which makes it less comfortable to control DAG usage via the ``use_dag`` argument in the plot configuration.

Instead, the **best way** of implementing DAG support is via the :py:func:`~dantro.plot_creators.pcr_ext.is_plot_func` decorator.
It provides the following arguments that have an effect on DAG usage:

- ``use_dag``: to enable or disable DAG usage. Disabled by default.
- ``required_dag_tags``: can be used to specify which tags are expected by the plot function; if these are not defined or not computed, an error will be raised.
- ``compute_only_required_dag_tags``: if the plot function defines required tags and ``compute_only is None``, the ``compute_only`` argument will be set such that only ``required_dag_tags`` are computed
- ``pass_dag_object_along``: passes the :py:class:`~dantro.dag.TransformationDAG` object to the plot function as ``dag`` keyword argument.

Decorator usage puts all the relevant arguments for using the DAG framework into one place: the definition of the plot function.


Special case: :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`, data selection and transformation has to occur based on data from the currently selected universe. 
This is taken care of automatically by dynamically setting the :py:meth:`~dantro.dag.TransformationDAG.select_base` property to the current universe.
Thus, the ``select`` argument acts as if selections were to happen directly from the universe.


Special case: :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator` has a harder job: It has to select data from the whole multiverse subspace, apply transformations to it, and finally combine it.
It does so fully within the DAG framework by building a separate DAG branch for each universe and bundling all them into a transformation that combines the data.

This happens via the ``select_and_combine`` argument.

**Important:** The ``select_and_combine`` argument behaves differently to the ``select`` argument of the DAG interface!
This is because it has to accomodate various further configuration parameters that control the selection of universes and the multidimensional combination of the selected data.

The ``select_and_combine`` argument expects the following keys:

- ``tags``: all keys given here will appear in the results dictionary. The values of these keys are dicts that contain the same parameters that can also be given to the ``select`` argument of the DAG interface.
  In other words: paths you would like to select form within each universe should be specified at ``select_and_combine.tags.<result_tag>.path`` rather than at ``select.<result_tag>.path``.
- ``base_path`` (optional): if given, this path is prepended to all paths given under ``tags``
- ``combination_method`` (optional, default: ``concat``): how to combine the selected and transformed data from the various universes. Available parameters:

    - ``concat``: attempts to preserve data types but is only possible if the universes fill a hypercube without holes
    - ``merge``: which is always possible, but leads to the data type falling back to float. Missing data will be ``np.nan`` in the results.

  The combination method can also be specified for each tag under ``select_and_combine.<result_tag>.combination_method``.
- ``subspace`` (optional): which multiverse subspace to work on. This is evaluated fully by the ``paramspace.ParamSpace.activate_subspace`` method.
  The subspace can also be specified for each tag under ``select_and_combine.<result_tag>.subspace``.

Remarks
"""""""
- The select operations on each universe set the ``omit_tag`` flag in order not to create a flood of only-internally-used tags
- File caching is hard-coded to be disabled for the initial select operation and for the operation that attached the parameter space coordinates to it. This behaviour cannot be influenced.
- The best place to cache is the result of the combination method.
- The regular ``select`` argument is still available, but it is applied only *after* the ``select_and_combine``-defined nodes were added and it does only act *globally*, i.e. not on *each* universe.

Example
"""""""
.. code-block:: yaml

    # Some plot configuration file
    ---
    my_plot:
      # ... some plot arguments here ...

      # Data selection via DAG framework
      select_and_combine:
        tags:
          foo: some/path/foo
          bar:
            path: some/path/bar
            transform:
              - mean: [!dag_prev ]
              - increment: [!dag_prev ]

        base_path: ~                # if given, prepended to `path` in tags

        # Default arguments, can be overwritten in each `tags` entry
        combination_method: concat  # can be `concat` (default) or `merge`
        subspace: ~                 # some subspace selection

      transform:
        - add: [!dag_tag foo, !dag_tag bar]
          tag: result
      # ... more DAG-related arguments here ...
