.. _pcr_psp:

Plots from Multidimensional Data
================================

The dantro plotting framework tries to make the plotting of multidimensional data as easy as possible.
This page describes how to define plot functions and plot configurations for these scenarios.

If you have not already done so, **make sure to read up on the corresponding nomenclature** (*universes* and *multiverses*, introduced :ref:`here <universes_and_multiverses>`) before continuing on this page.

.. contents::
    :local:
    :depth: 2

----

Plots from Universe Data
------------------------
To create plots that use data from a single universe, use the :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`.
It allows to specify a set of universes to create plots for and provides the plotting function with data from the selected universes.

The plot configuration for a universe plot requires as an additional argument a selection of which universes the plot should be created for.
This is done via the ``universes`` argument:

.. code-block:: yaml

    ---
    my_universe_plot:
      universes: all        # can also be:
                            #    1) 'first', 'any'
                            #    2) a dict specifying a multiverse subspace
                            #       to restrict the plots to
                            #    3) a list of (integer) universe IDs
                            #
      # ... more arguments


.. _uni_plot_with_dag:

Universe plots using DAG framework *(recommended)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the :ref:`creator-averse plot function definition <pcr_pyplot_recommended_sig>` and specify the ``creator`` in the plot configuration.
You can then use the :ref:`regular syntax <plot_creator_dag_args>` to select the desired data, based on the currently selected universe.

When using the recommended creator-averse plot function signature, the DAG is automatically enabled and allows to select data in the following way:

.. code-block:: yaml

    my_plot:
      creator: universe
      universes: all

      # Select data within the current universe
      select:
        some_data: data/MyModel/some/path/foo
        some_other_data:
          path: data/MyModel/some/path/bar
          transform:
            - mean: [!dag_prev ]
            - increment: [!dag_prev ]

      # Perform some transformation on the data
      transform:
        - add: [!dag_tag some_data, !dag_tag some_other_data]
          tag: result

      # ... further arguments

In this case, the available tags would be ``some_data``, ``some_other_data``, and ``result``.
Furthermore, for the universe plot creator, the ``uni`` tag is always available as well.

For more details, have a look at :ref:`plot_data_selection_uni` and :ref:`the general remarks on the transformation framework <pcr_pyplot_DAG_support>`.

Remarks
"""""""

* To access elements within each universe, you can use the ``uni`` tag and either do a selection of the desired element within the DAG framework or do it in the plot function, based on the ``uni`` result tag.
* Use the ``dag_options.select_path_prefix`` option to navigate to some base path, making subsequent path definitions in ``select`` a bit simpler.
  In the example above, the paths would just be ``some/path/foo`` and ``some/path/bar`` when setting ``dag_options.select_path_prefix`` to ``data/MyModel``, thus always starting paths within some base group.
* To traverse through some dict-like entry within the universe, you can also use the DAG framework:

    .. code-block:: yaml

        my_plot:
          creator: universe

          select:
            # This is equivalent to uni['cfg']['foo']['bar']['some_param']
            some_param:
              path: cfg
              with_previous_result: true
              transform:
                - getitem: foo
                - getitem: bar
                - getitem: some_param


Without DAG framework
^^^^^^^^^^^^^^^^^^^^^
Without the DAG framework, the data needs to be selected manually:

.. code-block:: python

    from dantro import DataManager
    from dantro.groups import ParamSpaceStateGroup as UniverseGroup
    from dantro.plot import is_plot_func, PlotHelper, UniversePlotCreator

    @is_plot_func(creator_type=UniversePlotCreator)
    def my_plot(dm: DataManager, *, uni: UniverseGroup, hlpr: PlotHelper,
                **additional_kwargs):
        """A universe-specific plot function using the data transformation
        framework and the plot helper framework.

        Args:
            dm: The DataManager, containing *all* data
            uni: The currently selected universe. Select the data from here.
            hlpr: The associated plot helper.
            **additional_kwargs: Anything else from the plot config. Ideally,
                specify these explicitly rather than gathering them via ``**``.
        """
        # Get the data
        x = uni['data/MyModel/foo']
        y = uni['data/MyModel/bar']

        # Plot the data
        hlpr.ax.plot(x, y)

        # Add some information from the universe configuration
        cfg = uni['cfg']
        some_param = cfg['MyModel']['some_param']
        hlpr.provide_defaults('set_title',
                              title="Some Parameter: {}".format(some_param))

        # Done. The plot helper saves the plot.

Note how the data selection is hard-coded in this example.
In other words, when *not* using the data selection and transformation framework, you have to either hard-code the selection or parametrize it, allowing to specify it via the plot configuration arguments.



----


Plots from Multiverse Data
--------------------------
To create plots that use data from *more than one* universe — henceforth called *multiverse data* — use the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator`.
This creator makes it possible to select and combine the data from all selected individual universes and provides the result of the combination to the plot function.

This requires the handling of multidimensional data and depends on the dimensionality of the chosen parameter space.
Say the selected data from each universe has dimensionality three and a parameter sweep was done over four dimensions, then the data provided to the plot function has seven dimensions.


.. _mv_plot_with_dag:

Multiverse plots using DAG framework *(recommended)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Again, use the :ref:`creator-averse plot function definition <pcr_pyplot_recommended_sig>` and specify the ``creator`` in the plot configuration.
For this creator, a :ref:`special syntax <plot_data_selection_mv>` exists to select and combine the multiverse data.

When using the recommended creator-averse plot function signature, the DAG is automatically enabled and allows to select data using the ``select_and_combine`` key:

.. code-block:: yaml

    ---
    my_plot:
      creator: multiverse

      # Multiverse data selection via DAG framework
      select_and_combine:
        fields:
          some_data: some/path/foo
          some_other_data:
            path: some/path/bar
            transform:
              - mean: [!dag_prev ]
              - increment: [!dag_prev ]

        base_path: data/MyModel     # ... to navigate to the model base group

        # Default values for combination method and subspace selection; can be
        # overwritten within the entries specified in ``fields``.
        combination_method: concat  # can be 'concat' (default) or 'merge'
        subspace: ~                 # some subspace selection

      transform:
        - add: [!dag_tag some_data, !dag_tag some_other_data]
          tag: result

Again, for more details, have a look at :ref:`plot_data_selection_mv` and :ref:`the general remarks on the transformation framework <pcr_pyplot_DAG_support>`.

.. hint::

    The subspace selection happens via `the paramspace package <https://pypi.org/project/paramspace/>`_.



.. _mv_plot_skipping:

Skipping multiverse plots
^^^^^^^^^^^^^^^^^^^^^^^^^
For skipping :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` plots, the ``expected_multiverse_ndim`` argument can optionally be specified in the plot configuration.
The argument specifies a set of dimensionalities with which plotting is possible; if the dimensionality of the associated :py:class:`~dantro.groups.psp.ParamSpaceGroup` is not part of this set, the plot will be skipped.

.. code-block:: yaml

    ---
    my_plot:
      creator: multiverse

      # Declare that this plot requires a 2-, 3-, or 4-dimensional associated
      # ParamSpaceGroup and should be skipped if this condition is not met
      expected_multiverse_ndim: [2,3,4]

      # ...

See :ref:`plot_mngr_skipping_plots` for general information about skipping of plots.
