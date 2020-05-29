.. _integrate_dantro:

Integrating :py:mod:`dantro` into your Project
==============================================

:py:mod:`dantro` works best if it's tightly integrated into your project.
The aim of this guide is to give step-by-step instructions of how :py:mod:`dantro` can be used to build a data processing pipeline for your project.

.. note::

    The examples given here focus on coveying the necessary steps for the pipeline integration, not so much on the :ref:`ideal structure <integrate_module_structure>` of the implementation.
    Thus, **we recommend going through this guide at least once before starting with the actual implementation** of the data processing pipeline for *your* project. 🤓

.. contents::
    :local:
    :depth: 2

----

Overview
--------
Let's assume you are in the following situation:

* You are working on a project that generates some form of structured data.
  The data itself can have very different properties. It ...

    * ... may be hierarchically organized, e.g. in HDF5 files
    * ... may contain data of very different kinds (numerical array-like, meta-data, plain text data, configuration files...), i.e. semantically heterogeneous data
    * ... may contain data that requires processing before becoming meaningful

* You want to be able to work with this data in a uniform way:

    * Load the generated data
    * Access and explore it
    * Transform it
    * Visualize it

* You will be going through this process more than once, such that putting in the effort to automate the above steps will pay off.

The result of this integration guide will be a **data processing pipeline:** an automated set of procedures that can be carried out on the generated data in order to **handle, transform, and visualize** it.
These procedures will be referred to as the *three stages of the processing pipeline*.

Setting up a tightly integrated pipeline will require more than a few lines of code, as you will see in this guide.
However, once implemented, the pipeline will be highly flexible, such that you can quickly configure it to your needs.
Overall, we think that the up-front time investment of setting up the pipeline will be paid-off by the everyday gains of using this framework and the automizations it provides.

.. hint::

    If you encounter any questions or issues with the integration, please raise an issue `on the project page <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro>`_.
    We are happy to assist and smooth out the pipeline integration process.

.. note::

    Some **remarks regarding the code examples** that follow below:

    * The sequence of examples below is tested and ensured to work with the version of dantro this documentation corresponds to.
    * The ``my_project`` string is used to refer to the Python project in which the processing pipeline is being defined in.
    * In general, every name of the form ``MyFoobar`` denotes that you can (and should) choose your own name for these data structures or variables.

    **Important:** For illustrational purposes, the code shown here is *not* modularized into different files but presented in a linear fashion.
    If you are looking for a minimal pipeline implementation, you can follow this approach.
    However, if you are building a processing pipeline that should be expandable and grow alongside your project, splitting these code chunks into multiple modules is highly recommended; see :ref:`integrate_module_structure`.

.. _integrate_data_gen:

Data Generation
---------------
For this guide, we define a simple data generator that will feed the example data processing pipeline.

This is of course only a **placeholder for your already-existing project**, e.g. a numerical simulation, an agent-based model, or some data collection routine.
The routine shown here is meant to illustrate which kinds of data structures can be worked with and in which manner.

Storyline
^^^^^^^^^
The storyline for our example data generator will be a numerical simulation.
Given a set of parameters, each simulation will create output data that includes:

* An HDF5 file with a bunch of (hierarchically nested) datasets

    * A ``random_walk`` time series
    * A simple agent-based model, which stores all its data to an ``abm`` group.
      It will write the state of each agent as well as some macroscopic observables into that group.

* The set of parameters that were used to generate the simulation output
* The accompanying logger output as a text file

Furthermore, the simulation will make use of the `paramspace package <https://pypi.org/project/paramspace/>`_ to generate simulation data not for a single set of parameters but for a whole multi-dimensional parameter space.

Preparations
^^^^^^^^^^^^
Disregarding the details of the numerical simulation for a moment, let's look at how it will be embedded, configured, and invoked.

First, we need some basic imports and definitions.
For this example, let's assume that you have ``base_out_path`` defined as the directory where simulation output should be stored in, and ``sim_cfg_path`` as the path to a YAML file that defines the parameters for the simulation.
(The ``project_cfg_path`` and ``plots_cfg_path`` will be used later on.)

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_generation_00
    :end-before:  ### End ---- data_generation_00
    :dedent: 4

Now, let's load the configuration and extract the simulation parameters.
Here, they will be a ``ParamSpace`` object (see `paramspace API reference <https://paramspace.readthedocs.io/en/latest/api/paramspace.paramspace.html>`_) which allows to easily define a set of different parameters to sweep over.
(For the actual values of the parameter space, see :ref:`below <integrate_data_gen_sim_params>`.)

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_generation_01
    :end-before:  ### End ---- data_generation_01
    :dedent: 4

Also, we need to prepare some output directory path, here: ``sim_out_dir``, where all the output for this specific simulation run should be stored.
The directory itself need not be created here.

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_generation_02
    :end-before:  ### End ---- data_generation_02
    :dedent: 4


Generating data and storing it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having extracted the relevant parameters, we now iterate over the parameter space.
For each set of parameters resulting from this iteration, we create the data using those parameters and store it inside the output directory:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_generation_03
    :end-before:  ### End ---- data_generation_03
    :dedent: 4

In this example, the ``generate_and_store_data`` function takes care of all of the tasks.
In your project, this might be done in *any* other way.

.. note::

    As the exact procedure of how the data is generated is not important, the corresponding source code of the above examples is omitted here.
    (If you are interested, you can still find it :ref:`below <integrate_data_gen_and_store_source_code>`.)

.. hint::

    If the above procedure is similar in your project, you may want to consider to *also* use `the paramspace package <https://pypi.org/project/paramspace/>`_ in your project to manage the parameters of your data generation procedure.

    To handle hyper-parameters, dantro makes use of ``ParamSpace`` objects in some other parts as well, e.g. for creating plots from simulations that were created in the above manner.
    Using paramspace for the generation routine can thus simplify the automation of data loading and visualization later on.

So far, so good: We now have some simulation output that we can use to feed the data processing pipeline.


Summary
^^^^^^^
The above serves as an example of how we can:

* Use a configuration file to define the parameters for a set of simulations
* Store the simulation output in a specific output directory for each set of parameters
* Store the configuration and parameter space alongside, such that we can later reproduce the simulation if we wanted to

Of course, all this might be quite different to what is needed to generate or collect actual data in your specific scenario.
This example merely illustrates one way to generate that data, in the hope that you can adapt it to your needs.

For now, the **important point** is: You are writing data to some output directory and storing the metadata (the configuration) alongside.
This data will be the input to the processing pipeline.

.. note::

    The data can be generated in *any* conceivable fashion;
    it is **not** required that it is generated by a Python project.
    Only the processing pipeline will be implemented as a Python project.


References
""""""""""
* :doc:`philosophy`
* :ref:`integrate_data_gen_and_store_source_code`
* :ref:`integrate_data_gen_sim_params`
* API reference :py:func:`~dantro._yaml.load_yml`, :py:func:`~dantro._yaml.write_yml`, `paramspace <https://paramspace.readthedocs.io/en/latest/api/paramspace.paramspace.html>`_

----

Stage 1: Data Loading and Handling
----------------------------------
Loading the generated data into a uniform data structure, :ref:`the data tree <phil_data_tree>`, is the first stage of the data processing pipeline.

The loading will be carried out by a custom :py:class:`~dantro.data_mngr.DataManager` that we will call ``MyDataManager``.
This specialization can be configured such that it adapts to the structure of the data that is being worked with, e.g. by using :ref:`specialized container or group types <spec_data_container>` for certain kinds of data.

Following dantro's :ref:`configurability philosophy <phil_configurability>`, all relevant parameters for loading will be consolidated into a configuration file.


Defining a custom :py:class:`~dantro.data_mngr.DataManager`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To specialize the :py:class:`~dantro.data_mngr.DataManager` for the pipeline, we can simply derive from it:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_loading_01
    :end-before:  ### End ---- data_loading_01
    :dedent: 4

In this case, ``MyDataManager`` has all available loaders available.
If desired, the available loaders can be controlled in a more granular fashion, see :ref:`spec_data_mngr`.

Furthermore, it was supplied with information about available group types.
We will use those below to build the initial tree structure.

The ``_HDF5_GROUP_MAP`` class variable is an example of a customization of one of the loaders.
In this case, the given mapping is used by the :py:class:`~dantro.data_loaders.load_hdf5.Hdf5LoaderMixin` to load appropriately labelled HDF5 groups not as the default dantro group type, but as the specified type, which can be a specialized version.


Initializing ``MyDataManager``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To instantiate ``MyDataManager``, we read the corresponding configuration entry from the project configuration and pass those parameters to it:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_loading_02
    :end-before:  ### End ---- data_loading_02
    :dedent: 4

As initialization parameters, we pass the following arguments:

.. literalinclude:: ../tests/cfg/integration.yml
    :language: yaml
    :start-after: ### Start -- dm_params
    :end-before:  ### End ---- dm_params
    :dedent: 0

These already include the so-called ``load_cfg``, i.e. a set of parameters that specifies which data should be loaded from where and how it should be stored in the data tree.

Furthermore, these parameters can be used to already generate a part of the data tree; this can make loading data easier in some scenarios.
Here, the ``create_groups`` argument creates the ``simulations`` group, a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup`, where each member is assumed to be the output of a single simulation.

The ``out_dir`` of the :py:class:`~dantro.data_mngr.DataManager` is a directory that is used to store output that is associated with the to-be-loaded data.
For example, the visualization output will end up in that directory.

Loading data
^^^^^^^^^^^^
Let's recap which data was written during :ref:`data generation <integrate_data_gen>`:

* ``pspace.yml`` stored the simulation parameters of *all* simulations.
* For each simulation, the following files were created:

    * ``data.h5`` is an HDF5 file with hierarchically structured numerical data
    * ``params.yml`` is the set of parameters for this *particular* simulation
    * ``sim.log`` is the plain text simulation log output

Basically, we want to represent the same structure in the data tree.
Thus, loading should carry out the following operations:

* Load the global ``pspace.yml`` and associate it with the already existing ``simulations`` group, such that it is aware of the parameter space.
* For each simulation output directory:

    * Load ``data.h5`` into a new group inside the ``simulations`` group.
    * Load simulation metadata (``sim.log`` and ``params.yml``) and store them alongside.

As mentioned above, all these load operations can be specified in the ``load_cfg``.
For the ``data.h5`` files, an entry of the ``load_cfg`` would look something like this:

.. literalinclude:: ../tests/cfg/integration.yml
    :language: yaml
    :start-after: # Load the binary output data from each simulation.
    :end-before:  enable_mapping: true
    :dedent: 4

This selects the relevant ``data.h5`` files inside the output directory using the ``glob_str`` argument and then uses ``path_regex`` to determine the ``target_path`` inside the ``simulations`` group.
The full load configuration is omitted here (you can inspect it :ref:`below <integrate_full_load_cfg>`).
For general information on the load configuration, see :ref:`here <data_mngr_loading_data>`.

With the load configuration already specified during initialization, loading the data into the data tree is a simple matter of invoking the :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` method:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_loading_03
    :end-before:  ### End ---- data_loading_03
    :dedent: 4

The (condensed) tree view shows which data was loaded into which part of the tree and provides some further information on the structure of the data.
As you see, the initial ``simulations`` group was populated with the output from the individual simulations, the HDF5 tree was unpacked, and the parameter and log output was stored alongside.
So: We preserved the hierarchical representation of the data, both from within the HDF5 file and from the directory structure.

Furthermore, the loader already applied a type mapping during loading: the ``data/abm/energy`` group is a :py:class:`~dantro.groups.time_series.TimeSeriesGroup`, which assumes that the underlying datasets represent a time series.

.. hint::

    :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg` also allows supplying new parameters or updating those given at initialization.

Once loaded, the tree can be navigated in a dict-like fashion:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_loading_04
    :end-before:  ### End ---- data_loading_04
    :dedent: 4


Summary
^^^^^^^
To recap, the following steps were carried out:

* We specialized a :py:class:`~dantro.data_mngr.DataManager`
* We then initialized it with arguments from the ``project_cfg``
* We loaded data as it was specified in a load configuration (also defined in ``project_cfg``)

With this, the first stage of the data processing pipeline is set up: We have automated the loading of data into the data tree.
If further data needs to be loaded or the shape of the data tree needs to be adjusted, the ``load_cfg`` can be changed accordingly.


References
""""""""""
* :doc:`data_io/data_mngr`
* :ref:`spec_data_mngr`
* :ref:`Usage examples <examples_loading_data>`
* API reference: :py:class:`~dantro.data_mngr.DataManager` and methods :py:meth:`~dantro.data_mngr.DataManager.__init__`, :py:meth:`~dantro.data_mngr.DataManager.load_from_cfg`, and :py:meth:`~dantro.data_mngr.DataManager.load`




Stage 2: Data Transformation
----------------------------
To couple to the :ref:`data transformation framework <dag_framework>`, the second stage of the processing pipeline, no special steps need to be taken.
As part of the data visualization stage, the plot creators take care of setting up everything and :ref:`passing the relevant configuration options <plot_creator_dag_usage>` directly to the data transformation framework.

However, to be able to conveniently :ref:`register additional data operations <register_data_ops>`, we suggest to add a dedicated module (e.g. ``data_ops.py``) to your project in which data operations can be defined and registered using the :py:func:`~dantro.utils.data_ops.register_operation` function.
It can look as simple as the following:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_transformation_01
    :end-before:  ### End ---- data_transformation_01
    :dedent: 4

Even if you do not have the need for custom operations at the point of building the integration, it is useful to already set up this module, such that it is easy to add further operations once you need them.

.. note::

    Make sure that this additional module is loaded when the rest of your project is loaded.
    If the :py:func:`~dantro.utils.data_ops.register_operation` calls are not interpreted, the operations will not be available.


Summary
^^^^^^^
* No additional steps *required*
* For having a place to define and register further data operations, adding a custom module is useful


References
""""""""""
* :doc:`data_io/data_ops` and :ref:`register_data_ops`
* :doc:`data_io/transform`





Stage 3: Visualization
----------------------
With the data tree loaded and the transformation framework ready, we merely need to set up the :doc:`plotting framework <plotting/plot_manager>`, which is orchestrated by the :py:class:`~dantro.plot_mngr.PlotManager`.

The process is similar to that with the :py:class:`~dantro.data_mngr.DataManager`:
We will create a specialized version of it, instantiate it, and provide a configuration that defines some common parameters.


Defining a custom :py:class:`~dantro.plot_mngr.PlotManager`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Akin to the customization achieved via ``MyDataManager``, we will define ``MyPlotManager`` as a customization of :py:class:`~dantro.plot_mngr.PlotManager`.
The customizations done there pertain mostly to :ref:`registering further plot creators <spec_plot_creators>`.

In this example, we will use dantro's existing plot creators.
Subsequently, ``MyPlotManager`` is simply a child of :py:class:`~dantro.plot_mngr.PlotManager` and does not require any further changes:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_viz_01
    :end-before:  ### End ---- data_viz_01
    :dedent: 4


Initializing ``MyPlotManager``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to have access to the data tree, ``MyPlotManager`` is associated with access to the ``MyDataManager`` instance from above.
Furthermore, we will use the configuration specified in the ``project_cfg`` during initialization, such that we can adjust ``MyPlotManager`` behaviour directly from the project configuration file:

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_viz_02
    :end-before:  ### End ---- data_viz_02
    :dedent: 4

The ``pm_cfg`` is used to specify some default behaviour of the manager, e.g. that it should raise exceptions instead of merely logging them:

.. literalinclude:: ../tests/cfg/integration.yml
    :language: yaml
    :start-after: ### Start -- pm_params
    :end-before:  ### End ---- pm_params
    :dedent: 0

As part of this initialization process, default arguments for the plot creators are also supplied via ``creator_init_kwargs``.
In this case, we configure these creators to use ``pdf`` as the default file extension.
For the ``ParamSpace``-supporting plot creators (see :doc:`plotting/creators/psp`), we specify the path to the :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup` inside the data tree.

Creating plots
^^^^^^^^^^^^^^
Creating plots is now as easy as invoking :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg` with a path to a configuration file (or with a dict containing the corresponding configuration).

Let's have a look at an example plot configuration and how it is invoked:

.. literalinclude:: ../tests/cfg/integration_plots.yml
    :language: yaml
    :start-after: ### Start -- plots_01
    :end-before:  ### End ---- plots_01
    :dedent: 0

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- data_viz_03
    :end-before:  ### End ---- data_viz_03
    :dedent: 4

Once invoked, the logger output will show the progress of the plotting procedure.
It will show that a plot named ``random_walk`` is created for each of the simulations, as specified in the plot configuration.
This is using the :py:class:`~dantro.plot_creators.pcr_psp.UniversePlotCreator`, which is capable of detecting the parameter space and which uses the capabilities of the :py:class:`~dantro.plot_mngr.PlotManager` to generate multiple plots.

.. hint::

    To plot only a subset of the plots configured in ``plots_cfg``, use the ``plot_only`` argument of :py:meth:`~dantro.plot_mngr.PlotManager.plot_from_cfg`.
    This is a useful parameter to make available via a CLI.

The plotting output will be saved to the output directory, which is the ``eval/{timestamp:}`` directory that ``MyDataManager`` created inside the data directory exactly for this purpose.

Extended example
""""""""""""""""
Let's look at a more involved example that plots mean random walk data from the parameter sweep (``mean_random_walk``) and the ABM's mean energy time series (``abm_mean_energy``):

.. literalinclude:: ../tests/cfg/integration_plots.yml
    :language: yaml
    :start-after: ### Start -- plots_02
    :end-before:  ### End ---- plots_02
    :dedent: 0

These plot configurations already do much more and are meant to illustrate the capabilities of the plotting framework.
Without going into detail, let's highlight some of the operations specified above:

* With the :py:class:`~dantro.plot_creators.pcr_psp.MultiversePlotCreator`, data from several simulations can be combined into a higher-dimensional array.
* The ``select_and_combine`` key controls which data to select from each simulation and how it should be combined into the higher-dimensional object.
* The ``transform`` key is used to control the :ref:`dag_framework`, e.g. to calculate the mean over some dimension of the data or label the dimensions accordingly.
* The ``facet_grid`` plot is a very versatile plotting function for high-dimensional data, which is why it is used here. See :ref:`here <dag_generic_facet_grid>` for more information.
* With the plot ``helpers``, the aesthetics of the plot can be changed, e.g. to set limits or labels right from the plot configuration.

The above example gives a glimpse of the possibilities of the plotting framework.
All of these features are already available as part of dantro.

Importantly, though, the plotting framework becomes much more capable once you specialize it to your needs.
For example, with the :py:class:`~dantro.plot_creators.pcr_ext.ExternalPlotCreator` and its built-in access to the :ref:`dag_framework`, you can easily define further plotting functions that form a bridge between selected and transformed data and its visualization.

.. hint::

    To re-use plot configurations, there is the :ref:`plot_cfg_inheritance` feature, which makes plot specifications much more modular and concise.
    It allows to outsource common parts of the plot configurations into a so-called "base configuration", and compose these back together using the ``based_on`` argument.

    This feature requires to specify a set of "base plot configurations", e.g. as defined in a ``base_plots_cfg.yml`` file.
    The path to this file or the content of it needs to be communicated to the :py:class:`~dantro.plot_mngr.PlotManager` at some point, e.g. via its :py:meth:`~dantro.plot_mngr.PlotManager.__init__` call.


Summary
^^^^^^^
To couple the data loading and transformation stages to the plotting framework, the following steps were necessary:

* Specialize a :py:class:`~dantro.plot_mngr.PlotManager`
* Instantiate it using arguments from a configuration file
* Configure plots in a configuration file
* Tell the ``MyPlotManager`` instance to generate plots from that configuration

With this, the data processing pipeline is complete: it automates the loading of data, its processing, and its visualization. 🎉🎉

.. note::

    Before repeating these steps for your project, make sure to study the :ref:`integrate_module_structure` section below.

References
""""""""""
* :doc:`plotting/plot_manager`
* :doc:`plotting/plot_creators`
* :ref:`spec_plot_creators`
* :doc:`plotting/plot_data_selection`
* :doc:`plotting/faq`




Closing Remarks
---------------
Of course, integration doesn't end here.
While this guide describes how the basic infrastructure of the pipeline can be implemented, you have many more possibilities to specialize the pipeline to your project's needs.

We hope that this guide helps in integrating dantro into your project!

.. note::

    If you encounter any difficulties with this process, have a question or suggestion, or need support of any other kind, feel free to `open an issue <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro/issues/new>`_ on the project page.
    We are looking forward to your feedback!


----

Further Information
-------------------

.. _integrate_module_structure:

Suggested module structure
^^^^^^^^^^^^^^^^^^^^^^^^^^
In the linearly presented code examples above, no particular module structure is apparent.
For the Python project your project's processing pipeline will be implemented in, we suggest the following structure:

.. code-block::

    ├ run_my_pipeline.py        # performs pipeline invocation
    └ my_project/               # contains the pipeline definition
      ├ __init__.py
      ├ data_io.py              # defines MyDataManager and custom containers
      ├ data_ops.py             # defines custom data operations
      ├ ...
      ├ pipeline_cfg.yml        # stores default pipeline configuration
      ├ base_plots_cfg.yml      # defines base plot configurations
      ├ ...
      ├ plot_funcs.py           # defines functions for ExternalPlotCreator
      └ plotting.py             # defines MyPlotManager

Here, ``run_my_pipeline.py`` is a script that determines which configuration files should be used and passed to ``MyDataManager`` and ``MyPlotManager``.
It can, for example, use `the argparse module <https://docs.python.org/3/library/argparse.html>`_ to provide a command line interface where the data directory and the configuration file paths can be specified.

What goes where?
""""""""""""""""
So... which code examples from above should be implemented in which module?

* Class definitions and specializations should all go into the modules inside ``my_project``
* Variable definitions (e.g. via CLI), instantiations of managers, and method calls should go into ``run_my_pipeline.py`` (for good measure: inside an ``if __name__ == "__main__"`` block)
* ... the only exception being calls to :py:func:`~dantro.utils.data_ops.register_operation`, which should be made in the ``data_ops`` module directly

Regarding configuration files, we suggest the following:

* Put the pipeline *default* values into ``pipeline_cfg.yml`` and use the entries from there to set up ``MyDataManager`` and ``MyPlotManager``, similar as the ``project_cfg.yml`` the example.
* Any *updates* to those defaults can then be done at runtime, e.g. via ``run_my_pipeline.py``
* Plot configurations of plots you frequently use should go into ``base_plots_cfg.yml``


Adapting to a growing project
"""""""""""""""""""""""""""""
Your project will certainly grow over time.
The above structure allows that your pipeline implementation grows alongside.
You can dynamically extend the above structure with submodules to allow a more granular module structure:

.. code-block::

    ├ run_my_pipeline.py
    └ my_project/
      ├ data_io/
        ├ __init__.py
        ├ some_custom_container.py
        ├ some_custom_group.py
        ├ ...
        └ some_custom_proxy.py
      ├ data_ops/
        ├ __init__.py
        ├ operations.py
        └ ...
      ├ plotting/
        ├ plot_funcs/
          ├ __init__.py
          ├ generic.py
          ├ ...
          └ multi_dim.py
        ├ __init__.py
        ├ some_plot_creator.py
        ├ ...
        └ some_custom_proxy.py
      ├ __init__.py
      ├ data_io.py
      ├ data_ops.py
      ├ ...
      ├ pipeline_cfg.yml
      ├ base_plots_cfg.yml
      ├ ...
      ├ plot_funcs.py
      └ plotting.py


.. hint::

    By adding additional imports from the new submodules to the top-level modules, you can avoid breaking imports.

Remarks
"""""""
* For robustly determining configuration file paths from within the python package, use ``pkg_resources.resource_filename`` (see `their docs <https://setuptools.readthedocs.io/en/latest/pkg_resources.html>`_)
* The dantro manager structures usually allow to pass strings instead of nested dicts for defining configurations, e.g. the ``plots_cfg``.
  Such a string is interpreted as a path to a YAML configuration file.
  This can alleviate loading the YAML files in the outer scope, e.g. the ``run_my_pipeline.py``.
* We are `considering <https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro/-/issues/163>`_ to add a CLI interface directly to dantro to alleviate the need to define a ``run_my_pipeline.py`` file manually.


.. _integrate_example_full_pipeline:

Example of a full pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^
For an example of a fully integrated data processing pipeline that makes use of most dantro features, have a look at `the Utopia project <https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/blob/master/python/utopya/utopya>`_.
The specializations described above are implemented in the ``datacontainer``, ``datagroup``, ``plotting`` and ``datamngr`` modules shown above.
User-defined plotting functions for the customized plot creators can be found in `a separate plotting module <https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/-/tree/master/python/model_plots>`_.


.. _integrate_data_gen_and_store_source_code:

Data generation and storage function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following example shows the data generation routine, consisting of a random walk and a (naive) implementation of a simple agent-based model (ABM).

.. literalinclude:: ../tests/test_integration.py
    :language: python
    :start-after: ### Start -- generate_and_store_data_function
    :end-before:  ### End ---- generate_and_store_data_function
    :dedent: 0

.. _integrate_data_gen_sim_params:

Simulation parameters
"""""""""""""""""""""
The corresponding simulation parameters are the following, which actually represent a two-dimensional parameter space (along dimensions ``seed`` of the internal random number generator, and ``max_step_size`` of the random walk).

.. literalinclude:: ../tests/cfg/integration.yml
    :language: yaml
    :start-after: ### Start -- sim_params
    :end-before:  ### End ---- sim_params
    :dedent: 0


.. _integrate_full_load_cfg:

Full load configuration
^^^^^^^^^^^^^^^^^^^^^^^
The following is the ``load_cfg`` used in the initialization of ``MyDataManager``:

.. literalinclude:: ../tests/cfg/integration.yml
    :language: yaml
    :start-after: ### Start -- dm_load_cfg
    :end-before:  ### End ---- dm_load_cfg
    :dedent: 4
