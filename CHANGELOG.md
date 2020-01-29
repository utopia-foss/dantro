# Changelog

`dantro` aims to adhere to [semantic versioning](https://semver.org/).

## v0.11.0
- !124 adds a condensed data tree representation (as proposed in #112)
- Minor changes:
    - !127 adds a `.coveragerc` file to control pytest-cov behaviour
- !126 changes the type of axis-specific configurations for the `PlotHelper` from list to dictionary
- #101 _removes_ the plot creator auto-detection via plot function signature (deprecated in 0.10, see #100).
- !128 adds the functionality to load plain text files into `StringContainer`
- !130 adds the `PlotHelper.attach_figure` method which allows assigning a custom figure.


## v0.10.2
- #108 adds the ability to unpack DAG results directly into the plot function invocation instead of passing them as the `data` keyword argument.
- #109 extends the operations database to allow importing callables or objects on-the-fly and use them in DAG transformations.
    - Adds the `import` operation which allows importing a module or callable (or any other object) directly, alleviating the need to register a new operation.
    - Adds the `call` operation to call a callable.
    - Adds the `import_and_call` operation, combining the two.


## v0.10.1
- !118 fixes an issue with the `MultiversePlotCreator` where the `select_and_combine` argument was erroneously passed-through to the plot function.
- #111 updates the documentation to reflect that the `paramspace` dependency is [now available on PyPI](https://pypi.org/project/paramspace/) and can be installed from there.


## v0.10.0
- !104 adds the `set_text` function to the `PlotHelper`
- !106 changes `PathMixin` such that detached objects now have `/<name>` as their path, which improves path handling. Furthermore, the `Link` object is now adjusted to this change and its tests are extended to a wide range of scenarios.
- #100 deprecates plot creator auto-detection via the plot function signature of `ExternalPlotCreator`. Instead, the `is_plot_func` decorator should be used.
- !107 changes the `xarray` version requirement from `0.12.1` to `0.13.0`. 
- !105 adds a transformation framework (#48) that allows caching of data operations (#96). It does so by implementing a directed acyclic graph of data transformations, where each node is uniquely represented by a hash. This hash can then be used reliably to determine cache hits. See the MR description for more information. Other minor changes alongside this MR:
    - Improve `LinkContainer`
    - Add `SizeOfMixin`, allowing to compute the size of a container's data
    - Incorporate `PathMixin` into `AbstractDataContainer`
    - Add `base_path` argument to `DataManager.load`
    - Add xarray- and numpy-related data loaders
    - Modularize the tests of the `utils` submodule
- !109 integrates the transformation framework into the plot creators (#99) and further extends the capabilities of the DAG.
    - Data selection and transformation is now built-in to the `BasePlotCreator`, making it available for all derived plot creators. There are specializations for some of the classes:
        - `ExternalPlotCreator` extends the `is_plot_func` decorator to control DAG integration and specify required tags
        - `UniversePlotCreator` sets the selected single universe as the basis of select operations.
        - `MultiversePlotCreator` has to take care of building a tree that selects and combines the data from all desired universes.
    - Improvements to the `TransformationDAG` and related classes:
        - Nodes can now be added programmatically using public methods.
        - Hash computation is much (factor 100) faster than in the implementation from !105.
        - Profiling information is more detailed now and also available in the `TransformationDAG` itself, aggregating information from all registered transformation nodes.
        - It is possible to set and change the reference that serves as basis for `select` operations.
- Other improvements:
    - #98 addresses an h5py deprecation warning regarding the default file mode
    - #106 adds the `AllAvailableLoadersMixin` to provide all available data loaders, making downstream import easier
    - #102 makes the documentation available [online](https://hermes.iup.uni-heidelberg.de/dantro_doc/master/html/)
    - #107 extends and improves the documentation

## v0.9.1
- !100 adds experimental (!) transformator capabilities to `ParamSpaceGroup.select`, improves logging, and resolves minor bugs and inconsistencies.
- !101 adds `base_path` argument to `ParamSpaceGroup.select`, allowing for all other paths to be specified relative to it.
- !102 fixes #88 and a bug in `UniversePlotCreator` when providing a `ParamSpace` as plot configuration.

## v0.9.0
- #76 and !91 improve working interactively with dantro, e.g. by providing the `__repr__` method and adding IPython key completion for group members.
- !95 makes it possible to use forward slashes in plot names, which will lead to the corresponding directories being created in the evaluation directory.
- #74 adds the `Link` class and `LinkContainer`, which allow having links from one container or group to another, given that they are all within the same data tree. Furthermore, this allows the `XrDataContainer` to use a linked container for coordinates (new coordinate modes: `linked`, `from_path`). Additionally, the `XrDataContainer` also supports coordinate modes `trivial` and `indices` now, which both assign the trivial indices to the dimension without the need for further arguments.
- #56 modularizes coordinate handling and adds `LabelledDataGroup`, which allows to work with all sorts of labelled data using the xarray selection interface. It not only allows to select a single element from the group, but follows the same selection syntax as in xarray and furthermore allows to perform a `deep` selection alongside, going down into the member data before combining the data. This brings along two specializations:
    - The `TimeSeriesGroup` now has the full feature set for its `isel` and `sel` methods.
    - There is the `HeterogeneousTimeSeriesGroup` which has high flexibility in how the labelled data is stored: containers of this group may represent data stored at irregular times, with overlapping or non-overlapping coordinate values, and also representing more than a single time snapshot.
- #22 adds the `logging` module, which implements custom log levels. These make conveying information to the user much more powerful by giving more granular control about the verbosity: instead of in effect having only the `debug` and `info` levels available, there are now the additional levels `trace`, `note`, `progress`, `hilight`, and `success`.

## v0.8.1
- #89 enables the `NetworkGroup` to handle one dimensional data that is not time-labelled.
- #90 fixes an issue during garbage collection of `Hdf5Proxy`


## v0.8.0
- #27 renames `ProxyMixin`-based classes to `ProxySupportMixin` to better communicate what they do and avoid confusion with `BaseDataProxy`-derived classes.
- Also with #27, it is possible to load HDF5 data into [`dask.array`s](http://docs.dask.org/en/latest/array.html), which allow to perform lazy operations on the data. This makes it hugely more comfortable to work with large amounts of data in dantro.
    - The `HDFDataProxy` can resolve HDF5 data as delayed `dask.array`s.
    - The `Hdf5LoaderMixin` now allows to pass parameters to created proxies, thus allowing to create proxies which `resolve_as_dask`.
    - The `dask.array` can be used as underlying data for the `XrDataContainer` while retaining the _exact_ same interface as with in-memory numpy data. This is possible due to the tight [integration of xarray with dask](http://xarray.pydata.org/en/stable/dask.html).
- #59 adds additional groups and mixins that allow handling indexed data:
    - `IndexedDataGroup` expects integer-parsable members and maintains ordering not by their string representation but by their integer representation.
    - The `IntegerItemAccessMixin` and `PaddedIntegerItemAccessMixin` provide convenient access to group members via integer keys; internally, keys are always strings.
    - The `ParamSpace`-related groups now use these mixins and groups rather than their own implementation; there are no changes to the public interface.
    - #79 adds a specialization of an `IndexedDataGroup`, namely the `TimeSeriesGroup`, which assumes that it holds members whose names refer to a point in time. The interface here slightly mimicks that of `xr.DataArray`.
- #79 also adds the `LockDataMixin`, incorporates it into `BaseDataGroup`, and adds the `ForwardAttrsMixin`, which is a more general form of the already existing `ForwardAttrsToDataMixin`.
- With #58, the `NetworkGroup` is now using the `TimeSeriesGroup` and `XrDataContainer`. It is now possible to also load node and edge properties into a graph.
- #75 adds dimension information to the info string of `XrDataContainer`s.
- #85 addresses a bug where the `UniversePlotCreator` failed plotting in cases where a multidimensional `ParamSpace` was associated with it but the data was only available for the default point in parameter space.
- dantro now requires `matplotlib >= 3.1`


## v0.7.1
- !83 Takes care of a deprecation warning regarding imports from the `collections` module.


## v0.7.0
- Infrastructure and Documentation
    - #43 Add a Sphinx-based documentation, currently containing only the API reference
    - #44 makes the dantro version number single-sourced in `dantro/__init__.py`

- Many new `PlotManager` and `PlotCreator` features:
    - With #40 and !41, the `PlotManager` can auto-detect which plot creator is to be used; this allows to leave out the `creator` key in the plot configuration and thus simplifies the configuration of a plot.
    - #63 greatly extends the configuration capabilities of the `PlotManager`: There now is a so-called "base" configuration, which plot configurations can use to base their parameters on; this is done via the `based_on` key.
        - #69 Allows `based_on` to be a sequence of base configuration names that are accumulated and then used as basis for the new plot.

- A new `PlotHelper` framework for the `ExternalPlotCreator`:
    - #46 allows the `ExternalPlotCreator` to enter a matplotlib RC parameter context in which a certain style is set.
    - #45 implements the `PlotHelper`, which provides a configuration-accessible interface for matplotlib functions invoked via `ExternalPlotCreator`.
    - #62 implements `PlotHelper` functions to set title, labels, limits, scale, legend, horizontal and vertical lines.
    - #64 adds the possibility to conveniently create animations when using the `ExternalPlotCreator` in combination with the `PlotHelper`
    - #68 makes the `PlotHelper` work axis-specific and thus allow defining helpers for different axes of a figure with subplots.
    - !72 extends `PlotHelper` capabilities and improves error messages

- New data container and proxy features:
    - #47 adds the `XrDataContainer`, which stores data as an [`xarray.DataArray`](http://xarray.pydata.org/en/stable/data-structures.html#dataarray) and associated dimension labels and coordinates by looking at the container attributes
   - #66 implements proxy support for `XrDataContainer`
    - #71 enables to reinstate a previously resolved proxy, thereby releasing the existing data, allowing it to go out of memory

- Miscellaneous improvements and tweaks
   - !39 makes minor improvements to info strings
   - #51 adds a `tree` property to `BaseDataGroup`, which returns the tree representation string of that group.
   - #54 lets HDF5 loader automatically convert encoded strings into python strings
   - #60 removes unnecessary log messages upon entering/exiting a data container's or group's `__init__` method.
   - #61, !53 makes `BaseDataContainer.__init__` non-abstract, which is not only more convenient but also more consistent (`BaseDataGroup` is not abstract either.) This also improves modularization of the `mixins` module.


## v0.6.1
- !44 Fixes a bug where the association of parameter dimensions in `UniversePlotCreator` was wrong and could lead to failing plots.

## v0.6.0
- #36 and !36 make the `out_dir` of `DataManager` more configurable and adds some other minor tweaks.
  It renames the format string segment `date` to `timestamp` (to be more general) and adds the `cfg_exists_action` argument to `PlotManager`, which allows to control the behaviour upon.
  Furthermore, the `read_yml` and `write_yml` functions now allow selecting a file mode.
- !37 adds the ability to specify a default type for the `new_container`
  function of `BaseDataGroup`. This choice is also respected by the HDF5
  loader, thus allowing a parent group to specify its childrens' type and not
  requiring them to use the mapping feature to be loaded as a custom type.
- !38 removes the edge and node attribute related features from `NetworkGroup`,
  because they do not work in a general enough manner.

## v0.5.3
- #35 makes changes to `NetworkGroup` to concur to the NetworkX interface and adds some tweaks to the `set_node_attributes` function.
- !35 allows matplotlib versions larger than the (python2 backwards compatible) version 2.2.3, which is important to keep up with new matplotlib features.

## v0.5.2
- !33 allows using transposed edge specifications in `NetworkGroup` for creation of an `nx.Graph`

## v0.5.1
- !32 fixes bug in hdf5 loader mixin.

## v0.5.0
- #33 improves package structure and modularization by creating sub-packages and moving class definitions into separate modules. This changes the import locations from `group` and `container` to `groups` and `containers`; all other import paths should remain valid.
- #29 implements a `NetworkGroup` that stores network data and enables the direct 
creation of a [`NetworkX`](https://networkx.github.io/documentation/stable/reference/classes/index.html) graph object (`Graph`, `DiGraph`, `MultiGraph`, `MultiDiGraph`)
with or without vertex properties (edge properties not yet implemented) from the data given in the members of the group.

## v0.4.1
- #32 fixes bugs that occurred in `ParamSpaceGroup` and `MultiversePlotCreator` if the associated `ParamSpace` had zero volume

## v0.4.0
- #24 adds a major new feature, the `ParamSpaceGroup`, which provides easy access to multidimensional datasets, represented by [`xarray.Dataset`](http://xarray.pydata.org/en/stable/data-structures.html#dataset)s.
   - It can be used in place of the group that holds the results of, e.g. simulations, carried out via a [`paramspace`](https://ts-gitlab.iup.uni-heidelberg.de/yunus/paramspace) parameter sweep.
   - Via the `select` function, a hyperslab of the underlying multidimensional data can be selected. The interface of this method is build with yaml configurations in mind, such that it can be used, e.g. in plot creators.
- !17 implements some changes necessary for allowing a smooth transition of `deeevoLab` from `deval` to `dantro`, as implemented in yunus/deeevoLab!52. The changes involve:
   - Adding an `ObjectContainer` class that can hold arbitrary objects.
   - Improving the item access interface by allowing lists as arguments, not only strings; this reduces split-and-join operations and makes the interface more versatile.
   - Improvements to the `BaseDataManager`:
      - The default load configuration can now be set via class variables
      - It is now possible to load an entry into the `attrs` of a group or container.
      - Add a `PickleLoaderMixin` to load pickled objects into an `ObjectContainer`.
   - Miscellaneous minor improvements to features, code formatting, and documentation.
- !22 resolves issues that created two kinds of deprecation warnings.
- #26 Test coverage of mixin classes improved; minor bug fixes.
- !26 Implenents the `unpack_data` feature of `DataManager.load` and allows the `YamlLoaderMixin` to load data into an `ObjectContainer`
- #31/!28 implement two new plot creators, based on `ExternalPlotCreator` that make it more convenient to plot data from `ParamSpaceGroup`s.

## v0.3.3
- !19 Restrict `paramspace` version to <2.0 in order to transition to a higher version in a more controlled manner.

## v0.3.2
- !18 With the `paramspace` yaml constructors having changed, it became necessary to change their usage in dantro. This should result in no changes to the behaviour of dantro.

## v0.3.1
- !16 Restrict matplotlib dependency to use version 2.2.3 until potential downstream issues (reg. dependencies of matplotlib) are resolved.

## v0.3
- !14 and #20: Extend the HDF5 loader to have the ability to load into custom container classes. The class is selected by a customaizable attribute of the group or dataset and a mapping from that attribute's value to a type.
- #10: Use American English in docstrings and logging messages


## v0.2
- #19: Test for multiple Python versions
- #20: Make it possible to create custom groups during `DataManager` initialisation


## v0.1
First minor release. Contains basic features. API is mostly established, but not yet final.

- #13: Implement the plotting framework
- #4: Implement `NumpyDataContainer`
- #3, #8, #9, #11: Implement the `DataManager`
- #2: Implement abstract base classes
- #1, #6, #16: Basic packaging, Readme, Changelog and GitLab CI/CD
