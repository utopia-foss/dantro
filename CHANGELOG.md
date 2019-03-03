# Changelog

`dantro` aims to adhere to [semantic versioning](https://semver.org/).

## v0.7.0
- #43 Add a Sphinx-based documentation, currently containing only the API reference
- With #40 and !41, the `PlotManager` can auto-detect which plot creator is to be used; this allows to leave out the `creator` key in the plot configuration and thus simplifies the configuration of a plot.
- #44 makes the dantro version number single-sourced in `dantro/__init__.py`


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
