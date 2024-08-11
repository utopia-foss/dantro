# Changelog

`dantro` aims to adhere to [semantic versioning](https://semver.org/).

## v0.20.0b2
- !352 adds the `parallel` key to the plot configuration, allowing to execute `ParamSpace`-based plots in separate threads or processes; this can significantly speed up plot creation if there are many plots to be made (and individual plots need more time than spawning the plot executor).
- !358 adds numpy >= 2.0 compatibility.

## v0.20.0b0
- !349 adds Windows support for dantro by ensuring paths are consistently formatted in Posix-style
- !355 improves automatic `col_wrap` in facet grid plots by using a simple optimization routine to maximize the fill ratio of the last row of a facet grid plot while trying to stay close to a square-like grid.
  The old behaviour (which may create lonely plots in the last row) is retained by setting `col_wrap: square`.

## v0.19.5
- !347 fixes empty axis label strings being ignored in the ``PlotHelper``

## v0.19.4
- !344 adds a shortcut syntax for same-name plot config inheritance.
- !344 allows to have empty plot config files, which previously threw an error.

#### Internal
- !342 fixes regressions in HDF5 proxy test caused by new h5py release
- !343 adds test environments for Python 3.12

## v0.19.3
- !341 allows using `!dag_result` placeholders in `style` and `animation` entries of plot configuration.

## v0.19.2
- !340 uses [`yayaml`](https://gitlab.com/blsqr/yayaml) as a YAML backend, including all the YAML-related functionality that was previously implemented directly in dantro.

## v0.19.1
- !339 restricts numpy to < 2.0 to avoid breaking changes later this year.

## v0.19.0
- !326 implements the `PathContainer` and the `DirectoryGroup` that are used to represent filesystem paths.
    - The `DirectoryGroup` can be used to represent nested filesystem structures.
    - These objects have a `fs_path` property that provides a `pathlib.Path` from which path-related operations can be carried out.
- !326 improves and extends data loading:
    - Additionally registers data loaders in a global registry such that setting up a custom `DataManager` with mixins is no longer required (for mixins that resemble class methods).
    - Improves globbing of files to load:
        - The `ignore` argument now also supports glob strings
        - Adds `include_directories` and `include_files` flags which can be used to filter out directories or files.
    - The `target_path` format string has additional keys available to compose the target path:
      `relpath`, `relpath_cleaned`, and `dirname_cleaned`.
      The latter two are replacing `/` in the path by `__`.
    - Implements the `fspath` data loader which loads filesystem paths as `PathContainer`s. By default, this creates a flat hierarchy.
    - Implements the `fstree` data loader which creates a representation of a filesystem tree inside the data tree using `DirectoryGroup` for directories and `PathContainer` for files.
- !326 implements a general `ObjectRegistry` class that can be used to keep track of types and objects throughout the package and make it easier for packages using dantro to supply their custom types.
    - Implemented group and container types are registered by their simple name and fully qualified name.
    - Decorators `@is_group` and `@is_container` are introduced that add class definitions to the respective registries.
- !326 improves the `BaseDataContainer` and `BaseDataGroup` interface:
    - `__init__` methods can now accept `parent` as argument.
      In cases where the parent object is known at the time of creation of the container or group, that information can be used during initialization.
- !326 makes `glob_paths` function available via `dantro.tools`.
- !328 adds `.chunk` data operation (for use with xarray objects).
- !327 adds an option to file cache `load_options` that forces unpacking of the `data` attribute from a dantro object.
- !322 adds tests for Python 3.11 to the CI

#### Bug fixes
- !327 addresses issues that appeared when using the data transformation framework with `dask.array`s:
    - A bug where the `DataManager` acted as unwanted cache layer despite `read.always` file cache parameter being set.
    - A bug where cache files were always overwritten, despite `write.allow_overwrite` being set to false.
- !334 addresses a visual glitch in scatter plots caused by the default `markeredge` color.
- !336 makes `setitem` and `setattr` operations pass through the object that they are operating on, such that they can be used properly in the data transformation framework.


## v0.18.10
- !325 allows to always load a computation result from the file cache.


## v0.18.9
- !324 fixes a bug from interface changes in the latest xarray release.


## v0.18.8
- !321 fixes a bug where `XarrayLoaderMixin` used `xr.load_` functions instead of the intended (and more memory-friendly) `xr.open_` functions.


## v0.18.7
- !319 address further incompatibilities with [matplotlib 3.6](https://matplotlib.org/stable/users/release_notes.html#version-3-6), especially in `ColorManager`
    - Colormaps are now retrieved via the new `matplotlib.colormaps` interface (instead of the deprecated `matplotlib.cm.get_cmap`)
- !320 allows `.plot.multiplot` to use the `ColorManager` syntax to specify custom colormaps from the config


## v0.18.6
- !317 address incompatibilities with [matplotlib 3.6](https://matplotlib.org/stable/users/release_notes.html#version-3-6)


## v0.18.5
- !315 adds `plt.errorbar` to `MULTIPLOT_FUNC_KINDS`, making it more easily accessible.
- !314 reworks and extends the `ColorManager`:
    - Now offers a shorthand syntax to define colorbar labels and colors via a single mapping.
    - Fixes subtle errors in the handling of parameters and the resulting output.
    - Adds a `ColorManager`-specific documentation page, including an integration example.
    - Allows creating color palettes using seaborn.
- !314 fixes an error where facet grid plots did not correctly display the passed data in an error message.

## v0.18.4
- !310 fixes a bug that occurred when using relative references like `!dag_prev` in the arguments to a meta-operation.
- !311 adds the `DataManager.available_loaders` property and expands the corresponding documentation.
- !311 adds some new data loaders:
    - `numpy_txt` to load plain text files into numpy arrays.
    - `pandas_csv` to load CSV files into data frames.
    - `pandas_generic` to load other file formats from [pandas I/O](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) into data frames.
- !312 generalizes the `path_regex` argument to `DataManager.load`, thus allowing multiple matching groups as well as addressing them by a name.


## v0.18.3
- !306 fixes a bug that prohibited re-creating a plot from the plot configuration file saved alongside the plot output.
- !307 adds hints to YAML-related error messages
- !308 adds the `force_compute` argument to `TransformationDAG.add_node`, ensuring that a node's result is always computed.
- !309 expand test cases for multiple uses of argument placeholders in meta-operations. (The aim was to address mutability issues observed in #288, but they could not be reproduced.)


## v0.18.2
- !305 fixes a regression in the `print_data` data operation, which led to uninformative output for xarray objects.


## v0.18.1
- !304 fixes a bug in `.plot.facet_grid` where a non-faceting plot with `size` and/or `aspect` arguments would cause an empty output.


## v0.18.0
- !263 implements various improvements to the plotting and data transformation framework, improving performance, communication and overall usability:
    - Allow caching whole `TransformationDAG` trees in order to avoid re-building them (which can be very time-consuming if there are many nodes)
    - Improve logging (formulation of messages, choice of level, removal)
    - Improve communication of run times, now only shown if beyond a threshold
    - Implement a significantly faster `cPickle`-based deep copy function which is used when building `TransformationDAG` and speeds up creation of large DAGs
    - Use `__slots__` in DAG placeholder classes to reduce memory load
    - In DAG, only load from cache file if the content was *not* already loaded
- !289 implements the ability to define optional positional and keyword arguments for meta-operations in the data transformation framework.
- !290 introduces the ability to *visualize* DAGs, which can be helpful for understanding the structure of the computations.
- !271 strongly reduces the time it takes to `import dantro` by delaying imports of dependencies
- !277 improves the [dantro documentation][dantro-docs] by adding cross-referencing to other sphinx-based docs and tweaking many minor aspects of the dantro docs.
- !278 implements the `TERMINAL_INFO` dict which holds information about the terminal size and can be updated using `dantro.tools.update_terminal_info()`.
- !279 adds the `set_margins` function to the `PlotHelper`, giving access to the [`ax.margins`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.margins.html) method.
- !280 reworks the structure of the `dantro.utils.data_ops` module, now available directly via `dantro.data_ops`.
  Furthermore, this implements the `is_operation` decorator which can be used to register functions as operations directly where they are defined.
  - !300 improves the documentation of the default data operations database.
- !281 completely restructures the plotting module (and some other modules):
    - Plotting related functionality is now condensed in `dantro.plot`, with plot creators having moved to `dantro.plot.creators`.
    - The `ExternalPlotCreator` is renamed to `PyPlotCreator`, highlighting that it works with `matplotlib.pyplot` as a backend.
    - The `dantro.groups.pspgrp` module was renamed to `dantro.groups.psp`
    - The `dantro.containers.xrdatactr` module was renamed to `dantro.containers.xr`
- !282 moves the former `ExternalPlotCreator`'s plot function loading functionality into `PlotManager`, making it a central component of the plotting framework rather than a creator's special feature.
    - This also makes `BasePlotCreator` a fully-functioning (no longer abstract) creator class, which is averse to any plotting backend.
    - Adds the customizable `PlotFuncResolver` class which now takes care of these tasks in a more decoupled setting.
- !291 expands the `PlotHelper` to support 3D projection plots and their z-axis.
  Furthermore, this implements the `scatter3d` facet grid plot.
- !292 migrates the `ColorManager`, which helps generating colormaps and norms, from [utopya][utopya-repo] over to dantro.
  In addition, the `!cmap` and `!cmap_norm` tags are implemented and allow to generate these objects from a YAML configuration.
    - !295 integrates the `ColorManager` into all `facet_grid`-related plot functions, where it will parse the `cmap` and `norm` arguments.
- !298 adds a base plot configuration pool to dantro, supplying useful defaults and exemplifying how such a configuration pool may look like.
    - There is a new example page in the documentation that showcases some of the base configurations.
    - Use of this config pool can be controlled via the `PlotManager`'s new `use_dantro_base_cfg_pool` argument.
    - Additionally, the new `shared_creator_init_kwargs` argument is added, which passes initialization arguments to plot creators, regardless of their name.
    - Adds the `.coords.transform` data operation, which can be used to apply a function to coordinate dimensions of a `xr.DataArray`.
    - Makes the tick locators and formatters from `matplotlib.dates` available in the plot helper.

#### Bug fixes
- !293 fixes a bug in the `make_facet_grid_plot` decorator that prohibited generating a correct colorbar for the faceted data.
- !296 fixes an error in the `set_labels` helper that prevented passing additional arguments like `labelpad`.
- !298 fixes a bug in the `set_ticks` helper that prevented setting ticks via a shorthand syntax.

#### Internal
- !267 performs some code clean-up and improves the sphinx setup
    - This may lead to implicit loss of Python 3.6 compatibility (which is no longer supported officially since a few releases ago).
    - Makes use of newer language features (like consistent use of f-strings)
- !280 separates the `dantro.utils.data_ops` module into a `dantro.data_ops` subpackage
- !285 expands and improves the `dantro._import_tools` module and improves tests
- !287 updates documentation after the main branch was renamed from `master` to `main`
- !298 reworks the figure generation functions to be part of the test suite and be controllable via environment variables.
- !302 drops tests for minimal versions of dependencies; and drops version requirements of packages *altogether*, aiming for a more flexible development. Scheduled CI pipelines ensure that package combinations continue working.

#### Deprecations
- !278 deprecates use of `dantro.tools.IS_A_TTY` and `dantro.tools.TTY_COLS` constants. This information should be retrieved from the `dantro.tools.TERMINAL_INFO` dict instead.
- !280 deprecates imports from `dantro.utils.data_ops`; use `dantro.data_ops` instead.
- !282 deprecates `creator_type` and `creator_name` arguments to the `is_plot_func` decorator; use `creator` instead.

#### Removals
- !270 drops official support and testing for Python 3.7
- !282 completely removes the plot creator auto detection feature, reducing implementation complexity.
- !294 removes the deprecated `errorbar` and `errorbands` non-faceting plot functions; their functionality is easily replaceable by the facet grid `errorbars`plot and its `use_bands` argument.
- !298 removes deprecated `PlotManager` initialization arguments: `plots_cfg`, `base_cfg` and `update_base_cfg`.



## v0.17.2
- !266 Updates versions of pre-commit hooks to improve compatibility
- !265 Adds CI test environment for Python 3.10
- !269 Update requirements to more recent and more compatible version combinations


## v0.17.1
- !261 allows skipping plots if a file already exists at the desired plot output path; to use this option, set the `exist_ok` argument of a plot or plot creator to `skip`.
- !262 improves and expands the `PlotHelper` and the `multiplot` plot function.
    - The `multiplot` function now accepts axis-level arguments, allowing to call different function sequences on each subplot.
      Furthermore, ad-hoc imports of function calls are now possible.
    - A workaround for a bug in the `errorbars` plot is added (addressing #261)
    - The `PlotHelper` is extended with new axis-level helpers (`annotate`, `call`) and figure-level helpers (`align_labels`, `subplots_adjust`, and `figcall`).


## v0.17.0
- !231 improves the performance of the ``LabelledDataGroup`` selection methods (when using the ``merge`` or ``auto`` combination method). A new combination method ``auto`` is added and set as default.
- !257 reduces memory usage (see #251) by postponing coordinate resolution to the time they are actually needed and removing unnecessary cache attributes.
- !258 changes the sphinx theme for the documentation (#283) and adds a dantro logo :tada:
- !259 fixes file cache reading for xarray 0.18
- !256 adds new features and various improvements for plotting with the data transformation framework:
    - New features:
        - Exclude tags starting with `.` or `_` from `compute_only: all` (#272)
        - In `UniversePlotCreator`, allow to specify individual universes by name (#281)
    - Improvements:
        - Make DAG construction faster by using cPickle-based deepcopying (relevant when having many 100k nodes in the DAG)
        - Improve log and error messages in `PlotManager` and data transformation framework, now including more time information
        - Improve how error messages from failing data operations show their arguments (#276)
        - Let xarray objects loaded from file cache not be renamed (#275)
        - Make it easier to conditionally skip a plot by providing the `raise_SkipPlot` data operation (#165)
        - The `print_data` data operation now allows specifying the output format using a format string, greatly simplifying debugging of DAGs
        - New data operations added (see !256 changes for details)



## v0.16.3
- !254 adds the `define` syntax to the data transformation framework, allowing to specify transformations in a dict-based fashion
- !254 improves `MultiversePlotCreator`:
    - The `select_and_combine.transform_after_combine` argument can now be used to apply transformations to the data *after* the combination happened.
    - The `select_and_combine.combination_method` argument can now also be used to specify a custom combination operation, which can be just any data operation available elsewhere in the data operation framework.


## v0.16.2
- !253 implements parallel loading of files via the `DataManager` and is usable for all data loaders.
  Refer to the [`DataManager.load` docstring](https://dantro.readthedocs.io/en/stable/api/dantro.data_mngr.html#dantro.data_mngr.DataManager.load) for more information.


## v0.16.1
#### Features
- !249 adds the `set_tick_locators` and `set_tick_formatters` methods to the `PlotHelper` to enable advanced tick settings.
- !243 implements [error handling](https://dantro.readthedocs.io/en/latest/data_io/transform.html#error-handling) into the data transformation framework
- !243 integrates error handling into the `MultiversePlotCreator` to allow selecting plot data from parameter spaces with missing data (#256)

#### Enhancements
- !247, !248, and !250 make dependency version specifications more compatible and make testing infrastructure more robust
- !243 slightly extends the available data operations


## v0.16.0
#### Features and Improvements
- !241 makes `based_on` allow lookup from the same plots configuration and allows specifying multiple pools of base plot configurations.
- !240 adds the `build_object_array` operation
- `GraphGroup` improvements:
    - !242 implements dropping missing (NaN) values in the node and edge data.
    - !242 adds the `align` argument for property data alignment to `GraphGroup.create_graph`.
    - !235 improves the warnings on changed graph size.

#### Breaking changes and deprecations
- With this release, we **drop support for Python 3.6**.
- !241 deprecates the `PlotManager` arguments `base_cfg` and `update_base_cfg` and replaces them by `base_cfg_pool`.
  Furthermore, `plots_cfg` is renamed to `default_plots_cfg`.



## v0.15.4
#### Bug Fixes
- !237 fixes the `set_suptitle` helper, now allowing to set the suptitle's y-position.


## v0.15.3
#### Enhancements
- !232 generalizes the `determine_encoding` interface, no longer requiring xarray data and more easily allowing to use the tool in custom plot functions outside of dantro.


## v0.15.2
#### Enhancements
- !233 Speeds up `import dantro` by about 50%; this is achieved by delaying imports of packages that take a long time to load.


## v0.15.1
#### Enhancements
- !229 Makes `facet_grid` animation more tolerant by `squeeze`ing out size-1 dimension coordinates.


## v0.15.0
#### Features and Improvements
- !202 adds [meta-operations](https://dantro.readthedocs.io/en/latest/data_io/transform.html#meta-operations) to the data transformation framework (#174), thereby allowing to define function-like constructs which help with modularization.
- !218 improves path handling within the data tree:
    - Item access now allows accessing the parent object (via `../`) or the object itself (`./`), similar to navigating within POSIX paths.
    - Addressing #220, error messages are improved to more accurately show where item access went wrong, and even provide a hint for the correct key.
- Features and improvements in the **plotting framework**:
    - !222 adds the [`multiplot`](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html) function allowing configuration-based consecutive calls of any plot function that does not create a new figure and operates on the current axis. Most [`matplotlib`](https://matplotlib.org) plot functions, as well as many [`seaborn`](http://seaborn.pydata.org/index.html) plot functions, are readily accessible.
    - !224 adds the `errorbars` plot, which supports faceting and is additionally available via `kind: errorbars` in the general `facet_grid` plot.
    - !211 makes it possible to [use data transformation results inside other parts of the plot configuration](https://dantro.readthedocs.io/en/latest/plotting/plot_data_selection.html#using-data-transformation-results-in-the-plot-configuration), e.g. to specify plot helper arguments or `multiplot` arguments.
    - !215 adds the [`auto_encoding` feauture](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html#auto-encoding-of-plot-layout) to the generic plot functions `facet_grid` and `errorbar`, allowing more data-averse plot configurations. (!221 and !224 improve it further.)
    - !224 adds the [`make_facet_grid_plot` decorator](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html#add-custom-plot-kinds-that-support-faceting) which simplifies defining plots that support faceting.
      Plot functions that are decorated like this will also become available as a plot `kind` in the general `facet_grid` plot.
    - !225 improves error messages upon invalid `based_on` argument.
    - Various `PlotHelper` improvements:
        - !210 adds the `set_ticks` helper function, enabling setting tick locations and labels.
        - !224 adds the seaborn-based `despine` helper, working on individual axes
        - !224 adds distinctions between figure-level helpers and helpers that operate on a single axis.
        - !224 adds the `track_handles_labels` method, which allows to keep track of artists that should appear in a legend.
          The `set_legend` helper is extended accordingly and the `set_figlegend` function is added which supplies the same functionality but for a figure legend.
        - !224 improves `set_suptitle` and `set_figlegend` such that the created artist no longer overlaps with the existing subplots grid.
        - !224 allows to provide an axis object to `select_axis` to sync the `PlotHelper` to that axis.
        - !225 lets the `PlotHelper` skip helper invocation if an axis has no artists associated with it; this reduces warnings arising from working on empty axes.
          This behavior is enabled by default, but can be controlled for each helper via the `skip_empty_axes` argument.
- !207 improves the computation time for data selection in the `GraphGroup`.
- !208 addresses #199 by adding the `keep_dim` option in the `GraphGroup` to specify dimensions that are not squeezed during data selection.
- !217 improves the `GraphGroup`, now storing selection information as graph attribute (in `g.graph`) whenever data is added to the graph object.
- !204 makes pickling of the data tree possible. If building the data tree takes a long time, storing its structure to a tree cache file and restoring it can bring a speed-up.
    - Data tree objects can be pickled and unpickled manually. To be more versatile, dantro now uses [dill](https://pypi.org/project/dill/) for pickling.
    - The `DataManager.dump` method can be used to store the full tree.
    - The `DataManager.restore` method allows to populate an existing `DataManager` with the content of a stored data tree, either clearing existing data or merging them.
    - !205 adds default file path handling, controlled via the `default_tree_cache_path` argument to the `DataManager` or a class variable.
- !220 improves error messages upon missing data operations
- !223 improves the `LabelledDataGroup` selection interface, making it more consistent with xarray.
- !226 improves the performance of `KeyOrderedDict` and `IndexedDataGroup` by using insertion hints. This reduces the insertion complexity to constant for in-order or hinted insertions.
- **Minor API additions:**
    - !204 implements `BaseDataGroup.clear` to remove all entries from a group.
    - !204 adds the `overwrite` argument to `BaseDataGroup.recursive_update`.
    - !204 adds the `BasicComparisonMixin`, which supplies a simple `__eq__` magic method.
    - !216 extends the operations database with commonly used operations and makes operations on the `nx.` module easier.

#### Breaking changes and deprecations
- As of !204, the `PickleLoaderMixin` no longer allows choosing which load function to use via a class variable but _always_ uses `dill.load`.
- !226 removes the `print_params` argument of the `hdf5` data loader, replacing it with a trimmed down `progress_params` argument.
- With !224, the `errorbar` plot is deprecated in favour of the `errorbars` plot, which supports faceting.
- With the changes to path handling in !218, there are the following notable changes that depart from the behavior of the previous interface:
    - In addition to the `/` character, names of data containers or group may no longer contain a set of characters (e.g. `*` or `?`) as these may interfere with operations on paths.
    - The `BaseDataGroup.__contains__` method now returns true if the given path-like argument (e.g. `foo/bar`) is a valid argument to `__getitem__`
    and there is an object at that path.
      Previously, this check was implemented independently, thus behaving slightly differently.
      For group- or container-like arguments, the behavior remains as it was: a non-recursive check whether the _object_ is part of this group, using `is` comparison.
    - The `BaseDataGroup.new_group` method used to raise a `KeyError` if an intermediate path segment was not available; now, the intermediate groups will automatically be created and no such error is raised.

#### Bug fixes
- !205 addresses scipy netcdf warnings by requiring a more recent version.
- !206 fixes a regression in the [generic `errorbar` and `errorbands` plot functions](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html) where size-1 dimensions were not always squeezed out.
- !215 fixes passing on the file format to the `FileWriters`' `savefig` function in cases where it cannot be deduced from the filename.
- !214 makes dantro compatible to the latest h5py version, addressing #212, and sets the minimum version to 3.1.
- !211 fixes a bug that lead to an outdated `logstr` after renaming a group or container.
- !224 addresses an issue where a custom `style` context was lost upon a switch of animation mode (#173)
- !223 fixes the handling of non-dimension coordinates and of the ``drop`` argument in the  `LabelledDataGroup` selection interface (#234)

#### Internal
- !209 addresses #125 by reformatting all code using [black](https://black.readthedocs.io/en/stable/).
- !209 sets up [pre-commit infrastructure](https://pre-commit.com/) to automate code formatting.
- !218 moves custom dantro exception types into their own module and provides a common base class.



## v0.14.1
- !199 and !201 update the GitLab CI/CD configuration using latest GitLab features, e.g. to show code coverage inside Merge Request diffs.
- !200 fixes an error in data operation `create_mask` when `data.name` was `None`


## v0.14.0
#### Features and Improvements
- `PlotHelper` extensions and improvements (#94)
    - !181 improves error message handling, now composing a single error message for _all_ encountered errors and providing axis-specific information on the errors.
    - !181 extends `set_legend` to allow gathering handles and labels from already existing `matplotlib.legend.Legend` objects.
    - !181 improves docstrings of helper methods to convey more clearly which kinds of arguments are expected.
- The plotting framework now *experimentally* supports skipping of plots.
  Skipping is triggered via a custom `SkipPlot` exception, that users may raise in their plot functions.
  Additionally, the `MultiversePlotCreator` allows skipping a plot if the dimensionality of the associated multiverse is not in a set of expected dimensionalities.
- Documentation:
    - !163 adds the paper published in the [Journal of Open Source Software](https://joss.theoj.org) and corresponding information on how to cite it.
    - !187 adds links to the source files from which example code is included into the documentation.
    - !189 improves the names of the introductory guides, as proposed in #190.
    - !190 makes using the IPython directive possible, simplifying the embedding of code examples, addressing #188.
    - !190 adds a section describing the "Universe and Multiverse" terminology.
- Improvements of the README
    - !186 adds a dependency table and adds the `dev` installation extra to include all development-related dependencies.
    - !193 adds installation instructions for `conda` as an alternative to `pip` (see #184)

#### Bug fixes
- !185 renames licensing-related files in order to concur with the official LGPLv3 criteria and let [licensee](https://github.com/licensee/licensee/) correctly detect it.
- !184 addresses a bug in the `PlotHelper` that prevented helper invocations after the first error, see #181.


## v0.13.4
- !191 fixes a bug that prohibited a coordinate to be named `tolerance` in `UniversePlotCreator`, see #192.


## v0.13.3
- !183 fixes a bug in the specification of the `np.`, `xr.` and `scipy.` data operations.


## v0.13.2
- !181 adds DAG-based generic plot functions [`errorbar` and `errorbands`](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html).


## v0.13.1
- !178 extends the `count_unique` data operation to be applied along dimensions
- !174 fixes the handling of NaN values by the data operation `count_unique`
- !175 adjusts the `PlotManager` to allow UNIX shell wildcards in `plot_only` and gives more informative errors when unsupported characters are used in the plot name.
- !176 allows showing per-operation profiling statistics during DAG result computation, controlled by the `TransformationDAG.verbosity` attribute.
- !177 adds the `recursive_getitem` data operation
- !179 adds commonly-used builtin operations, e.g. `any` and `all`
- !180 adds the `ParamSpaceStateGroup.coords` property, allowing to retrieve the coordinates within the associated parameter space.
  Furthermore, a [documentation entry on the `ParamSpaceGroup`](https://dantro.readthedocs.io/en/latest/data_structures/groups/psp.html) is added.


## v0.13.0
#### Features and Improvements
- !170 adds an [integration guide](https://dantro.readthedocs.io/en/latest/integrating.html) to the documentation, illustrating how a dantro-based data processing pipeline can be built and integrated into a project.
- !158 allows the loading of external property data in the ``GraphGroup`` (#145).
- Extension of the [`data_ops` module](https://dantro.readthedocs.io/en/latest/data_io/data_ops.html):
    - !161 extends the operations database with commonly used operations and makes operations on `np.`, `xr.` and `scipy.` modules easier.
    - !160 adds the `expression` operation for evaluating [`sympy`](https://www.sympy.org/en/index.html) expressions
    - !165 adds the `lambda` operation, which can be used to define callables, and the [`curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) operation that requires such a callable.
    - !168 adds the `expand_object_array` operation, allowing to unpack an `object`-dtype array of arrays into a higher-dimensional array.
- Various [transformation framework](https://dantro.readthedocs.io/en/latest/data_io/transform.html) improvements:
    - !164 extends the [DAG minimal syntax](https://dantro.readthedocs.io/en/latest/data_io/transform.html#minimal-syntax) to allow specifying both positional and keyword arguments.
    - !160 adds [DAG syntax parsers](https://dantro.readthedocs.io/en/latest/data_io/dag_op_hooks.html) that can simplify DAG specification.
      Currently, this includes only a parser for the `expression` operation (#149).
    - !166 adds [DAG usage examples](https://dantro.readthedocs.io/en/latest/data_io/examples.html) to the documentation.
- !169 adds various improvements to the plotting framework:
    - Allow debugging individual plot configurations using the `debug` option in the plot configuration.
    - Improve error messages in `PlotManager`
    - Can now control the `PlotHelper`'s behaviour upon exceptions
    - The `PlotManager.plot_from_cfg` method is now also ignoring plot configurations with names starting with `.` (additionally to those starting with `_`)
    - Add a [plot configuration reference](https://dantro.readthedocs.io/en/latest/plotting/plot_cfg_ref.html) page to the documentation
- !171 allows to use `load_yml` and `write_yml` with paths that include the `~` to specify the current user's home directory.


#### Bug fixes
- !158 fixes a bug (#151) which led to edge property data not being associated correctly with the edges.
- !169 addresses #161, which prohibited specifying `out_dir` when plotting using `PlotManager.plot_from_cfg`.


## v0.12.5
- !159 fixes a bug (#147) which led to duplicate DAG cache files after a storage function error.


## v0.12.4
- !151 adds documentation of the `GraphGroup`.


## v0.12.3
- !153 adds the DAG-based generic `facet_grid` plot function that wraps [`xarray.plot`](http://xarray.pydata.org/en/stable/plotting.html) functionality and makes plots of high-dimensional data very convenient.
    - !157 extends the `facet_grid` plot function with animation support.
      This makes it possible to represent one further data dimension via the ``frames`` specifier.
    - More information can be found in [the documentation](https://dantro.readthedocs.io/en/latest/plotting/plot_functions.html).
- !154 adds the possibility to dynamically enter or exit animation mode from any `ExternalPlotCreator`-managed plot function.
- !155 addresses a bug (#141) that prohibited passing containers during initialization of `LabelledDataGroup` objects or objects of derived classes.


## v0.12.2
- !152 addresses a bug (#138) that prohibited using the short syntax in the `select_and_combine` field of the `MultiversePlotCreator`


## v0.12.1
- !149 updates dantro's requirements lists and extends the CI to also test for lower bounds of version requirements.


## v0.12.0
- As of this release, dantro is licensed under the [LGPLv3+ license](COPYING.md), added in !133.
- !141 adds automatic deployment of dantro to the [PyPI](https://pypi.org/project/dantro/).
- !134 adds a [Contribution Guide](CONTRIBUTING.md) and a [Code of Conduct](CODE_OF_CONDUCT.md).
- Furthermore, the dantro documentation is now deployed to Read the Docs, both for [stable versions](https://dantro.readthedocs.io/en/stable/) and for the [latest version](https://dantro.readthedocs.io/en/latest/).
  See !140 and !143 for more information.
- #92 adds a test job for a Python 3.8 environment to the CI pipeline
- #132 updates the graph-related vocabulary to `graph`, `nodes`, and `edges`.
- Various documentation improvements
    - #124 and !136 fix all broken references in the documentation and the docstrings and improve the Sphinx configuration.
    - !135 adds usage examples and includes code snippets from tests, thus automatically making sure that they work as intended.
    - Additionally, the CI now exits with a warning if Sphinx emitted any warnings, and a log file is made available via the job artifacts to inspect the Sphinx error log.
    - #117 improves, restructures, and extents the documentation, now covering the full range of dantro applications.
    - #135 eliminates typos and grammar issues in the documentation making it consistently use American English.


#### Important notes on upgrading
- Due to the changes introduced in !92, the netcdf4 package is no longer a dependency required by dantro.
  It is replaced by the more commonly used scipy package.
  To ensure that no interference occurs between a remaining installation of netcdf4 and the new dependencies, we suggest to uninstall it using `pip uninstall netcdf4`.
- With #132, the former `NetworkGroup` (now `GraphGroup`) and some of its attributes are renamed.


## v0.11.2
- #127 allows to disable `DataManager.load` operations via keyword argument; this is useful e.g. when passing arguments via recursively updated configuration hierarchies.


## v0.11.1
- #126 makes it possible to overwrite existing plots


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
- #111 updates the documentation to reflect that the `paramspace` dependency is [now available on PyPI][paramspace-pypi] and can be installed from there.


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
    - #102 makes the documentation available [online](https://dantro.readthedocs.io/)
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
   - It can be used in place of the group that holds the results of, e.g. simulations, carried out via a [`paramspace`][paramspace] parameter sweep.
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
- !19 Restrict [`paramspace`][paramspace] version to <2.0 in order to transition to a higher version in a more controlled manner.

## v0.3.2
- !18 With the [`paramspace`][paramspace] YAML constructors having changed, it became necessary to change their usage in dantro. This should result in no changes to the behaviour of dantro.

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


[dantro-docs]: https://dantro.readthedocs.io/
[dantro-repo]: https://gitlab.com/utopia-project/dantro
[utopya-repo]: https://gitlab.com/utopia-project/utopya
[utopia-project]: https://utopia-project.org/
[paramspace]: https://gitlab.com/blsqr/paramspace
[paramspace-pypi]: https://pypi.org/project/paramspace/
