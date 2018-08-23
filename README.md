# dantro

The `dantro` package — from *data* and *dentro* (gr., for tree) – is a Python package to work with hierarchically organized data.
It allows loading possibly heterogeneous data into a tree-like structure that can be conveniently handled for data manipulation, analysis, plotting, ... you name it.

It emerged as an abstraction of the [`deval`](https://ts-gitlab.iup.uni-heidelberg.de/yunus/deval) package and focusses on the loading of data and supplying convenient classes to work with.
While being developed alongside the [Utopia project](https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia), it aims to remain general enough to be usable outside of that context.

#### What you will find in this README
* A few words about the [package structure](#package-structure)
* and the [testing framework](#testing-framework)
* [How to use `dantro`](#how-to-use-dantro) in your own project

#### Further reading
* [`CHANGELOG.md`](CHANGELOG.md)
* _(link to documentation)_
* ...

## Package structure

### Philosophy
`dantro` aims to be rather general, trying to not impose restrictions on:
* types of data to work with,
* the ways to load that data,
* the ways to process data,
* or the ways to plot data.

A common approach is to just use the package to strictly define a common interface and outsource all specializations to the projects that need to use it.
However, this can result in needing to write (and re-write!) a lot of code outside of this package, which can become hard to maintain.

In order to avoid this, `dantro` not only supplies an interface, but also provides classes that can be easily customised to fit the needs of a certain use case.

#### Enter: mixin classes.
This refers to the idea that functionality can be added to classes using [multiple inheritance](https://docs.python.org/3/tutorial/classes.html#multiple-inheritance).

For example, if a `DataManager` is desired that needs a certain load functionality, this is specified simply by _additionally_ inheriting the mixin class:

```python
from dantro.data_mngr import DataManager
from dantro.data_loaders import YamlLoaderMixin

class MyDataManager(YamlLoaderMixin, DataManager):
    """My data manager can load YAML files."""
    pass  # Done here. Nothing else to do.
```

This concept is extended also to ways how `DataContainer` classes can be specialized.

### Modules
Before diving deeper, an overview over all `dantro` modules:

* `abc` and `base` define the interface for the following classes:
   * `BaseDataContainer`: a general data container
   * `BaseDataGroup`: a group gathers multiple containers (or other groups)
   * `BaseDataAttrs`: every container can store metadata in such an instance
   * `BaseDataProxy`: can be a proxy for data, postponing loading to when it is needed
   * `AbstractPlotCreator`: defines the interface between `PlotManager` and `BasePlotCreator` classes (implemented elsewhere, see below)
* `container` and `group` implement some non-abstract classes for use as containers or groups
* `mixins` define general purpose mixin classes that can be used when defining a custom data container
* `data_mngr` defines the `DataManager` class:
   * an extended `BaseDataGroup` which serves as the _root_ of a tree
   * provides the ability to load data into the data tree
   * allows dict-like access
   * is associated with a directory
   * can be extended using mixin classes from the `data_loaders` module
* `proxy` holds the classes that "placeholder" objects can be created from
* `plot_creators` is a sub-package with the following modules:
   * `pcr_base` holds the implementation of the `BasePlotCreator`
   * `pcr_*` modules hold implementations of derived classes
* `plot_mngr` implements the `PlotManager`, which handles the configuration of plots and passes it on to the `BasePlotCreator`-derived classes
* `tools` holds general-purpose tools and helper functions


### Testing framework
To assert correct functionality, `pytest`s are written alongside new features. They are also part of the continuous integration.

`dantro` is tested for Python 3.6 and 3.7.
Test coverage and pipeline status can be seen on [the project page](https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro).


## Installation
If the project you want to use `dantro` with uses a virtual environment, enter it now.

Before being able to install `dantro`, one external dependency that is not on the python package index, needs to be installed, `paramspace`:
```
$ pip install git+ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/yunus/paramspace.git
```

After that, installation of `dantro` can happen directly via `pip`:
```
$ pip install git+ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/utopia/dantro.git
```
This will automatically resolve the remaining dependencies.

If you do not have SSH keys available, use the HTTPS link. To install a certain branch, tag, or commit, see the [`pip` documentation](https://pip.pypa.io/en/stable/reference/pip_install/#git).


#### For developers
If you would like to contribute to `dantro` (yeah!), you should clone the repository to a local directory:

```
git clone ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/utopia/dantro.git
```

For development purposes, it makes sense to work in a [python3 virtual environment](https://docs.python.org/3/library/venv.html) specific for dantro:
```
$ python3 -m venv ~/.virtualenvs/dantro
$ source ~/.virtualenvs/dantro/bin/activate
(dantro) $ 
```

To run the tests, the test dependencies also need to be installed. To do so, navigate to the cloned repository, enter it and perform an installation using:
```
(dantro) $ cd dantro
(dantro) $ pip install .[test_deps]
```

To then perform the tests, call:
```
(dantro) $ python -m pytest -v tests/ --cov=dantro --cov-report=term-missing
```

Alternatively, with the [`tox`](https://tox.readthedocs.io/en/latest/) framework, it is possible to select different python environments for testing.
Given that the interpreter is available, the test for a specific environment can be carried out with the following command:
```
(dantro) $ tox -e py37
```
_Note:_ that this relies on `paramspace` being available to `tox`, i.e.: it either needs to be installed in the virtual environment or as part of system-site-packages.



## How to use dantro
A few examples of how to use `dantro`.

Often times, there are many possibilities and options available.
We advise to use `ipython` and its `? module.i.want.to.look.up` command to get the docstrings.


### How to create a custom data container
As an example, let's look at the implementation of the `MutableSequenceContainer`, a container that is meant to store mutable sequences:
```python
# Import the python abc we want to adhere to
from collections.abc import MutableSequence

# Import base mixin classes (others can be found in the mixin module)
from dantro.base import BaseDataContainer, ItemAccessMixin, CollectionMixin, CheckDataMixin


class MutableSequenceContainer(CheckDataMixin, ItemAccessMixin, CollectionMixin, BaseDataContainer, MutableSequence):
    """The MutableSequenceContainer stores data that is sequence-like"""
    # ...
```
The steps to arrive at this point are as follows:

The [`collections.abc` python module](https://docs.python.org/3/library/collections.abc.html) is what also specifies the interface for python-internal classes.
There, it says that the `MutableSequence` inherits from `Sequence` and has the following abstract methods: `__getitem__`, `__setitem__`, `__delitem__`, `__len__`, and `insert`.

As we want the resulting container to adhere to this interface, we set `MutableSequence` as the first class to inherit from.

The `BaseDataContainer` is what makes this object a data container. It implements some methods, but leaves others abstract.

Now, we need to supply implementations of these abstract methods. That is the job of the following two (reading from right to left) mixin classes.  
In this case, the `Sequence` interface has to be fulfilled. As a `Sequence` is nothing more than a `Collection` with item access, we can fulfill this by inheriting from the `CollectionMixin` and the `ItemAccessMixin`.

The `CheckDataMixin` is an example of how functionality can be added to the container while still adhering to the interface. This mixin checks the provided data before storing it and allows specifying whether unexpected data should lead to warnings or exceptions.

Some methods will still remain abstract, e.g. `insert`. These are the only ones that need to be manually instantiated.

#### Using a data container
Once defined, instantiation is easy:
```python
dc = MutableSequenceContainer(name="my_mutable_sequence",
                              data=[16, 8, 4])

# Insert values
dc.insert(0, 2)
dc.insert(0, 1)

# Item access and collection interface
assert 16 in dc  # True
assert 32 in dc  # False
print(dc[0])     # 16

for num in dc:
    print(num, end=", ")
# 16, 8, 4, 2, 1, 
```

### How to create a custom data manager
As an example, the [`utopya.DataManager`](https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/blob/master/python/utopya/utopya/datamanager.py):

```python
import dantro as dtr
import dantro.data_mngr
from dantro.data_loaders import YamlLoaderMixin, Hdf5LoaderMixin

from utopya.datacontainer import NumpyDC

class DataManager(Hdf5LoaderMixin, YamlLoaderMixin, dtr.data_mngr.DataManager):
    """This class manages the data that is written out by Utopia simulations.

    It is based on the dantro.DataManager class and adds the functionality for
    specific loader functions that are needed in Utopia: Hdf5 and Yaml.
    """

    # Tell the HDF5 loader which container class to use
    _HDF5_DSET_DEFAULT_CLS = NumpyDC

```

That's all. The only thing not visible here is the definition of `NumpyDC`, but this happens as described above. (In fact, [nothing more](https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia/blob/master/python/utopya/utopya/datacontainer.py) is done there.)


#### How to load data
To load data, a data manager needs to be instantiated.
```python
dm = DataManager(data_dir="~/output_dir/my_data")

# Now, data can be loaded using the `load` command:
dm.load("some_data",       # where to load the data to
        loader="yaml",     # which loader to use
        glob_str="*.yml")  # which files to find and load
```

#### Using a pre-defined load configuration
For a known structure of output data, it makes sense to pre-define the configuration somewhere. It can then be passed to the `DataManager` during initialization.

Again, an example from `utopya`:
```yaml
data_manager:
  # Where to create the output directory for this DataManager, relative to
  # the run directory of the Multiverse.
  out_dir: eval/{date:}
  # The {date:} placeholder is replaced by the current timestamp such that
  # future DataManager instances that operate on the same data directory do
  # not create collisions.

  # Supply a default load configuration for the DataManager
  load_cfg:
    # Load the frontend configuration files from the config/ directory
    # Each file refers to a level of the configuration that is supplied to
    # the Multiverse: base <- user <- model <- run <- update
    cfg:
      loader: yaml
      glob_str: 'config/*.yml'
      required: true
      path_regex: config/(\w+)_cfg.yml
      target_path: cfg/{match:}

    # Load the configuration files that are generated for _each_ simulation
    # These hold all information that is available to a single simulation and
    # are in an explicit, human-readable form.
    uni_cfg:
      loader: yaml
      glob_str: universes/uni*/config.yml
      required: true
      path_regex: universes/uni(\d+)/config.yml
      target_path: uni/{match:}/cfg

    # Load the binary output data from each simulation.
    data:
      loader: hdf5_proxy
      glob_str: universes/uni*/data.h5
      required: true
      path_regex: universes/uni(\d+)/data.h5
      target_path: uni/{match:}/data

    # The resulting data tree is then:
    #  └┬ cfg
    #     └┬ base
    #      ├ meta
    #      ├ model
    #      ├ run
    #      └ update
    #   └ uni
    #     └┬ 0
    #        └┬ cfg
    #         └ data
    #           └─ ...         
    #      ├ 1
    #      ...
```

Once the `DataManager` is configured this way, it becomes very easy to load data:
```python
dm = DataManager(data_dir="~/output_dir/my_data", load_cfg=load_cfg_dict)
dm.load_from_cfg()

# Access the data
dm['cfg']['meta']['something_something']
# ...
```


### How to create plots
For creation of plots, `dantro` provides the `PlotManager` and the `PlotCreator` classes.

The `PlotManager` does not actually carry out any plots. Its purpose is to handle the configuration of the `PlotCreator` classes; those implement the actual plotting functionality.  
This way, the plots can be configured in a consistent way, profiting from the shared interface and the already implemented functions, while keeping the flexibility of having multiple ways to create plots.

A brief usage example:
```python
# Load the data given a load configuration
dm = DataManager(data_dir="/path/to/the/data/to/plot", load_cfg=load_cfg)
dm.load_from_cfg()

# Initialize a plot manager and provide it with that data manager
pm = PlotManager(dm=dm)

# Perform a single plot:
pm.plot("my_plot",            # name of the plot
        creator="external",   # plot creator to use
        # all further: kwargs to that plot creator
        module=".basic", plot_func="lineplot", y="vectors/values")
```

Like the `DataManager`, the `PlotManager` also provides a `plot_from_cfg` method, which allows passing a pre-existing configuration to generate multiple plots.

This is illustrated with examples using the `ExternalPlotCreator`, a class that makes it easy to use external scripts to create plots.
Multiple ways of loading a plotting function are supplied:
* It allows importing external scripts that receive a `DataManager` and an `out_path` as arguments.
* All other arguments of the configuration are passed along
* The script can do whatever it wants, also meaning that it _has_ to do everything by itself (getting data, saving plots, closing figures ...)
* Plotting functions can be imported from three locations:
   * The included `ext_funcs` subpackage, which currently supplies the `lineplot` method
   * An already importable module, i.e. one that is installed or can be found in `sys.path`
   * A module loaded from a file

A configuration example would be the following:
```yaml
values_over_time:  # this will also be the final name of the plot (without extension)
  # Select the creator to use
  creator: external
  # NOTE: This has to be known to `PlotManager` under this name.
  #       It can also be set as default during `PlotManager` initialisation.

  # Specify the module to find the plot_function in
  module: .basic  # Uses the dantro-internal plot functions

  # Specify the name of the plot function to load from that module
  plot_func: lineplot

  # The data manager is passed to that function as first positional argument.
  # Also, the generated output path is passed as `out_path` keyword argument.

  # All further kwargs on this level are passed on to that function.
  # Specify how to get to the data in the data manager
  x: vectors/times
  y: vectors/values

  # Specify styling
  fmt: go-
  # ...

my_fancy_plot:
  # Select the creator to use
  creator: external

  # This time, get the module from a file
  module_file: /path/to/my/fancy/plotting/script.py
  # NOTE Can also be a relative path, if `base_module_file_dir` was set

  # Get the plot function from that module
  plot_func: my_plot_func

  # All further kwargs on this level are passed on to that function.
  # ...
```
This will create two plots: `values_over_time` and `my_fancy_plot`. Both are using `ExternalPlotCreator` (known to `PlotManager` by name `external`) and are loading certain functions to use for plotting.


#### Using parameter sweeps for creating multiple plots
Using the `paramspace` package, it is also easily possible to sweep over a configuration _on the level of single output files_, i.e.: each point in parameter space is a single call to a plot creator.

```yaml
multiple_plots: !pspace
  creator: external
  module: .basic
  plot_func: lineplot

  # All further kwargs on this level are passed on to that function.
  x: vectors/times

  # Create multiple plots with different y-values
  y: !pdim
    default: vectors/values
    values:
      - vectors/values
      - vectors/more_values
```
This will create two _files_, one with `values` over `times`, one with `more_values` over `times`. Having multiple lines in one plot would be the job of the plotting function used.

#### More creators
In the future, a `DeclarativePlotCreator` will allow to specify – in a declarative syntax – how the data should be loaded, potentially transformed, and then represented in a plot.
