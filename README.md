# dantro

The `dantro` package — from *data* and *dentro* (gr., for tree) – is a Python package to work with hierarchically organised data.
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

A common approach is to just use the package to strictly define a common interface and outsource all specialisations to the projects that need to use it.
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

This concept is extended also to ways how `DataContainer` classes can be specialised.

### Data structures
Before diving deeper, let's get an overview over important `dantro` classes:


### Testing framework


## How to use dantro
