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

### Modules
Before diving deeper, an overview over all `dantro` modules:

* `abc` and `base` define the interface for the following classes:
   * `BaseDataContainer`: a general data container
   * `BaseDataGroup`: a group gathers multiple containers (or other groups)
   * `BaseDataAttrs`: every container can store metadata in such an instance
   * `BaseDataProxy`: can be a proxy for data, postponing loading to when it is needed
* `container` and `group` implement some non-abstract classes for use as containers or groups
* `mixins` define general purpose mixin classes that can be used when defining a custom data container
* `data_mngr` defines the `DataManager` class:
   * an extended `BaseDataGroup` which serves as the _root_ of a tree
   * provides the ability to load data into the data tree
   * allows dict-like access
   * is associated with a directory
   * can be extended using mixin classes from the `data_loaders` module
* `proxy` holds the classes that "placeholder" objects can be created from
* `tools` holds general-purpose tools and helper functions


### Testing framework
To assert correct functionality, `pytest`s are written alongside new features. They are also part of the continuous integration.

`dantro` is tested for Python 3.6. Test coverage and pipeline status can be seen on [the project page](https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro).


## How to use dantro
A few examples of how to use `dantro`.

Often times, there are many possibilities and options available.
We advise to use `ipython` and its `? module.i.want.to.look.up` command to get the docstrings.

### Installation
If the project you want to use `dantro` with uses a virtual environment, enter it now.

Installation can happen directly via `pip`:
```
$ pip install git+ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/utopia/dantro.git
```
This will automatically resolve the needed dependencies.
<!-- TODO it won't do with `paramspace` added! Adjust this ... -->

If you do not have SSH keys available, use the HTTPS link. To install a certain branch, tag, or commit, see the [`pip` documentation](https://pip.pypa.io/en/stable/reference/pip_install/#git).

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
_wip_

#### How to load data
_wip_

### Example code
_wip_

### How to run the tests
To run the tests, the test dependencies also need to be installed. Clone the repository to a local directory, then:
```
$ source ~/.virtualenvs/dantro/bin/activate
(dantro) $ cd dantro
(dantro) $ pip install .[test_deps]
(dantro) $ python -m pytest -v tests/ --cov=dantro --cov-report=term-missing
```
