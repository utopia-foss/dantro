# `dantro`: handle, transform, and visualize hierarchically structured data

`dantro`—from *data* and *dentro* (Greek for *tree*)—is a Python package that provides a uniform interface for hierarchically structured and semantically heterogeneous data.
It is built around three main features:

* **data handling:** loading heterogeneous data into a tree-like data structure and providing a uniform interface for it
* **data transformation:** performing arbitrary operations on the data, if necessary using lazy evaluation
* **data visualization:** creating a visual representation of the processed data

Together, these stages constitute a **data processing pipeline**: an automated sequence of predefined, configurable operations.
Akin to a Continuous Integration pipeline, a data processing pipeline provides a uniform, consistent, and easily extensible infrastructure that contributes to more efficient and reproducible workflows.
This can be beneficial especially in a scientific context, for instance when handling data that was generated by computer simulations.

`dantro` is meant to be **integrated** into projects and to be used to set up such a data processing pipeline.
It is designed to be **easily customizable** to the requirements of the project it is integrated into, even if the involved data is hierarchically structured or semantically heterogeneous.
Furthermore, it allows a **configuration-based specification** of all operations via YAML configuration files; the resulting pipeline can then be controlled entirely via these configuration files and without requiring code changes.

The `dantro` package is **open source software** released under the [LGPLv3+](COPYING.md).
It was developed alongside the [Utopia project](https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia), but is an independent package.

For more information on `dantro`, its features, philosophy, and integration, please visit its **documentation** at [`dantro.readthedocs.io`](https://dantro.readthedocs.io/).



## Installing dantro
The `dantro` package is available [on the Python Package Index](https://pypi.org/project/dantro/).
The recommended way of installing it is via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install dantro
```

Note that — in order to make full use of `dantro`'s features — it is meant to be *integrated* into your project and customized to its needs.
Usage examples and an integration guide can be found in the [package documentation](https://dantro.readthedocs.io/).



## Developing dantro
### Installation for developers
For installation of versions that are not on the PyPI, `pip` allows specifying a git repository:

```bash
pip install git+<clone-link>@<some-branch-name>
```

Here, replace `clone-link` with the clone URL of this project and `some-branch-name` with the name of the branch that you want to install the package from (see the [`pip` documentation](https://pip.pypa.io/en/stable/reference/pip_install/#git) for details).
Alternatively, omit the `@` and everything after it.
If you do not have SSH keys available, use the HTTPS link.

If you would like to contribute to `dantro` (yeah!), you should clone the repository to a local directory:

```bash
git clone ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/utopia/dantro.git
```

For development purposes, it makes sense to work in a specific [virtual environment](https://docs.python.org/3/library/venv.html) for dantro and install dantro in editable mode:

```bash
$ python3 -m venv ~/.virtualenvs/dantro
$ source ~/.virtualenvs/dantro/bin/activate
(dantro) $ pip install -e ./dantro
```

### Testing framework
To assert correct functionality, tests are written alongside all features.
These tests are carried out with continuous integration.
For development, dantro advertises a test-driven approach.

`dantro` is tested for Python 3.6 through 3.8 using a Continuous Integration pipeline.
Test coverage and pipeline status can be seen on [the project page](https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro).

#### Installing test dependencies
The [`pytest`](https://pytest.org/en/latest/) and [`tox`](https://tox.readthedocs.io/en/latest/) packages are used as testing frameworks.

To install the dependencies required for performing tests, enter the virtual environment, navigate to the cloned repository, and perform the installation using:

```bash
(dantro) $ cd dantro
(dantro) $ pip install .[test_deps]
```

#### Running tests
To run all [defined tests](tests/), call:

```bash
(dantro) $ python -m pytest -v tests/ --cov=dantro --cov-report=term-missing
```
This also provides a coverage report, showing the lines that are *not* covered by the tests.

Alternatively, with [`tox`](https://tox.readthedocs.io/en/latest/), it is possible to select different python environments for testing.
Given that the interpreter is available, the test for a specific environment can be carried out with the following command:

```bash
(dantro) $ tox -e py37
```


### Documentation
#### Locally building the documentation
To build `dantro`'s documentation locally via [Sphinx](https://www.sphinx-doc.org/), install the required dependencies and invoke the `make doc` command:

```bash
(dantro) $ pip install .[doc_deps]
(dantro) $ cd doc
(dantro) $ make doc
```

You can then view the documentation by opening the `doc/_build/html/index.html` file.

_Note:_ Sphinx is configured such that warnings will be regarded as errors, making detection of markup mistakes easier.
You can inspect the error logs gathered in the `doc/build_errors.log` file.
For Python-related Sphinx referencing errors, see the [`doc/.nitpick-ignore` file](doc/.nitpick-ignore) for exceptions

#### Documentation Environment
When developing dantro and pushing to the feature branch, the `build:doc` job of the CI pipeline additionally creates a documentation preview.
The result can either be downloaded from the job artifacts or the deployed GitLab environment.

Upon warnings or errors in the build, the job will exit with an orange warning sign.
You can inspect the `build_errors.log` file via the exposed CI artifacts.
