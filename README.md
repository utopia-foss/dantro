# `dantro`

The `dantro` package — from *data* and *dentro* (gr., for tree) – is a Python package to work with hierarchically organized data.
It allows loading possibly heterogeneous data into a tree-like structure that can be conveniently handled for data manipulation, analysis, and plotting.

While being developed alongside the [Utopia project](https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia), dantro aims to remain general enough to be usable outside of that context as well.

This README focuses primarily on informing about the procedures to get dantro installed.
For more information on the philosophy and usage of dantro, have a look at the [online documentation](https://hermes.iup.uni-heidelberg.de/dantro_doc/master/html/).
The development history can be inspected in the [changelog](CHANGELOG.md).



## Installing dantro
If the project you want to use `dantro` with uses a virtual environment, enter it now.

To then install dantro, use pip and the clone link of this project:

```bash
(some-venv) $ pip install git+<clone-link>
```

If you do not have SSH keys available, use the HTTPS link.
To install a certain branch, tag, or commit, see the [`pip` documentation](https://pip.pypa.io/en/stable/reference/pip_install/#git).
Available release branches can be found [on the project page](https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro/-/branches/all?utf8=✓&search=release%2F).



## Developing dantro
### Installation for developers
If you would like to contribute to `dantro` (yeah!), you should clone the repository to a local directory:

```bash
git clone ssh://git@ts-gitlab.iup.uni-heidelberg.de:10022/utopia/dantro.git
```

For development purposes, it makes sense to work in a [python3 virtual environment](https://docs.python.org/3/library/venv.html) specific for dantro and install dantro in editable mode:

```bash
$ python3 -m venv ~/.virtualenvs/dantro
$ source ~/.virtualenvs/dantro/bin/activate
(dantro) $ pip install -e ./dantro
```

### Testing framework
To assert correct functionality, tests are written alongside new features.
They are also part of the continuous integration.
For development, dantro advertises a test-driven approach.

`dantro` is tested for Python 3.6 and 3.7.
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

_Note:_ This relies on `paramspace` being available to `tox`, i.e.: it either needs to be installed in the virtual environment or as part of the system-site-packages.




## Documentation
### Online Documentation
The current documentation is available [online](https://hermes.iup.uni-heidelberg.de/dantro_doc/master/html/).

Additionally, when developing dantro and pushing to the feature branch, the `build:doc` job of the CI pipeline also creates a documentation.
The result can either be downloaded from the job artifacts or from the deployed environment.

### Local Documentation
To build `dantro`'s documentation locally, install the required dependencies and invoke the `make doc` command:

```bash
(dantro) $ pip install .[doc_deps]
(dantro) $ cd doc
(dantro) $ make doc
```

You can then view the documentation by opening the `doc/_build/html/index.html` file.
