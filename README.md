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

The `dantro` package is **open source software** released under the LGPLv3+ license (see [copyright notice](#copyright) below).
It was developed alongside the [Utopia project][utopia-project], but is an independent package.

We describe the motivation and scope of `dantro` in more detail in [this publication in the Journal of Open Source Software][dantro-joss-doi].
For more information on the package, its features, philosophy, and integration, please visit its **documentation** at [`dantro.readthedocs.io`][dantro-docs].
If you encounter any issues with `dantro` or have suggestions or questions of any kind, please open an issue via the [**project page**][dantro-project].



## Installing dantro
The `dantro` package is available [on the Python Package Index][pypi-dantro] and [via `conda-forge`][conda-forge-dantro].

If you are unsure which installation method works best for you, we recommend to use `conda`.

Note that — in order to make full use of `dantro`'s features — it is meant to be *integrated* into your project and customized to its needs.
Basic usage examples and an integration guide can be found in the [package documentation][dantro-docs].


### Installation via [`conda`][conda]
As a first step, install [Anaconda][Anaconda] or [Miniconda][Miniconda], if you have not already done so.
You can then use the following command to install dantro and its dependencies:

```bash
$ conda install -c conda-forge dantro
```


### Installation via [`pip`][pip]
If you already have a Python installation on your system, you probably already have `pip` installed as well.
To install dantro and its dependencies, invoke the following command:

```bash
$ pip install dantro
```

In case the `pip` command is not available, follow [these instructions][pip-installation] to install it or switch to the `conda`-based installation.
_Note_ that if you have both Python 2 and Python 3 installed, you might have to use the `pip3` command instead.



### Dependencies
`dantro` is implemented for [Python >= 3.6][Python3] and depends on the following Python packages:

| Package Name                  | Minimum Version  | Purpose                  |
| ----------------------------- | ---------------- | ------------------------ |
| [numpy][numpy]                | 1.19.4           | |
| [xarray][xarray]              | 0.16             | For labelled N-dimensional arrays |
| [dask][dask]                  | 2.10.1           | To work with large data |
| [toolz][toolz]                | 0.10             | For [dask.delayed][dask-delayed]
| [distributed][distributed]    | 2.10             | For distributed computing |
| [scipy][scipy]                | 1.5.3            | As engine for NetCDF files |
| [sympy][sympy]                | 1.6.1            | For symbolic math operations |
| [h5py][h5py]                  | 3.1              | For reading HDF5 datasets |
| [matplotlib][matplotlib]      | 3.2.1            | For data visualization |
| [seaborn][seaborn]            | 0.11             | For advanced data visualization |
| [networkx][networkx]          | 2.5              | For network visualization |
| [ruamel.yaml][ruamelyaml]     | 0.16.10          | For parsing YAML configuration files |
| [dill][dill]                  | 0.3.3            | For advanced pickling |
| [paramspace][paramspace]      | 2.5              | For dictionary- or YAML-based parameter spaces |



## Developing dantro
### Installation for developers
For installation of versions that are not on the PyPI, `pip` allows specifying an URL to a git repository:

```bash
$ pip install git+<clone-link>@<some-branch-name>
```

Here, replace `clone-link` with the clone URL of this project and `some-branch-name` with the name of the branch that you want to install the package from (see the [`pip` documentation][pip-install-docs] for details).
Alternatively, omit the `@` and everything after it.
If you do not have SSH keys available, use the HTTPS link.

If you would like to contribute to `dantro` (yeah!), you should clone the repository to a local directory:

```bash
$ git clone <clone-link>
```

For development purposes, it makes sense to work in a specific [virtual environment][venv] for dantro and install dantro in editable mode:

```bash
$ python3 -m venv ~/.virtualenvs/dantro
$ source ~/.virtualenvs/dantro/bin/activate
(dantro) $ pip install -e ./dantro
```


### Additional dependencies
For development purposes, the following additional packages are required.

| Package Name                  | Minimum Version  | Purpose                  |
| ----------------------------- | ---------------- | ------------------------ |
| [pytest][pytest]              | 3.4              | Testing framework        |
| [pytest-cov][pytest-cov]      | 2.5.1            | Coverage report          |
| [tox][tox]                    | 3.1.2            | Test environments        |
| [Sphinx][sphinx]              | 2.4 (< 3.0)      | Documentation generator  |
| [sphinx_rtd_theme][sphinxrtd] | 0.5              | Documentation HTML theme |
| [pre-commit][pre-commit]      | 2.8.2            | For [commit hooks](#commit-hooks) |
| [black][black]                | 20.8b1           | For code formatting      |

To install these development-related dependencies, enter the virtual environment, navigate to the cloned repository, and perform the installation using:

```bash
(dantro) $ cd dantro
(dantro) $ pip install -e .[dev]
```

With these dependencies having been installed, make sure to set up the git hook that allows pre-commit to run before making a commit:

```bash
(dantro) $ pre-commit install
```

For more information on commit hooks, see [the commit hooks section below](#commit-hooks).


### Testing framework
To assert correct functionality, tests are written alongside all features.
The [`pytest`][pytest] and [`tox`][tox] packages are used as testing frameworks.

All tests are carried out for Python 3.6 through 3.9 using the GitLab CI/CD and the newest versions of all [dependencies](#dependencies).
When merging to the master branch, `dantro` is additionally tested against the specified _minimum_ versions.

Test coverage and pipeline status can be seen on [the project page][dantro-project].


#### Running tests
To run all [defined tests](tests/), call:

```bash
(dantro) $ python -m pytest -v tests/ --cov=dantro --cov-report=term-missing
```
This also provides a coverage report, showing the lines that are *not* covered by the tests.

Alternatively, with [`tox`][tox], it is possible to select different python environments for testing.
Given that the interpreter is available, the test for a specific environment can be carried out with the following command:

```bash
(dantro) $ tox -e py37
```


### Documentation
#### Locally building the documentation
To build `dantro`'s documentation locally via [Sphinx][sphinx], install the required dependencies and invoke the `make doc` command:

```bash
(dantro) $ cd doc
(dantro) $ make doc
```

You can then view the documentation by opening the `doc/_build/html/index.html` file.

_Note:_ Sphinx is configured such that warnings will be regarded as errors, making detection of markup mistakes easier.
You can inspect the error logs gathered in the `doc/build_errors.log` file.
For Python-related Sphinx referencing errors, see the [`doc/.nitpick-ignore` file](doc/.nitpick-ignore) for exceptions

#### GitLab Documentation Environment
When developing dantro and pushing to the feature branch, the `build:doc` job of the CI pipeline additionally creates a documentation preview.
The result can either be downloaded from the job artifacts or the deployed GitLab environment.

Upon warnings or errors in the build, the job will exit with an orange warning sign.
You can inspect the `build_errors.log` file via the exposed CI artifacts.


### Commit hooks
To streamline dantro development, a number of automations are used which take care of code formatting and perform some basic checks.
These automations are managed by [pre-commit][pre-commit] and are run when invoking `git commit` (hence the name).

If these so-called hooks determine a problem, they will display an error and you will not be able to commit just yet.
Some of the hooks automatically fix the error (e.g.: removing whitespace), others require some manual action on your part.
_Either way,_ you will have to stage these changes manually (using `git add`, as usual).
To check which changes were made by the hooks, use `git diff`.

Once you applied the requested changes, invoke `git commit` anew.
This will again trigger the hooks, but — with all issues resolved — the hooks should now all pass and lead you to the usual commit message prompt.

The most notable hooks are:

* [black][black]: The uncompromising code formatter
* [isort][isort]: Systematically sorts Python `import` statements

Both [isort][isort] and [black][black] are configured in the [`pyproject.toml`](pyproject.toml) file.
For the other hooks' configuration, see [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
All hooks are also being run in the [GitLab CI/CD](.gitlab-ci.yml) `check:hooks` job.

If you have trouble setting up the hooks or if they create erroneous results, please let us know.



## Troubleshooting
### Install test and/or documentation dependencies when using `zsh`
If you use a `zsh` terminal (default for macOS users since Catalina) and try to install extra requirements like the test and/or documentation dependencies, you will probably get an error similar to `zsh: no matches found: .[test_deps]`.
This can be fixed by escaping the square brackets, i.e. writing `.\[test_deps\]` or  `.\[doc_deps\]`.




## Copyright
dantro is licensed under the [GNU Lesser General Public License Version 3][LGPLv3] or any later version.

### Copyright Notice

    dantro -- a python package for handling and plotting hierarchical data
    Copyright (C) 2018 – 2020  dantro developers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

A copy of the [GNU General Public License Version 3][GPLv3], and the [GNU Lesser General Public License Version 3][LGPLv3] extending it, is distributed with the source code of this program; see [`COPYING`](COPYING) and [`COPYING.LESSER`](COPYING.LESSER), respectively.


### Copyright Holders

The copyright holders of dantro are collectively referred to as _dantro developers_ in the respective copyright notices and disclaimers.

dantro has been developed by (in alphabetical order):

* Unai Fischer Abaigar
* Benjamin Herdeanu
* Daniel Lake
* Yunus Sevinchan
* Jeremias Traub
* Julian Weninger

Contact the developers via: [`dantro-dev@iup.uni-heidelberg.de`][devmail]

[dantro-project]: https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro
[dantro-docs]: https://dantro.readthedocs.io/
[pypi-dantro]: https://pypi.org/project/dantro/
[conda-forge-dantro]: https://anaconda.org/conda-forge/dantro

[utopia-project]: https://ts-gitlab.iup.uni-heidelberg.de/utopia/utopia

[dantro-joss-doi]: https://doi.org/10.21105/joss.02316

[pip]: https://pip.pypa.io/en/stable/
[pip-installation]: https://pip.pypa.io/en/stable/installing/
[conda]: https://conda.io/en/latest/
[Anaconda]: https://www.anaconda.com/download
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html

[Python3]: https://www.python.org/downloads/
[numpy]: https://numpy.org
[scipy]: https://www.scipy.org
[xarray]: http://xarray.pydata.org/en/stable/
[dask]: https://dask.org
[toolz]: https://toolz.readthedocs.io/en/latest/
[dask-delayed]: https://docs.dask.org/en/latest/delayed.html
[distributed]: https://distributed.dask.org/en/latest/
[h5py]: http://www.h5py.org
[sympy]: https://www.sympy.org/
[matplotlib]: https://matplotlib.org
[networkx]: https://networkx.github.io
[ruamelyaml]: https://yaml.readthedocs.io/en/latest/
[dill]: https://pypi.org/project/dill/
[paramspace]: https://pypi.org/project/paramspace/

[pytest]: https://pytest.org/en/latest/
[pytest-cov]: https://pytest-cov.readthedocs.io/en/latest/
[tox]: https://tox.readthedocs.io/en/latest/
[sphinx]: https://www.sphinx-doc.org/
[sphinxrtd]: https://sphinx-rtd-theme.readthedocs.io/en/stable/
[pre-commit]: https://pre-commit.com
[black]: https://black.readthedocs.io/en/stable/
[isort]: https://pycqa.github.io/isort/

[pip-install-docs]: https://pip.pypa.io/en/stable/reference/pip_install/#git
[venv]: https://docs.python.org/3/library/venv.html

[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html
[LGPLv3]: https://www.gnu.org/licenses/lgpl-3.0.en.html
[devmail]: mailto:dantro-dev@iup.uni-heidelberg.de
