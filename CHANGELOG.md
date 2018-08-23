# Changelog

`dantro` aims to adhere to [semantic versioning](https://semver.org/).

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
