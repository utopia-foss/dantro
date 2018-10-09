# Changelog

`dantro` aims to adhere to [semantic versioning](https://semver.org/).

## v0.4 _(WIP)_
- !17 implements some changes necessary for allowing a smooth transition of `deeevoLab` from `deval` to `dantro`, as implemented in yunus/deeevoLab!52. The changes involve:
   - Adding an `ObjectContainer` class that can hold arbitrary objects.
   - Improving the item access interface by allowing lists as arguments, not only strings; this reduces split-and-join operations and makes the interface more versatile.
   - Improvements to the `BaseDataManager`:
      - The default load configuration can now be set via class variables
      - It is now possible to load an entry into the `attrs` of a group or container.
      - Add a `PickleLoaderMixin` to load pickled objects into an `ObjectContainer`.
   - Miscellaneous minor improvements to features, code formatting, and documentation.
- !22 resolves issues that created two kinds of deprecation warnings.

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
