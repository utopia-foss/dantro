# Release Guide

This document describes the procedure of working on new dantro releases and the procedure of releasing a new dantro version. New releases are automatically deployed to the [PyPI](https://pypi.org/project/dantro/).

Below, replace all mentions of `vX.Y` with the to-be-released version number.
For patch releases, not all of the steps below apply; see [below](#bangbang-important-remarks).


## Working on a New Release
Issues and MRs that are to be worked on for a specific release are gathered under a new milestone called `Version X.Y`.

MRs that are being merged under that milestone should do the following:

* Bump the dantro version to a _pre-release_ version of `vX.Y`, starting at `vX.Y.0a0` and incrementing the trailing number for each further MR.
* Add a [changelog](CHANGELOG.md) entry under the `vX.Y` section (still marked "WIP").

After merging the first MR of a new milestone, a new _release branch_ should be created, which can be used for testing the pre-release versions of dantro:

```bash
git checkout master
git pull
git checkout -b release/vX.Y
git push --set-upstream origin release/vX.Y
```

After further MRs are merged, the release branch should also be updated:

```bash
git checkout master
git pull
git checkout release/vX.Y
git merge master
git push
```

:warning: Wildcard release branches (e.g., `v0.x`) should **not** be updated with pre-release versions!
See [important remarks](#bangbang-important-remarks) below for more info.


## Release Procedures

1. Check that all issues and MRs associated with the milestone are finished
1. Locally, create a new branch `prepare-release-vX.Y` and apply the following changes:
    * Adjust the version number in `dantro/__init__.py` to no longer be a pre-release version.
    * Remove the `WIP` flag from the corresponding changelog section for this release.
1. Commit both changes together with the following message: `Prepare release of vX.Y`
1. Push the branch and open a Merge Request, using the corresponding MR template. Follow additional procedures as described there.
    * On all branches starting with `prepare-release-`, an additional `test_minimal_deps` stage is carried out in the CI. It runs all tests anew, but with the lowest specified version of all dependencies.
    * Versions are specified in the `setup.py` file using the `>=` comparator. The lower bound is created by strictly requiring the specified version using `==`.
    * It *might* become necessary to update these lower version bounds if the corresponding jobs fail. Try to choose the lowest possible version that makes the pipeline pass for *all* Python versions.
1. After the MR is merged, pull the latest changes and update the release branch.
    ```bash
    git checkout master
    git pull
    git checkout release/vX.Y    # if not created previously, create it now
    git merge master
    git push
    ```
1. Update potentially existing "wildcard release branches", e.g. `v0.x`, should always be up to date with the latest minor release.
    ```bash
    git checkout release/vX.x
    git pull
    git merge release/vX.Y
    git push
    ```
1. Create a new release in the GitLab:
    * Create a new tag (Repository -> Tags -> New Tag)
    * Name the tag `vX.Y.Z`, including a patch release number (starting at zero)
    * Leave the "message" empty
    * As release notes, add the content of the corresponding section from `CHANGELOG.md` (best copied from the raw view in order to not lose any markdown syntax)
1. Update the project badges:
    * Navigate to Settings -> General -> Badges
    * Update the "latest release" badge with the latest version number
    * Update the "docs" badge's label with the new version number and the link with the new documentation link (deployed from the `release/vX.Y` branch)
1. Done! :tada:


## :bangbang: Important Remarks
* Patch releases (`vX.Y.Z`) should not have their own milestone but be gathered under the corresponding `vX.Y` milestone!
* Patch releases should not have their own release branch, but update the corresponding `release/vX.Y` (and potential wildcard release branches)!
* Wildcard release branches should **NOT** be updated with pre-release versions!
* Patch releases do not necessarily require pre-releases. If it's just a fix requiring a single MR, the version number can directly be bumped.
