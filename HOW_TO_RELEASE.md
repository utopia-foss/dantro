# Release Guide

This document describes the procedure of working on new dantro releases and the procedure of releasing a new dantro version. New releases are automatically deployed to the [PyPI](https://pypi.org/project/dantro/).

Below, replace all mentions of `vX.Y` with the to-be-released version number.
For patch releases, not all of the steps below apply; see [below](#bangbang-important-remarks).


## Working on a New Release
Issues and MRs that are to be worked on for a specific release are gathered under a new milestone called `Version X.Y`.

MRs that are being merged under that milestone should do the following:

* Bump the dantro version to a _pre-release_ version of `vX.Y`, starting at `vX.Y.0a0` and incrementing the trailing number for each further MR.
* Add a [changelog](CHANGELOG.md) entry under the `vX.Y` section (still marked "WIP").

### Release branches *(optional, see remarks below)*
After merging the first MR of a new milestone, a new *release branch* can be created, which can be used for testing the pre-release versions of dantro:

```bash
git checkout main
git pull
git checkout -b release/vX.Y
git push --set-upstream origin release/vX.Y
```

After further MRs are merged, the release branch should also be updated:

```bash
git checkout main
git pull
git checkout release/vX.Y
git merge main
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
1. After the MR is merged, pull the latest changes
    ```bash
    git checkout main
    git pull
1. *Optionally,* update the release branches (but see remarks below):
    ```bash
    # Specific release branch
    git checkout release/vX.Y    # if not created previously, create it now
    git merge main
    git push

    # Potentially existing wildcard release branches, e.g. v0.x (should always be up to date with the latest minor release.)
    git checkout release/vX.x
    git pull
    git merge release/vX.Y
    git push
    ```
1. Create a new release in the GitLab:
    * [Deployments -> Releases -> New Release](https://gitlab.com/utopia-project/dantro/-/releases/new)
    * Name the tag `vX.Y.Z`, including a patch release number (starting at zero)
        * This should match the package `__version__`, also for pre-releases
        * For pre-release versions, versions look like `vX.Y.ZaA` or `vX.Y.ZbB` (with `A` and `B` being numbers)
    * Select the `main` branch to create the release from.
        * If release branches were created, select those instead.
    * Leave the tag's "message" empty.
    * For release notes, add the content of the corresponding section from `CHANGELOG.md` (best copied from the raw view in order to not lose any markdown syntax).
1. Done! :tada:


## :bangbang: Important Remarks
* Patch releases (`vX.Y.Z`)  and pre-releases (`vX.Y.Z.aA`) should *not* have their own milestone but be gathered under the corresponding `vX.Y` milestone!
* Patch releases do not necessarily require pre-releases. If it's just a fix requiring a single MR and a quick release, the version number can directly be bumped.
* Regarding **release branches**:
    * This was done for *every* release prior to dantro v0.18; since then it is only done sporadically.
    * Patch releases should not have their own release branch, but update the corresponding `release/vX.Y` (and potential wildcard release branches)!
    * Wildcard release branches should **NOT** be updated with pre-release versions!
