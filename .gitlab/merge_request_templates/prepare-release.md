<!-- 1 - Set as MR title: Prepare release of vX.Y -->
<!-- 2 - Adjust the following quick commands: -->
/label ~release
/milestone %"Version X.Y"

<!-- 3 - Fill in the MR description and the checklist below. -->

This MR prepares the vX.Y release of dantro.


### Can this MR be accepted?
- [ ] Set version number in [`dantro/__init__.py`](dantro/__init__.py)
   - Removed the pre-release specifier.
   - Version is now: `X.Y.0`
- [ ] Prepared [changelog](CHANGELOG.md) for release
   - Removed "WIP" in section heading
   - If necessary, re-ordered and cleaned-up the corresponding section
- [ ] Pipeline passes without warnings
- [ ] Approved by @  <!-- only necessary if there are substantial changes -->

<!-- 4 - If you are not allowed to merge, assign a maintainer now. -->
/assign @