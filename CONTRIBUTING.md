# Contributing to dantro

We are happy that you found your way to this document.

:sparkles: **Thank you for contributing to dantro!** :sparkles:


## How to Contribute

The dantro package is open source software.
We thus strive to make every stage of development public, sharing our advances, and incorporating community contributions.
To achieve this transparency, we coordinate the whole process openly via this GitLab project.

The dantro package is licensed under the [GNU Lesser General Public License Version 3](https://www.gnu.org/licenses/lgpl-3.0.en.html).
As a contributor to the project and source code you agree that your contributions are published and may be distributed according to this license.
At your discretion you can be acknowledged in the [list of copyright holders](README.md#copyright-holders); please use your first contribution to dantro to inform us about your choice.

In brief, to contribute to dantro, the following steps are involved:

1. [Get access](#gitlab-account) to the dantro GitLab project
1. [Write an issue](#issues) to ask your question, report a bug, or suggest a feature
1. If applicable, [open a merge request](#merge-requests) to provide changes to the source code

This document is here to guide you through the contribution process.

We are aware that the procedure involves a number of steps and might appear lengthy or complicated upon first look.
However, each step is motivated by the aim to let dantro be a reliable and well-maintainable open source software project.
In our opinion, this is best achieved by adhering to some common best-practices, communicating clearly, working as a team, and learning from each other.

If you have questions about contributing, please do not hesitate to [contact us](#direct-contact).


### GitLab Account

The dantro repository is hosted on the private GitLab instance of the [TS-CCEES][ts_hp] research group at the [Institute of Environmental Physics (IUP) Heidelberg][iup_hp].

You can get an account via [the **registration page**](https://ts-gitlab.iup.uni-heidelberg.de/users/sign_in).
Your account will be an ["external" user account][ext_user], which gives you access to all public projects, including dantro, and allows to open issues there.

If you would like to contribute with code development, [request access to the dantro project by clicking this link][request_access], or by visiting the project page.
If you have questions regarding your account, feel free to [drop us an e-mail][devmail].


### Issues

To ask questions, report a bug, or suggest a new feature or enhancement for dantro, we encourage you to [open an issue][new_issue] in the dantro GitLab project.

The issue constitutes the **planning phase** of a contribution to dantro.
This is especially important for larger code changes, which require planning of the implementation [and its tests](#coding-guidelines).

When opening an issue, we provide several templates which you may find useful for structuring your issue description and providing the required information.
If you do *not* find them useful, don't worry about it: you don't have to use them.
We are happy about feedback in any form! :)


### Merge Requests

After a proper discussion of a task and its possible implementation in an issue, the Merge Request (MR) constitutes the **execution phase** of a contribution.

If you are willing to implement the changes, open a merge request by clicking on the *"Create merge request"* button in the issue.
Then, use one of the available templates to provide a description of the changes (even if brief), and add the corresponding labels.

When opening MRs like this, GitLab will automatically mark the MR as a draft by prefixing `Draft: ` to its title.
This indicates, that this MR is currently being worked on.
To further specify that *you* are working on this MR, it's best to assign the MR to your self, e.g. via the right-hand sidebar.

Once you are ready with an implementation, follow these steps to get your MR ready for merging.

1. Adjust the MR description to provide information on how your code changes and additions solve the problem or implement the feature.
1. Go through the checklist provided in the MR templates. It contains points like the following:
    - Appropriate tests are implemented and the pipeline passes
    - The code is well-documented
    - ...
1. Remove the `Draft: ` in the title to denote that the MR is ready for review.
1. Assign a reviewer, e.g. one of the maintainers. They will guide you through the review process and the remaining steps to merge the MR.

After all discussions are resolved (by you or the reviewer), your MR will be merged. :tada:

Again: **Thank you for taking the time to contribute to dantro!!**


## Coding Guidelines

To remain a maintainable and future-proof project, we aim to adhere to the following guidelines:

- **Code should be legible and follow a consistent style**
    - Ideally, write code that documents itself, alleviating the need for many inline comments.
    - For a succint summary, see [The Zen of Python][zen].
    - To ensure consistent code formatting, we let [black][black] do all the hard work. See the section on [pre-commit hooks](README.md#commit-hooks) for more information.

- **Code should be documented**
    - Always add a docstring with a one-line description and a documentation of the parameters. Use [Google-style docstrings][google_docstrings].
    - For higher-level documentation, add a page to the [Sphinx documentation](doc/).

- **Every implementation needs a test**
    - This allows dantro to grow without breaking apart.
    - For more difficult features, the test implementation should be planned alongside the issue, prior to the implementation.
    - The code coverage report is an indicator for whether sufficient tests are implemented. However, keep in mind that the coverage percentage does not indicate *semantic* coverage of the implemented code!
    - Have a look at the [pytest documentation][pytest] for more information on how to implement tests.

- **Code needs to be reviewed before merging**
    - Two pairs of eyes see more than one. You will see that this greatly improves code quality and at the same time provides knowledge transfer.
    - Review is done in the [merge requests](#merge-requests).
    - Reviewers typically are the maintainers, but we also invite other contributors for a code review.

- **The git history is *reasonably* clean**
    - Only a conscientiously written git history is a useful git history.
    - Every MR should either clean up the new commits (e.g. by interactively rebasing onto master) or set the GitLab's squash option.

If you have questions or remarks regarding any of these points, we are happy to answer them or provide guidance throughout any part of the contribution process.


## Code of Conduct

Everybody participating in and contributing to this project is expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
You can report any unacceptable behavior via the interface provided by GitLab or by directly contacting the [developer team][devmail]!


## Direct Contact

Of course, you can always contact the developer team [directly via e-mail][devmail] if you have questions regarding dantro or this contribution guide.

[devmail]: mailto:dantro-dev@iup.uni-heidelberg.de
[ts_hp]: https://ts.iup.uni-heidelberg.de/
[iup_hp]: https://www.iup.uni-heidelberg.de/
[ext_user]: https://docs.gitlab.com/ee/user/permissions.html#external-users-core-only
[dantro_project]: https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro
[request_access]: https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro/-/project_members/request_access
[new_issue]: https://ts-gitlab.iup.uni-heidelberg.de/utopia/dantro/issues/new?issue
[pep8]: https://www.python.org/dev/peps/pep-0008/
[black]: https://black.readthedocs.io/en/stable/
[zen]: https://www.python.org/dev/peps/pep-0020/
[google_docstrings]: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
[pytest]: https://docs.pytest.org/en/latest/contents.html
