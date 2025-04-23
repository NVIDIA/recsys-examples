# Contributing to recsys-examples

If you are interested in contributing to recsys-examples, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/recsys-examples/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `recsys-examples/print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
    - The recsys-examples team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA/recsys-examples/blob/main/README.md)
    to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA/recsys-examples/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
    or [help wanted](https://github.com/NVIDIA/recsys-examples/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Get familiar with the [Coding Guidelines](#coding-guidelines).
5. Code! Make sure to update unit tests!
6. When done, [create your pull request](https://github.com/NVIDIA/recsys-examples/compare).
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a recsys-examples developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!


### Managing PR labels

Each PR must be labeled to indicate whether the change is a feature, improvement, bugfix, or documentation change.

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project board](https://github.com/NVIDIA/recsys-examples/projects?query=is%3Aopen).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:
- Type:
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.


# Coding Guidelines
To maintain code quality and consistency, please adhere to the following coding guidelines:
## Naming Conventions
### Type Names
In general, a type name (including classes and type aliases) is a concatenated string of tokens, each starting with a capital letter. For example: ShardedEmbedding.
### Variable Names
A variable name is typically a concatenated string of lowercase letters, with underscores separating words. For example: use_mixed_precision. If the variable is a class member and is used only within the class, it must start with an underscore. E.g. self._plan
### Function Names
All accessors, member functions, and global functions should be written in lowercase with underscores between words. For example: get_hstu_config().

## Comments and Documentation
* Use comments judiciously to explain complex logic.
* Document public methods and classes using docstrings in accordance with PEP 257.
## Testing
* Write unit tests for any new features or bug fixes.
* Ensure all tests pass before submitting your pull request.
## Linting
Make sure your code adheres to linting standards. You can run the lint tests using the following command:
```python
pre-commit run -a
```
By following these guidelines, you help ensure that contributions are integrated smoothly into the project. Thank you for contributing to **recsys-examples**! We look forward to your contributions!

## Signing Your Work
We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.
To sign off on a commit you simply use the --signoff (or -s) option when committing your changes:

$ git commit -s -m "Add cool feature."
This will append the following to your commit message:

Signed-off-by: Your Name <your@email.com>
Full text of the DCO:
```
  Developer Certificate of Origin
  Version 1.1
  
  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129
  
  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  Developer's Certificate of Origin 1.1
  
  By making a contribution to this project, I certify that:
  
  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
  
  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
  
  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
  
  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
