# Contributing

Thank you for your interest in this project.

## Contribution status

This project is currently **not accepting external contributions**. Pull requests
from outside NVIDIA may be closed without review. Bug reports and feature
requests are welcome as [GitHub issues](https://github.com/NVIDIA/auto-ontology-eval/issues).

**Do not report security vulnerabilities through GitHub issues.** Follow the
process in [SECURITY.md](SECURITY.md) instead.

The rest of this document describes the workflow for maintainers and for
contributions once they are accepted.

## Development setup

Follow [Prerequisites](README.md#prerequisites) and [Setup](README.md#setup) in
the README, then install the dev dependency group:

```bash
uv sync --group dev
```

## Before opening a pull request

Run the same checks as CI:

```bash
uv run ruff check ontology_sql_eval
uv run ruff format --check ontology_sql_eval
uv run pytest
```

- Keep pull requests focused on a single change.
- Add or update tests for behavior changes.
- Add the Apache-2.0 SPDX header to new source files, matching existing files.
- Do not commit secrets, credentials, `.env` files, or dataset files that the
  dataset READMEs say must be downloaded at runtime.
- Pull requests require review and approval before merging.

## Signing your work

All commits must be signed off under the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/).
The sign-off certifies that you wrote the change or otherwise have the right to
submit it under the project's open source license.

Sign off each commit with `-s`:

```bash
git commit -s -m "Add a useful change"
```

This appends a line like the following to the commit message:

```
Signed-off-by: Your Name <your@email.com>
```

Use your real name. Pull requests with unsigned commits will not be accepted.

<details>
<summary>Full text of the DCO</summary>

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution and all
    personal information I submit with it (including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

</details>

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
