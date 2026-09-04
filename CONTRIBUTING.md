# Contributing to rai-toolkit

Bug reports and pull requests are welcome. For a substantive change, start with
an [issue](https://github.com/wandb/rai-toolkit/issues) so the scope and expected
behaviour are clear.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

## Starting work

Issues labeled [`status: available`](https://github.com/wandb/rai-toolkit/labels/status%3A%20available)
have been scoped by a maintainer and are ready for contribution. The label
describes the readiness of the issue, not whether someone has already started
working on it.

Before starting, check the issue's Development section and the open pull
requests for linked work. If an open pull request already addresses the issue,
join that discussion instead of duplicating the work.

When you begin:

1. Create a branch and open a draft pull request as soon as you have an initial
   commit.
2. Add `Closes #NN` to the pull request description.
3. Briefly describe your approach and the tests you plan to run.

The linked draft pull request is the signal that work has started. You do not
need to post a `/claim` comment or wait for a status label change. Keep the pull
request in draft while it is in progress, then mark it ready when the change and
local validation are complete.

## Pull request guidelines

- Keep each pull request focused on one change.
- Link the issue with `Closes #NN` when the pull request resolves it.
- Explain what changed and call out any behaviour or compatibility
  considerations.
- Cover behavioural changes with tests.
- List the exact local validation commands you ran and their results.
- Update documentation when user-visible behaviour changes.
- AI-assisted contributions are fine. Say so in the pull request description,
  and be ready to explain and rework any line when asked.

## License headers
<!--- REUSE-IgnoreStart -->

Every source file carries an SPDX header reflecting:
- Year and copyright owner
- SPDX license identifier: `SPDX-License-Identifier: Apache-2.0`
- Package name: `SPDX-PackageName: rai-toolkit`

This is automated with [FSFE REUSE](https://reuse.software/dev/#tool) using the
template in `.reuse/templates/`:

```shell
reuse annotate --license Apache-2.0 --copyright 'CoreWeave, Inc.' --year 2026 \
--template default_template --merge-copyrights $FILE
```

Do not blindly add headers to every file. Assigning the wrong copyright owner
is a real risk. Understand who owns a contribution before annotating it.

Licensing state and the SPDX bill of materials can be validated and generated
with:

```shell
reuse lint
reuse spdx
```

By submitting a contribution you agree it is licensed under Apache-2.0 (see
`LICENSE`).

<!--- REUSE-IgnoreEnd -->

## Security issues

Email **contact@wandb.ai** privately. Don't open a public issue for
vulnerabilities.

## Quick start for new contributors

- Scoped starter work lives under the [good first issue](https://github.com/wandb/rai-toolkit/labels/good%20first%20issue) label.
- Framework mappings (ISO/IEC 42001 #6, Colorado AI Act #7, NYC LL144 #8) mirror the existing NIST AI RMF mapping structure in `rai_toolkit/compliance/`.
- The example policy pack under `rai_toolkit/policies/packs/example_enterprise_pack/` shows the policy format.
- Lint policies with `rai policies lint`.
