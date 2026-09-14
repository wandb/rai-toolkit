# Contributing to rai-toolkit

Bug reports and pull requests are welcome. For a substantive change, start with
an [issue](https://github.com/wandb/rai-toolkit/issues) so the scope and expected
behaviour are clear.

## Setup

Use Python 3.10 or newer. From the repository root, install the core library
and lightweight test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the quality tools separately, using the versions pinned in
[CI](.github/workflows/ci.yml):

```bash
python -m pip install "ruff==0.16.6" "reuse[charset-normalizer]==6.2.0"
```

The `all` extra also installs the demo and optional integrations. It is not
needed for the core contributor checks below.

## Required local checks

Run these checks from the repository root before marking a pull request ready:

```bash
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 .
reuse --no-multiprocessing lint
git diff --check
git diff --cached --check
```

Check the committed patch against the latest main branch as well:

```bash
git fetch origin main
git diff --check origin/main...HEAD
```

Include the exact commands and results in the pull request, including any
skipped tests. With only `.[dev]` installed, tests requiring optional Weave
dependencies are skipped. Maintainers independently verify the checks before
merge. CI also runs the core suite on Python 3.10, 3.11, and 3.12 and checks
package builds and installation.

## Optional offline Weave checks

Use a separate environment so the core environment stays lightweight. These
tests mock external services and do not require provider API keys. Run them
when working on Weave integration and include the results in the pull request;
they are optional for unrelated changes.

```bash
python -m venv .venv-weave
source .venv-weave/bin/activate
python -m pip install -e ".[dev,weave]"
python -m pytest -q
```

CI tests both the latest compatible Weave release and the supported minimum.
To check the minimum in the same Weave environment, install its pinned version
and rerun the suite:

```bash
python -m pip install -e ".[dev,weave]" "weave==0.52.40"
python -m pytest -q
deactivate
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

## Scorer contributions

When adding a scorer, follow the [scorer-authoring guide](docs/scorer_authoring.md)
for the result contract, unassessed rows, evidence validation, and offline tests.

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
- [`docs/model_adapters.md`](docs/model_adapters.md) is the contract every model adapter is held to — the `predict` signature, where retrieved context may and may not go, the standard metadata keys, lazy optional SDK imports, and `from_args`. Read it before adding or changing an adapter under `rai_toolkit/models/`, and add a `ModelAdapterContractTests` subclass (`tests/model_adapter_contract.py`) to the adapter's test module.
- Lint policies with `rai policies lint`.
