# Contributing to rai-toolkit

Bug reports and PRs are welcome. For anything substantive, open an
[issue](https://github.com/wandb/rai-toolkit/issues) first.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

## Pull request guidelines

- One change per PR.
- Cover behavioural changes with a test.

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
