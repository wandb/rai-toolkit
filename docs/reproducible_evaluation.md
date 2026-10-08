# Reproducible evaluation

A small, offline example of the thing an evaluation has to be before its numbers can be trusted: the
same source revision and the same fixtures produce the same rows, and the artifacts say what
produced them.

The example ships in [`docs/examples/pii_pattern_replay/`](examples/pii_pattern_replay/): six
hand-authored responses, three of which carry a pattern the shipped `RegexPIIScorer` is configured
to detect, scored once with the scorer itself. Nothing is sampled, downloaded, judged by a model, or
recorded from a real session.

## Run it

From a git checkout (the script reads its own revision, so a source archive is not a supported way
to reproduce it). The revision this example was implemented at is
`42284034c9b39621a2138fa3cd4848951d3bc0ab` (`docs: add an offline, reproducible PII-pattern replay
example`), the first commit of pull request
[#111](https://github.com/wandb/rai-toolkit/pull/111); no release contains it yet. That commit
predates the LF pin in `.gitattributes`, so the clone below turns end-of-line conversion off, which
is what keeps the fixture bytes the manifest hashes:

```bash
git clone --config core.autocrlf=false https://github.com/wandb/rai-toolkit.git
cd rai-toolkit
git fetch origin pull/111/head
git checkout 42284034c9b39621a2138fa3cd4848951d3bc0ab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python docs/examples/pii_pattern_replay/run.py --output-dir /tmp/rai-pii-replay-a --verify
python docs/examples/pii_pattern_replay/run.py --output-dir /tmp/rai-pii-replay-b --verify
python -m pytest -q tests/test_reproducible_evaluation.py
```

Every `run.json` a run writes records the checkout's own revision as well, so a reader can tell
which revision produced an artifact.

Two runs write byte-identical `results.json`. `--verify` returns zero only when the input hashes, the
rows and the summary match the committed expectations, and it fails when a row comes back
unassessed. Without `--verify` the run still refuses to score an input whose checksum does not match
`fixture_manifest.json`.

## What it measures, and what it does not

Six responses are replayed and each is scored once. The expected rows are in
[`expected_results.json`](examples/pii_pattern_replay/expected_results.json); the summary is

```json
{
  "attempted_rows": 6,
  "assessed_rows": 6,
  "unassessed_rows": 0,
  "passed_rows": 3,
  "failed_rows": 3,
  "mean_score": 0.5,
  "pass_rate": 0.5,
  "coverage": 1.0
}
```

Mean score and pass rate are computed over assessed rows; coverage is assessed over attempted. The
example does **not** measure whether a model protects personal data, whether the data in a response
is real, or whether any regulation is satisfied. It measures whether an evaluation can be run and
reproduced from its own inputs.

`000-00-0000` is an intentionally invalid SSN (it cannot belong to anyone), and the shipped `ssn`
pattern matches it, so the row is reported as a failure. That is the recorded behaviour of the
scorer, preserved on purpose: an example that quietly special-cased it would be an example of
hiding a false positive rather than of measuring one.

## The artifacts

| File | What it is |
|---|---|
| `responses.json` | the six synthetic responses, provenance `synthetic_hand_authored` |
| `fixture_manifest.json` | the source fixture, its revision and SHA-256, the selected rows, the response file's SHA-256, provenance and license |
| `expected_results.json` | the rows and summary a correct run produces, reviewed against the issue's table |
| `run.py` | the replay, and the writer of `results.json` and `run.json` |
| *`results.json`* (written) | one row per case with every `ScorerResult` field, plus the summary |
| *`run.json`* (written) | what produced the run: fixture identities and hashes, scorer class and configuration, installed toolkit version, source revision and whether tracked files were dirty, Python version, installed distributions, and one UTC timestamp |

`results.json` is pinned and deterministic: sorted keys, two-space indentation, a trailing newline. It
holds no timestamp, which is why two runs in different directories agree byte for byte. `run.json` is
allowed to differ between environments (that is what it is for), and it deliberately records no
environment variables, credentials or absolute local paths.

`run.json` keeps two versions apart, because they mean different things: `toolkit_version` is what
`pip` installed (which can be an earlier release), while `source_revision.commit` is the checkout the
code was read from, with `tracked_dirty` saying whether that checkout was clean. A run from
unreleased source therefore reports the previous package version next to the real revision instead of
pretending they are the same.

## Where the data comes from

The prompts are the toolkit's own bundled sample, `rai_toolkit/examples/pii_probes.json`
(Apache-2.0), part of this repository, not a downloaded research dataset. Row indices 0, 3 and 5 are used, in that
order, and the fixture's `expected` text stays where it belongs: as source context, not as a ground
truth classification for a regex-only replay. The manifest pins the file's SHA-256 and the revision it
was measured at.
