<!--
SPDX-FileCopyrightText: 2026 Karan Nisar
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: rai-toolkit
-->

# Writing a scorer

A scorer turns one model response and its available evidence into a
`ScorerResult`. Start with a precise measurement: what is being graded, what
evidence is required, what a passing result means, and when the row cannot be
assessed. This guide describes the interfaces and retrieval scorers on `main`.
The citation-correctness scorer in [PR #27](https://github.com/wandb/rai-toolkit/pull/27)
is still under review and is not part of this guide's supported API.

## The result contract

Subclass [`BaseScorer`](../rai_toolkit/scorers/base.py) and implement
`score(output, input="", context="", **kwargs)`. Give the scorer a stable `name`,
a description, the risk `category` it measures, and a `threshold`.
Use distinct names for distinct configured scorers: evaluation stores results
by scorer name, so duplicate names can overwrite results.

| Field | Authoring rule |
| --- | --- |
| `score` | A finite number from 0 to 1; higher means better or safer. Validate raw values before normalization. |
| `passed` | For an assessed row, apply the declared threshold or explicitly document another decision rule. |
| `category` | The risk category actually measured, such as `MIT-3.1`; this determines aggregation and policy context. |
| `explanation` | Describe the result or missing evidence in terms a report reader can verify. |
| `details` | Keep JSON-safe evidence and audit data, including `scorer_name` and any skip reason. |
| `assessed` | `True` only when the scorer produced a usable measurement. |

Reject non-finite raw values, booleans where a numeric score is expected,
invalid ranges, and malformed judge output before calling `ScoreNormalizer`.
Do not rely on normalization to validate these inputs: `from_scale` clips
values, and broader non-finite validation is still tracked in
[issue #38](https://github.com/wandb/rai-toolkit/issues/38).

An unassessed row normally uses `score=0.0`, `passed=False`, `assessed=False`,
and `details["skipped"]` with a stable reason. The zero and false values are
placeholders, not a measured failure. The evaluation pipeline excludes these
rows from score and pass-rate aggregates and reports them as coverage gaps.
Code consuming a result directly must check `assessed` before interpreting it.
If you use `ScoreNormalizer.aggregate_scores` directly, filter unassessed
results yourself; that helper does not filter them.

## A small offline example

This scorer measures literal reference-text inclusion. It does not establish
factual correctness, entailment, or source attribution.