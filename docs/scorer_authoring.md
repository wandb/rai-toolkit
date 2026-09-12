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

```python
from rai_toolkit.scorers import BaseScorer, ScorerResult


class ReferenceTextScorer(BaseScorer):
    name = "reference_text"
    description = "Checks whether the response contains the supplied reference text"
    category = "MIT-3.1"
    threshold = 1.0

    def score(self, output, input="", context="", **kwargs):
        expected = kwargs.get("expected")
        if not isinstance(expected, str) or not expected.strip():
            return ScorerResult(
                score=0.0,
                passed=False,
                category=self.category,
                explanation="Unassessed: a non-empty reference string is required.",
                details={"scorer_name": self.name, "skipped": "missing_reference"},
                assessed=False,
            )

        score = 1.0 if expected in output else 0.0
        return ScorerResult(
            score=score,
            passed=score >= self.threshold,
            category=self.category,
            explanation="Reference text found." if score else "Reference text absent.",
            details={"scorer_name": self.name, "reference_span": expected},
        )


scorer = ReferenceTextScorer()
result = scorer.score("The limit is 10.", expected="limit is 10")
assert result.assessed and result.passed
assert not scorer.score("The limit is 10.").assessed
```

The built-in pipeline forwards a row's non-empty `expected` and `rubrics`
values as scorer keyword arguments. It does not forward arbitrary dataset
columns. For `context`, it prefers the model response's non-empty
`metadata["retrieved_context"]` over the dataset context. See
[`RAIEvaluationPipeline`](../rai_toolkit/evaluation/pipeline.py) before adding
new input requirements.

For asynchronous I/O, implement `score_async` and keep the required `score`
method's behavior explicit. The base implementation of `score_async` calls
`score` directly; it does not move blocking work to a worker thread.

## Choose skip rules for the measurement

Missing required context, a missing reference answer, or an unusable judge reply
is a coverage gap. Return an unassessed result with a specific explanation.
The pipeline also converts scorer exceptions into unassessed results. Do not
catch an exception and turn it into a successful score.

Refusal handling depends on what the scorer measures. `GroundednessScorer`
skips rows whose reference expects refusal or boundary-setting.
`ContextPrecisionScorer` and `ContextRecallScorer` grade retrieval, so they can
still assess the context when the generator refuses. Copying a skip rule from
another scorer without checking the measurement can remove valid evidence.

When introducing a skip reason, check
[`_classify_unassessed_reason`](../rai_toolkit/assessment/assessor.py). It recognizes
some reasons explicitly and otherwise falls back to a generic report label.
Add a report regression if the new reason needs a specific human-facing label.

## Verify evidence before storing it

`GroundednessScorer` records verified evidence in `details["supporting_spans"]`
and `details["contradicting_spans"]`. Each entry has this shape:

```json
{
  "response_span": "Notices are required",
  "context_span": "Notices are required for denied applicants."
}
```

Both strings must be present in the corresponding response or context. The
existing verifier allows whitespace and quote normalization when matching,
then stores the matched slice from the original text. A matching quote proves
that the evidence exists, not that the judge's semantic conclusion is correct.
Do not use prompt envelopes, source labels alone, or a quotation from another
source as evidence for the target passage.

For an LLM judge, mock `_call_judge` in offline tests. Check response types,
required fields, indexes, duplicates, and coverage before calculating a result.
Keep rejected evidence counts and enough sanitized audit data to explain a
rejection. When a score cannot be derived safely from the remaining evidence,
return an unassessed result. State explicitly whether the judge's score is
authoritative, validated against a rubric, or only advisory.

## Keep retrieval denominators tied to the input

The three retrieval scorers reuse the chunk parsing and envelope rendering in
[`llm_judges.py`](../rai_toolkit/scorers/llm_judges.py). Their private helpers are
implementation references, not a stable public parser API.

- `RetrievalRelevanceScorer` grades each parsed chunk independently and derives
  its aggregate from validated per-chunk verdicts.
- `ContextPrecisionScorer` divides needed chunks by all parsed chunks. It groups
  exact duplicate passage content after removing source labels and normalizing
  whitespace. If any copy is needed, the lowest-index copy is kept as needed;
  later copies are not. A group with no needed copy stays unnecessary.
- `ContextRecallScorer` binds each reference item to one occurrence of a verified
  reference span. Repeated text needs separate occurrences. Items must cover the
  full reference without overlap; missing, duplicate, or overlapping coverage
  cannot silently shrink the denominator.

Recall merges adjacent pieces with the same support verdict into stretches on
the whitespace-normalized reference, then divides supported stretch length by
total stretch length. Spaces between opposite-verdict stretches are outside
both lengths. This measures text coverage, not an equally weighted count of
facts. Equivalent splitting of the same supported or unsupported text must not
change the score, and adding unsupported reference content must reduce it.
Evidence for a supported reference item must verify in the original retrieved
chunk; unverifiable support is downgraded. The original judge score is advisory.

## Add the scorer to an assessment

For a task-specific scorer, pass an instance through `additional_scorers` to
`Assessor` or `RAIEvaluationPipeline`. For example, after configuring the model,
datasets, and compliance profile as in the README, add
`additional_scorers=[ReferenceTextScorer()]` to the constructor.
This is also how the optional RAG scorers are enabled; exporting a class does
not register it for every assessment.

For a new built-in scorer, export it from
[`rai_toolkit.scorers`](../rai_toolkit/scorers/__init__.py). Add it to
[`SCORER_REGISTRY`](../rai_toolkit/compliance/scorer_registry.py) only when the
associated compliance profiles should select it by default. Verify resolution
through the mapping engine and both evaluation backends. A specialized scorer
that requires inputs most rows lack is usually better left opt-in.

## Validation before review

Use offline cases that distinguish the intended behavior from a plausible bug:

- A measured pass, a measured failure, and missing-evidence cases; guard paths
  should assert that no judge call was made.
- Malformed, non-object, contradictory, incomplete, and non-finite judge replies.
- Evidence that exists, evidence that was invented, and evidence in the wrong
  passage, retaining the original matched text in accepted results.
- Duplicate chunks, repeated reference spans, equivalent partitions, and growing
  missing content when the measurement uses a denominator.
- Async execution, configured names, and coverage-gap reporting when affected.

Useful examples are [`test_groundedness_scorer.py`](../tests/test_groundedness_scorer.py),
[`test_retrieval_relevance.py`](../tests/test_retrieval_relevance.py),
[`test_context_precision_recall.py`](../tests/test_context_precision_recall.py),
and [`test_composite_scorer.py`](../tests/test_composite_scorer.py).
Run the affected tests and the checks in [the CI workflow](../.github/workflows/ci.yml).
Document whether validation was offline or used a live model; offline parser
and evidence checks do not establish judge accuracy on a real dataset.
