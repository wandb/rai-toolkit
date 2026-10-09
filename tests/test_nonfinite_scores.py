# SPDX-FileCopyrightText: 2026 Abhinav Garg
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""A judge score of NaN or infinity is un-assessed, not a grade.

``ScoreNormalizer`` clamps with ``min``/``max``, and both swallow NaN
silently: ``min(1.0, nan)`` returns 1.0, so a NaN reply used to normalize to
a perfect 1.0 and inflate the aggregate with a row the judge never graded.
``-Infinity`` clamped the other way and dragged the aggregate down. The
normalizer now raises on non-finite input, and these tests pin that along
with the un-assessed contract for both shared-judge and groundedness paths,
keeping finite clamping/inversion working as controls.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any
from unittest.mock import Mock

import pytest

from rai_toolkit.assessment.assessor import _classify_unassessed_reason
from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework
from rai_toolkit.evaluation.pipeline import RAIEvaluationPipeline
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.llm_judges import FairnessJudge, GroundednessScorer
from rai_toolkit.scorers.normalizer import ScoreNormalizer

# Both judge paths that parse a ``score`` field off the judge reply: the
# shared base implementation (FairnessJudge adds nothing to it) and
# GroundednessScorer, which parses its own score alongside evidence spans.
JUDGE_CLASSES = (FairnessJudge, GroundednessScorer)

# Every non-finite score a judge can emit, in both the wire form Python's
# JSON parser accepts as a literal (``NaN``) and the string form a judge may
# quote. ``float("NaN")`` succeeds for both, so neither is caught by the
# numeric-parse guard.
NON_FINITE_SCORES: tuple[tuple[Any, str], ...] = (
    ("NaN", "NaN"),
    (float("nan"), "NaN"),
    ("Infinity", "Infinity"),
    (float("inf"), "Infinity"),
    ("-Infinity", "-Infinity"),
    (float("-inf"), "-Infinity"),
)

PARSER_FAILURE_REASON = "scorer integration / parser failure"


def _judge(scorer_class: type, reply: Any) -> Any:
    """Offline scorer whose judge call returns ``reply`` without any network."""
    scorer = scorer_class(api_key="offline-test")
    scorer._call_judge = Mock(return_value=reply)
    return scorer


def _score_row(scorer: Any) -> Any:
    """Score a row that reaches the judge on every path under test.

    Non-empty context clears the groundedness/factuality context gate, and
    the expected text is factual so the behavioral-refusal gate does not
    short-circuit before the judge is called.
    """
    return scorer.score(
        "Revenue was $10 million.",
        input="What was revenue?",
        context="[fin-1] The company's revenue was $10 million.",
        expected="Answer with the revenue figure from the context.",
    )


class _StubModel(BaseModel):
    """Echoes a canned answer so the pipeline runs fully offline."""

    name = "stub-model"

    async def predict(
        self, input_text: str, context: str = "", **kwargs: Any
    ) -> ModelResponse:
        return ModelResponse(output="Revenue was $10 million.", metadata={})


def _empty_profile() -> ComplianceProfile:
    """Profile with no categories: scorers come from ``additional_scorers``."""
    return ComplianceProfile(
        name="Non-finite score regression",
        framework=Framework.MIT_AI_RISK,
        categories=[],
    )


# --------------------------------------------------------------------------
# Scorer contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
@pytest.mark.parametrize("raw_score,label", NON_FINITE_SCORES)
def test_non_finite_judge_score_is_unassessed(
    scorer_class: type, raw_score: Any, label: str
) -> None:
    scorer = _judge(scorer_class, {"score": raw_score, "explanation": "graded"})

    result = _score_row(scorer)

    assert result.assessed is False
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["skipped"] == "non_finite_judge_score"
    assert result.details["raw_score"] == label
    assert result.details["scorer_name"] == scorer.name


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
@pytest.mark.parametrize("raw_score,label", NON_FINITE_SCORES)
def test_non_finite_judge_score_classifies_as_parser_failure(
    scorer_class: type, raw_score: Any, label: str
) -> None:
    scorer = _judge(scorer_class, {"score": raw_score})

    result = _score_row(scorer)

    assert _classify_unassessed_reason(result) == PARSER_FAILURE_REASON


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
@pytest.mark.parametrize("raw_score,label", NON_FINITE_SCORES)
def test_non_finite_judge_score_result_is_json_strict(
    scorer_class: type, raw_score: Any, label: str
) -> None:
    scorer = _judge(scorer_class, {"score": raw_score, "explanation": "graded"})

    result = _score_row(scorer)

    # The JSON report serializes with allow_nan=False; a stored float("nan")
    # would raise ValueError here and take the whole report down.
    payload = json.dumps(asdict(result), allow_nan=False)
    assert label in payload


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
def test_non_finite_values_elsewhere_in_reply_are_sanitized(
    scorer_class: type,
) -> None:
    """A nested NaN must not survive into the audit record either."""
    scorer = _judge(
        scorer_class,
        {
            "score": float("nan"),
            "explanation": "graded",
            "confidence": float("nan"),
            "sub_scores": [float("inf"), 2.0, {"detail": float("-inf")}],
        },
    )

    result = _score_row(scorer)

    json.dumps(asdict(result), allow_nan=False)
    response = result.details["judge_response"]
    assert response["score"] == "NaN"
    assert response["confidence"] == "NaN"
    assert response["sub_scores"] == ["Infinity", 2.0, {"detail": "-Infinity"}]


def test_finite_score_with_non_finite_sibling_field_is_json_strict() -> None:
    """A finite grade does not make the rest of the reply finite."""
    scorer = _judge(
        FairnessJudge, {"score": 3, "explanation": "ok", "confidence": float("nan")}
    )

    result = _score_row(scorer)

    assert result.assessed is True
    assert result.score == 1.0
    json.dumps(asdict(result), allow_nan=False)
    assert result.details["judge_response"]["confidence"] == "NaN"


def test_groundedness_non_finite_score_claims_no_evidence() -> None:
    """An un-assessed row must not carry evidence spans from a bad reply."""
    scorer = _judge(
        GroundednessScorer,
        {
            "score": float("nan"),
            "supporting_spans": [
                {
                    "response_span": "Revenue was $10 million.",
                    "context_span": "The company's revenue was $10 million.",
                }
            ],
            "contradicting_spans": [],
        },
    )

    result = _score_row(scorer)

    assert result.assessed is False
    assert result.details["supporting_spans"] == []
    assert result.details["contradicting_spans"] == []


# --------------------------------------------------------------------------
# Finite controls: clamping and inversion must keep working
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
@pytest.mark.parametrize(
    "raw_score,expected",
    [
        (0, 0.0),
        (1, 1 / 3),
        (3, 1.0),
        (7, 1.0),  # clamped to the top of the 0-3 scale
        (-2, 0.0),  # clamped to the bottom
    ],
)
def test_finite_scores_still_normalize_and_clamp(
    scorer_class: type, raw_score: float, expected: float
) -> None:
    scorer = _judge(scorer_class, {"score": raw_score, "explanation": "graded"})

    result = _score_row(scorer)

    assert result.assessed is True
    assert result.score == pytest.approx(expected)
    assert result.details["raw_score"] == pytest.approx(float(raw_score))
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize(
    "raw_score,expected",
    [(0.0, 1.0), (1.5, 0.5), (3.0, 0.0), (9.0, 0.0), (-4.0, 1.0)],
)
def test_finite_inversion_still_clamps(raw_score: float, expected: float) -> None:
    """Inverted risk scales (higher raw = worse) keep their finite behavior."""
    assert ScoreNormalizer.from_scale(
        raw_score, max_value=3.0, invert=True
    ) == pytest.approx(expected)


NON_FINITE_FLOATS = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("raw_score", NON_FINITE_FLOATS)
def test_normalizer_rejects_non_finite_raw_score(raw_score: float) -> None:
    """The clamp would swallow these, so the normalizer refuses them outright."""
    with pytest.raises(ValueError, match="raw_score must be finite"):
        ScoreNormalizer.from_scale(raw_score, 3)


@pytest.mark.parametrize("max_value", NON_FINITE_FLOATS)
def test_normalizer_rejects_non_finite_max_value(max_value: float) -> None:
    with pytest.raises(ValueError, match="max_value must be finite"):
        ScoreNormalizer.from_scale(1, max_value)


@pytest.mark.parametrize("raw_score", NON_FINITE_FLOATS)
def test_compliance_scale_rejects_non_finite_raw_score(raw_score: float) -> None:
    with pytest.raises(ValueError, match="raw_score must be finite"):
        ScoreNormalizer.from_compliance_scale(raw_score)


# --------------------------------------------------------------------------
# Pipeline aggregation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
@pytest.mark.parametrize(
    "bad_score,good_score,expected_mean",
    [
        # NaN used to clamp to 1.0 and inflate a failing category to 0.5.
        (float("nan"), 0, 0.0),
        ("NaN", 0, 0.0),
        # -Infinity used to clamp to 0.0 and drag a passing category to 0.5.
        (float("-inf"), 3, 1.0),
        ("-Infinity", 3, 1.0),
        (float("inf"), 0, 0.0),
    ],
)
def test_pipeline_excludes_non_finite_row_from_aggregation(
    scorer_class: type, bad_score: Any, good_score: float, expected_mean: float
) -> None:
    scorer = scorer_class(api_key="offline-test")
    # Row order is the dataset order: the bad reply first, the real grade second.
    scorer._call_judge = Mock(
        side_effect=[{"score": bad_score}, {"score": good_score}]
    )
    pipeline = RAIEvaluationPipeline(
        ComplianceMappingEngine(), additional_scorers=[scorer]
    )
    dataset = [
        {
            "input": "What was revenue?",
            "context": "[fin-1] The company's revenue was $10 million.",
            "expected": "Answer with the revenue figure from the context.",
        },
        {
            "input": "What was revenue last year?",
            "context": "[fin-2] Prior-year revenue was $8 million.",
            "expected": "Answer with the prior-year revenue from the context.",
        },
    ]

    results = asyncio.run(
        pipeline.run_evaluation(
            model=_StubModel(),
            profile=_empty_profile(),
            dataset=dataset,
            name="non-finite regression",
        )
    )

    category = scorer.category
    summary = results.summary[category]
    assert summary["unassessed_items"] == 1
    assert summary["total_items"] == 1
    assert summary["mean_score"] == pytest.approx(expected_mean)
    # The valid row alone decides the aggregate; the un-assessed row neither
    # inflates nor dilutes it.
    assert results.overall_score == pytest.approx(expected_mean)

    unassessed = [
        sr
        for item in results.items
        for sr in item.scores.values()
        if not sr.assessed
    ]
    assert len(unassessed) == 1
    assert unassessed[0].details["skipped"] == "non_finite_judge_score"
    assert _classify_unassessed_reason(unassessed[0]) == PARSER_FAILURE_REASON
    json.dumps([asdict(sr) for sr in unassessed], allow_nan=False)


@pytest.mark.parametrize("scorer_class", JUDGE_CLASSES)
def test_pipeline_aggregates_two_finite_rows_unchanged(scorer_class: type) -> None:
    """Control: with no non-finite reply the aggregate is the plain mean."""
    scorer = scorer_class(api_key="offline-test")
    scorer._call_judge = Mock(side_effect=[{"score": 0}, {"score": 3}])
    pipeline = RAIEvaluationPipeline(
        ComplianceMappingEngine(), additional_scorers=[scorer]
    )
    dataset = [
        {
            "input": "What was revenue?",
            "context": "[fin-1] The company's revenue was $10 million.",
            "expected": "Answer with the revenue figure from the context.",
        },
        {
            "input": "What was revenue last year?",
            "context": "[fin-2] Prior-year revenue was $8 million.",
            "expected": "Answer with the prior-year revenue from the context.",
        },
    ]

    results = asyncio.run(
        pipeline.run_evaluation(
            model=_StubModel(),
            profile=_empty_profile(),
            dataset=dataset,
            name="finite control",
        )
    )

    summary = results.summary[scorer.category]
    assert summary["unassessed_items"] == 0
    assert summary["total_items"] == 2
    assert summary["mean_score"] == pytest.approx(0.5)
