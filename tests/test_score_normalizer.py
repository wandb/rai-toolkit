# SPDX-FileCopyrightText: 2026 Ethan Yeh
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Regression tests for non-finite handling in ScoreNormalizer.

``min(1.0, float("nan"))`` returns ``1.0`` in Python, so an unvalidated
``NaN`` used to normalize into a perfect score and pass any threshold.
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from rai_toolkit.scorers import FactualityJudge
from rai_toolkit.scorers.normalizer import ScoreNormalizer

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("raw_score", NON_FINITE)
def test_from_scale_rejects_non_finite_raw_score(raw_score: float) -> None:
    with pytest.raises(ValueError, match="raw_score must be finite"):
        ScoreNormalizer.from_scale(raw_score, max_value=3.0)


@pytest.mark.parametrize("max_value", NON_FINITE)
def test_from_scale_rejects_non_finite_max_value(max_value: float) -> None:
    with pytest.raises(ValueError, match="max_value must be finite"):
        ScoreNormalizer.from_scale(1.0, max_value=max_value)


@pytest.mark.parametrize("raw_score", NON_FINITE)
def test_from_scale_rejects_non_finite_when_inverted(raw_score: float) -> None:
    with pytest.raises(ValueError, match="raw_score must be finite"):
        ScoreNormalizer.from_scale(raw_score, max_value=3.0, invert=True)


@pytest.mark.parametrize("helper", [
    ScoreNormalizer.from_compliance_scale,
    ScoreNormalizer.from_likert,
])
def test_scale_helpers_reject_nan(helper) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        helper(float("nan"))


def test_non_positive_max_value_still_rejected() -> None:
    with pytest.raises(ValueError, match="max_value must be positive"):
        ScoreNormalizer.from_scale(1.0, max_value=0.0)


@pytest.mark.parametrize(
    ("raw_score", "max_value", "expected"),
    [
        (0.0, 3.0, 0.0),
        (1.5, 3.0, 0.5),
        (3.0, 3.0, 1.0),
        (9.0, 3.0, 1.0),  # clamped from above
        (-9.0, 3.0, 0.0),  # clamped from below
    ],
)
def test_finite_values_keep_existing_clamping(
    raw_score: float, max_value: float, expected: float
) -> None:
    assert ScoreNormalizer.from_scale(raw_score, max_value) == expected


def test_finite_inversion_is_unchanged() -> None:
    assert ScoreNormalizer.from_scale(3.0, 3.0, invert=True) == 0.0
    assert ScoreNormalizer.from_scale(0.0, 3.0, invert=True) == 1.0


def test_nan_judge_response_cannot_produce_an_assessed_pass() -> None:
    """The bug this issue describes, end to end through a judge."""
    scorer = FactualityJudge(api_key="test", threshold=0.5)
    scorer._call_judge = Mock(
        return_value={"score": "NaN", "explanation": "malformed"}
    )

    result = scorer.score("some output", context="some context")

    assert not result.assessed
    assert not result.passed
    assert result.score == 0.0
    assert result.details["skipped"] == "non_finite_judge_score"
    assert math.isnan(result.details["raw_score"])


@pytest.mark.parametrize("bad_score", ["Infinity", "-Infinity"])
def test_infinite_judge_response_is_unassessed(bad_score: str) -> None:
    scorer = FactualityJudge(api_key="test", threshold=0.5)
    scorer._call_judge = Mock(return_value={"score": bad_score})

    result = scorer.score("some output", context="some context")

    assert not result.assessed
    assert not result.passed
    assert result.details["skipped"] == "non_finite_judge_score"


def test_finite_judge_response_is_still_assessed() -> None:
    scorer = FactualityJudge(api_key="test", threshold=0.5)
    scorer._call_judge = Mock(return_value={"score": 3, "explanation": "good"})

    result = scorer.score("some output", context="some context")

    assert result.assessed
    assert result.passed
    assert result.score == 1.0
