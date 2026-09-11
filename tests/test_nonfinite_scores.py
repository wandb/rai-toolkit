# SPDX-FileCopyrightText: 2026 Abhinav Garg
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Non-finite scores must be rejected at every layer, not silently averaged in."""

from __future__ import annotations

import pytest

from rai_toolkit.scorers.base import ScorerResult
from rai_toolkit.scorers.normalizer import ScoreNormalizer


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_scorer_result_rejects_nonfinite(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScorerResult(score=bad_value, passed=False, category="test", explanation="x")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_normalizer_from_scale_rejects_nonfinite_raw_score(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScoreNormalizer.from_scale(bad_value, max_value=3.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_normalizer_from_scale_rejects_nonfinite_max_value(bad_value: float) -> None:
    with pytest.raises(ValueError):
        ScoreNormalizer.from_scale(1.0, max_value=bad_value)


def test_aggregate_scores_skips_unassessed_results() -> None:
    results = [
        ScorerResult(
            score=0.8, passed=True, category="a", explanation="ok", assessed=True
        ),
        ScorerResult(
            score=0.6, passed=True, category="b", explanation="ok", assessed=True
        ),
        ScorerResult(
            score=0.0, passed=False, category="c", explanation="skip", assessed=False
        ),
    ]
    avg = ScoreNormalizer.aggregate_scores(results)
    # Should average only the two assessed results: (0.8 + 0.6) / 2 = 0.7
    assert avg == pytest.approx(0.7)


def test_aggregate_scores_weighted_skips_unassessed_results() -> None:
    results = [
        ScorerResult(
            score=0.8, passed=True, category="a", explanation="ok", assessed=True
        ),
        ScorerResult(
            score=0.0, passed=False, category="b", explanation="skip", assessed=False
        ),
    ]
    avg = ScoreNormalizer.aggregate_scores(results, weights={"a": 2.0, "b": 5.0})
    assert avg == pytest.approx(0.8)


def test_aggregate_scores_all_unassessed_returns_zero() -> None:
    results = [
        ScorerResult(
            score=0.0, passed=False, category="a", explanation="skip", assessed=False
        ),
    ]
    assert ScoreNormalizer.aggregate_scores(results) == 0.0


def test_aggregate_scores_empty_returns_zero() -> None:
    assert ScoreNormalizer.aggregate_scores([]) == 0.0


# Regression guards -- valid values still work
@pytest.mark.parametrize("good_value", [0.0, 0.5, 1.0])
def test_scorer_result_accepts_valid_finite_scores(good_value: float) -> None:
    result = ScorerResult(
        score=good_value, passed=True, category="test", explanation="ok"
    )
    assert result.score == good_value


def test_normalizer_from_scale_valid() -> None:
    assert ScoreNormalizer.from_scale(2.0, max_value=3.0) == pytest.approx(2.0 / 3.0)
