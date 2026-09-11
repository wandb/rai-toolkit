# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Score normalization: converts various scoring scales to a unified 0-1 range."""

from __future__ import annotations

import math

from rai_toolkit.scorers.base import ScorerResult


class ScoreNormalizer:
    """Normalizes scores from different scales to a unified 0.0-1.0 range.

    Supports common scoring patterns:
    - 0-3 compliance scale (PwC/Lilly style): divide by 3
    - 0-5 Likert scale: divide by 5
    - 0-10 scale: divide by 10
    - 0-100 percentage: divide by 100
    - Boolean pass/fail: 1.0 or 0.0
    - Inverted scores (higher = worse, e.g. toxicity): 1.0 - normalized
    """

    @staticmethod
    def from_scale(raw_score: float, max_value: float, invert: bool = False) -> float:
        """Normalize a score from an arbitrary scale to 0.0-1.0.

        Args:
            raw_score: The raw score value.
            max_value: The maximum possible value on the scale.
            invert: If True, invert so higher raw = lower normalized (for risk scores).

        Returns:
            Normalized score between 0.0 and 1.0.
        """
        if not math.isfinite(raw_score):
            raise ValueError(f"raw_score must be finite, got {raw_score}")
        if not math.isfinite(max_value) or max_value <= 0:
            raise ValueError(
                f"max_value must be a finite positive number, got {max_value}"
            )
        normalized = max(0.0, min(1.0, raw_score / max_value))
        return 1.0 - normalized if invert else normalized

    @staticmethod
    def from_boolean(passed: bool) -> float:
        """Convert a boolean pass/fail to a score."""
        return 1.0 if passed else 0.0

    @staticmethod
    def from_compliance_scale(raw_score: float) -> float:
        """Normalize from the 0-3 compliance scale (Fully/Mostly/Partially/Non-Compliant)."""
        return ScoreNormalizer.from_scale(raw_score, max_value=3.0)

    @staticmethod
    def from_likert(raw_score: float) -> float:
        """Normalize from a 1-5 Likert scale."""
        return ScoreNormalizer.from_scale(raw_score - 1.0, max_value=4.0)

    @staticmethod
    def apply_threshold(score: float, threshold: float) -> bool:
        """Determine if a normalized score passes the threshold."""
        return score >= threshold

    @staticmethod
    def aggregate_scores(
        results: list[ScorerResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted average of multiple scorer results.

        Args:
            results: List of ScorerResult objects.
            weights: Optional dict mapping category -> weight. Defaults to equal weights.

        Un-assessed results are skipped entirely: their placeholder scores
        are not a signal and must not dilute the real ones.

        Returns:
            Weighted average score between 0.0 and 1.0.
        """
        assessed = [r for r in results if r.assessed]
        if not assessed:
            return 0.0

        if weights is None:
            return sum(r.score for r in assessed) / len(assessed)

        total_weight = 0.0
        weighted_sum = 0.0
        for result in assessed:
            w = weights.get(result.category, 1.0)
            weighted_sum += result.score * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0
