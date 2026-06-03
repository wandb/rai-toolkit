# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Base scorer interface — the core abstraction users implement for custom scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorerResult:
    """Universal scorer output. Same structure regardless of platform.

    Attributes:
        score: 0.0 to 1.0 where higher means better/safer.
        passed: Whether the score meets the threshold.
        category: Risk category ID this result maps to (e.g. "MIT-1.1").
        explanation: Human-readable reasoning for the score.
        details: Scorer-specific metadata (raw outputs, sub-scores, etc.).
        assessed: True when the scorer produced a real signal. False when
            the scorer could not evaluate this row (missing context, parser
            failure, scorer error). Un-assessed results are excluded from
            aggregations and never trigger policy violations — surface the
            gap honestly instead of inflating with neutral defaults.
    """

    score: float
    passed: bool
    category: str
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)
    assessed: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


class BaseScorer(ABC):
    """Abstract scorer interface. Subclass this to create custom scorers.

    Example::

        class MedicalAccuracyScorer(BaseScorer):
            name = "medical_accuracy"
            description = "Checks clinical claims against approved sources"
            category = "MIT-3.1"
            threshold = 0.8

            def score(self, output, input="", context="", **kwargs):
                # Your domain-specific scoring logic
                is_accurate = check_claims(output, context)
                return ScorerResult(
                    score=1.0 if is_accurate else 0.0,
                    passed=is_accurate,
                    category=self.category,
                    explanation="All claims verified" if is_accurate else "Unverified claims found",
                )
    """

    name: str = ""
    description: str = ""
    category: str = ""
    threshold: float = 0.5

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        threshold: float | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if category is not None:
            self.category = category
        if threshold is not None:
            self.threshold = threshold

        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        """Score a model output. Override this method.

        Args:
            output: The model's response text.
            input: The original user input/query.
            context: Retrieved context (for RAG systems).
            **kwargs: Additional scorer-specific arguments.

        Returns:
            ScorerResult with score, passed, category, explanation, and details.
        """
        ...

    async def score_async(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        """Async version of score. Override for async scorers (e.g. LLM judges).

        Default implementation calls the sync score method.
        """
        return self.score(output=output, input=input, context=context, **kwargs)

    def score_batch(
        self, items: list[dict[str, Any]]
    ) -> list[ScorerResult]:
        """Score multiple items. Override for batch-optimized implementations.

        Args:
            items: List of dicts with keys matching score() parameters.

        Returns:
            List of ScorerResult, one per item.
        """
        return [self.score(**item) for item in items]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category!r}, threshold={self.threshold})"
