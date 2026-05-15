"""Composite scorers — combine multiple scorers into a single assessment."""

from __future__ import annotations

from typing import Any

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.normalizer import ScoreNormalizer


class CompositeScorer(BaseScorer):
    """Combines multiple scorers into a single composite score.

    Useful for creating domain-specific assessment suites that aggregate
    multiple risk dimensions into one pass/fail decision.

    Example::

        composite = CompositeScorer(
            name="rag_trust",
            scorers=[FactualityJudge(), RegexPIIScorer(), KeywordToxicityScorer()],
            weights={"MIT-3.1": 2.0, "MIT-2.1": 1.5, "MIT-1.2": 1.0},
            threshold=0.7,
            fail_fast=True,  # Fail immediately if any scorer fails
        )
        result = composite.score(output="...", input="...", context="...")
    """

    name = "CompositeScorer"
    description = "Aggregates multiple scorers into a composite assessment"

    def __init__(
        self,
        scorers: list[BaseScorer],
        weights: dict[str, float] | None = None,
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize composite scorer.

        Args:
            scorers: List of scorers to run.
            weights: Optional weights per category. Higher = more important.
            fail_fast: If True, the composite fails if ANY scorer fails.
            **kwargs: Passed to BaseScorer.__init__.
        """
        super().__init__(**kwargs)
        self.scorers = scorers
        self.weights = weights
        self.fail_fast = fail_fast

    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        results: list[ScorerResult] = []

        for scorer in self.scorers:
            result = scorer.score(output=output, input=input, context=context, **kwargs)
            results.append(result)

        # Compute aggregate score
        aggregate = ScoreNormalizer.aggregate_scores(results, self.weights)

        # Determine pass/fail
        if self.fail_fast:
            passed = all(r.passed for r in results)
        else:
            passed = ScoreNormalizer.apply_threshold(aggregate, self.threshold)

        # Build explanation
        failed_scorers = [r for r in results if not r.passed]
        if failed_scorers:
            explanations = [
                f"{r.category}: {r.explanation}" for r in failed_scorers
            ]
            explanation = f"Failed checks: {'; '.join(explanations)}"
        else:
            explanation = f"All {len(results)} checks passed"

        return ScorerResult(
            score=aggregate,
            passed=passed,
            category=self.category or "composite",
            explanation=explanation,
            details={
                "individual_results": [
                    {
                        "scorer": scorer.name,
                        "category": result.category,
                        "score": result.score,
                        "passed": result.passed,
                        "explanation": result.explanation,
                    }
                    for scorer, result in zip(self.scorers, results)
                ],
                "aggregate_score": aggregate,
                "scorers_run": len(results),
                "scorers_passed": sum(1 for r in results if r.passed),
                "scorers_failed": sum(1 for r in results if not r.passed),
            },
        )
