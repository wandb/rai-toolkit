# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Composite scorers: combine multiple scorers into a single assessment."""

from __future__ import annotations

import asyncio
from typing import Any

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.llm_judges import LLMJudgeScorer
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

        return self._combine_results(results)

    async def score_async(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        """Asynchronously score each child before combining its result.

        Child scorers run sequentially to preserve the composite's existing
        execution behaviour. Sync-delegating children run in a worker thread;
        genuine asynchronous overrides are awaited directly.
        """
        results = []
        for scorer in self.scorers:
            # Match the Weave adapter's handling of the bundled sync delegates.
            if type(scorer).score_async in {
                BaseScorer.score_async,
                LLMJudgeScorer.score_async,
            }:
                result = await asyncio.to_thread(
                    scorer.score, output=output, input=input, context=context, **kwargs
                )
            else:
                result = await scorer.score_async(
                    output=output, input=input, context=context, **kwargs
                )
            results.append(result)

        return self._combine_results(results)

    def _combine_results(self, results: list[ScorerResult]) -> ScorerResult:
        """Build a composite result from child results.

        Un-assessed results are retained in details for coverage reporting but
        never contribute to the aggregate or pass/fail decision.
        """
        assessed_results = [result for result in results if result.assessed]
        unassessed_count = len(results) - len(assessed_results)

        if not assessed_results:
            return ScorerResult(
                score=0.0,
                passed=False,
                category=self.category or "composite",
                explanation="Un-assessed: no child scorers produced a signal.",
                details=self._details(
                    results,
                    aggregate=0.0,
                    assessed_count=0,
                    unassessed_count=unassessed_count,
                ),
                assessed=False,
            )

        # Aggregate only assessed results, which also renormalizes weights.
        aggregate = ScoreNormalizer.aggregate_scores(assessed_results, self.weights)

        # Determine pass/fail
        if self.fail_fast:
            passed = all(r.passed for r in assessed_results)
        else:
            passed = ScoreNormalizer.apply_threshold(aggregate, self.threshold)

        # Build explanation
        failed_scorers = [r for r in assessed_results if not r.passed]
        if failed_scorers:
            explanations = [
                f"{r.category}: {r.explanation}" for r in failed_scorers
            ]
            explanation = f"Failed checks: {'; '.join(explanations)}"
        else:
            explanation = f"All {len(assessed_results)} assessed checks passed"

        return ScorerResult(
            score=aggregate,
            passed=passed,
            category=self.category or "composite",
            explanation=explanation,
            details=self._details(
                results,
                aggregate=aggregate,
                assessed_count=len(assessed_results),
                unassessed_count=unassessed_count,
            ),
        )

    def _details(
        self,
        results: list[ScorerResult],
        *,
        aggregate: float,
        assessed_count: int,
        unassessed_count: int,
    ) -> dict[str, Any]:
        return {
            "individual_results": [
                {
                    "scorer": scorer.name,
                    "category": result.category,
                    "score": result.score,
                    "passed": result.passed,
                    "explanation": result.explanation,
                    "assessed": result.assessed,
                }
                for scorer, result in zip(self.scorers, results)
            ],
            "aggregate_score": aggregate,
            "scorers_run": len(results),
            "scorers_assessed": assessed_count,
            "scorers_unassessed": unassessed_count,
            "scorers_passed": sum(1 for r in results if r.assessed and r.passed),
            "scorers_failed": sum(
                1 for r in results if r.assessed and not r.passed
            ),
        }
