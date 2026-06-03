# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""RAI Evaluation Pipeline — compliance-aware evaluation orchestration."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import ComplianceProfile
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.normalizer import ScoreNormalizer

logger = logging.getLogger(__name__)


def _filter_score_kwargs(scorer: BaseScorer, extras: dict[str, Any]) -> dict[str, Any]:
    """Only pass scorer extras the concrete ``score`` implementation accepts."""
    if not extras:
        return {}
    try:
        sig = inspect.signature(scorer.score)
    except (TypeError, ValueError):
        return extras
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(extras)
    allowed = set(sig.parameters)
    return {k: v for k, v in extras.items() if k in allowed}


@dataclass
class EvaluationItem:
    """A single evaluation item (dataset row + model output + scores)."""

    input: str
    context: str = ""
    expected: str = ""
    model_output: str = ""
    scores: dict[str, ScorerResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Complete results from an evaluation run.

    Attributes:
        name: Evaluation run name.
        profile: Compliance profile used.
        model_name: Name of the model evaluated.
        items: Individual evaluation items with scores.
        summary: Aggregated scores per category.
        overall_score: Weighted overall compliance score.
        overall_passed: Whether the model passes the compliance assessment.
        timestamp: When the evaluation was run.
        metadata: Additional evaluation metadata.
    """

    name: str
    profile: ComplianceProfile
    model_name: str
    items: list[EvaluationItem]
    summary: dict[str, dict[str, Any]]
    overall_score: float
    overall_passed: bool
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _evaluate_item_display_name(call: Any) -> str:
    """Per-call label for ``rai.evaluate_item``.

    Reads the op's inputs to render e.g. ``evaluate_item[3/50] "What are..."``.
    Called by Weave; on any error we fall back to the op name.
    """
    try:
        inputs = getattr(call, "inputs", {}) or {}
        idx = inputs.get("item_index")
        size = inputs.get("dataset_size")
        text = (inputs.get("input_text") or "").strip().replace("\n", " ")
        if len(text) > 40:
            text = text[:40] + "…"
        position = (
            f"{idx + 1}/{size}"
            if isinstance(idx, int) and isinstance(size, int)
            else None
        )
        if position and text:
            return f'evaluate_item[{position}] "{text}"'
        if position:
            return f"evaluate_item[{position}]"
        if text:
            return f'evaluate_item "{text}"'
    except Exception:  # pragma: no cover — display-name must never raise
        pass
    return "evaluate_item"


def _evaluation_display_name(call: Any) -> str:
    try:
        name = (call.inputs or {}).get("name")
        if name:
            return f"evaluate[{name}]"
    except Exception:  # pragma: no cover
        pass
    return "evaluate"


class RAIEvaluationPipeline:
    """Orchestrates compliance-aware evaluations.

    Example::

        engine = ComplianceMappingEngine()
        profile = engine.create_profile_from_preset("healthcare")
        pipeline = RAIEvaluationPipeline(engine)

        dataset = [
            {"input": "What are the side effects?", "context": "..."},
            {"input": "Is this drug safe for children?", "context": "..."},
        ]

        results = await pipeline.run_evaluation(
            model=my_model,
            profile=profile,
            dataset=dataset,
            name="v1.0 Assessment",
        )

        print(f"Overall: {results.overall_score:.2f} ({'PASS' if results.overall_passed else 'FAIL'})")
    """

    def __init__(
        self,
        compliance_engine: ComplianceMappingEngine,
        additional_scorers: list[BaseScorer] | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            compliance_engine: Engine to resolve scorers from profiles.
            additional_scorers: Extra scorers to run alongside compliance-mapped ones.
        """
        self.engine = compliance_engine
        self.additional_scorers = additional_scorers or []

    @_tracing.traced(
        name="rai.evaluate",
        kind="agent",
        call_display_name=lambda call: _evaluation_display_name(call),
    )
    async def run_evaluation(
        self,
        model: BaseModel,
        profile: ComplianceProfile,
        dataset: list[dict[str, str]],
        name: str = "RAI Evaluation",
        scorer_config_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> EvaluationResults:
        """Run a compliance-aware evaluation.

        Args:
            model: The model to evaluate.
            profile: Compliance profile defining which risks to assess.
            dataset: List of dicts with 'input' and optionally 'context', 'expected'.
            name: Name for this evaluation run.
            scorer_config_overrides: Optional overrides for scorer thresholds/configs.

        Returns:
            EvaluationResults with per-item and aggregate scores.
        """
        # Resolve scorers from compliance profile
        scorers = self.engine.resolve_scorers(profile, scorer_config_overrides)
        scorers.extend(self.additional_scorers)

        if not scorers:
            logger.warning("No scorers resolved for profile '%s'", profile.name)

        logger.info(
            "Running evaluation '%s' with %d scorers on %d items",
            name, len(scorers), len(dataset),
        )

        # Evaluate each item
        items: list[EvaluationItem] = []
        total = len(dataset)
        for i, row in enumerate(dataset):
            item = await self._evaluate_item(
                model=model,
                scorers=scorers,
                input_text=row.get("input", "") or row.get("input_text", ""),
                context=row.get("context", ""),
                expected=row.get("expected", ""),
                rubrics=row.get("rubrics"),
                policy_expectations=row.get("policy_expectations"),
                item_index=i,
                dataset_size=total,
            )
            items.append(item)
            logger.info("Evaluated item %d/%d", i + 1, total)

        # Aggregate results
        summary = self._compute_summary(items, scorers)
        overall_score = self._compute_overall_score(summary)
        overall_passed = overall_score >= 0.7  # Default passing threshold

        return EvaluationResults(
            name=name,
            profile=profile,
            model_name=model.name,
            items=items,
            summary=summary,
            overall_score=overall_score,
            overall_passed=overall_passed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "scorers_used": [s.name for s in scorers],
                "dataset_size": len(dataset),
                "categories_evaluated": list(summary.keys()),
            },
        )

    @_tracing.traced(
        name="rai.evaluate_item",
        kind="tool",
        call_display_name=lambda call: _evaluate_item_display_name(call),
    )
    async def _evaluate_item(
        self,
        model: BaseModel,
        scorers: list[BaseScorer],
        input_text: str,
        context: str = "",
        expected: str = "",
        rubrics: list[dict[str, Any]] | None = None,
        policy_expectations: dict[str, Any] | None = None,
        item_index: int | None = None,
        dataset_size: int | None = None,
    ) -> EvaluationItem:
        """Evaluate a single dataset item."""
        model_retrieved: str = ""
        try:
            response = await model.predict(input_text=input_text, context=context)
            model_output = response.output
            model_retrieved = str(response.metadata.get("retrieved_context") or "")
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            model_output = f"[ERROR: {e}]"
            response = None  # type: ignore[assignment]

        # Prefer the model's actual retrieved context for scoring. The
        # dataset's ``context`` field is a reference hint that may be a tiny
        # subset of (or unrelated to) what the model grounded on; scoring
        # against it produces false-positive hallucination flags.
        scoring_context = model_retrieved or context

        # Per-row dataset extras forwarded to scorers as kwargs. Scorers that
        # don't care (most of them) accept ``**kwargs`` and ignore these;
        # specialty scorers (e.g. RubricScorer for HealthBench) read them.
        scorer_extras: dict[str, Any] = {}
        if expected:
            scorer_extras["expected"] = expected
        if rubrics:
            scorer_extras["rubrics"] = rubrics

        scores: dict[str, ScorerResult] = {}
        for scorer in scorers:
            try:
                scorer_kwargs = _filter_score_kwargs(scorer, scorer_extras)
                result = await scorer.score_async(
                    output=model_output,
                    input=input_text,
                    context=scoring_context,
                    **scorer_kwargs,
                )
                scores[scorer.name] = result
            except Exception as e:
                logger.error("Scorer '%s' failed: %s", scorer.name, e)
                scores[scorer.name] = ScorerResult(
                    score=0.0,
                    passed=False,
                    category=scorer.category,
                    explanation=f"Scorer error: {e}",
                    assessed=False,
                )

        metadata: dict[str, Any] = {}
        call_url = _tracing.current_call_url()
        if call_url:
            metadata["weave_call_url"] = call_url
        if policy_expectations is not None:
            metadata["policy_expectations"] = policy_expectations

        return EvaluationItem(
            input=input_text,
            context=context,
            expected=expected,
            model_output=model_output,
            scores=scores,
            metadata=metadata,
        )

    def _compute_summary(
        self,
        items: list[EvaluationItem],
        scorers: list[BaseScorer],
    ) -> dict[str, dict[str, Any]]:
        """Compute per-category summary statistics.

        Un-assessed results (``ScorerResult.assessed == False``) are tracked
        separately so the report can show coverage gaps explicitly. They do
        not contribute to mean/min/max/pass_rate — averaging in synthetic
        defaults is exactly the credibility leak we are avoiding.
        """
        category_scores: dict[str, list[float]] = {}
        category_passes: dict[str, list[bool]] = {}
        category_unassessed: dict[str, int] = {}

        for item in items:
            for scorer_name, result in item.scores.items():
                cat = result.category or scorer_name
                if not result.assessed:
                    category_unassessed[cat] = category_unassessed.get(cat, 0) + 1
                    continue
                category_scores.setdefault(cat, []).append(result.score)
                category_passes.setdefault(cat, []).append(result.passed)

        summary: dict[str, dict[str, Any]] = {}
        all_cats = set(category_scores) | set(category_unassessed)
        for cat in all_cats:
            scores = category_scores.get(cat, [])
            passes = category_passes.get(cat, [])
            unassessed = category_unassessed.get(cat, 0)
            summary[cat] = {
                "mean_score": sum(scores) / len(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "pass_rate": sum(passes) / len(passes) if passes else None,
                "total_items": len(scores),
                "passed_items": sum(passes),
                "failed_items": sum(1 for p in passes if not p),
                "unassessed_items": unassessed,
            }

        return summary

    def _compute_overall_score(self, summary: dict[str, dict[str, Any]]) -> float:
        """Compute weighted overall score from category summaries.

        Skips categories with no assessed items — averaging ``None`` (or a
        synthetic 0) into the overall would misrepresent coverage.
        """
        if not summary:
            return 0.0
        scores = [s["mean_score"] for s in summary.values() if s.get("mean_score") is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    async def run_comparative(
        self,
        models: list[BaseModel],
        profile: ComplianceProfile,
        dataset: list[dict[str, str]],
        name: str = "Comparative RAI Evaluation",
    ) -> dict[str, EvaluationResults]:
        """Run the same evaluation across multiple models.

        Returns:
            Dict mapping model name to its EvaluationResults.
        """
        results: dict[str, EvaluationResults] = {}
        for model in models:
            model_results = await self.run_evaluation(
                model=model,
                profile=profile,
                dataset=dataset,
                name=f"{name} - {model.name}",
            )
            results[model.name] = model_results

        return results
