# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Weave evaluation runner: runs rai_toolkit evaluations via weave.Evaluation."""

from __future__ import annotations

import logging
from typing import Any

import weave

from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import ComplianceProfile
from integrations.weave_integration.scorers import (
    WeaveRAIScorer,
    get_weave_builtin_scorers,
    make_weave_rai_scorer,
)

logger = logging.getLogger(__name__)


class WeaveEvaluationRunner:
    """Runs RAI evaluations using weave.Evaluation.

    Bridges the rai_toolkit evaluation pipeline to Weave's native
    evaluation system for full tracing, dashboards, and comparison.

    Example::

        runner = WeaveEvaluationRunner(engine)

        results = await runner.run(
            model=weave_model,
            profile=profile,
            dataset=dataset,
            name="v1.0 RAI Assessment",
            include_weave_builtins=True,
        )
    """

    def __init__(self, compliance_engine: ComplianceMappingEngine) -> None:
        self.engine = compliance_engine

    def build_scorer_list(
        self,
        profile: ComplianceProfile,
        include_weave_builtins: bool = True,
        additional_scorers: list[weave.Scorer] | None = None,
    ) -> list[weave.Scorer]:
        """Assemble the Weave scorer list (RAI judges + optional Weave built-ins)."""
        rai_scorers = self.engine.resolve_scorers(profile)
        weave_scorers: list[weave.Scorer] = [
            make_weave_rai_scorer(s) for s in rai_scorers
        ]
        if include_weave_builtins:
            builtin_map = get_weave_builtin_scorers(profile.category_ids)
            for _cat_id, scorers in builtin_map.items():
                weave_scorers.extend(scorers)
        if additional_scorers:
            weave_scorers.extend(additional_scorers)
        return weave_scorers

    def build_evaluation(
        self,
        profile: ComplianceProfile,
        dataset: list[dict[str, Any]] | weave.Dataset,
        name: str = "RAI Evaluation",
        include_weave_builtins: bool = True,
        additional_scorers: list[weave.Scorer] | None = None,
    ) -> weave.Evaluation:
        """Construct a ``weave.Evaluation`` without running it."""
        weave_scorers = self.build_scorer_list(
            profile,
            include_weave_builtins=include_weave_builtins,
            additional_scorers=additional_scorers,
        )
        logger.info(
            "Built Weave evaluation '%s' with %d scorers",
            name, len(weave_scorers),
        )
        return weave.Evaluation(
            dataset=dataset,
            scorers=weave_scorers,
            name=name,
        )

    @weave.op()
    async def run(
        self,
        model: weave.Model,
        profile: ComplianceProfile,
        dataset: list[dict[str, Any]] | weave.Dataset,
        name: str = "RAI Evaluation",
        include_weave_builtins: bool = True,
        additional_scorers: list[weave.Scorer] | None = None,
    ) -> Any:
        """Run a compliance-aware evaluation using weave.Evaluation.

        Args:
            model: A weave.Model to evaluate.
            profile: Compliance profile defining risk categories.
            dataset: Evaluation dataset (list of dicts or weave.Dataset).
            name: Evaluation name for Weave UI.
            include_weave_builtins: Also run matching Weave built-in scorers.
            additional_scorers: Extra Weave scorers to include.

        Returns:
            Weave evaluation summary dict (from ``summarize``).
        """
        evaluation = self.build_evaluation(
            profile,
            dataset,
            name=name,
            include_weave_builtins=include_weave_builtins,
            additional_scorers=additional_scorers,
        )
        return await evaluation.evaluate(model)

    async def get_detailed_evaluation(
        self,
        model: weave.Model,
        profile: ComplianceProfile,
        dataset: list[dict[str, Any]] | weave.Dataset,
        name: str = "RAI Evaluation",
        include_weave_builtins: bool = True,
        additional_scorers: list[weave.Scorer] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return per-row ``EvaluationResults`` from Weave plus the summary dict.

        Calls ``evaluate()`` (not ``get_eval_results`` directly) so the run
        registers in the Weave UI's Evals tab. The tab filters on op_name
        ``Evaluation.evaluate`` and only that method carries it. Per-row
        results are captured by transparently wrapping
        ``evaluation.get_eval_results`` on the instance; ``evaluate`` calls
        ``get_eval_results`` internally, so we get both the UI registration
        and the row-level data without re-running the model.
        """
        evaluation = self.build_evaluation(
            profile,
            dataset,
            name=name,
            include_weave_builtins=include_weave_builtins,
            additional_scorers=additional_scorers,
        )
        if not hasattr(evaluation, "get_eval_results"):
            raise RuntimeError(
                "weave.Evaluation.get_eval_results is missing; upgrade weave "
                "to a recent 0.52+ release for assessment integration."
            )

        captured: list[Any] = []
        original_get_eval_results = evaluation.get_eval_results

        async def _capturing_get_eval_results(model_: Any) -> Any:
            results = await original_get_eval_results(model_)
            captured.append(results)
            return results

        # Bypass pydantic's __setattr__; bind the wrapper as an instance attr
        # so `self.get_eval_results(...)` inside `evaluate()` resolves to it.
        object.__setattr__(evaluation, "get_eval_results", _capturing_get_eval_results)

        summary = await evaluation.evaluate(model)
        if not captured:
            raise RuntimeError(
                "weave.Evaluation.evaluate did not invoke get_eval_results; "
                "cannot recover per-row results."
            )
        return captured[0], summary

    @weave.op()
    async def run_comparative(
        self,
        models: list[weave.Model],
        profile: ComplianceProfile,
        dataset: list[dict[str, Any]] | weave.Dataset,
        name: str = "Model Comparison",
        include_weave_builtins: bool = True,
    ) -> dict[str, Any]:
        """Run the same evaluation across multiple models for leaderboard.

        Returns:
            Dict mapping model name to evaluation results.
        """
        results: dict[str, Any] = {}
        for model in models:
            model_name = getattr(model, "model_name", model.__class__.__name__)
            result = await self.run(
                model=model,
                profile=profile,
                dataset=dataset,
                name=f"{name} - {model_name}",
                include_weave_builtins=include_weave_builtins,
            )
            results[model_name] = result

        return results
