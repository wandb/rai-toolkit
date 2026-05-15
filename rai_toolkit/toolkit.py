"""RAIToolkit — top-level orchestrator for the Responsible AI toolkit.

Provides a unified interface for compliance mapping, evaluation, and guardrails.
"""

from __future__ import annotations

from typing import Any

from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework
from rai_toolkit.evaluation.datasets import DatasetLoader
from rai_toolkit.evaluation.pipeline import EvaluationResults, RAIEvaluationPipeline
from rai_toolkit.evaluation.report import ComplianceReport
from rai_toolkit.guardrails.guarded_model import GuardedModel
from rai_toolkit.models.base import BaseModel
from rai_toolkit.scorers.base import BaseScorer


class RAIToolkit:
    """Unified interface for the Responsible AI toolkit.

    Example::

        from rai_toolkit import RAIToolkit

        # Initialize
        toolkit = RAIToolkit()

        # Create a compliance profile
        profile = toolkit.create_profile(industry="healthcare")

        # See what scorers will be used
        scorers = toolkit.get_scorers(profile)

        # Run evaluation
        dataset = toolkit.load_dataset("datasets/rag_qa_dataset.csv")
        results = await toolkit.evaluate(model, profile, dataset)

        # Generate report
        report = toolkit.generate_report(results)
        print(report.to_summary())

        # Wrap model with guardrails
        guarded = toolkit.create_guarded_model(
            model=model,
            scorers=scorers,
        )
    """

    def __init__(
        self,
        custom_scorers: dict[str, type[BaseScorer]] | None = None,
    ) -> None:
        """Initialize the toolkit.

        Args:
            custom_scorers: Dict mapping class names to custom scorer classes.
                          These are registered with the compliance engine.
        """
        self.engine = ComplianceMappingEngine(custom_scorers=custom_scorers)
        self.pipeline = RAIEvaluationPipeline(self.engine)

    # --- Compliance ---

    def get_frameworks(self) -> list[Framework]:
        """List available compliance frameworks."""
        return self.engine.get_frameworks()

    def get_industry_presets(self) -> dict[str, list[str]]:
        """List available industry presets."""
        return self.engine.get_industry_presets()

    def create_profile(
        self,
        industry: str | None = None,
        framework: Framework = Framework.MIT_AI_RISK,
        category_ids: list[str] | None = None,
        name: str = "RAI Assessment",
    ) -> ComplianceProfile:
        """Create a compliance profile.

        Args:
            industry: Use an industry preset (healthcare, financial_services, etc.).
            framework: Compliance framework. Defaults to MIT AI Risk Repository.
            category_ids: Explicit list of category IDs. Ignored if industry is set.
            name: Profile name.

        Returns:
            ComplianceProfile ready for evaluation.
        """
        if industry:
            return self.engine.create_profile_from_preset(
                industry=industry, name=name, framework=framework
            )
        elif category_ids:
            return self.engine.create_profile(
                framework=framework,
                category_ids=category_ids,
                name=name,
            )
        else:
            raise ValueError("Provide either 'industry' or 'category_ids'")

    def get_scorers(self, profile: ComplianceProfile) -> list[BaseScorer]:
        """Resolve scorers for a compliance profile."""
        return self.engine.resolve_scorers(profile)

    def get_nist_coverage(self, profile: ComplianceProfile) -> dict[str, Any]:
        """Get NIST AI RMF coverage for a profile."""
        return self.engine.get_nist_coverage(profile)

    def get_eu_ai_act_coverage(self, profile: ComplianceProfile) -> dict[str, Any]:
        """Get EU AI Act coverage for a profile."""
        return self.engine.get_eu_ai_act_coverage(profile)

    # --- Datasets ---

    @staticmethod
    def load_dataset(path: str) -> list[dict[str, str]]:
        """Load a dataset from file (CSV, JSON, JSONL)."""
        return DatasetLoader.from_file(path)

    @staticmethod
    def create_dataset(items: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Create a dataset from a list of dicts."""
        return DatasetLoader.from_list(items)

    # --- Evaluation ---

    async def evaluate(
        self,
        model: BaseModel,
        profile: ComplianceProfile,
        dataset: list[dict[str, str]],
        name: str = "RAI Evaluation",
        additional_scorers: list[BaseScorer] | None = None,
    ) -> EvaluationResults:
        """Run a compliance-aware evaluation.

        Args:
            model: The model to evaluate.
            profile: Compliance profile defining risk categories.
            dataset: Evaluation dataset.
            name: Evaluation run name.
            additional_scorers: Extra scorers beyond the compliance-mapped ones.

        Returns:
            EvaluationResults with per-item and aggregate scores.
        """
        if additional_scorers:
            self.pipeline.additional_scorers = additional_scorers
        return await self.pipeline.run_evaluation(
            model=model, profile=profile, dataset=dataset, name=name
        )

    async def compare_models(
        self,
        models: list[BaseModel],
        profile: ComplianceProfile,
        dataset: list[dict[str, str]],
        name: str = "Model Comparison",
    ) -> dict[str, EvaluationResults]:
        """Compare multiple models on the same evaluation."""
        return await self.pipeline.run_comparative(
            models=models, profile=profile, dataset=dataset, name=name
        )

    # --- Reports ---

    @staticmethod
    def generate_report(results: EvaluationResults) -> ComplianceReport:
        """Generate a compliance report from evaluation results."""
        return ComplianceReport(results)

    # --- Guardrails ---

    @staticmethod
    def create_guarded_model(
        model: BaseModel,
        input_guardrails: list | None = None,
        output_guardrails: list | None = None,
        scorers: list[BaseScorer] | None = None,
        block_on_scorer_fail: bool = False,
    ) -> GuardedModel:
        """Wrap a model with guardrails and scorer checks.

        Args:
            model: The model to protect.
            input_guardrails: Guardrails to check user input.
            output_guardrails: Guardrails to check model output.
            scorers: Scorers to run on output (flag or block).
            block_on_scorer_fail: Block response if any scorer fails.

        Returns:
            GuardedModel with the full safety pipeline.
        """
        return GuardedModel(
            model=model,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            output_scorers=scorers,
            block_on_scorer_fail=block_on_scorer_fail,
        )
