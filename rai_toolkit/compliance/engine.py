# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Compliance Mapping Engine — the central orchestrator.

Maps compliance frameworks to risk categories to scorers.
This is the primary interface for building compliance-aware RAI evaluations.
"""

from __future__ import annotations

from typing import Any

from rai_toolkit.compliance.frameworks import (
    ComplianceProfile,
    Framework,
    INDUSTRY_PRESETS,
    RiskCategory,
)
from rai_toolkit.compliance.mit_taxonomy import (
    MIT_TAXONOMY,
    MIT_DOMAINS,
    get_all_categories,
    get_categories_by_domain,
    get_scorable_categories,
)
from rai_toolkit.compliance.nist_mapping import (
    NIST_FUNCTIONS,
    NIST_TO_MIT_MAPPING,
    get_mit_categories_for_nist,
)
from rai_toolkit.compliance.eu_ai_act_mapping import (
    EU_AI_ACT_REQUIREMENTS,
    get_all_required_mit_categories,
    get_mit_categories_for_eu_requirement,
)
from rai_toolkit.compliance.scorer_registry import (
    SCORER_REGISTRY,
    get_scorer_mapping,
)
from rai_toolkit.scorers.base import BaseScorer


# Lazy import to avoid circular dependencies
_SCORER_CLASS_CACHE: dict[str, type[BaseScorer]] = {}


def _resolve_scorer_class(class_name: str) -> type[BaseScorer] | None:
    """Resolve a scorer class name to the actual class."""
    if class_name in _SCORER_CLASS_CACHE:
        return _SCORER_CLASS_CACHE[class_name]

    # Try importing from known modules
    modules = [
        "rai_toolkit.scorers.llm_judges",
        "rai_toolkit.scorers.programmatic",
        "rai_toolkit.scorers.composite",
    ]
    for module_path in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, BaseScorer):
                _SCORER_CLASS_CACHE[class_name] = cls
                return cls
        except ImportError:
            continue
    return None


class ComplianceMappingEngine:
    """Maps compliance frameworks to risk categories to scorers.

    Example::

        engine = ComplianceMappingEngine()

        # List available frameworks
        frameworks = engine.get_frameworks()

        # Get all MIT risk categories
        categories = engine.get_categories(Framework.MIT_AI_RISK)

        # Use an industry preset
        profile = engine.create_profile_from_preset("healthcare", name="My Assessment")

        # Or pick categories manually
        profile = engine.create_profile(
            framework=Framework.MIT_AI_RISK,
            category_ids=["MIT-1.1", "MIT-2.1", "MIT-3.1"],
            name="Custom Assessment",
        )

        # Resolve scorers for the profile
        scorers = engine.resolve_scorers(profile)

        # Get NIST mapping
        nist_map = engine.get_nist_coverage(profile)
    """

    def __init__(self, custom_scorers: dict[str, type[BaseScorer]] | None = None) -> None:
        """Initialize the engine with optional custom scorer classes.

        Args:
            custom_scorers: Dict mapping class names to scorer classes.
                          These override or extend the built-in scorer registry.
        """
        if custom_scorers:
            _SCORER_CLASS_CACHE.update(custom_scorers)

    @staticmethod
    def get_frameworks() -> list[Framework]:
        """Return all supported compliance frameworks."""
        return list(Framework)

    @staticmethod
    def get_categories(framework: Framework) -> list[RiskCategory]:
        """Get all risk categories for a framework.

        Currently the MIT AI Risk Repository is the primary taxonomy.
        NIST and EU AI Act map to MIT categories under the hood.
        """
        if framework == Framework.MIT_AI_RISK:
            return get_all_categories()
        elif framework == Framework.NIST_AI_RMF:
            # NIST MEASURE function maps to scorable MIT categories
            mit_ids = get_mit_categories_for_nist("MEASURE")
            return [MIT_TAXONOMY[id_] for id_ in mit_ids if id_ in MIT_TAXONOMY]
        elif framework == Framework.EU_AI_ACT:
            mit_ids = get_all_required_mit_categories()
            return [MIT_TAXONOMY[id_] for id_ in mit_ids if id_ in MIT_TAXONOMY]
        return []

    @staticmethod
    def get_domains(framework: Framework) -> dict[str, list[str]]:
        """Get domain structure for a framework."""
        if framework == Framework.MIT_AI_RISK:
            return MIT_DOMAINS
        elif framework == Framework.NIST_AI_RMF:
            return {fid: NIST_TO_MIT_MAPPING.get(fid, []) for fid in NIST_FUNCTIONS}
        elif framework == Framework.EU_AI_ACT:
            return {
                req.title: req.mit_category_ids
                for req in EU_AI_ACT_REQUIREMENTS.values()
            }
        return {}

    @staticmethod
    def get_industry_presets() -> dict[str, list[str]]:
        """Return available industry presets with their risk category IDs."""
        return INDUSTRY_PRESETS

    def create_profile(
        self,
        framework: Framework,
        category_ids: list[str],
        name: str = "RAI Assessment",
        industry: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComplianceProfile:
        """Create a compliance profile from selected categories.

        Args:
            framework: The compliance framework.
            category_ids: List of risk category IDs to include.
            name: Profile name.
            industry: Optional industry context.
            metadata: Optional additional metadata.

        Returns:
            ComplianceProfile with resolved categories.

        Raises:
            ValueError: If a category_id is not found in the taxonomy.
        """
        categories = []
        for cat_id in category_ids:
            cat = MIT_TAXONOMY.get(cat_id)
            if cat is None:
                raise ValueError(
                    f"Unknown risk category: {cat_id}. "
                    f"Available: {sorted(MIT_TAXONOMY.keys())}"
                )
            categories.append(cat)

        return ComplianceProfile(
            name=name,
            framework=framework,
            categories=categories,
            industry=industry,
            metadata=metadata or {},
        )

    def create_profile_from_preset(
        self,
        industry: str,
        name: str | None = None,
        framework: Framework = Framework.MIT_AI_RISK,
    ) -> ComplianceProfile:
        """Create a profile from an industry preset.

        Args:
            industry: Industry name (healthcare, financial_services, government, general).
            name: Optional profile name. Defaults to "{Industry} RAI Assessment".
            framework: Compliance framework. Defaults to MIT AI Risk Repository.

        Returns:
            ComplianceProfile with preset categories.

        Raises:
            ValueError: If the industry preset is not found.
        """
        preset = INDUSTRY_PRESETS.get(industry)
        if preset is None:
            raise ValueError(
                f"Unknown industry preset: {industry}. "
                f"Available: {sorted(INDUSTRY_PRESETS.keys())}"
            )
        return self.create_profile(
            framework=framework,
            category_ids=preset,
            name=name or f"{industry.replace('_', ' ').title()} RAI Assessment",
            industry=industry,
        )

    def resolve_scorers(
        self,
        profile: ComplianceProfile,
        scorer_config_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[BaseScorer]:
        """Instantiate all scorers needed for a compliance profile.

        Args:
            profile: The compliance profile to resolve scorers for.
            scorer_config_overrides: Optional overrides for scorer configuration.

        Returns:
            List of instantiated BaseScorer objects.
        """
        overrides = scorer_config_overrides or {}
        scorers: list[BaseScorer] = []
        seen_classes: set[str] = set()

        for category in profile.categories:
            mapping = get_scorer_mapping(category.id)
            if mapping is None:
                continue

            for class_name in mapping.scorer_classes:
                if class_name in seen_classes:
                    continue
                seen_classes.add(class_name)

                cls = _resolve_scorer_class(class_name)
                if cls is None:
                    continue

                # Merge default config with overrides
                config = mapping.scorer_config.get(class_name, {})
                if class_name in overrides:
                    config.update(overrides[class_name])

                # Set category from the mapping
                config.setdefault("category", category.id)

                try:
                    scorer = cls(**config)
                    scorers.append(scorer)
                except Exception:
                    # Skip scorers that fail to instantiate (e.g. missing API key)
                    continue

        return scorers

    def get_scorable_categories(self, profile: ComplianceProfile) -> list[RiskCategory]:
        """Return only the categories in the profile that have associated scorers."""
        return [c for c in profile.categories if c.id in SCORER_REGISTRY]

    def get_unscorable_categories(self, profile: ComplianceProfile) -> list[RiskCategory]:
        """Return categories that cannot be automatically scored."""
        return [c for c in profile.categories if c.id not in SCORER_REGISTRY]

    def get_nist_coverage(self, profile: ComplianceProfile) -> dict[str, dict[str, Any]]:
        """Map a profile's categories to NIST AI RMF functions.

        Returns a dict of NIST functions with their coverage status.
        """
        category_ids = set(profile.category_ids)
        coverage: dict[str, dict[str, Any]] = {}

        for func_id, func in NIST_FUNCTIONS.items():
            mapped_mits = set(NIST_TO_MIT_MAPPING.get(func_id, []))
            covered = category_ids & mapped_mits
            coverage[func_id] = {
                "function": func.name,
                "description": func.description,
                "total_categories": len(mapped_mits),
                "covered_categories": len(covered),
                "coverage_pct": len(covered) / len(mapped_mits) * 100 if mapped_mits else 0,
                "covered_ids": sorted(covered),
                "missing_ids": sorted(mapped_mits - category_ids),
                "capabilities": func.rai_capabilities,
            }

        return coverage

    def get_eu_ai_act_coverage(self, profile: ComplianceProfile) -> dict[str, dict[str, Any]]:
        """Map a profile's categories to EU AI Act requirements.

        Returns a dict of EU AI Act requirements with their coverage status.
        """
        category_ids = set(profile.category_ids)
        coverage: dict[str, dict[str, Any]] = {}

        for req_id, req in EU_AI_ACT_REQUIREMENTS.items():
            required_mits = set(req.mit_category_ids)
            covered = category_ids & required_mits
            coverage[req_id] = {
                "article": req.article,
                "title": req.title,
                "description": req.description,
                "total_categories": len(required_mits),
                "covered_categories": len(covered),
                "coverage_pct": len(covered) / len(required_mits) * 100 if required_mits else 0,
                "covered_ids": sorted(covered),
                "missing_ids": sorted(required_mits - category_ids),
                "capabilities": req.rai_capabilities,
            }

        return coverage
