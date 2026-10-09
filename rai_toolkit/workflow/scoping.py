# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Risk-aware test scoping.

Given an ``ApplicationProfile``, return a configured ``Assessor`` plus a
human-readable ``ScopingDecision`` explaining *why* each choice was made.
The explanation is important: reviewers need to see the scope rationale to
trust the assessment that follows.

Rules are deliberately simple and data-driven, keep them readable so the
whitepaper can show the full table without hand-waving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rai_toolkit.assessment import Assessor
from rai_toolkit.examples import DEMO_EXAMPLE_BUNDLES
from rai_toolkit.models.base import BaseModel
from rai_toolkit.policies.engine import PolicyEngine, load_policy_set
from rai_toolkit.workflow.profile import (
    ApplicationProfile,
    DeploymentContext,
    Industry,
    RiskTier,
)


@dataclass
class ScopingDecision:
    """Human-readable record of how the assessment was scoped.

    This is the artifact the RAI team reads to verify that the right tests
    ran for this app. Every line corresponds to a decision the scoping
    logic made, with the trait that triggered it.
    """

    preset: str
    datasets: list[str]
    run_redteam: bool
    redteam_max_severity: int
    dataset_limit: int | None
    policies_dir: str | None
    weave_project: str | None
    weave_entity: str | None
    effective_risk_tier: RiskTier
    rationale: list[str] = field(default_factory=list)
    redteam_max_error_rate: float | None = None

    def as_markdown(self) -> str:
        lines = [
            "**Scoping decisions**",
            "",
            f"- Preset: `{self.preset}`",
            f"- Datasets: `{', '.join(self.datasets) if self.datasets else '-'}`",
            f"- Red-team: {'on' if self.run_redteam else 'off'} "
            f"(max severity {self.redteam_max_severity})",
            _error_budget_markdown(self),
            f"- Dataset row cap: {self.dataset_limit if self.dataset_limit else '-'}",
            f"- Effective risk tier: {self.effective_risk_tier.value}",
        ]
        if self.weave_project:
            who = f"{self.weave_entity}/{self.weave_project}" if self.weave_entity else self.weave_project
            lines.append(f"- Weave traces: `{who}`")
        lines.append("")
        lines.append("**Why:**")
        for r in self.rationale:
            lines.append(f"- {r}")
        return "\n".join(lines)


def _error_budget_markdown(decision: ScopingDecision) -> str:
    if not decision.run_redteam:
        return "- Red-team execution-error budget: N/A (red-team off)"
    if decision.redteam_max_error_rate is None:
        return "- Red-team execution-error budget: N/A (not recorded)"
    return f"- Red-team execution-error budget: {decision.redteam_max_error_rate:.1%}"


_SAMPLE_DATA_TYPE_EXTRA_DATASETS: dict[str, list[str]] = {
    "pii": ["pii-extraction-probes"],
    "phi": ["pii-extraction-probes"],
}

# Risk-tier → red-team severity cap. Higher tier = allow harsher attacks.
_SEVERITY_CAP: dict[RiskTier, int] = {
    RiskTier.LOW: 2,
    RiskTier.MEDIUM: 3,
    RiskTier.HIGH: 4,
    RiskTier.CRITICAL: 5,
}

# Risk-tier → dataset row cap. Higher tier = more evidence required.
_DATASET_LIMIT: dict[RiskTier, int | None] = {
    RiskTier.LOW: 10,
    RiskTier.MEDIUM: 25,
    RiskTier.HIGH: 50,
    RiskTier.CRITICAL: None,  # full dataset
}

# These policies apply to every built-in industry preset. Domain-specific
# additions stay explicit here until first-class policy-pack loading lands.
_COMMON_BUILTIN_POLICY_FILENAMES = (
    "eu_ai_act_article_15.yaml",
    "fairness_baseline.yaml",
)
_HEALTHCARE_POLICY_FILENAMES = ("healthcare_hipaa.yaml",)


def _resolve_policies_dir(policies_dir: str | Path | None) -> str | None:
    if policies_dir is not None:
        return str(policies_dir)
    return str(_builtin_policy_directory())


def _builtin_policy_directory() -> Path:
    """Return the packaged directory containing built-in policy files."""
    return Path(__file__).resolve().parents[1] / "policies" / "examples"


def _builtin_policy_files(industry: Industry) -> list[Path]:
    """Return the built-in policy files selected for one industry preset.

    Built-ins are fail-closed: a missing directory, an unclassified YAML file,
    or a duplicate selection is a packaging/registry error rather than a reason
    to silently run a reduced policy set.
    """
    directory = _builtin_policy_directory()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Built-in policy directory is missing: {directory}. "
            "Pass policies_dir explicitly to use a user-supplied policy directory."
        )

    filenames = list(_COMMON_BUILTIN_POLICY_FILENAMES)
    if industry is Industry.HEALTHCARE:
        filenames.extend(_HEALTHCARE_POLICY_FILENAMES)

    if len(filenames) != len(set(filenames)):
        raise ValueError("Built-in policy registry selects a filename more than once.")

    classified = set(_COMMON_BUILTIN_POLICY_FILENAMES) | set(
        _HEALTHCARE_POLICY_FILENAMES
    )
    available = {path.name for path in directory.glob("*.yaml")}
    unclassified = sorted(available - classified)
    if unclassified:
        raise ValueError(
            "Built-in policy registry does not classify: " + ", ".join(unclassified)
        )

    missing = sorted(classified - available)
    if missing:
        raise FileNotFoundError(
            "Built-in policy registry references missing file(s): " + ", ".join(missing)
        )
    return [directory / filename for filename in filenames]


def _effective_risk_tier(
    profile: ApplicationProfile,
    rationale: list[str],
) -> RiskTier:
    """Escalate the self-declared risk tier based on trait evidence.

    Rule: PHI, biometric, credit, or legal data auto-escalate to at least
    HIGH. Autonomous action or handling minors auto-escalates to HIGH. This
    stops an app team from self-declaring LOW on a high-risk surface.
    """
    declared = profile.risk_tier
    escalate_to_high = False
    reasons: list[str] = []

    for flag in ("phi", "biometric", "credit", "legal", "minors"):
        if flag in profile.data_types:
            escalate_to_high = True
            reasons.append(flag)
    if "autonomous_action" in profile.capabilities:
        escalate_to_high = True
        reasons.append("autonomous_action")

    if escalate_to_high and declared in (RiskTier.LOW, RiskTier.MEDIUM):
        rationale.append(
            f"Escalated risk tier from {declared.value} → high because the "
            f"declared traits {reasons} are inherently high-risk."
        )
        return RiskTier.HIGH
    return declared


def scope_assessor(
    profile: ApplicationProfile,
    model: BaseModel,
    policies_dir: str | Path | None = None,
) -> tuple[Assessor, ScopingDecision]:
    """Build a ``Assessor`` configured for this app, with a paper-trail.

    The ``model`` is passed separately because the profile only carries a
    *reference* (``model_ref`` is a string); loading/constructing the model
    is the caller's job. The UI and CLI both do this.
    """
    rationale: list[str] = []

    preset = profile.industry.value
    rationale.append(f"Preset = `{preset}` (from industry={profile.industry.value}).")

    effective_tier = _effective_risk_tier(profile, rationale)

    datasets = list(profile.dataset_overrides) if profile.dataset_overrides else []
    if datasets:
        rationale.append(f"Datasets overridden by submitter: {datasets}.")
    elif profile.allow_sample_datasets:
        datasets = list(DEMO_EXAMPLE_BUNDLES.get(profile.industry.value, []))
        if not datasets:
            raise ValueError(
                f"No demo datasets are configured for industry "
                f"`{profile.industry.value}`. Provide dataset_overrides instead."
            )
        rationale.append(f"Sample/demo datasets explicitly enabled: {datasets}.")
    else:
        raise ValueError(
            "Dataset selection is required for assessments. Provide "
            "dataset_overrides for a real assessment, or enable sample datasets "
            "for a demo run."
        )
    for tag in profile.data_types:
        if profile.allow_sample_datasets:
            for ds in _SAMPLE_DATA_TYPE_EXTRA_DATASETS.get(tag, []):
                if ds not in datasets:
                    datasets.append(ds)
                    rationale.append(
                        f"Added sample dataset `{ds}` because data_types includes `{tag}`."
                    )
        elif tag in _SAMPLE_DATA_TYPE_EXTRA_DATASETS:
            rationale.append(
                f"Sample dataset(s) for data_types `{tag}` were not auto-added; "
                "explicit datasets are required for this assessment."
            )

    severity_cap = _SEVERITY_CAP[effective_tier]
    rationale.append(
        f"Red-team max severity = {severity_cap} "
        f"(risk tier = {effective_tier.value})."
    )

    strict_error_budget = (
        profile.industry is not Industry.GENERAL
        or effective_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
    )
    redteam_max_error_rate = 0.0 if strict_error_budget else 0.10

    # Public-facing deployments include adversarial users by default; bump
    # the severity cap one rung (clamped at 5) so we cover at least the
    # next attack tier the in-tree catalog defines. ``external`` (known
    # customers behind auth) stays at the tier-derived cap.
    if profile.deployment_context == DeploymentContext.PUBLIC and severity_cap < 5:
        bumped = severity_cap + 1
        rationale.append(
            f"Red-team max severity bumped {severity_cap} → {bumped} because "
            "deployment_context = `public` (unauthenticated / adversarial by default)."
        )
        severity_cap = bumped

    # Public deployment always runs redteam; internal low-risk can skip.
    run_redteam = True
    if (
        profile.deployment_context == DeploymentContext.INTERNAL
        and effective_tier == RiskTier.LOW
    ):
        run_redteam = False
        rationale.append(
            "Red-team skipped: internal deployment + low risk tier. "
            "Re-enable manually for belt-and-suspenders coverage."
        )
    if run_redteam:
        rationale.append(
            f"Red-team execution-error budget = {redteam_max_error_rate:.1%} "
            f"(industry = {profile.industry.value}, risk tier = {effective_tier.value})."
        )
    else:
        rationale.append(
            "Red-team execution-error budget not applied because red-team is off."
        )

    dataset_limit = _DATASET_LIMIT[effective_tier]
    rationale.append(
        f"Dataset row cap = {dataset_limit if dataset_limit else 'full dataset'} "
        f"(risk tier = {effective_tier.value})."
    )

    resolved_policies = _resolve_policies_dir(policies_dir)
    policies_engine: PolicyEngine | None = None
    if policies_dir is not None:
        # A caller-selected directory is a complete override: its files are
        # loaded exactly as supplied, without mixing in or filtering built-ins.
        if resolved_policies:
            rationale.append(
                f"Policies loaded from user-supplied `{resolved_policies}`."
            )
    else:
        policy_files = _builtin_policy_files(profile.industry)
        policies_engine = PolicyEngine([load_policy_set(path) for path in policy_files])
        rationale.append(
            "Built-in policy files selected: "
            + ", ".join(f"`{path.name}`" for path in policy_files)
            + "."
        )

    if profile.weave_project:
        rationale.append(
            f"Weave observability enabled → traces pushed to "
            f"`{profile.weave_entity + '/' if profile.weave_entity else ''}"
            f"{profile.weave_project}`."
        )

    assessor = Assessor(
        model=model,
        preset=preset,
        datasets=datasets,
        policies_dir=resolved_policies if policies_dir is not None else None,
        policies_engine=policies_engine,
        run_redteam=run_redteam,
        redteam_max_severity=severity_cap,
        redteam_max_error_rate=redteam_max_error_rate,
        extra_redteam_sources=list(profile.extra_redteam_sources),
        dataset_limit=dataset_limit,
        weave_project=profile.weave_project,
        weave_entity=profile.weave_entity,
    )

    decision = ScopingDecision(
        preset=preset,
        datasets=datasets,
        run_redteam=run_redteam,
        redteam_max_severity=severity_cap,
        dataset_limit=dataset_limit,
        policies_dir=resolved_policies,
        weave_project=profile.weave_project,
        weave_entity=profile.weave_entity,
        effective_risk_tier=effective_tier,
        redteam_max_error_rate=redteam_max_error_rate,
        rationale=rationale,
    )
    return assessor, decision
