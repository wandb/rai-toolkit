# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ApplicationProfile — the intake record for an app requesting RAI review.

This is the *input* the RAI team receives from an app team. Everything that
scopes the assessment run (which datasets, which scorers, which red-team
severity cap, which policies) is derived from this profile — see
``rai_toolkit.workflow.scoping``.
"""

from __future__ import annotations

import enum
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


class Industry(str, enum.Enum):
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    GOVERNMENT = "government"
    GENERAL = "general"


class DeploymentContext(str, enum.Enum):
    INTERNAL = "internal"     # employees only
    EXTERNAL = "external"     # known customers behind auth
    PUBLIC = "public"         # unauthenticated / general public


class RiskTier(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Data-sensitivity tags an app team can declare. Scoping reads these to
# decide which policies/scorers apply. Keep the list short and opinionated;
# an app team should be able to fill this out in under 30 seconds.
DATA_TYPE_CHOICES = (
    "pii",              # personally identifiable info
    "phi",              # protected health info (HIPAA)
    "financial",        # account numbers, transactions
    "minors",           # users under 18
    "biometric",        # face, voice, fingerprint
    "employment",       # hiring / HR decisions
    "credit",           # credit / lending decisions
    "legal",            # legal advice / court-adjacent
)

CAPABILITY_CHOICES = (
    "qa",               # Q&A / retrieval
    "summarization",
    "code_generation",
    "advice",           # medical / legal / financial advice
    "content_moderation",
    "decision_support", # influences a human decision
    "autonomous_action",# takes actions without human approval
)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    slug = _SLUG_RE.sub("-", s.lower()).strip("-")
    return slug or "app"


@dataclass
class ApplicationProfile:
    """Intake record describing the app being submitted for RAI review.

    ``app_id`` is the stable handle; derive it from ``name`` if not provided.
    ``risk_tier`` is self-declared by the submitter but the decision engine
    may override it upward based on declared data_types (e.g. any app
    handling ``phi`` is escalated to at least HIGH).
    """

    name: str
    description: str
    owner_team: str
    owner_email: str
    industry: Industry = Industry.GENERAL
    deployment_context: DeploymentContext = DeploymentContext.INTERNAL
    risk_tier: RiskTier = RiskTier.MEDIUM
    data_types: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    model_ref: str = ""        # e.g. "my_pkg:build_model" — used when adapter == "python_class"
    model_adapter: str = "python_class"  # "python_class" | "openai_compatible"
    model_adapter_args: dict[str, Any] = field(default_factory=dict)
    dataset_overrides: list[str] = field(default_factory=list)
    allow_sample_datasets: bool = False
    # Optional third-party red-team sources merged into the in-tree catalog
    # during assessment. Recognised values: ``"pyrit"``, ``"garak"``. Skipped
    # silently if the corresponding extra isn't installed.
    extra_redteam_sources: list[str] = field(default_factory=list)
    # Apply NeMo Guardrails as input/output rails on the model under
    # review during assessment. Requires the ``[nemo]`` extra. Wraps the
    # loaded model in a ``GuardedModel`` before assessing.
    enable_nemo_guardrails: bool = False
    weave_project: str | None = None
    weave_entity: str | None = None
    notes: str = ""
    submitted_by: str = ""
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    app_id: str = ""

    def __post_init__(self) -> None:
        if not self.app_id:
            self.app_id = _slugify(self.name)
        if isinstance(self.industry, str):
            self.industry = Industry(self.industry)
        if isinstance(self.deployment_context, str):
            self.deployment_context = DeploymentContext(self.deployment_context)
        if isinstance(self.risk_tier, str):
            self.risk_tier = RiskTier(self.risk_tier)
        unknown = [d for d in self.data_types if d not in DATA_TYPE_CHOICES]
        if unknown:
            raise ValueError(
                f"Unknown data_types: {unknown}. Allowed: {DATA_TYPE_CHOICES}"
            )
        unknown_caps = [c for c in self.capabilities if c not in CAPABILITY_CHOICES]
        if unknown_caps:
            raise ValueError(
                f"Unknown capabilities: {unknown_caps}. Allowed: {CAPABILITY_CHOICES}"
            )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["industry"] = self.industry.value
        out["deployment_context"] = self.deployment_context.value
        out["risk_tier"] = self.risk_tier.value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplicationProfile":
        return cls(**data)
