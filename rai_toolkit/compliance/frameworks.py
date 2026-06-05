# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Data models for compliance frameworks, risk categories, and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Framework(Enum):
    """Supported compliance frameworks."""

    MIT_AI_RISK = "mit_ai_risk_repository"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"


class RiskSeverity(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskCategory:
    """A single risk category within a compliance framework.

    Attributes:
        id: Unique identifier (e.g. "MIT-1.1", "NIST-MEASURE-2.6").
        domain: Top-level domain name.
        subdomain: Specific risk subdomain.
        description: Detailed description of the risk.
        framework: Which compliance framework this belongs to.
        severity: Default severity level.
        scorer_ids: List of scorer class names that cover this category.
        tags: Optional tags for filtering (e.g. ["rag", "healthcare"]).
    """

    id: str
    domain: str
    subdomain: str
    description: str
    framework: Framework
    severity: RiskSeverity = RiskSeverity.MEDIUM
    scorer_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class ComplianceProfile:
    """A selected subset of risk categories for a specific assessment.

    Attributes:
        name: Profile name (e.g. "Healthcare RAG Assessment").
        framework: Primary compliance framework.
        categories: Selected risk categories to evaluate.
        industry: Optional industry context.
        metadata: Additional profile metadata.
    """

    name: str
    framework: Framework
    categories: list[RiskCategory]
    industry: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category_ids(self) -> list[str]:
        """List of selected category IDs."""
        return [c.id for c in self.categories]

    @property
    def domains(self) -> list[str]:
        """Unique domain names in this profile."""
        return list(dict.fromkeys(c.domain for c in self.categories))

    def get_categories_by_domain(self, domain: str) -> list[RiskCategory]:
        """Filter categories by domain name."""
        return [c for c in self.categories if c.domain == domain]

    def get_categories_by_severity(self, severity: RiskSeverity) -> list[RiskCategory]:
        """Filter categories by severity level."""
        return [c for c in self.categories if c.severity == severity]


# Industry presets: commonly selected risk categories per industry
INDUSTRY_PRESETS: dict[str, list[str]] = {
    "healthcare": [
        "MIT-1.1",   # Discrimination/bias: critical for patient equity
        "MIT-1.2",   # Toxic content
        "MIT-2.1",   # Privacy (PHI/PII): HIPAA
        "MIT-3.1",   # False/misleading info: clinical accuracy
        "MIT-3.2",   # Info ecosystem: medical misinformation
        "MIT-5.1",   # Overreliance: clinical decision support
        "MIT-7.1",   # System failures: safety-critical
        "MIT-7.2",   # Transparency: explainability for clinicians
    ],
    "financial_services": [
        "MIT-1.1",   # Discrimination: fair lending
        "MIT-1.3",   # Unequal performance: across demographics
        "MIT-2.1",   # Privacy: financial data
        "MIT-2.2",   # Security: fraud prevention
        "MIT-3.1",   # False info: financial advice accuracy
        "MIT-4.2",   # Fraud/manipulation
        "MIT-5.1",   # Overreliance: automated trading/decisions
        "MIT-7.2",   # Transparency: regulatory explainability
    ],
    "government": [
        "MIT-1.1",   # Discrimination: civil rights
        "MIT-1.3",   # Unequal performance
        "MIT-2.1",   # Privacy: citizen data
        "MIT-2.2",   # Security: national security
        "MIT-3.1",   # False info: public trust
        "MIT-5.1",   # Overreliance
        "MIT-5.2",   # Loss of human agency
        "MIT-7.2",   # Transparency: public accountability
    ],
    "general": [
        "MIT-1.1",   # Discrimination/bias
        "MIT-1.2",   # Toxic content
        "MIT-2.1",   # Privacy
        "MIT-3.1",   # False/misleading info
        "MIT-5.1",   # Overreliance
        "MIT-7.2",   # Transparency
    ],
}
