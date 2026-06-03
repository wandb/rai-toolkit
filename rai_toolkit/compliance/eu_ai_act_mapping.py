# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""EU AI Act compliance mapping.

Maps EU AI Act requirements to RAI toolkit capabilities and MIT risk categories.
Full enforcement of high-risk AI requirements: August 2, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EUAIActRequirement:
    """An EU AI Act requirement with mapping to toolkit capabilities.

    Attributes:
        id: Requirement identifier.
        article: EU AI Act article reference.
        title: Short title.
        description: What the requirement mandates.
        risk_tier: Which AI Act risk tier this applies to.
        mit_category_ids: MIT risk categories that help address this requirement.
        rai_capabilities: How the RAI toolkit helps with compliance.
    """

    id: str
    article: str
    title: str
    description: str
    risk_tier: str  # "prohibited", "high_risk", "limited_risk", "minimal_risk"
    mit_category_ids: list[str] = field(default_factory=list)
    rai_capabilities: list[str] = field(default_factory=list)


EU_AI_ACT_REQUIREMENTS: dict[str, EUAIActRequirement] = {
    "EU-HR-1": EUAIActRequirement(
        id="EU-HR-1",
        article="Article 9",
        title="Risk Management System",
        description=(
            "High-risk AI systems shall have a risk management system established, "
            "implemented, documented, and maintained as a continuous iterative process."
        ),
        risk_tier="high_risk",
        mit_category_ids=[
            "MIT-1.1", "MIT-1.2", "MIT-2.1", "MIT-2.2",
            "MIT-3.1", "MIT-5.1", "MIT-7.1", "MIT-7.2",
        ],
        rai_capabilities=[
            "Compliance Mapping Engine provides systematic risk identification",
            "Evaluation pipeline quantifies risks with normalized scores",
            "Compliance reports document risk management activities",
        ],
    ),
    "EU-HR-2": EUAIActRequirement(
        id="EU-HR-2",
        article="Article 10",
        title="Data and Data Governance",
        description=(
            "Training, validation, and testing datasets shall be subject to "
            "appropriate data governance and management practices."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-1.1", "MIT-1.3", "MIT-2.1"],
        rai_capabilities=[
            "Dataset versioning and publishing",
            "Bias evaluation on training/test data",
            "PII detection in datasets",
        ],
    ),
    "EU-HR-3": EUAIActRequirement(
        id="EU-HR-3",
        article="Article 11",
        title="Technical Documentation",
        description=(
            "Technical documentation shall be drawn up before the high-risk AI "
            "system is placed on the market and shall be kept up to date."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-7.2", "MIT-7.3"],
        rai_capabilities=[
            "Evaluation traces provide technical audit trail",
            "Versioned prompts and model configs document system behavior",
            "Compliance reports generate structured documentation",
        ],
    ),
    "EU-HR-4": EUAIActRequirement(
        id="EU-HR-4",
        article="Article 12",
        title="Record-Keeping",
        description=(
            "High-risk AI systems shall technically allow for automatic recording "
            "of events (logs) over their lifetime."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-7.1", "MIT-7.2"],
        rai_capabilities=[
            "Traces capture every model interaction with full I/O",
            "Production monitoring logs scorer results continuously",
            "Feedback/annotations create human review records",
        ],
    ),
    "EU-HR-5": EUAIActRequirement(
        id="EU-HR-5",
        article="Article 13",
        title="Transparency and Information",
        description=(
            "High-risk AI systems shall be designed and developed to ensure "
            "their operation is sufficiently transparent."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-5.1", "MIT-7.2"],
        rai_capabilities=[
            "Explainability scorer evaluates reasoning clarity",
            "Transparency scorer checks limitation disclosure",
            "Full trace visibility into model decision-making",
        ],
    ),
    "EU-HR-6": EUAIActRequirement(
        id="EU-HR-6",
        article="Article 14",
        title="Human Oversight",
        description=(
            "High-risk AI systems shall be designed to be effectively overseen "
            "by natural persons during their period of use."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-5.1", "MIT-5.2"],
        rai_capabilities=[
            "Human-in-the-loop annotation workflows",
            "Feedback collection on model outputs",
            "Guardrails enable human override of AI decisions",
        ],
    ),
    "EU-HR-7": EUAIActRequirement(
        id="EU-HR-7",
        article="Article 15",
        title="Accuracy, Robustness, and Cybersecurity",
        description=(
            "High-risk AI systems shall be designed to achieve appropriate levels "
            "of accuracy, robustness, and cybersecurity."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-2.2", "MIT-3.1", "MIT-7.1"],
        rai_capabilities=[
            "Factuality and hallucination scoring measures accuracy",
            "Security scorer detects adversarial vulnerabilities",
            "Comparative evaluations track robustness across versions",
        ],
    ),
    "EU-HR-8": EUAIActRequirement(
        id="EU-HR-8",
        article="Article 9(7)",
        title="Bias Testing and Mitigation",
        description=(
            "Testing shall include examination for possible biases that may "
            "affect health and safety of persons or lead to discrimination."
        ),
        risk_tier="high_risk",
        mit_category_ids=["MIT-1.1", "MIT-1.3"],
        rai_capabilities=[
            "Fairness scorer evaluates demographic bias",
            "Bias benchmarks test across protected characteristics",
            "Industry presets pre-select bias-related risk categories",
        ],
    ),
}


def get_requirement(requirement_id: str) -> EUAIActRequirement | None:
    """Look up an EU AI Act requirement."""
    return EU_AI_ACT_REQUIREMENTS.get(requirement_id)


def get_requirements_by_tier(risk_tier: str) -> list[EUAIActRequirement]:
    """Get all requirements for a given risk tier."""
    return [r for r in EU_AI_ACT_REQUIREMENTS.values() if r.risk_tier == risk_tier]


def get_mit_categories_for_eu_requirement(requirement_id: str) -> list[str]:
    """Get MIT risk categories that address an EU AI Act requirement."""
    req = EU_AI_ACT_REQUIREMENTS.get(requirement_id)
    return req.mit_category_ids if req else []


def get_all_required_mit_categories() -> list[str]:
    """Get all MIT risk categories needed for full EU AI Act high-risk compliance."""
    categories: set[str] = set()
    for req in EU_AI_ACT_REQUIREMENTS.values():
        categories.update(req.mit_category_ids)
    return sorted(categories)
