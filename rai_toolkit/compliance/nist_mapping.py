# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST AI Risk Management Framework mapping.

Maps NIST AI RMF 1.0 functions and categories to RAI toolkit capabilities
and Weave features (when used with the Weave integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rai_toolkit.compliance.frameworks import Framework, RiskCategory, RiskSeverity


@dataclass
class NISTFunction:
    """A NIST AI RMF function with its subfunctions.

    Attributes:
        id: Function identifier (e.g. "GOVERN", "MAP", "MEASURE", "MANAGE").
        name: Full function name.
        description: What this function covers.
        subfunctions: Mapping of subfunction IDs to descriptions and actions.
        rai_capabilities: How the RAI toolkit addresses this function.
    """

    id: str
    name: str
    description: str
    subfunctions: dict[str, dict[str, str]] = field(default_factory=dict)
    rai_capabilities: list[str] = field(default_factory=list)


NIST_FUNCTIONS: dict[str, NISTFunction] = {
    "GOVERN": NISTFunction(
        id="GOVERN",
        name="Govern",
        description="Cultivate and implement a culture of risk management",
        subfunctions={
            "GV-1": {
                "description": "Policies for AI risk management are established",
                "rai_action": "Publish governance policies as versioned objects",
            },
            "GV-2": {
                "description": "Roles and responsibilities are defined",
                "rai_action": "Define compliance profiles with ownership metadata",
            },
            "GV-3": {
                "description": "Workforce diversity and AI expertise",
                "rai_action": "Multi-stakeholder feedback via annotation workflows",
            },
            "GV-4": {
                "description": "Organizational practices are in place",
                "rai_action": "Automated evaluation pipelines and monitoring",
            },
        },
        rai_capabilities=[
            "Compliance profiles define governance structure",
            "Versioned prompts track policy changes over time",
            "Dataset versioning maintains evaluation history",
        ],
    ),
    "MAP": NISTFunction(
        id="MAP",
        name="Map",
        description="Identify and categorize AI risks in context",
        subfunctions={
            "MAP-1": {
                "description": "Context is established and understood",
                "rai_action": "Industry presets auto-select relevant risk categories",
            },
            "MAP-2": {
                "description": "Categorization of AI risks",
                "rai_action": "MIT AI Risk Repository taxonomy with 23 subcategories",
            },
            "MAP-3": {
                "description": "AI capabilities and limitations identified",
                "rai_action": "Model evaluation against risk-specific test datasets",
            },
            "MAP-4": {
                "description": "Risks mapped to organizational impact",
                "rai_action": "Severity levels and compliance scoring per category",
            },
        },
        rai_capabilities=[
            "Compliance Mapping Engine maps frameworks to scorers",
            "MIT taxonomy provides comprehensive risk categorization",
            "Industry presets accelerate risk identification",
        ],
    ),
    "MEASURE": NISTFunction(
        id="MEASURE",
        name="Measure",
        description="Analyze, assess, and track identified risks",
        subfunctions={
            "MEASURE-1": {
                "description": "Appropriate methods for risk measurement",
                "rai_action": "Hybrid scoring: programmatic + LLM-as-a-Judge + built-in",
            },
            "MEASURE-2": {
                "description": "AI systems evaluated for trustworthiness",
                "rai_action": "Evaluation pipeline with compliance-mapped scorers",
            },
            "MEASURE-3": {
                "description": "Mechanisms for tracking risks over time",
                "rai_action": "Evaluation history and model comparison leaderboards",
            },
            "MEASURE-4": {
                "description": "Feedback incorporated into measurement",
                "rai_action": "Human annotation and feedback collection workflows",
            },
        },
        rai_capabilities=[
            "RAIEvaluationPipeline orchestrates compliance-aware evaluations",
            "Score normalization enables cross-scorer comparison",
            "Comparative evaluations across model versions",
        ],
    ),
    "MANAGE": NISTFunction(
        id="MANAGE",
        name="Manage",
        description="Allocate resources to mapped and measured risks",
        subfunctions={
            "MANAGE-1": {
                "description": "Risks are prioritized and acted upon",
                "rai_action": "Severity-based prioritization in compliance profiles",
            },
            "MANAGE-2": {
                "description": "Strategies to maximize benefits and minimize harms",
                "rai_action": "Runtime guardrails block harmful content in real-time",
            },
            "MANAGE-3": {
                "description": "Risks from third-party entities managed",
                "rai_action": "Model-agnostic evaluation works across any LLM provider",
            },
            "MANAGE-4": {
                "description": "Risk treatments are monitored",
                "rai_action": "Production monitoring with automated alerts",
            },
        },
        rai_capabilities=[
            "GuardedModel provides runtime input/output safety",
            "Production monitors track scorer results on live traffic",
            "Automated alerts via Slack/webhooks on threshold breaches",
        ],
    ),
}


# Map NIST functions to MIT risk categories for cross-framework navigation
NIST_TO_MIT_MAPPING: dict[str, list[str]] = {
    "GOVERN": [],  # Governance is process-level, not scorer-measurable
    "MAP": [
        "MIT-1.1", "MIT-1.2", "MIT-1.3",
        "MIT-2.1", "MIT-2.2",
        "MIT-3.1", "MIT-3.2",
        "MIT-4.1", "MIT-4.2", "MIT-4.3",
        "MIT-5.1", "MIT-5.2",
        "MIT-7.1", "MIT-7.2",
    ],
    "MEASURE": [
        "MIT-1.1", "MIT-1.2", "MIT-1.3",
        "MIT-2.1", "MIT-2.2",
        "MIT-3.1", "MIT-3.2",
        "MIT-5.1",
        "MIT-7.2",
    ],
    "MANAGE": [
        "MIT-1.2",  # Toxicity → guardrails
        "MIT-2.1",  # Privacy → PII guardrails
        "MIT-2.2",  # Security → injection guardrails
        "MIT-3.1",  # Hallucination → monitoring
    ],
}


def get_nist_function(function_id: str) -> NISTFunction | None:
    """Look up a NIST AI RMF function."""
    return NIST_FUNCTIONS.get(function_id)


def get_mit_categories_for_nist(function_id: str) -> list[str]:
    """Get MIT risk category IDs relevant to a NIST function."""
    return NIST_TO_MIT_MAPPING.get(function_id, [])
