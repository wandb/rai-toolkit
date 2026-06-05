# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Responsible AI Toolkit: Production-grade RAI evaluation, compliance, and guardrails."""

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.models.base import BaseModel
from rai_toolkit.guardrails.base import BaseGuardrail, GuardrailResult
from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import (
    Framework,
    RiskCategory,
    ComplianceProfile,
)
from rai_toolkit.evaluation.pipeline import RAIEvaluationPipeline
from rai_toolkit.toolkit import RAIToolkit

from rai_toolkit.policies import PolicyEngine, Policy, PolicyViolation
from rai_toolkit.redteam import AttackRunner, RedTeamReport, ATTACK_CATALOG
from rai_toolkit.examples import ExampleDescriptor, ExampleRegistry
from rai_toolkit.assessment import Assessor, AssessmentResult

__version__ = "0.1.0"

__all__ = [
    "BaseScorer",
    "ScorerResult",
    "BaseModel",
    "BaseGuardrail",
    "GuardrailResult",
    "ComplianceMappingEngine",
    "Framework",
    "RiskCategory",
    "ComplianceProfile",
    "RAIEvaluationPipeline",
    "RAIToolkit",
    "PolicyEngine",
    "Policy",
    "PolicyViolation",
    "AttackRunner",
    "RedTeamReport",
    "ATTACK_CATALOG",
    "ExampleDescriptor",
    "ExampleRegistry",
    "Assessor",
    "AssessmentResult",
]
