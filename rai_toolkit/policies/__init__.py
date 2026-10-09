# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Policy-as-code engine.

Lets customers encode compliance rules as versioned, reviewable YAML files.
Policies are evaluated against ScorerResult objects produced by the RAI pipeline,
producing PolicyViolation records with severity, framework references, and
remediation hints.

Example::

    from rai_toolkit.policies import PolicyEngine

    engine = PolicyEngine.from_directory("policies/")
    violations = engine.evaluate(scorer_results)

    for v in violations:
        print(f"[{v.severity}] {v.policy_name}: {v.message}")
"""

from rai_toolkit.policies.schema import (
    Policy,
    PolicySet,
    PolicySeverity,
    PolicyTrigger,
    PolicyViolation,
)
from rai_toolkit.policies.engine import PolicyEngine
from rai_toolkit.policies.expectations import (
    evaluate_policy_expectations,
    has_policy_expectations,
)

__all__ = [
    "Policy",
    "PolicySet",
    "PolicySeverity",
    "PolicyTrigger",
    "PolicyViolation",
    "PolicyEngine",
    "evaluate_policy_expectations",
    "has_policy_expectations",
]
