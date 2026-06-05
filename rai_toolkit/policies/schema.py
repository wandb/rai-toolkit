# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Policy-as-code schema.

Defines the data model for YAML policy files. Pydantic validates each loaded
policy and gives actionable error messages when customer-authored policies are
malformed.
"""

from __future__ import annotations

import enum
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PolicySeverity(str, enum.Enum):
    """Severity levels. Consumers can decide blocking vs. advisory behavior."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        return {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]


class PolicyTrigger(BaseModel):
    """Conditions under which the policy fires.

    At least one trigger field must be populated. Multiple fields are combined
    with logical AND (all conditions must hold).
    """

    scorer_failed: str | None = Field(
        default=None,
        description="Policy fires when a scorer with this name produces passed=False.",
    )
    category: str | None = Field(
        default=None,
        description="Policy fires on scorer results tagged with this risk category.",
    )
    score_below: float | None = Field(
        default=None,
        description="Policy fires when score < this threshold.",
        ge=0.0,
        le=1.0,
    )
    score_above: float | None = Field(
        default=None,
        description="Policy fires when score > this threshold.",
        ge=0.0,
        le=1.0,
    )
    output_contains: list[str] = Field(
        default_factory=list,
        description="Policy fires when the model output contains any of these substrings (case-insensitive).",
    )
    output_missing: list[str] = Field(
        default_factory=list,
        description="Policy fires when the model output is missing ALL of these substrings.",
    )
    output_matches: str | None = Field(
        default=None,
        description="Policy fires when the model output matches this regex.",
    )
    input_contains: list[str] = Field(
        default_factory=list,
        description=(
            "Policy fires only when the user input contains any of these "
            "substrings (case-insensitive). Use to scope a policy to a "
            "particular query population, e.g. only check emergency-"
            "escalation policies on queries that mention emergency-indicating "
            "symptoms."
        ),
    )
    input_missing: list[str] = Field(
        default_factory=list,
        description=(
            "Policy fires only when the user input is missing ALL of these "
            "substrings (case-insensitive)."
        ),
    )
    input_matches: str | None = Field(
        default=None,
        description="Policy fires only when the user input matches this regex.",
    )

    @field_validator("output_matches", "input_matches")
    @classmethod
    def _validate_regex(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        return v

    def is_empty(self) -> bool:
        return not any(
            [
                self.scorer_failed,
                self.category,
                self.score_below is not None,
                self.score_above is not None,
                self.output_contains,
                self.output_missing,
                self.output_matches,
                self.input_contains,
                self.input_missing,
                self.input_matches,
            ]
        )


class Policy(BaseModel):
    """A single policy rule.

    Policies are the atomic unit of customer-defined compliance logic. Each
    policy encodes one expectation and, when violated, emits a `PolicyViolation`
    that flows into the assessment report.
    """

    name: str = Field(description="Unique policy identifier (kebab-case recommended).")
    description: str = Field(description="Human-readable explanation of what this policy protects against.")
    severity: PolicySeverity = PolicySeverity.MEDIUM
    trigger: PolicyTrigger
    frameworks: list[str] = Field(
        default_factory=list,
        description="Framework references, e.g. ['EU-AI-Act-Art-15', 'HIPAA-164.312', 'NIST-MEASURE-2.7'].",
    )
    remediation: str | None = Field(
        default=None,
        description="Suggested fix shown in reports when this policy is violated.",
    )
    tags: list[str] = Field(default_factory=list, description="Free-form tags for filtering.")
    enabled: bool = True

    @field_validator("trigger")
    @classmethod
    def _validate_trigger_not_empty(cls, v: PolicyTrigger) -> PolicyTrigger:
        if v.is_empty():
            raise ValueError("Policy trigger must specify at least one condition")
        return v

    @field_validator("name")
    @classmethod
    def _validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9\-_]*$", v):
            raise ValueError(
                "Policy name must be lowercase alphanumeric with hyphens or underscores"
            )
        return v


class PolicySet(BaseModel):
    """A collection of policies, typically loaded from one YAML file.

    Represents a logical grouping, e.g. "healthcare policies" or
    "EU AI Act Article 15 policies."
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    policies: list[Policy]

    @field_validator("policies")
    @classmethod
    def _unique_names(cls, v: list[Policy]) -> list[Policy]:
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate policy names in set: {dupes}")
        return v


class PolicyViolation(BaseModel):
    """A single violation emitted when a policy's trigger conditions match.

    Violations flow into the assessment report. One scorer result may produce
    multiple violations if it matches multiple policies.
    """

    policy_name: str
    severity: PolicySeverity
    message: str
    frameworks: list[str] = Field(default_factory=list)
    remediation: str | None = None
    scorer_name: str | None = None
    category: str | None = None
    score: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    def format(self) -> str:
        """Render a single-line human summary."""
        prefix = f"[{self.severity.value.upper()}]"
        frameworks = f" ({', '.join(self.frameworks)})" if self.frameworks else ""
        return f"{prefix} {self.policy_name}{frameworks}: {self.message}"
