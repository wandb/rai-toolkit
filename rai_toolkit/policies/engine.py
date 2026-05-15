"""Policy engine — loads and evaluates policy sets against scorer results."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from rai_toolkit.policies.schema import (
    Policy,
    PolicySet,
    PolicyTrigger,
    PolicyViolation,
)
from rai_toolkit.scorers.base import ScorerResult

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Evaluates scorer results against a collection of policies.

    Policies are typically loaded from YAML files on disk. The engine is
    stateless after construction: call ``evaluate()`` with fresh scorer results
    as many times as you like.

    Example::

        engine = PolicyEngine.from_directory("policies/")
        violations = engine.evaluate(scorer_results, model_output="...")

        high_severity = [v for v in violations if v.severity.numeric >= 3]
    """

    def __init__(self, policy_sets: Sequence[PolicySet]) -> None:
        self.policy_sets: list[PolicySet] = list(policy_sets)
        self._all_policies: list[Policy] = [
            p for ps in self.policy_sets for p in ps.policies if p.enabled
        ]
        names = [p.name for p in self._all_policies]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            logger.warning("Duplicate policy names across sets: %s", dupes)

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyEngine":
        return cls([load_policy_set(path)])

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.yaml") -> "PolicyEngine":
        """Load every YAML file in a directory as a separate PolicySet."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"{directory} is not a directory")

        policy_sets: list[PolicySet] = []
        for f in sorted(directory.glob(pattern)):
            try:
                policy_sets.append(load_policy_set(f))
            except Exception as e:
                logger.error("Failed to load policy set %s: %s", f, e)
                raise

        if not policy_sets:
            logger.warning("No policy files matched %s in %s", pattern, directory)

        return cls(policy_sets)

    @property
    def policies(self) -> list[Policy]:
        return list(self._all_policies)

    def evaluate(
        self,
        scorer_results: Iterable[ScorerResult],
        model_output: str = "",
        model_input: str = "",
    ) -> list[PolicyViolation]:
        """Evaluate every policy against every scorer result.

        Args:
            scorer_results: Results produced by the RAI evaluation pipeline.
            model_output: The raw model output text. Needed for content-based
                triggers (``output_contains``, ``output_missing``, ``output_matches``).
            model_input: The user-supplied input text. Needed for input-scoped
                triggers (``input_contains``, ``input_missing``, ``input_matches``).
                Empty string disables input-scoped triggers (they fail to match,
                so policies that require an input condition simply do not fire).

        Returns:
            Violations, sorted by severity descending then policy name.
        """
        results = list(scorer_results)
        violations: list[PolicyViolation] = []

        for policy in self._all_policies:
            for sr in results:
                violation = _match_policy_against_result(
                    policy, sr, model_output, model_input
                )
                if violation is not None:
                    violations.append(violation)

            if not results:
                v = _match_content_only_policy(policy, model_output, model_input)
                if v is not None:
                    violations.append(v)

        violations.sort(key=lambda v: (-v.severity.numeric, v.policy_name))
        return violations

    def filter(self, framework: str | None = None, tag: str | None = None) -> "PolicyEngine":
        """Return a new engine containing only matching policies."""
        filtered_sets: list[PolicySet] = []
        for ps in self.policy_sets:
            filtered = [
                p
                for p in ps.policies
                if (framework is None or framework in p.frameworks)
                and (tag is None or tag in p.tags)
            ]
            if filtered:
                filtered_sets.append(
                    PolicySet(
                        name=ps.name,
                        description=ps.description,
                        version=ps.version,
                        policies=filtered,
                    )
                )
        return PolicyEngine(filtered_sets)


def load_policy_set(path: str | Path) -> PolicySet:
    """Load and validate a single YAML policy file."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load policy files. Install with `pip install pyyaml`."
        ) from e

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Policy file {path} must contain a YAML mapping at the top level")

    return PolicySet.model_validate(raw)


def _match_policy_against_result(
    policy: Policy, sr: ScorerResult, model_output: str, model_input: str = ""
) -> PolicyViolation | None:
    """Apply a policy to one scorer result. Returns a violation if triggered.

    Un-assessed scorer results (``assessed=False``) never trigger a violation.
    A scorer that could not evaluate a row produced no signal — counting that
    as a failure would be the same kind of synthetic-default credibility leak
    we are trying to avoid.
    """
    if not getattr(sr, "assessed", True):
        return None

    trigger = policy.trigger

    if trigger.scorer_failed and trigger.scorer_failed != getattr(sr, "name", None):
        scorer_name = _extract_scorer_name(sr)
        if scorer_name != trigger.scorer_failed:
            return None
        if sr.passed:
            return None

    if trigger.category and sr.category != trigger.category:
        return None

    if trigger.score_below is not None and sr.score >= trigger.score_below:
        return None

    if trigger.score_above is not None and sr.score <= trigger.score_above:
        return None

    if not _check_content_triggers(trigger, model_output, model_input):
        return None

    if (
        trigger.scorer_failed is None
        and trigger.category is None
        and trigger.score_below is None
        and trigger.score_above is None
    ):
        return None

    message = _build_violation_message(policy, sr)
    return PolicyViolation(
        policy_name=policy.name,
        severity=policy.severity,
        message=message,
        frameworks=policy.frameworks,
        remediation=policy.remediation,
        scorer_name=_extract_scorer_name(sr),
        category=sr.category,
        score=sr.score,
        evidence={"explanation": sr.explanation, "details": sr.details},
    )


def _match_content_only_policy(
    policy: Policy, model_output: str, model_input: str = ""
) -> PolicyViolation | None:
    """Apply a policy that only references content (no scorer conditions)."""
    trigger = policy.trigger
    scorer_conditions = any(
        [
            trigger.scorer_failed,
            trigger.category,
            trigger.score_below is not None,
            trigger.score_above is not None,
        ]
    )
    if scorer_conditions:
        return None

    if not _check_content_triggers(trigger, model_output, model_input):
        return None

    return PolicyViolation(
        policy_name=policy.name,
        severity=policy.severity,
        message=policy.description,
        frameworks=policy.frameworks,
        remediation=policy.remediation,
        evidence={"output_preview": model_output[:200]},
    )


def _check_content_triggers(
    trigger: PolicyTrigger, model_output: str, model_input: str = ""
) -> bool:
    """Return True iff all content-based triggers match.

    Substring lists (``*_contains`` / ``*_missing``) match case-insensitively
    against the raw text. Regex triggers (``*_matches``) match against the raw
    text and are case-sensitive unless the pattern includes an inline ``(?i)``
    flag — mirroring the existing convention.

    A trigger with no content conditions returns True (vacuously satisfied);
    scorer-only policies are filtered out earlier in the caller.
    """
    output_lower = (model_output or "").lower()
    input_lower = (model_input or "").lower()

    if trigger.output_contains:
        if not any(s.lower() in output_lower for s in trigger.output_contains):
            return False

    if trigger.output_missing:
        if any(s.lower() in output_lower for s in trigger.output_missing):
            return False

    if trigger.output_matches:
        if not re.search(trigger.output_matches, model_output or ""):
            return False

    if trigger.input_contains:
        if not any(s.lower() in input_lower for s in trigger.input_contains):
            return False

    if trigger.input_missing:
        if any(s.lower() in input_lower for s in trigger.input_missing):
            return False

    if trigger.input_matches:
        if not re.search(trigger.input_matches, model_input or ""):
            return False

    return True


def _build_violation_message(policy: Policy, sr: ScorerResult) -> str:
    base = policy.description
    detail = f" (score={sr.score:.2f}, category={sr.category})"
    return base + detail


def _extract_scorer_name(sr: ScorerResult) -> str | None:
    """ScorerResult doesn't carry the scorer's name natively; check details."""
    return sr.details.get("scorer_name") if sr.details else None
