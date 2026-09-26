# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Deterministic scoring for recorded agent tool invocations."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from rai_toolkit.scorers.base import BaseScorer, ScorerResult


def _json_path(path: str, key: str) -> str:
    """Append a dictionary key to a stable, JSON-safe diagnostic path."""
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=True)}]"


def _json_issue(
    value: Any,
    path: str,
    seen: set[int] | None = None,
) -> tuple[str, str] | None:
    """Return the first strict-JSON validation location and safe reason."""
    if value is None or isinstance(value, (str, bool)):
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else (path, "non_finite_number")
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return path, "cyclic_reference"
    if isinstance(value, list):
        seen.add(identity)
        try:
            for index, item in enumerate(value):
                issue = _json_issue(item, f"{path}[{index}]", seen)
                if issue:
                    return issue
        finally:
            seen.remove(identity)
        return None
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return path, "non_string_key"
        seen.add(identity)
        try:
            for key, item in value.items():
                issue = _json_issue(item, _json_path(path, key), seen)
                if issue:
                    return issue
        finally:
            seen.remove(identity)
        return None
    return path, "unsupported_type"


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON values while keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            left.keys() == right.keys()
            and all(_same_json(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def _unassessed(
    category: str,
    name: str,
    reason: str,
    explanation: str,
    **diagnostics: Any,
) -> ScorerResult:
    return ScorerResult(
        score=0.0,
        passed=False,
        category=category,
        explanation=explanation,
        details={"scorer_name": name, "skipped": reason, **diagnostics},
        assessed=False,
    )


def _invalid(
    category: str,
    name: str,
    skipped: str,
    explanation: str,
    location: str,
    reason: str,
) -> ScorerResult:
    """Return an unassessed result without copying the rejected raw value."""
    return _unassessed(
        category,
        name,
        skipped,
        explanation,
        validation_location=location,
        validation_reason=reason,
    )


class ToolCallAccuracyScorer(BaseScorer):
    """Score expected agent tool calls against a complete invocation trace.

    The scorer intentionally consumes structured events supplied by the caller;
    it never attempts to infer calls from response text or tool-result prose.

    Example: