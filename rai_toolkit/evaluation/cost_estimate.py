# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Rough FinOps estimates for assessment runs (no Weave import)."""

from __future__ import annotations

import os
from typing import Any

# Per-token USD (aligned with integrations/weave_integration/costs.MODEL_PRICING)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 2.5e-6, "completion": 10.0e-6},
    "gpt-4o-mini": {"prompt": 0.15e-6, "completion": 0.6e-6},
    "gpt-4-turbo": {"prompt": 10.0e-6, "completion": 30.0e-6},
    "gpt-3.5-turbo": {"prompt": 0.5e-6, "completion": 1.5e-6},
    "claude-sonnet-4-6": {"prompt": 3.0e-6, "completion": 15.0e-6},
    "claude-haiku-4-5": {"prompt": 0.8e-6, "completion": 4.0e-6},
    "claude-opus-4-6": {"prompt": 15.0e-6, "completion": 75.0e-6},
}


_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_ASSUMED_PROMPT_TOKENS = 800
_ASSUMED_COMPLETION_TOKENS = 150
_AVAILABLE_NOTE = (
    "Upper bound: assumes every scorer column is one paid LLM call; "
    "regex/programmatic scorers cost $0. For live spend, rely on Weave's "
    "built-in per-op cost tracking when weave.init() is on."
)


def _resolve_judge_model(judge_model: str | None) -> str:
    """Explicit non-empty argument, then RAI_JUDGE_MODEL, then the default."""
    explicit = (judge_model or "").strip()
    if explicit:
        return explicit
    from_env = (os.environ.get("RAI_JUDGE_MODEL") or "").strip()
    if from_env:
        return from_env
    return _DEFAULT_JUDGE_MODEL


def estimate_assessment_run_cost(
    eval_results: Any,
    preset: str,
    judge_model: str | None = None,
) -> dict[str, Any] | None:
    """Upper-bound USD estimate; see integrations docstring for semantics.

    An empty evaluation returns None. A requested model absent from
    ``MODEL_PRICING`` is unavailable: no substitute price and not zero dollars.
    """
    items = getattr(eval_results, "items", None) or []
    if not items:
        return None
    model = _resolve_judge_model(judge_model)
    n = len(items)
    m = max(len(getattr(it, "scores", {}) or {}) for it in items)
    calls = n * max(m, 1)
    token_assumptions = {
        "prompt": _ASSUMED_PROMPT_TOKENS,
        "completion": _ASSUMED_COMPLETION_TOKENS,
    }
    common = {
        "preset": preset,
        "requested_judge_model": model,
        "assumed_llm_calls_upper_bound": calls,
        "assumed_tokens_per_call": token_assumptions,
    }
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return {
            **common,
            "status": "unavailable",
            "reason": "unknown_model_pricing",
            "judge_model_for_pricing": None,
            "estimated_usd_upper_bound": None,
            "note": (
                f"Pricing is unavailable for requested judge model {model!r}: "
                "it is not in the local price table. No substitute price was "
                "used, and this is not a $0 estimate."
            ),
        }
    usd = calls * (
        _ASSUMED_PROMPT_TOKENS * pricing["prompt"]
        + _ASSUMED_COMPLETION_TOKENS * pricing["completion"]
    )
    return {
        **common,
        "status": "available",
        "judge_model_for_pricing": model,
        "estimated_usd_upper_bound": round(usd, 4),
        "note": _AVAILABLE_NOTE,
    }
