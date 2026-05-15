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


def estimate_assessment_run_cost(
    eval_results: Any,
    preset: str,
    judge_model: str | None = None,
) -> dict[str, Any] | None:
    """Upper-bound USD estimate; see integrations docstring for semantics."""
    items = getattr(eval_results, "items", None) or []
    if not items:
        return None
    model = judge_model or os.environ.get("RAI_JUDGE_MODEL", "gpt-4o-mini")
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    n = len(items)
    m = max(len(getattr(it, "scores", {}) or {}) for it in items)
    calls = n * max(m, 1)
    tin, tout = 800, 150
    usd = calls * (tin * pricing["prompt"] + tout * pricing["completion"])
    return {
        "preset": preset,
        "judge_model_for_pricing": model,
        "estimated_usd_upper_bound": round(usd, 4),
        "assumed_llm_calls_upper_bound": calls,
        "assumed_tokens_per_call": {"prompt": tin, "completion": tout},
        "note": (
            "Upper bound: assumes every scorer column is one paid LLM call; "
            "regex/programmatic scorers cost $0. For live spend, rely on Weave's "
            "built-in per-op cost tracking when weave.init() is on."
        ),
    }
