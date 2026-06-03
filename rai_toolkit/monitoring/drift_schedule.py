# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Continuous compliance / drift monitoring — planning helpers.

One-shot :class:`rai_toolkit.assessment.Assessor` runs produce a point-in-time
artifact. Enterprise buyers typically want **periodic reassessment** when
models, data, or regulations change. This module encodes conservative default
intervals and a small config object; it does **not** run schedulers or poll
Weave (that belongs in your job runner or Weave monitors).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftMonitorConfig:
    """User-tunable knobs for a monitoring / reassessment job."""

    preset: str = "healthcare"
    interval_days: int | None = None
    """If set, overrides :func:`recommended_reassessment_interval_days`."""
    alert_on_policy_regression: bool = True
    """When True, any new high/critical policy violation should page on-call."""
    compare_to_baseline_run_id: str | None = None
    """Optional prior assessment ``run_id`` for delta reporting."""


def recommended_reassessment_interval_days(preset: str) -> int:
    """Return a conservative default reassessment cadence in days.

    High-risk regulated presets get shorter intervals; general presets longer.
    These are **policy suggestions**, not legal advice.
    """
    p = (preset or "").lower()
    if p in ("healthcare", "financial_services", "government"):
        return 30
    if p in ("legal", "fintech"):
        return 30
    return 90
