# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""W&B run lifecycle helpers for the Streamlit demo.

Bridges a synchronous "do work in Streamlit" block to a single W&B run so
every Weave trace produced inside the block is searchable in the W&B UI by
the same ``run_id`` / ``run_url`` that owns the run page.

The bridge has two halves:

1. ``wandb.init(project, entity, ...)`` creates the run.
2. ``weave.attributes({"wandb": {...}})`` tags every Weave call recorded
   inside the context with the run's id / name / url / path.

Both halves are best-effort: if ``wandb`` or ``weave`` isn't importable, or
if the call fails (no api key, network blip, …), the context manager still
yields ``None`` so callers can keep doing the actual work without the demo
falling over on the W&B side.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _gate_state(passed: bool | None) -> str:
    if passed is None:
        return "N/A"
    return "PASS" if passed else "FAIL"


@contextlib.contextmanager
def wandb_run_context(
    project: str | None,
    entity: str | None = None,
    *,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    job_type: str = "rai-assessment",
) -> Iterator[Any]:
    """Open a W&B run and tag every Weave call inside the block with its id.

    Parameters mirror :func:`wandb.init` (``project``, ``entity``, ``name``,
    ``config``, ``tags``, ``notes``, ``job_type``). The yielded value is
    the live ``wandb.Run`` (or ``None`` when wandb isn't reachable).

    All Weave calls created inside this block are decorated with a
    ``wandb`` attribute group::

        {
          "run_id":   "<short id>",
          "run_name": "<display name>",
          "run_url":  "https://wandb.ai/<entity>/<project>/runs/<id>",
          "run_path": "<entity>/<project>/<id>",
          "project":  "<project>",
          "entity":   "<entity>",
        }

    The run is closed with ``wandb.finish()`` on exit, success or exception;
    that satisfies the "at the end terminate the W&B run" requirement
    even when the assessment pipeline raises.
    """
    if not project:
        # Caller didn't configure W&B / Weave at all. Degrade silently so
        # this helper can be wrapped around any assessment run without the
        # caller having to gate on it.
        yield None
        return

    try:
        import wandb
    except ImportError:
        logger.warning(
            "wandb not installed; skipping W&B run. "
            "Install with `pip install -e \".[weave]\"` to enable."
        )
        yield None
        return

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            tags=tags or None,
            notes=notes,
            job_type=job_type,
            reinit=True,
        )
    except Exception as e:
        logger.warning("wandb.init(%s/%s) failed: %s. Continuing without W&B run.",
                       entity, project, e)
        yield None
        return

    attrs: dict[str, Any] = {
        "wandb": {
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "run_url": getattr(run, "url", None),
            "run_path": "/".join(getattr(run, "path", []) or []) or None,
            "project": getattr(run, "project", project),
            "entity": getattr(run, "entity", entity),
        }
    }

    weave_attrs_cm: Any = contextlib.nullcontext()
    try:
        import weave

        weave_attrs_cm = weave.attributes(attrs)
    except Exception as e:
        # ``weave.attributes`` is a best-effort tag; its absence shouldn't
        # break the run. The wandb run still completes; traces just won't
        # carry the run_id in their attribute panel.
        logger.debug("weave.attributes(%s) skipped: %s", attrs, e)

    try:
        with weave_attrs_cm:
            yield run
    finally:
        try:
            wandb.finish()
        except Exception as e:
            logger.debug("wandb.finish() failed: %s", e)


def summarize_assessment_run(
    run: Any,
    *,
    result: Any,
    submission_id: str,
) -> None:
    """Mirror an :class:`AssessmentResult` onto the active W&B run.

    Logs the headline gates (verdict, eval score, red-team severity gate,
    red-team error budget, policy violations, duration) as ``wandb.summary``
    keys so the run page shows the assessment outcome at a glance, not just an
    empty stub.

    No-op when ``run`` is ``None`` (W&B not reachable) or when ``result``
    is ``None`` (assessment failed before producing a result).
    """
    if run is None or result is None:
        return
    try:
        import wandb
    except ImportError:
        return
    try:
        redteam_report = getattr(result, "redteam_summary", None)
        redteam_summary = redteam_report or {}
        summary: dict[str, Any] = {
            "submission_id": submission_id,
            "verdict": "PASS" if getattr(result, "overall_passed", False) else "FAIL",
            "evaluation_score": float(getattr(result, "evaluation_overall_score", 0.0) or 0.0),
            "composite_score": float(getattr(result, "overall_score", 0.0) or 0.0),
            "redteam_severity_gate": (
                "N/A"
                if redteam_report is None
                else _gate_state(getattr(result, "redteam_severity_gate_passed", None))
            ),
            "redteam_severity_threshold":
                getattr(result, "redteam_severity_gate_threshold", None),
            "redteam_error_budget_gate": (
                "N/A"
                if redteam_report is None
                else _gate_state(getattr(result, "redteam_error_budget_passed", None))
            ),
            "redteam_error_budget": getattr(result, "redteam_error_budget", None),
            "redteam_error_rate": redteam_summary.get("error_rate"),
            "redteam_total_assessed": redteam_summary.get("total_assessed"),
            "redteam_total_errors": redteam_summary.get("total_errors"),
            "policy_violations": len(getattr(result, "policy_violations", []) or []),
            "duration_seconds": float(getattr(result, "duration_seconds", 0.0) or 0.0),
        }
        # ``wandb.summary`` is the canonical "final value of this run" surface
        # and is what lights up the leaderboard column on the project page.
        wandb.summary.update({k: v for k, v in summary.items() if v is not None})

        # Also emit one ``wandb.log`` row so the run history (Charts tab) has
        # a single point per assessment, handy when users plot
        # evaluation_score across resubmissions of the same app.
        wandb.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    except Exception as e:
        logger.debug("summarize_assessment_run failed: %s", e)
