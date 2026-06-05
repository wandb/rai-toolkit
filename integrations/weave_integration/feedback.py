# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Weave annotations for the human-in-the-loop layer.

When a reviewer pins a chat turn as a ``ManualFinding`` or signs off on a
submission, those actions belong on the *same* Weave trace that produced
the assessment. That's what makes the audit trail complete:
auto-eval traces + reviewer judgments live side by side.

Also exposes :func:`probe_thread`, a no-op-safe context manager for
grouping reviewer-probe model calls into one Weave conversation thread
instead of one top-level trace per turn.

All helpers are best-effort: they no-op silently when Weave isn't
initialized or when the underlying API isn't reachable. The reviewer
flow must never break because of trace annotation.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def probe_thread(thread_id: str | None) -> Iterator[None]:
    """Group probe model calls into a single Weave thread.

    Streamlit reruns the page on every chat turn, so each
    ``model.predict(...)`` is its own top-level Weave trace. Calling
    ``predict`` inside this context manager tags it with a stable
    ``thread_id``, and Weave's UI renders the turns as one conversation
    thread (Threads tab), which is what reviewers actually want when
    probing the model.

    No-op when Weave isn't initialized or ``thread_id`` is falsy, so
    callers can use this unconditionally without a guard.
    """
    if not thread_id:
        yield
        return
    try:
        import weave

        with weave.thread(thread_id):
            yield
    except Exception as e:
        logger.debug("probe_thread(%s) skipped: %s", thread_id, e)
        yield


def _get_call(call_id: str) -> Any | None:
    """Look up a Weave ``Call`` by ID. Returns ``None`` on any failure.

    We're attaching feedback *after* the cert op has finished, so
    ``weave.require_current_call`` won't help. Pull the live client
    directly and call ``get_call(id)`` on it.
    """
    if not call_id:
        return None
    try:
        from weave.trace.context.weave_client_context import (
            require_weave_client,
        )

        client = require_weave_client()
        return client.get_call(call_id)
    except Exception as e:
        logger.debug("get_call(%s) failed: %s", call_id, e)
        return None


def attach_manual_finding(
    call_id: str | None,
    *,
    user_input: str,
    model_output: str,
    severity: str,
    note: str,
    pinned_by: str,
) -> str | None:
    """Attach a pinned chat turn to the cert trace as Weave feedback.

    Adds two pieces of feedback for ergonomic UI rendering, mirroring
    :func:`attach_reviewer_decision`:
      * a ``rai.manual_finding`` payload with full details, and
      * a severity-keyed reaction (ℹ️ info, 🟢 low, 🟡 medium, 🟠 high,
        🔴 critical) so the Weave call list surfaces pinned findings at
        a glance. Same affordance the reviewer approve/reject reactions
        give for the final decision.

    Returns the feedback ID for the structured payload (the reaction
    failure is non-fatal). ``None`` when Weave isn't reachable.
    """
    if not call_id:
        return None
    call = _get_call(call_id)
    if call is None:
        return None

    feedback_id: str | None = None
    try:
        feedback_id = call.feedback.add(
            "rai.manual_finding",
            payload={
                "severity": severity,
                "note": note,
                "user_input": user_input,
                "model_output": model_output,
                "pinned_by": pinned_by,
            },
            creator=pinned_by or None,
        )
    except Exception as e:
        logger.debug("attach_manual_finding payload failed: %s", e)

    emoji = {
        "info": "ℹ️",
        "low": "🟢",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴",
    }.get((severity or "").lower())
    if emoji is not None:
        try:
            call.feedback.add_reaction(emoji, creator=pinned_by or None)
        except Exception as e:
            logger.debug("attach_manual_finding reaction failed: %s", e)

    return feedback_id


def attach_reviewer_decision(
    call_id: str | None,
    *,
    decision: str,
    reviewer: str,
    notes: str | None,
    auto_recommendation: str | None = None,
) -> str | None:
    """Record the human approve/request_changes/reject on the cert trace.

    Adds two pieces of feedback for ergonomic UI rendering:
      * a ``rai.reviewer_decision`` payload with full details, and
      * a reaction (✅ approve, 📝 request_changes, ❌ reject) so the
        Weave call list shows the decision at a glance.

    Returns the feedback ID for the structured payload (the reaction
    failure is non-fatal). ``None`` when Weave isn't reachable.
    """
    if not call_id:
        return None
    call = _get_call(call_id)
    if call is None:
        return None

    feedback_id: str | None = None
    try:
        feedback_id = call.feedback.add(
            "rai.reviewer_decision",
            payload={
                "decision": decision,
                "reviewer": reviewer,
                "notes": notes or "",
                "auto_recommendation": auto_recommendation,
            },
            creator=reviewer or None,
        )
    except Exception as e:
        logger.debug("attach_reviewer_decision payload failed: %s", e)

    emoji = {"approve": "✅", "request_changes": "📝", "reject": "❌"}.get(
        decision.lower(), None
    )
    if emoji is not None:
        try:
            call.feedback.add_reaction(emoji, creator=reviewer or None)
        except Exception as e:
            logger.debug("attach_reviewer_decision reaction failed: %s", e)

    return feedback_id
