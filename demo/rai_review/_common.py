# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Shared helpers for the RAI review Streamlit app.

Keeps the page files short and focused on layout; all workflow glue (model
loading, running a assessment, saving artifacts) lives here.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Make the repo root importable when the app is launched from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rai_toolkit.assessment import AssessmentResult
from rai_toolkit.workflow import (
    ApplicationProfile,
    ReviewRegistry,
    Submission,
    SubmissionStatus,
    auto_decide,
    new_submission_id,
    scope_assessor,
)

WORKSPACE = _REPO_ROOT / "rai_workspace"


@st.cache_resource
def get_registry() -> ReviewRegistry:
    return ReviewRegistry(WORKSPACE)


def load_model(model_ref: str) -> Any:
    """Import a ``BaseModel`` by dotted path or ``module:attr`` form."""
    if ":" in model_ref:
        module_path, attr = model_ref.split(":", 1)
    elif "." in model_ref:
        module_path, attr = model_ref.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid model ref `{model_ref}`. "
            "Use `package.module:ClassName` or `package.module.ClassName`."
        )
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    return obj() if callable(obj) else obj


def load_model_from_profile(profile: ApplicationProfile) -> Any:
    """Load the model the profile points at: Python class or OpenAI endpoint.

    The intake form lets the submitter pick the adapter; the chat panel
    and the assessment pipeline both call this so they probe the same
    thing. When the profile has ``enable_nemo_guardrails=True``, the
    loaded model is wrapped in :class:`GuardedModel` with NeMo input +
    output rails so every prediction (eval rows, red-team attacks,
    reviewer probes) flows through NeMo's checks and shows up in the
    trace tree as ``rai.guardrail.nemo.*``.
    """
    if profile.model_adapter == "openai_compatible":
        from rai_toolkit.models.openai_compatible import from_args
        args = dict(profile.model_adapter_args)
        if not args.get("model"):
            raise ValueError("OpenAI-compatible adapter requires a `model` field.")
        if "name" not in args:
            args["name"] = profile.app_id
        model = from_args(args)
    else:
        model = load_model(profile.model_ref)

    if getattr(profile, "enable_nemo_guardrails", False):
        try:
            from integrations.nemo_integration import NeMoGuardrail
            from rai_toolkit.guardrails import GuardedModel

            nemo = NeMoGuardrail()
            model = GuardedModel(
                model=model,
                input_guardrails=[nemo],
                output_guardrails=[nemo],
            )
        except Exception as e:
            # Don't fail the whole load if NeMo can't initialize (missing
            # extra, bad colang config, etc.); surface it loudly so the
            # reviewer sees the toggle didn't take effect.
            raise RuntimeError(
                f"Could not apply NeMo Guardrails (install with "
                f"`pip install \"rai-toolkit[nemo]\"`): {e}"
            ) from e

    return model


def run_assessment(
    profile: ApplicationProfile,
    dataset_limit_override: int | None = None,
) -> tuple[Submission, AssessmentResult | None, str | None]:
    """Run the scoped assessment pipeline for a submitted profile.

    Returns the submission (populated with scoping + result + auto-decision),
    the result, and an error message if the run failed.

    ``dataset_limit_override`` forces a lower row cap than the risk tier
    would otherwise demand, used by the Streamlit demo-mode switch so
    first-time runs finish in ~30s instead of several minutes.
    """
    registry = get_registry()
    submission_id = new_submission_id(profile)
    submission = Submission(
        submission_id=submission_id,
        profile=profile,
        status=SubmissionStatus.SUBMITTED,
    )
    submission.transition(SubmissionStatus.RUNNING, actor=profile.submitted_by or "system")

    try:
        model = load_model_from_profile(profile)
    except Exception as e:
        registry.save_submission(submission)
        adapter_label = (
            f"adapter `{profile.model_adapter}` (model={profile.model_adapter_args.get('model')})"
            if profile.model_adapter != "python_class"
            else f"`{profile.model_ref}`"
        )
        return submission, None, f"Could not load model {adapter_label}: {e}"

    assessor, scoping = scope_assessor(profile, model)
    if dataset_limit_override is not None:
        assessor.dataset_limit = dataset_limit_override
        scoping.dataset_limit = dataset_limit_override
        scoping.rationale.append(
            f"Dataset row cap lowered to {dataset_limit_override} by demo-mode override."
        )
    submission.scoping = scoping
    registry.save_submission(submission)

    try:
        result = asyncio.run(assessor.run())
    except Exception as e:
        return submission, None, f"Assessment run failed: {e}"

    submission.assessment_result = result.to_dict()
    submission.decision = auto_decide(result, profile)
    submission.transition(
        SubmissionStatus.UNDER_REVIEW,
        actor="rai-engine",
        note=f"Auto-recommendation: {submission.decision.auto_recommendation.value}",
    )
    registry.save_submission(submission)
    try:
        registry.save_html_report(submission_id, result.to_html())
    except Exception:
        pass
    return submission, result, None


_STATUS_BADGE = {
    SubmissionStatus.DRAFT.value:            ("#666",    "#eee"),
    SubmissionStatus.SUBMITTED.value:        ("#0057c2", "#e6f0ff"),
    SubmissionStatus.RUNNING.value:          ("#8a5a00", "#fbf3d6"),
    SubmissionStatus.UNDER_REVIEW.value:     ("#6b2a8a", "#f3e6fb"),
    SubmissionStatus.CHANGES_REQUESTED.value: ("#8a5a00", "#fbf3d6"),
    SubmissionStatus.APPROVED.value:         ("#0a7a2f", "#e5f5ea"),
    SubmissionStatus.REJECTED.value:         ("#a1201b", "#fbe6e4"),
}


def status_badge(status: str) -> str:
    fg, bg = _STATUS_BADGE.get(status, ("#333", "#eee"))
    label = status.replace("_", " ").title()
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
        f'font-size:12px;font-weight:600;color:{fg};background:{bg}">{label}</span>'
    )
