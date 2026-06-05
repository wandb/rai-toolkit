# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""App-team intake form.

Collects the ``ApplicationProfile`` and kicks off a scoped assessment run.
On completion, navigates the reviewer to the review page for the submission.

Implementation note: this page deliberately does NOT use ``st.form``.
Streamlit forms submit when Enter is pressed inside any text input, not just
when the submit button is clicked, which triggers the several-minute
assessment pipeline when the user was only trying to move between fields.
We use individual widgets + a plain ``st.button`` so only an explicit click
runs the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make _common importable when Streamlit runs the page directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import get_registry, load_model_from_profile, run_assessment  # noqa: E402

from rai_toolkit.workflow import (  # noqa: E402
    CAPABILITY_CHOICES,
    DATA_TYPE_CHOICES,
    ApplicationProfile,
    DeploymentContext,
    Industry,
    RiskTier,
    scope_assessor,
)

st.set_page_config(page_title="New submission · RAI", layout="wide")
st.title("New submission")
st.caption(
    "Fill this out as the app team. Nothing runs until you click "
    "**Submit for review** at the bottom."
)

st.subheader("Application")
c1, c2 = st.columns(2)
name = c1.text_input("App name", value="Meridian Clinical Triage", key="name")
owner_team = c2.text_input("Owner team", value="Meridian Health ML", key="owner_team")

description = st.text_area(
    "What does this app do?",
    value="Answers patient-reported symptoms with triage guidance. Internal clinicians only.",
    key="description",
)

c3, c4 = st.columns(2)
owner_email = c3.text_input("Primary contact email", value="ml-lead@meridianhealth.example", key="owner_email")
submitted_by = c4.text_input("Submitting on behalf of", value="reviewer@example.com", key="submitted_by")

st.subheader("Context")
c5, c6, c7 = st.columns(3)
industry = c5.selectbox(
    "Industry",
    [i.value for i in Industry],
    index=0,
    help="Drives the preset and default evaluation datasets.",
    key="industry",
)
deployment = c6.selectbox(
    "Deployment context",
    [d.value for d in DeploymentContext],
    index=1,
    help="`internal` employees · `external` known customers · `public` unauthenticated.",
    key="deployment",
)
risk_tier = c7.selectbox(
    "Self-declared risk tier",
    [r.value for r in RiskTier],
    index=1,
    help="The engine may escalate this upward if declared data types are sensitive.",
    key="risk_tier",
)

st.subheader("Data & capabilities")
data_types = st.multiselect(
    "Data types processed",
    DATA_TYPE_CHOICES,
    default=["phi", "pii"],
    help="Used to expand datasets, scorers, and policies. PHI/biometric/credit/legal "
    "auto-escalate risk tier to at least HIGH.",
    key="data_types",
)
capabilities = st.multiselect(
    "Capabilities",
    CAPABILITY_CHOICES,
    default=["qa", "advice", "decision_support"],
    help="`autonomous_action` auto-escalates risk tier to HIGH.",
    key="capabilities",
)

st.subheader("Model")
adapter = st.radio(
    "How does the RAI team reach the app?",
    options=["python_class", "openai_compatible"],
    format_func=lambda s: {
        "python_class": "Python class in this repo",
        "openai_compatible": "OpenAI-compatible HTTP endpoint",
    }[s],
    horizontal=True,
    key="adapter",
    help="Pick `Python class` for code that lives in the repo. Pick "
    "`OpenAI-compatible endpoint` to point at any deployed service that "
    "speaks chat-completions: OpenAI, Azure, vLLM, Ollama, LiteLLM, "
    "internal proxies, anything.",
)

model_ref = ""
adapter_args: dict[str, str] = {}
if adapter == "python_class":
    model_ref = st.text_input(
        "Model reference (`package.module:ClassName`)",
        value="demo_app.triage_assistant:build_model",
        help="Dotted path importable from the repo root. The shipped "
        "`demo_app.triage_assistant:build_model` is a small healthcare "
        "RAG you can probe end-to-end.",
        key="model_ref",
    )
else:
    oc1, oc2 = st.columns(2)
    adapter_args["model"] = oc1.text_input(
        "Model name on the endpoint",
        value="gpt-4o-mini",
        help="As the endpoint expects it. e.g. `gpt-4o-mini`, `llama3:8b`, "
        "`meta-llama/Llama-3-8B-Instruct`.",
        key="adapter_model",
    )
    adapter_args["base_url"] = oc2.text_input(
        "Base URL (blank = public OpenAI)",
        placeholder="https://my-vllm.example/v1",
        key="adapter_base_url",
    )
    adapter_args["api_key"] = st.text_input(
        "API key (blank = OPENAI_API_KEY env var)",
        type="password",
        key="adapter_api_key",
    )
    adapter_args["system_prompt"] = st.text_area(
        "System prompt (optional, pasted from the app team)",
        placeholder="You are a clinical triage assistant…",
        key="adapter_system_prompt",
    )
dataset_overrides_raw = st.text_input(
    "Dataset slugs (comma-separated)",
    help="For real assessments, provide explicit dataset slugs. Leave blank only in demo mode.",
    key="dataset_overrides_raw",
)
demo_mode = st.checkbox(
    "Demo mode (sample datasets, cap to 5 rows)",
    value=True,
    help="Allows bundled sample fixture datasets and overrides the risk-tier row cap "
    "for a fast demo run. Turn off for a real assessment.",
    key="demo_mode",
)

st.subheader("Red-team & guardrails")
rcol1, rcol2, rcol3 = st.columns(3)
enable_pyrit = rcol1.checkbox(
    "Run PyRIT attacks",
    value=False,
    help="Run microsoft/PyRIT single-turn attacks (skeleton-key, many-shot, "
    "flip, context-compliance) alongside the in-tree red-team catalog. "
    "Requires `pip install \"rai-toolkit[pyrit]\"`.",
    key="enable_pyrit",
)
enable_garak = rcol2.checkbox(
    "Run Garak probes",
    value=False,
    help="Run NVIDIA/garak probes (DAN, prompt-injection, encoding-bypass, "
    "toxic continuation). Requires `pip install \"rai-toolkit[garak]\"`.",
    key="enable_garak",
)
enable_nemo = rcol3.checkbox(
    "Apply NeMo Guardrails",
    value=False,
    help="Wrap the model under review in NVIDIA NeMo Guardrails (input + "
    "output rails) before assessing. Every eval / red-team / probe call "
    "flows through NeMo and shows up in the trace tree. Requires "
    "`pip install \"rai-toolkit[nemo]\"`.",
    key="enable_nemo",
)
if enable_pyrit:
    # Surface load failures (missing install OR e.g. a version mismatch in a
    # transitive dep like openai) before the user kicks off a multi-minute
    # assessment. Without this the assessor would just log at WARNING and the
    # checked box silently does nothing.
    try:
        from integrations.pyrit_integration.adapter import (
            _PYRIT_IMPORT_ERROR,
            PYRIT_INSTALLED,
        )
    except Exception as _e:
        PYRIT_INSTALLED = False
        _PYRIT_IMPORT_ERROR = _e
    if not PYRIT_INSTALLED:
        detail = f"\n\nUnderlying import error:\n\n`{_PYRIT_IMPORT_ERROR!r}`" if _PYRIT_IMPORT_ERROR else ""
        st.warning(
            "PyRIT could not be loaded in the active Python environment, so "
            "PyRIT attacks will be skipped and no PyRIT traces will appear. "
            "Install (or repair) with `pip install \"rai-toolkit[pyrit]\"`."
            + detail
        )

if enable_garak:
    try:
        from integrations.garak_integration.adapter import GARAK_INSTALLED
    except Exception:
        GARAK_INSTALLED = False
    if not GARAK_INSTALLED:
        st.warning(
            "Garak is not installed in the active Python environment, so Garak "
            "probes will be skipped and no Garak traces will appear."
        )

st.subheader("Observability")
st.caption(
    "Every evaluate_item / red-team attack / judge call is piped into W&B Weave "
    "so the reviewer can drill into any finding. Project and entity are required."
)
wcol1, wcol2 = st.columns(2)
weave_project = wcol1.text_input(
    "Weave project",
    placeholder="my-rai-reviews",
    key="weave_project",
)
weave_entity = wcol2.text_input(
    "W&B entity (team/user)",
    placeholder="my-team",
    key="weave_entity",
)

notes = st.text_area(
    "Notes for the reviewer",
    value="First submission for the v1 triage assistant. Internal pilot only. Clinicians review every output before it reaches a patient. Please flag any PHI leakage or unsafe triage advice.",
    key="notes",
)

st.markdown("---")
submitted = st.button("Submit for review", type="primary", key="submit_btn")


if submitted:
    dataset_overrides = [
        s.strip() for s in dataset_overrides_raw.split(",") if s.strip()
    ]
    base_required = {
        "name": name,
        "description": description,
        "owner_team": owner_team,
        "owner_email": owner_email,
        "Weave project": weave_project,
        "W&B entity": weave_entity,
    }
    if adapter == "python_class":
        base_required["model reference"] = model_ref
    else:
        base_required["adapter model name"] = adapter_args.get("model", "")
    if not demo_mode:
        base_required["dataset slugs"] = ", ".join(dataset_overrides)

    missing = [label for label, value in base_required.items() if not value]
    if missing:
        st.error(f"Required fields missing: {', '.join(missing)}")
        st.stop()

    try:
        profile = ApplicationProfile(
            name=name,
            description=description,
            owner_team=owner_team,
            owner_email=owner_email,
            industry=industry,
            deployment_context=deployment,
            risk_tier=risk_tier,
            data_types=data_types,
            capabilities=capabilities,
            model_ref=model_ref,
            model_adapter=adapter,
            model_adapter_args={k: v for k, v in adapter_args.items() if v},
            dataset_overrides=dataset_overrides,
            allow_sample_datasets=demo_mode,
            extra_redteam_sources=(
                (["pyrit"] if enable_pyrit else [])
                + (["garak"] if enable_garak else [])
            ),
            enable_nemo_guardrails=enable_nemo,
            weave_project=weave_project,
            weave_entity=weave_entity,
            notes=notes,
            submitted_by=submitted_by,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Dry-scope before the heavy pipeline so the user sees what's about to run
    # and a rough duration estimate. The real scoping happens again inside
    # run_assessment; this preview is read-only.
    try:
        _preview_model = load_model_from_profile(profile)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()
    try:
        _, preview_scope = scope_assessor(profile, _preview_model)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    registry = get_registry()
    registry.save_profile(profile)
    demo_override = 5 if demo_mode else None
    row_cap = demo_override if demo_override is not None else (preview_scope.dataset_limit or 100)
    # Rough estimate: one model call + ~6 LLM-judge calls per row, plus ~12 red-team calls.
    expected_calls = row_cap * 7 + (12 if preview_scope.run_redteam else 0)
    est_seconds = int(expected_calls * 0.9)  # ~0.9s per OpenAI call average

    with st.status(
        f"Running scoped assessment (~{est_seconds}s expected)…",
        expanded=True,
    ) as status:
        st.write(f"Application `{profile.app_id}` registered.")
        st.write(
            f"**Scope:** preset `{preview_scope.preset}` · datasets "
            f"`{', '.join(preview_scope.datasets)}` · risk tier "
            f"`{preview_scope.effective_risk_tier.value}` · "
            f"row cap `{demo_override if demo_override is not None else (preview_scope.dataset_limit or 'full')}` · "
            f"red-team `{'on sev≤' + str(preview_scope.redteam_max_severity) if preview_scope.run_redteam else 'off'}`"
        )
        st.write(
            f"This will issue roughly **{expected_calls} OpenAI calls** "
            f"(≈ {row_cap} rows × 7 judges + {12 if preview_scope.run_redteam else 0} red-team). "
            "Logs stream to your terminal; re-open this page after completion to see the review."
        )
        if profile.weave_project:
            st.write(
                f"Traces streaming to Weave project `{profile.weave_project}`. "
                "Open the Weave UI now to watch calls land live."
            )
        st.write("Running pipeline")
        submission, result, err = run_assessment(profile, dataset_limit_override=demo_override)
        if err:
            status.update(label="Assessment failed", state="error")
            st.error(err)
            st.stop()
        verdict = "PASS" if result.overall_passed else "FAIL"
        sev_gate = "PASS" if result.redteam_severity_gate_passed else "FAIL"
        st.write(
            f"Assessment complete in {result.duration_seconds:.1f}s. "
            f"Verdict **{verdict}**: evaluation gate {result.evaluation_overall_score:.1%}, "
            f"red-team severity gate (sev >= {result.redteam_severity_gate_threshold or '-'}) **{sev_gate}**, "
            f"policy violations {len(result.policy_violations)}."
        )
        st.write(
            f"Engine recommends: **{submission.decision.auto_recommendation.value}** "
            f"(pending reviewer sign-off)."
        )
        status.update(label="Submission ready for review", state="complete")

    st.success(f"Submitted: `{submission.submission_id}`")
    st.page_link(
        "pages/3_Review.py",
        label=f"Review `{submission.submission_id}` →",
    )
