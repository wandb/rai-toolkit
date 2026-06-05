# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Review page: findings + remediation + approve/reject.

This is the artifact the RAI team signs off on. Shows the scoping rationale
(why these tests ran), the headline verdict, per-framework findings, the
auto-generated remediation list, and a reviewer action block.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import get_registry, load_model_from_profile, status_badge  # noqa: E402

from rai_toolkit.assessment.report_view import AssessmentReportView  # noqa: E402
from rai_toolkit.workflow.submission import (  # noqa: E402
    Decision,
    ManualFinding,
    reconcile_manual_findings,
    submit_decision,
)

st.set_page_config(page_title="Review · RAI", layout="wide")

registry = get_registry()
subs = registry.list_submissions()
if not subs:
    st.info("No submissions yet. Create one first.")
    st.page_link("pages/1_New_submission.py", label="New submission →")
    st.stop()

default_id = st.session_state.get("active_submission") or subs[0].submission_id
picked = st.selectbox(
    "Submission",
    [s.submission_id for s in subs],
    index=[s.submission_id for s in subs].index(default_id) if default_id in [s.submission_id for s in subs] else 0,
)
st.session_state["active_submission"] = picked
submission = registry.load_submission(picked)

# Header -----------------------------------------------------------------

header_cols = st.columns([4, 2])
with header_cols[0]:
    st.title(submission.profile.name)
    st.caption(
        f"`{submission.submission_id}` · {submission.profile.owner_team} "
        f"· submitted {submission.created_at[:19]}"
    )
with header_cols[1]:
    st.markdown(status_badge(submission.status.value), unsafe_allow_html=True)
    if submission.decision:
        st.caption(
            f"Auto-recommendation: **{submission.decision.auto_recommendation.value}**"
        )

st.markdown("---")

# Scoping ----------------------------------------------------------------

if submission.scoping:
    with st.expander("Scoping rationale: why these tests ran", expanded=False):
        st.markdown(submission.scoping.as_markdown())

# Assessment result ---------------------------------------------------

result = submission.assessment_result
if not result:
    st.warning("No assessment result yet. The run may still be in progress.")
    st.stop()

# Single source of truth across all three render surfaces (standalone HTML
# report, Weave panel, this Streamlit page). The view builder accepts either
# a live AssessmentResult or the dict form Streamlit reads from disk.
view = AssessmentReportView.from_result(result)


def _gate_chip(label: str, state: str, note: str = "") -> str:
    color = "#0a5c2a" if state == "PASS" else "#8a1a1a"
    bg = "#d4f4dd" if state == "PASS" else "#fce0e0"
    label_html = f"{label} ({note})" if note else label
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f"font-weight:600;font-size:11px;background:{bg};color:{color};"
        f'margin-right:6px;">{label_html}: {state}</span>'
    )


verdict_html = (
    f'<span style="display:inline-block;padding:4px 12px;border-radius:12px;'
    f"font-weight:700;font-size:13px;letter-spacing:0.3px;"
    f"background:{'#d4f4dd' if view.verdict == 'PASS' else '#fce0e0'};"
    f"color:{'#0a5c2a' if view.verdict == 'PASS' else '#8a1a1a'};"
    f'margin-right:10px;">VERDICT: {view.verdict}</span>'
)
st.markdown(
    verdict_html
    + "".join(_gate_chip(g.label, g.state, g.threshold_note) for g in view.gates),
    unsafe_allow_html=True,
)

if view.rationale:
    border_color = "#d84a4a" if view.verdict == "FAIL" else "#2da55a"
    bg = "#fff5f5" if view.verdict == "FAIL" else "#f5fff7"
    with st.container():
        st.markdown(
            f'<div style="border-left:3px solid {border_color};background:{bg};'
            f"padding:8px 12px;border-radius:4px;margin:8px 0 14px;"
            f'font-size:13px;color:#555;"><b>Why {view.verdict}.</b><ul style='
            f'"margin:6px 0 0 18px;padding:0;">'
            + "".join(f"<li>{line}</li>" for line in view.rationale)
            + "</ul></div>",
            unsafe_allow_html=True,
        )

metric_cols = st.columns(4)
metric_cols[0].metric(view.scores[0].label, f"{view.scores[0].percent:.1%}")
metric_cols[1].metric(view.scores[1].label, f"{view.scores[1].percent:.1%}")
metric_cols[2].metric(view.scores[2].label, f"{view.scores[2].percent:.1%}")
metric_cols[3].metric("Policy violations", view.policy_violations_count)
st.caption(view.disclaimer + " " + view.framework_coverage_footnote)

if view.weave_trace_url:
    st.markdown(f"**Weave trace:** [{view.weave_trace_url}]({view.weave_trace_url})")


def _clip_text(value, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _row_label(row: dict) -> str:
    index = row.get("index")
    return f"dataset row {index + 1}" if isinstance(index, int) else "dataset row"


policy_violations = result.get("policy_violations", [])
if policy_violations:
    with st.expander("Policy violation evidence: where each finding came from", expanded=True):
        for violation in policy_violations:
            evidence = violation.get("evidence") or {}
            row = evidence.get("dataset_row") or {}
            severity = ((violation.get("severity") or {}).get("value") if isinstance(violation.get("severity"), dict) else violation.get("severity")) or "unknown"
            title_bits = [
                f"[{str(severity).upper()}]",
                violation.get("policy_name", "policy"),
                _row_label(row) if row else "no dataset row captured",
            ]
            with st.container(border=True):
                st.markdown(" **·** ".join(title_bits))
                meta = []
                if violation.get("scorer_name"):
                    meta.append(f"scorer `{violation['scorer_name']}`")
                if violation.get("category"):
                    meta.append(f"category `{violation['category']}`")
                if violation.get("score") is not None:
                    meta.append(f"score `{violation['score']:.2f}`")
                if meta:
                    st.caption(" · ".join(meta))
                st.write(violation.get("message", ""))

                c1, c2 = st.columns(2)
                if row.get("input"):
                    c1.markdown("**Input**")
                    c1.write(_clip_text(row.get("input"), 500))
                if row.get("model_output"):
                    c2.markdown("**Model output**")
                    c2.write(_clip_text(row.get("model_output"), 500))
                if row.get("expected"):
                    st.markdown("**Expected**")
                    st.write(_clip_text(row.get("expected"), 500))
                if row.get("context"):
                    with st.expander("Context", expanded=False):
                        st.write(row.get("context"))
                if evidence.get("explanation"):
                    st.markdown("**Scorer explanation**")
                    st.write(_clip_text(evidence.get("explanation"), 700))
                if evidence.get("weave_call_url"):
                    st.markdown(f"[Open row trace in Weave]({evidence['weave_call_url']})")

# Rationale + remediation (the review-gate payload) ----------------------

decision = submission.decision
if decision:
    st.subheader("Why this recommendation")
    for line in decision.rationale:
        st.markdown(f"- {line}")

    if decision.remediation:
        st.subheader("Remediation: what to change before resubmission")
        for item in decision.remediation:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{item.title}**")
                c2.caption(f"severity: {item.severity}")
                st.markdown(f"*{item.detail}*")
                if item.suggestion:
                    st.markdown(f"**How to fix:** {item.suggestion}")
                if item.frameworks:
                    st.caption("Frameworks: " + ", ".join(item.frameworks))
    else:
        st.success("No blocking findings. Engine recommends approval.")

# Framework coverage ----------------------------------------------------

with st.expander("Framework coverage: evidence organized by reviewer-facing vocabulary", expanded=False):
    st.caption(view.framework_coverage_footnote)
    for f in view.frameworks:
        cols = st.columns([4, 2, 1])
        cols[0].write(f.label)
        if f.is_not_applicable:
            cols[1].write("Not applicable")
            cols[2].write("N/A")
        else:
            cols[1].write(f.coverage_label)
            cols[2].write(f.status)
        for finding_line in f.findings:
            st.caption(f"  · {finding_line}")

# Findings for review ----------------------------------------------------

if view.findings:
    with st.expander(
        f"Findings for review: total {view.findings_count}",
        expanded=False,
    ):
        st.caption(
            "Scorer-threshold matches without row-level policy_expectations are "
            "surfaced here as reviewer findings, not as policy violations. A finding "
            "is evidence for a human, not a verdict."
        )
        for fnd in view.findings:
            with st.container(border=True):
                title_bits = []
                if fnd.policy_name:
                    title_bits.append(f"**`{fnd.policy_name}`**")
                title_bits.append(f"`{fnd.scorer}`")
                title_bits.append(fnd.category)
                title_bits.append(fnd.row_label)
                st.markdown(" · ".join(title_bits))
                st.write(fnd.reason)
                if fnd.weave_trace_url:
                    st.markdown(f"[Open row trace in Weave]({fnd.weave_trace_url})")

# Un-assessed coverage gaps ---------------------------------------------

if view.coverage_gaps_count:
    with st.expander(
        f"Un-assessed coverage gaps: total {view.coverage_gaps_count}",
        expanded=False,
    ):
        st.caption(
            "Un-assessed rows are excluded from gates, averages, and pass-rates. "
            "They surface here so reviewers can decide whether the gap matters "
            "for their use case."
        )
        for gap in view.coverage_gaps:
            cols = st.columns([3, 4, 1])
            cols[0].markdown(f"`{gap.scorer}`")
            cols[1].write(gap.reason)
            cols[2].write(str(gap.count))

# Successful red-team attacks -------------------------------------------

sev_gate_passed = all(
    g.state == "PASS" for g in view.gates if g.key == "severity"
)
if view.redteam_successful_attacks:
    with st.expander(
        f"Successful red-team attacks: total {len(view.redteam_successful_attacks)}",
        expanded=not sev_gate_passed,
    ):
        if not sev_gate_passed:
            st.error(
                f"Red-team severity gate failed: at least one attack at "
                f"sev >= {view.severity_gate_threshold_label} succeeded. This is "
                f"what's driving the FAIL verdict from the red-team side."
            )
        for atk in view.redteam_successful_attacks[:25]:
            cols = st.columns([3, 2, 1, 1])
            cols[0].markdown(f"`{atk.attack_id}`")
            cols[1].caption(atk.category)
            cols[2].write(atk.severity_label)
            if atk.weave_trace_url:
                cols[3].markdown(f"[trace]({atk.weave_trace_url})")

# Interactive probing (chat) --------------------------------------------

st.markdown("---")
st.subheader("Interactive probing")
st.caption(
    "Chat with the same model the assessment ran against. Pin any turn as "
    "a manual finding to fold it into the approval rationale. If Weave is on, "
    "your probing is also traced under this submission."
)

chat_key = f"chat_history__{submission.submission_id}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []  # list of {role, content, finding_pinned}

# Lazy-load the model on first probe; cache on the page session.
model_state_key = f"model__{submission.submission_id}"
if model_state_key not in st.session_state:
    try:
        st.session_state[model_state_key] = load_model_from_profile(submission.profile)
    except Exception as e:
        st.error(f"Cannot start chat. Model wouldn't load: {e}")
        st.session_state[model_state_key] = None

probe_model = st.session_state.get(model_state_key)

# Existing manual findings.
if submission.manual_findings:
    with st.container(border=True):
        st.caption(
            f"**{len(submission.manual_findings)} manual finding(s) pinned**"
        )
        for f in submission.manual_findings:
            st.markdown(
                f"- *[{f.severity}]* `{f.user_input[:60]}…` → `{f.model_output[:80]}…`"
                + (f"  \n  > {f.note}" if f.note else "")
            )

for turn in st.session_state[chat_key]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Wrap the chat input in a container so ``st.chat_input`` pins to the
# bottom of *this section* instead of the page viewport. Otherwise the
# chatbox always renders below "Reviewer action" no matter where this
# code lives, which breaks the natural reviewer flow (probe → decide).
probe_box = st.container()
if probe_model is not None:
    user_msg = probe_box.chat_input(
        "Probe the model: try edge cases, jailbreaks, ambiguous queries…"
    )
    if user_msg:
        # Build a multi-turn context from prior turns. ``BaseModel.predict``
        # only takes ``input_text`` + ``context``, so we serialize the
        # rolling chat history into ``context`` (the demo apps already
        # forward ``context`` into their system-prompt block, so prior
        # turns reach the LLM as prior conversation rather than being
        # dropped). Without this, the reviewer's chat is single-turn and
        # the model can't reason about anything said earlier.
        prior_turns = st.session_state[chat_key]
        history_lines = []
        for turn in prior_turns:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {turn['content']}")
        chat_context = (
            "Prior conversation (most recent last):\n" + "\n\n".join(history_lines)
            if history_lines
            else ""
        )

        st.session_state[chat_key].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("…")
            import asyncio

            # Group every probe turn for this submission under one Weave
            # thread so the UI renders them as one conversation in the
            # Threads tab rather than as N scattered top-level traces.
            # (Streamlit reruns on every message, so each predict() is its
            # own trace by construction; ``thread_id`` is what links them.)
            try:
                from integrations.weave_integration import probe_thread
            except Exception:
                from contextlib import nullcontext as probe_thread  # type: ignore

            thread_id = f"probe-{submission.submission_id}"
            try:
                with probe_thread(thread_id):
                    response = asyncio.run(
                        probe_model.predict(input_text=user_msg, context=chat_context)
                    )
                output = response.output
            except Exception as e:
                output = f"_[probe failed: {e}]_"
            placeholder.markdown(output)
        st.session_state[chat_key].append({"role": "assistant", "content": output})
        st.rerun()

    # Pin-as-finding controls: operate on the most recent assistant turn.
    history = st.session_state[chat_key]
    if len(history) >= 2 and history[-1]["role"] == "assistant":
        last_user = history[-2]["content"]
        last_bot = history[-1]["content"]
        with st.expander("Pin last turn as a manual finding", expanded=False):
            sev = st.selectbox(
                "Severity",
                ["info", "low", "medium", "high", "critical"],
                index=2,
                key=f"sev__{submission.submission_id}",
            )
            note = st.text_input(
                "Why is this a finding?",
                placeholder="e.g. fabricated drug dose; ungrounded confident claim; jailbreak succeeded",
                key=f"note__{submission.submission_id}",
            )
            pinned_by = st.text_input(
                "Pinned by",
                value=st.session_state.get("reviewer", ""),
                key=f"pinned_by__{submission.submission_id}",
            )
            if st.button("Pin finding", key=f"pin_btn__{submission.submission_id}"):
                submission.manual_findings.append(
                    ManualFinding(
                        user_input=last_user,
                        model_output=last_bot,
                        severity=sev,
                        note=note,
                        pinned_by=pinned_by or "",
                    )
                )
                reconcile_manual_findings(submission)
                registry.save_submission(submission)
                # Mirror the pin onto the assessment's Weave trace as feedback so the
                # human-in-the-loop step is visible inside the same trace as
                # the auto-eval evidence. No-op if Weave isn't enabled.
                assessment_call_id = (submission.assessment_result or {}).get(
                    "weave_call_id"
                )
                if assessment_call_id:
                    try:
                        from integrations.weave_integration import attach_manual_finding

                        attach_manual_finding(
                            assessment_call_id,
                            user_input=last_user,
                            model_output=last_bot,
                            severity=sev,
                            note=note,
                            pinned_by=pinned_by or "",
                        )
                    except Exception:
                        pass  # annotation is cosmetic; never block the pin
                st.success("Pinned. Auto-recommendation refreshed.")
                st.rerun()

# Reviewer action --------------------------------------------------------

st.markdown("---")
st.subheader("Reviewer action")

if submission.status.value in ("approved", "rejected"):
    st.info(
        f"Decision already recorded: **{submission.decision.decision.value}** "
        f"by {submission.decision.approved_by or 'unknown'} "
        f"on {submission.decision.decided_at[:19]}."
    )
    if submission.decision.reviewer_notes:
        st.caption(f"Notes: {submission.decision.reviewer_notes}")
else:
    reviewer = st.text_input("Reviewer email", value=st.session_state.get("reviewer", ""))
    st.session_state["reviewer"] = reviewer
    notes = st.text_area(
        "Reviewer notes",
        placeholder="Document any override of the auto-recommendation. The paper trail matters.",
    )
    def _record_decision(decision: Decision) -> None:
        submit_decision(submission, decision, reviewer, notes)
        registry.save_submission(submission)
        # Mirror the human decision onto the assessment's Weave trace so the
        # full audit picture (auto-eval + reviewer judgment) lives on
        # one call. No-op if Weave isn't enabled.
        assessment_call_id = (submission.assessment_result or {}).get("weave_call_id")
        if assessment_call_id:
            try:
                from integrations.weave_integration import attach_reviewer_decision

                auto = getattr(
                    getattr(submission.decision, "auto_recommendation", None),
                    "value",
                    None,
                )
                attach_reviewer_decision(
                    assessment_call_id,
                    decision=decision.value,
                    reviewer=reviewer,
                    notes=notes,
                    auto_recommendation=auto,
                )
            except Exception:
                pass  # annotation is cosmetic; never block the decision

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Approve", type="primary", disabled=not reviewer):
            _record_decision(Decision.APPROVE)
            st.success("Approved.")
            st.rerun()
    with action_cols[1]:
        if st.button("Request changes", disabled=not reviewer):
            _record_decision(Decision.REQUEST_CHANGES)
            st.warning("Changes requested.")
            st.rerun()
    with action_cols[2]:
        if st.button("Reject", disabled=not reviewer):
            _record_decision(Decision.REJECT)
            st.error("Rejected.")
            st.rerun()

# History ----------------------------------------------------------------

with st.expander("State history", expanded=False):
    for h in submission.history:
        st.caption(
            f"{h.at[:19]}  -  {h.from_status} → **{h.to_status}**  "
            f"({h.actor or 'system'})"
            + (f" · {h.note}" if h.note else "")
        )
