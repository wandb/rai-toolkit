# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""AI Governance Review — entry page.

Run with::

    streamlit run demo/rai_review/app.py

The multipage navigation (Inbox / New submission / Review) is served from
the ``pages/`` directory alongside this file.
"""

from __future__ import annotations

import streamlit as st

from _common import get_registry, status_badge

st.set_page_config(
    page_title="AI Governance Review",
    page_icon=":shield:",
    layout="wide",
)

st.title("AI Governance Review")
st.caption(
    "The review gate between your AI application team and production. "
    "Submit an app, get a scoped assessment, receive an approval decision with remediation."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Submit")
    st.markdown(
        "App teams fill out an intake profile describing what the app does, "
        "what data it handles, and where it's deployed. Risk tier and applicable "
        "tests are derived automatically from those traits."
    )
    st.page_link("pages/1_New_submission.py", label="Start a new submission →")

with col2:
    st.subheader("2. Assess")
    st.markdown(
        "The toolkit picks datasets, scorers, red-team severity, and policies "
        "based on the profile — then runs the full assessment pipeline. "
        "Every choice is recorded in the scoping rationale."
    )

with col3:
    st.subheader("3. Decide")
    st.markdown(
        "The governance team sees the findings with auto-generated remediation and the "
        "engine's recommendation (approve / request changes / reject). The "
        "reviewer makes the final call, and can override with a documented reason."
    )
    st.page_link("pages/3_Review.py", label="Open the reviewer inbox →")

st.markdown("---")

reg = get_registry()
subs = reg.list_submissions()

st.subheader(f"Workspace · {len(subs)} submission(s)")
if not subs:
    st.info(
        "No submissions yet. Head to **New submission** to create the first one. "
        "For a quick demo, use `demo_app.triage_assistant:build_model` as the "
        "model ref — it's a bundled healthcare RAG you can probe end-to-end."
    )
else:
    for s in subs[:10]:
        with st.container(border=True):
            a, b, c = st.columns([3, 2, 2])
            a.markdown(
                f"**{s.profile.name}** · `{s.submission_id}`  \n"
                f"<small>{s.profile.owner_team} · {s.profile.industry.value} · "
                f"submitted {s.created_at[:19]}</small>",
                unsafe_allow_html=True,
            )
            b.markdown(status_badge(s.status.value), unsafe_allow_html=True)
            if s.decision:
                c.caption(
                    f"Auto: {s.decision.auto_recommendation.value} · "
                    f"Decision: {s.decision.decision.value}"
                )

st.caption(
    f"Workspace on disk: `rai_workspace/` · "
    f"{len(reg.list_profiles())} application(s) registered."
)
