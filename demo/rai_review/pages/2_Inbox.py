# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Inbox — list every submission, filter by status."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import get_registry, status_badge  # noqa: E402

from rai_toolkit.workflow import SubmissionStatus  # noqa: E402

st.set_page_config(page_title="Inbox · RAI", layout="wide")
st.title("Reviewer inbox")

registry = get_registry()
subs = registry.list_submissions()

if not subs:
    st.info("No submissions yet.")
    st.page_link("pages/1_New_submission.py", label="Create the first submission →")
    st.stop()

status_filter = st.multiselect(
    "Filter by status",
    [s.value for s in SubmissionStatus],
    default=[
        SubmissionStatus.UNDER_REVIEW.value,
        SubmissionStatus.CHANGES_REQUESTED.value,
    ],
)
if status_filter:
    subs = [s for s in subs if s.status.value in status_filter]

for s in subs:
    with st.container(border=True):
        cols = st.columns([3, 2, 2, 2])
        cols[0].markdown(
            f"**{s.profile.name}** · `{s.submission_id}`  \n"
            f"<small>{s.profile.owner_team} · {s.profile.industry.value} · "
            f"risk {s.scoping.effective_risk_tier.value if s.scoping else s.profile.risk_tier.value}</small>",
            unsafe_allow_html=True,
        )
        cols[1].markdown(status_badge(s.status.value), unsafe_allow_html=True)
        if s.decision:
            cols[2].caption(
                f"Auto: **{s.decision.auto_recommendation.value}**"
            )
            cols[2].caption(f"Chosen: **{s.decision.decision.value}**")
        else:
            cols[2].caption("—")
        with cols[3]:
            if st.button("Open", key=f"open-{s.submission_id}"):
                st.session_state["active_submission"] = s.submission_id
                st.switch_page("pages/3_Review.py")
