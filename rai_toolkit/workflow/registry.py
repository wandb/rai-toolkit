# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Filesystem-backed review registry.

Not a database — just JSON on disk under ``rai_workspace/`` so the demo is
self-contained and reproducible. Swap this class for a real backend when
the review gate graduates past MWP.

Layout::

    rai_workspace/
      applications/<app_id>.json     # profile only (intake record)
      submissions/<submission_id>.json
      reports/<submission_id>.html   # generated HTML report, if present
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rai_toolkit.workflow.profile import ApplicationProfile
from rai_toolkit.workflow.submission import (
    Submission,
    deserialize_submission,
    serialize_submission,
)


DEFAULT_WORKSPACE = Path("rai_workspace")


class ReviewRegistry:
    """Simple disk store for applications, submissions, and reports."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_WORKSPACE
        self.apps_dir = self.root / "applications"
        self.subs_dir = self.root / "submissions"
        self.reports_dir = self.root / "reports"
        for d in (self.apps_dir, self.subs_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Applications --------------------------------------------------------

    def save_profile(self, profile: ApplicationProfile) -> Path:
        path = self.apps_dir / f"{profile.app_id}.json"
        import json
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_profile(self, app_id: str) -> ApplicationProfile:
        path = self.apps_dir / f"{app_id}.json"
        import json
        return ApplicationProfile.from_dict(json.loads(path.read_text()))

    def list_profiles(self) -> list[ApplicationProfile]:
        import json
        out: list[ApplicationProfile] = []
        for f in sorted(self.apps_dir.glob("*.json")):
            try:
                out.append(ApplicationProfile.from_dict(json.loads(f.read_text())))
            except Exception:
                continue
        return out

    # Submissions ---------------------------------------------------------

    def save_submission(self, submission: Submission) -> Path:
        path = self.subs_dir / f"{submission.submission_id}.json"
        path.write_text(serialize_submission(submission), encoding="utf-8")
        return path

    def load_submission(self, submission_id: str) -> Submission:
        path = self.subs_dir / f"{submission_id}.json"
        return deserialize_submission(path.read_text(encoding="utf-8"))

    def list_submissions(self) -> list[Submission]:
        out: list[Submission] = []
        for f in sorted(self.subs_dir.glob("sub-*.json"), reverse=True):
            try:
                out.append(deserialize_submission(f.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def list_submissions_for_app(self, app_id: str) -> list[Submission]:
        return [s for s in self.list_submissions() if s.profile.app_id == app_id]

    # Reports -------------------------------------------------------------

    def save_html_report(self, submission_id: str, html: str) -> Path:
        path = self.reports_dir / f"{submission_id}.html"
        path.write_text(html, encoding="utf-8")
        return path

    def report_path(self, submission_id: str) -> Path | None:
        path = self.reports_dir / f"{submission_id}.html"
        return path if path.exists() else None
