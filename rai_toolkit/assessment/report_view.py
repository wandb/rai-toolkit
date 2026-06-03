# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Renderer-agnostic view model for an :class:`AssessmentResult`.

Three render surfaces consume this view:

* :func:`rai_toolkit.assessment.assessor._render_html` — the standalone HTML
  report attached to ``AssessmentResult.to_html``.
* :func:`integrations.weave_integration.views.render_assessment_html` — the
  Weave panel HTML published via ``weave.set_view``.
* ``demo/rai_review/pages/3_Review.py`` — the Streamlit reviewer UI.

The three surfaces differ in *presentation* (sandboxed CSS panel sizing,
interactive Streamlit widgets, downloadable static HTML) but share the same
*structure*: the same gates, the same scores, the same framework table, the
same findings, the same coverage gaps. This module pulls every shared
derivation into one place so a future field doesn't have to be added in
three independent renderers (which is how earlier bugs slipped in —
e.g. the view computing un-assessed counts from a field that didn't exist).

Each renderer keeps its own surface-specific bells (chat probing,
decision buttons, evidence drill-downs) — only the report data passes
through this view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rai_toolkit.assessment.assessor import AssessmentResult


# Severity-to-CSS-class mapping for red-team attack badges. Keeping this in
# the view (not the HTML template) so all surfaces label attacks identically.
_SEVERITY_LABELS: dict[int, tuple[str, str]] = {
    5: ("critical", "CRIT (5)"),
    4: ("critical", "CRIT (4)"),
    3: ("high", "HIGH (3)"),
    2: ("medium", "MED (2)"),
    1: ("low", "LOW (1)"),
}


@dataclass
class GateRow:
    """One of the four verdict gates.

    Each gate evaluates independently against the assessment, and the overall
    verdict is the AND of all four. Surfacing them as a flat list keeps every
    renderer from re-deriving "did this gate pass?".
    """

    key: str
    label: str
    state: str
    threshold_note: str = ""


@dataclass
class ScoreRow:
    """One row in the Scores card.

    The three percentages a reviewer reads first: eval gate, red-team
    resistance, policy health. Notes carry the threshold or the gate
    threshold so the bar isn't context-free.
    """

    key: str
    label: str
    percent: float
    state: str
    note: str = ""


@dataclass
class FrameworkRow:
    """One framework's scorer coverage.

    NIST GOVERN (and any framework with no scorer-measurable categories)
    renders as N/A with a single span across the coverage and status
    columns. Coverage label is pre-formatted so renderers don't have to
    decide between "N/A" and "67% of mapped scorers".
    """

    label: str
    coverage_percent: float
    coverage_label: str
    status: str
    is_not_applicable: bool
    findings: list[str] = field(default_factory=list)


@dataclass
class FindingRow:
    """A scorer-threshold match surfaced for human review.

    ``row_label`` is pre-resolved from ``dataset_row.index`` to "row N"
    (1-based) so renderers don't have to remember where row identity lives
    on the underlying finding dict. ``policy_name`` carries the deferred
    policy that fired — important context when the finding's `message`
    field doesn't repeat the policy slug.
    """

    scorer: str
    category: str
    row_label: str
    reason: str
    policy_name: str = ""
    weave_trace_url: str | None = None


@dataclass
class CoverageGapRow:
    """A bucket of un-assessed scorer runs grouped by scorer + reason."""

    scorer: str
    reason: str
    count: int


@dataclass
class AttackRow:
    """A successful red-team attack with severity rendered as label + class."""

    attack_id: str
    category: str
    severity: int
    severity_label: str
    severity_class: str
    weave_trace_url: str | None = None


@dataclass
class AssessmentReportView:
    """Structured, renderer-agnostic view of an :class:`AssessmentResult`.

    Build via :meth:`from_result`. Renderers must consume only this object —
    never poke ``AssessmentResult`` fields directly — so behavior stays
    consistent across the standalone report, the Weave panel, and the
    Streamlit reviewer UI.
    """

    # Identity
    title: str
    model_name: str
    preset: str
    run_id: str
    content_hash_short: str
    duration_seconds: float

    # Verdict
    verdict: str
    gates: list[GateRow]
    rationale: list[str]

    # Scores card
    scores: list[ScoreRow]
    policy_violations_count: int
    findings_count: int
    coverage_gaps_count: int

    # Tables
    frameworks: list[FrameworkRow]
    findings: list[FindingRow]
    coverage_gaps: list[CoverageGapRow]

    # Red-team
    redteam_attacks_total: int
    redteam_resistance: float
    redteam_attack_success: float
    redteam_successful_attacks: list[AttackRow]

    # Pass-through for renderer-specific evidence rendering
    policy_violations: list[dict[str, Any]]
    policy_assessment: dict[str, Any]

    # Surface-agnostic strings
    severity_gate_threshold: int
    severity_gate_threshold_label: str
    weave_trace_url: str | None
    disclaimer: str
    framework_coverage_footnote: str

    @classmethod
    def from_result(cls, result: "AssessmentResult | dict[str, Any]") -> "AssessmentReportView":
        """Project an :class:`AssessmentResult` (or its dict serialization) into
        renderer-agnostic shape.

        Accepting both shapes lets Streamlit (which reads
        ``submission.assessment_result`` from disk as a dict) and the Weave
        panel (which receives a live ``AssessmentResult``) share the same
        builder. All gate-state derivations, label resolutions, and percentage
        formatting happen here.
        """
        if isinstance(result, dict):
            result = _DictResult(result)
        return cls._from_attrs(result)

    @classmethod
    def _from_attrs(cls, result: Any) -> "AssessmentReportView":
        bd = result.score_breakdown or {}
        violations = result.policy_violations or []
        findings = list(result.review_findings or [])
        framework_list = result.frameworks or []

        verdict = "PASS" if result.overall_passed else "FAIL"
        eval_gate = "PASS" if result.evaluation_overall_passed else "FAIL"
        sev_gate = "PASS" if result.redteam_severity_gate_passed else "FAIL"
        framework_gate = (
            "PASS"
            if all(getattr(f, "passed", False) for f in framework_list)
            else "FAIL"
        )
        policy_gate = (
            "FAIL"
            if any(
                _violation_severity_value(v) in ("critical", "high")
                for v in violations
            )
            else "PASS"
        )

        threshold = int(result.redteam_severity_gate_threshold or 0)
        threshold_label = str(threshold) if threshold else "—"

        gates = [
            GateRow("eval", "eval gate", eval_gate),
            GateRow("framework", "framework gate", framework_gate),
            GateRow(
                "severity",
                "red-team severity gate",
                sev_gate,
                threshold_note=f"sev ≥ {threshold_label}",
            ),
            GateRow("policy", "policy gate", policy_gate),
        ]

        # Scores rows: keep order stable so all surfaces render identically.
        scores = [
            ScoreRow(
                "evaluation_gate",
                "Evaluation gate",
                float(result.evaluation_overall_score),
                "PASS" if result.evaluation_overall_passed else "FAIL",
                note="threshold 70%",
            ),
            ScoreRow(
                "redteam_resistance",
                "Red-team resistance",
                float(bd.get("red_team_resistance", 0) or 0),
                "PASS" if sev_gate == "PASS" else "FAIL",
                note=f"severity gate sev ≥ {threshold_label}",
            ),
            ScoreRow(
                "policy_health",
                "Policy health",
                float(bd.get("policy_health", 0) or 0),
                "PASS" if policy_gate == "PASS" else "FAIL",
            ),
        ]

        frameworks = [_framework_row(f) for f in framework_list]
        finding_rows = [_finding_row(f) for f in findings if isinstance(f, dict)]
        gap_rows = [
            CoverageGapRow(
                scorer=str(g.get("scorer", "scorer")),
                reason=str(g.get("reason", "—")),
                count=int(g.get("count") or 0),
            )
            for g in (result.coverage_gaps or [])
            if isinstance(g, dict) and int(g.get("count") or 0) > 0
        ]

        rt_summary = result.redteam_summary or {}
        attacks_total = int(rt_summary.get("total") or 0)
        success_rate = float(rt_summary.get("overall_success_rate") or 0)
        resistance = 1.0 - success_rate
        rt_attack_rows: list[AttackRow] = []
        for row in rt_summary.get("results") or []:
            if not isinstance(row, dict) or not row.get("succeeded"):
                continue
            rt_attack_rows.append(_attack_row(row))

        subtitle_hash = (result.content_hash or "")[:8]

        return cls(
            title="AI Governance Assessment",
            model_name=result.model_name,
            preset=result.preset,
            run_id=result.run_id,
            content_hash_short=subtitle_hash,
            duration_seconds=float(result.duration_seconds or 0),
            verdict=verdict,
            gates=gates,
            rationale=list(result.verdict_rationale or []),
            scores=scores,
            policy_violations_count=len(violations),
            findings_count=len(finding_rows),
            coverage_gaps_count=sum(g.count for g in gap_rows),
            frameworks=frameworks,
            findings=finding_rows,
            coverage_gaps=gap_rows,
            redteam_attacks_total=attacks_total,
            redteam_resistance=resistance,
            redteam_attack_success=success_rate,
            redteam_successful_attacks=rt_attack_rows,
            policy_violations=[_violation_to_dict(v) for v in violations],
            policy_assessment=dict(result.policy_assessment or {}),
            severity_gate_threshold=threshold,
            severity_gate_threshold_label=threshold_label,
            weave_trace_url=result.weave_trace_url,
            disclaimer=(
                "Each gate is evaluated independently. The toolkit does not "
                "certify your system against any framework, it organizes "
                "evidence for human review."
            ),
            framework_coverage_footnote=(
                "Coverage of evidence types reviewers asked about, not a "
                "compliance score. Frameworks are organized through the MIT "
                "AI Risk Repository taxonomy, with NIST AI RMF functions and "
                "EU AI Act articles mapped onto the same categories."
            ),
        )


def _framework_row(f: Any) -> FrameworkRow:
    status = str(getattr(f, "status", "FAIL"))
    is_na = status == "N/A" or bool(getattr(f, "is_not_applicable", False))
    coverage = float(getattr(f, "coverage_percent", 0) or 0)
    if is_na:
        coverage_label = "Not applicable. Process-level requirement, no scorer-measurable controls."
    else:
        coverage_label = f"{coverage * 100:.0f}% of mapped scorers"
    return FrameworkRow(
        label=str(getattr(f, "framework", "")),
        coverage_percent=coverage,
        coverage_label=coverage_label,
        status=status,
        is_not_applicable=is_na,
        findings=list(getattr(f, "findings", []) or []),
    )


def _finding_row(fnd: dict[str, Any]) -> FindingRow:
    row_dict = fnd.get("dataset_row") or {}
    row_idx = row_dict.get("index") if isinstance(row_dict, dict) else None
    row_label = f"row {row_idx + 1}" if isinstance(row_idx, int) else "—"
    return FindingRow(
        scorer=str(fnd.get("scorer_name") or fnd.get("scorer") or "—"),
        category=str(fnd.get("category") or fnd.get("mit_category") or "—"),
        row_label=row_label,
        reason=str(
            fnd.get("message")
            or fnd.get("explanation")
            or fnd.get("reason")
            or "—"
        ),
        policy_name=str(fnd.get("policy_name") or ""),
        weave_trace_url=fnd.get("weave_call_url"),
    )


def _attack_row(row: dict[str, Any]) -> AttackRow:
    severity = int(row.get("severity") or 0)
    css_class, label = _SEVERITY_LABELS.get(severity, ("low", f"SEV {severity}"))
    return AttackRow(
        attack_id=str(row.get("attack_id") or "attack"),
        category=str(row.get("category") or "—"),
        severity=severity,
        severity_label=label,
        severity_class=css_class,
        weave_trace_url=row.get("weave_call_url"),
    )


class _DictFrameworkRow:
    """Attribute-access shim for a serialized FrameworkAssessment dict."""

    __slots__ = ("framework", "coverage_percent", "status", "findings", "is_not_applicable", "passed")

    def __init__(self, d: dict[str, Any]) -> None:
        self.framework = d.get("framework", "")
        self.coverage_percent = d.get("coverage_percent", 0.0)
        self.status = d.get("status", "FAIL")
        self.findings = list(d.get("findings") or [])
        self.is_not_applicable = self.status == "N/A" or bool(d.get("is_not_applicable"))
        self.passed = self.status in ("PASS", "N/A")


class _DictResult:
    """Attribute-access shim over a serialized ``AssessmentResult`` dict.

    Used by :meth:`AssessmentReportView.from_result` when the caller has the
    JSON form of a result on hand (e.g. Streamlit reads
    ``submission.assessment_result`` as a dict). Exposes every attribute the
    view builder reads, so the builder stays single-source.
    """

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d
        self._frameworks = [
            _DictFrameworkRow(f) for f in (d.get("frameworks") or []) if isinstance(f, dict)
        ]

    def __getattr__(self, name: str) -> Any:
        if name == "frameworks":
            return self._frameworks
        return self._d.get(name)


def _violation_severity_value(v: Any) -> str:
    if isinstance(v, dict):
        sev = v.get("severity")
        if isinstance(sev, dict):
            return str(sev.get("value") or "").lower()
        return str(sev or "").lower()
    sev = getattr(v, "severity", None)
    return str(getattr(sev, "value", sev) or "").lower()


def _violation_to_dict(v: Any) -> dict[str, Any]:
    """Best-effort serialization of a policy violation for renderer pass-through.

    Each renderer formats violation evidence differently (the standalone HTML
    report has full row drill-down, the Weave panel shows a compact table,
    Streamlit shows containers with input/output excerpts). Keep the data
    intact and let each surface format it.
    """
    if isinstance(v, dict):
        return dict(v)
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(v):
            return asdict(v)
    except Exception:
        pass
    return {
        "policy_name": getattr(v, "policy_name", None),
        "severity": _violation_severity_value(v),
        "message": getattr(v, "message", ""),
        "category": getattr(v, "category", None),
        "scorer_name": getattr(v, "scorer_name", None),
        "score": getattr(v, "score", None),
        "evidence": getattr(v, "evidence", None),
        "frameworks": list(getattr(v, "frameworks", []) or []),
    }
