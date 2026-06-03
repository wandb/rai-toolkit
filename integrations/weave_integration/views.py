# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Rendered views for toolkit results.
"""

from __future__ import annotations

import logging
from html import escape
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.assessment.assessor import AssessmentResult
from rai_toolkit.assessment.report_view import AssessmentReportView
from rai_toolkit.models.base import ModelResponse

logger = logging.getLogger(__name__)


_WEAVE_VIEW_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fafafa; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       padding: 20px 22px 24px; color: #1a1a1a; font-size: 13px;
       max-width: 1100px; margin: 0 auto; }

/* Title + subtitle */
.title { font-size: 20px; font-weight: 700; margin-bottom: 4px; color: #1a1a1a; }
.subtitle { font-size: 12px; color: #777; margin-bottom: 18px; }

/* Verdict band — visually distinct so it can't be missed */
.verdict-band { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
                padding: 14px 16px; margin-bottom: 16px; }
.verdict-band.fail { border-left: 4px solid #d84a4a; }
.verdict-band.pass { border-left: 4px solid #2da55a; }
.verdict-line { display: flex; gap: 10px; align-items: center;
                flex-wrap: wrap; margin-bottom: 10px; }
.verdict-badge { display: inline-block; padding: 5px 14px; border-radius: 14px;
                 font-weight: 700; font-size: 14px; letter-spacing: 0.4px; }
.verdict-badge.FAIL { background: #fce0e0; color: #8a1a1a; }
.verdict-badge.PASS { background: #d4f4dd; color: #0a5c2a; }
.gate-chip { display: inline-block; padding: 3px 10px; border-radius: 12px;
             font-weight: 600; font-size: 11px; letter-spacing: 0.3px;
             background: #ececec; color: #555; }
.rationale { font-size: 12.5px; color: #444; line-height: 1.55; }
.rationale b { color: #1a1a1a; }

/* Cards */
.card { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
        margin-bottom: 14px; overflow: hidden; }
.card-hdr { background: #f3f3f3; padding: 8px 14px; font-size: 11px;
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
            color: #444; border-bottom: 1px solid #e6e6e6; }
.card-body { padding: 14px 16px; }
.card-foot { padding: 9px 14px; font-size: 11px; color: #666;
             font-style: italic; border-top: 1px solid #f0f0f0;
             background: #fafafa; }

/* Scores: three-metric row */
.scores-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.score-cell { padding: 4px 0; }
.score-label { font-size: 11px; color: #666; text-transform: uppercase;
               letter-spacing: 0.4px; margin-bottom: 6px; }
.score-value { font-size: 22px; font-weight: 700; color: #1a1a1a;
               margin-bottom: 4px; }
.score-note { font-size: 11px; color: #888; }
.score-bar { display: block; width: 100%; height: 6px; background: #ececec;
             border-radius: 3px; margin: 4px 0 6px; overflow: hidden; }
.score-bar > div { height: 100%; background: #2da55a; }
.score-bar.neutral > div { background: #9aa0a6; }
.score-bar.fail > div { background: #d84a4a; }
.score-bar.warn > div { background: #d9a441; }

/* Tables */
.coverage-bar { display: inline-block; vertical-align: middle; width: 70px;
                height: 6px; background: #ececec; border-radius: 3px;
                margin-right: 8px; overflow: hidden; }
.coverage-bar > div { height: 100%; background: #6b7280; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #f0f0f0;
         vertical-align: top; }
th { font-weight: 600; color: #666; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.4px; }
.na-row td { color: #888; font-style: italic; }
.sev { font-weight: 600; font-size: 10px; padding: 2px 6px; border-radius: 8px;
       text-transform: uppercase; }
.sev.critical { background: #5b1a1a; color: #fff; }
.sev.high { background: #fce0e0; color: #8a1a1a; }
.sev.medium { background: #fff2cc; color: #7a5a00; }
.sev.low { background: #e8f0ff; color: #1a4a8a; }
.empty-state { color: #0a5c2a; font-style: normal; padding: 12px 14px;
               background: #f5fff7; }
.empty { color: #999; font-style: italic; }
code { font-family: SF Mono, Menlo, monospace; font-size: 11px;
       background: #f3f3f3; padding: 1px 5px; border-radius: 3px; }
a { color: #1f5fbf; text-decoration: none; font-weight: 600; }
a:hover { text-decoration: underline; }
"""


def _link_or_text(label: str, url: str | None) -> str:
    safe_label = escape(label)
    if not url:
        return safe_label
    safe_url = escape(url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_label}</a>'


def _score_cell(label: str, percent: float, bar_cls: str, note: str) -> str:
    """Render one of the three big-number metrics in the Scores card.

    ``bar_cls`` selects the bar colour: empty string = green (pass),
    ``"fail"`` = red, ``"warn"`` = amber, ``"neutral"`` = gray (used for the
    red-team resistance metric, whose own severity gate — not this rate — drives
    the verdict).
    """
    width = max(0.0, min(100.0, percent * 100))
    return (
        '<div class="score-cell">'
        f'<div class="score-label">{escape(label)}</div>'
        f'<div class="score-value">{percent * 100:.1f}%</div>'
        f'<div class="score-bar {bar_cls}"><div style="width:{width:.1f}%"></div></div>'
        f'<div class="score-note">{escape(note)}</div>'
        "</div>"
    )


def render_assessment_html(result: AssessmentResult) -> str:
    """Render a ``AssessmentResult`` as a Weave view-panel HTML doc.

    Mirrors the standalone assessment report (Figure 7) layout: a framed
    verdict band with independent gate chips, a three-metric Scores row,
    framework coverage, policy violations (shown even when empty so the reader
    can see the gate ran), and red-team. The Findings and Un-assessed coverage
    gaps cards are panel-specific additions that surface evidence the figure
    omits. Every cross-surface piece flows through
    :class:`AssessmentReportView`; this function only handles panel-specific
    HTML (card sizing, badge colours, table markup).
    """
    if not isinstance(result, AssessmentResult):
        # Defensive: unknown payload type, render minimal placeholder so the
        # Weave panel still shows *something* rather than failing the op.
        return f"<pre>{escape(repr(result))}</pre>"

    view = AssessmentReportView.from_result(result)

    # Verdict band: badge + independent gate chips + paragraph rationale.
    band_cls = "fail" if view.verdict == "FAIL" else "pass"
    gate_chips = "".join(
        f'<span class="gate-chip">{escape(g.label)}'
        + (f" ({escape(g.threshold_note)})" if g.threshold_note else "")
        + f" {escape(g.state)}</span>"
        for g in view.gates
    )
    rationale_text = " ".join(escape(line) for line in view.rationale)
    rationale_html = (
        f'<div class="rationale"><b>Why {escape(view.verdict)}.</b> {rationale_text}</div>'
        if rationale_text
        else ""
    )

    # Red-team counts, reused by the score note and the red-team card header.
    rt_total = view.redteam_attacks_total
    rt_succeeded = len(view.redteam_successful_attacks)
    rt_resisted = max(0, rt_total - rt_succeeded)

    # Scores: three big numbers, one per independent gate.
    eval_note = view.scores[0].note
    rt_note = view.scores[1].note
    if rt_total:
        rt_note = f"{rt_resisted} of {rt_total} attacks resisted; {rt_note}"
    pv = view.policy_violations_count
    policy_note = f"{pv} violation{'s' if pv != 1 else ''} across configured policies"
    scores_card = (
        '<div class="card"><div class="card-hdr">Scores</div>'
        '<div class="card-body"><div class="scores-row">'
        + _score_cell(
            view.scores[0].label,
            view.scores[0].percent,
            "" if view.scores[0].state == "PASS" else "fail",
            eval_note,
        )
        + _score_cell(view.scores[1].label, view.scores[1].percent, "neutral", rt_note)
        + _score_cell(
            view.scores[2].label,
            view.scores[2].percent,
            "" if view.scores[2].state == "PASS" else "fail",
            policy_note,
        )
        + "</div></div>"
        f'<div class="card-foot">{escape(view.disclaimer)}</div></div>'
    )

    # Framework coverage: framework + scorer coverage, two columns. N/A rows
    # (e.g. NIST GOVERN) span the coverage cell with their explanation.
    fw_rows = ""
    for f in view.frameworks:
        if f.is_not_applicable:
            fw_rows += (
                f'<tr class="na-row"><td>{escape(f.label)}</td>'
                f"<td>{escape(f.coverage_label)}</td></tr>"
            )
            continue
        pct = max(0.0, min(1.0, f.coverage_percent))
        fw_rows += (
            f"<tr><td>{escape(f.label)}</td>"
            f'<td><span class="coverage-bar"><div style="width:{pct * 100:.0f}%"></div></span>'
            f"{pct * 100:.0f}%</td></tr>"
        )
    if not fw_rows:
        fw_rows = '<tr><td colspan="2" class="empty">No frameworks assessed.</td></tr>'
    framework_card = (
        '<div class="card"><div class="card-hdr">Framework coverage</div>'
        '<div class="card-body"><table><thead>'
        "<tr><th>Framework</th><th>Scorers exercised</th></tr></thead>"
        f"<tbody>{fw_rows}</tbody></table></div>"
        f'<div class="card-foot">{escape(view.framework_coverage_footnote)}</div></div>'
    )

    # Findings card (panel-only): scorer-threshold matches surfaced for review.
    finding_rows = ""
    for fnd in view.findings[:8]:
        trace_html = (
            _link_or_text("trace", fnd.weave_trace_url) if fnd.weave_trace_url else "—"
        )
        policy_cell = f"<code>{escape(fnd.policy_name)}</code>" if fnd.policy_name else "—"
        finding_rows += (
            f"<tr><td>{policy_cell}</td>"
            f"<td><code>{escape(fnd.scorer)}</code></td>"
            f"<td>{escape(fnd.category)}</td>"
            f"<td>{escape(fnd.row_label)}</td>"
            f"<td>{escape(fnd.reason)}</td>"
            f"<td>{trace_html}</td></tr>"
        )
    if not finding_rows and view.findings_count:
        finding_rows = (
            f'<tr><td colspan="6" class="empty">'
            f"{view.findings_count} finding(s) recorded; details unavailable in this view."
            "</td></tr>"
        )
    findings_card = (
        f'<div class="card"><div class="card-hdr">Findings for review · total {view.findings_count}</div>'
        f'<div class="card-body"><table><thead>'
        f"<tr><th>Policy</th><th>Scorer</th><th>Category</th><th>Row</th><th>Reason</th><th>Trace</th></tr>"
        f"</thead><tbody>{finding_rows}</tbody></table></div>"
        f'<div class="card-foot">'
        f"Scorer-threshold matches without row-level <code>policy_expectations</code> "
        f"are surfaced here as reviewer findings, not as policy violations. A finding "
        f"is evidence for a human, not a verdict.</div></div>"
        if view.findings_count
        else ""
    )

    # Un-assessed coverage gaps (panel-only): scorer runs excluded from gates.
    unassessed_rows = ""
    for gap in view.coverage_gaps:
        unassessed_rows += (
            f"<tr><td><code>{escape(gap.scorer)}</code></td>"
            f"<td>{escape(gap.reason)}</td>"
            f"<td>{gap.count}</td></tr>"
        )
    unassessed_total = view.coverage_gaps_count
    unassessed_card = (
        f'<div class="card"><div class="card-hdr">Un-assessed coverage gaps · total {unassessed_total}</div>'
        f'<div class="card-body"><table><thead>'
        f"<tr><th>Scorer</th><th>Reason</th><th>Rows affected</th></tr>"
        f"</thead><tbody>{unassessed_rows}</tbody></table></div>"
        f'<div class="card-foot">'
        f"Un-assessed rows are excluded from gates, averages, and pass-rates. "
        f"They surface here so reviewers can decide whether the gap matters for "
        f"their use case.</div></div>"
        if unassessed_total
        else ""
    )

    # Policy violations: shown even when empty so the reader can see the gate ran.
    # The pass-through dicts preserve evidence.weave_call_url for trace links.
    if pv:
        viol_rows = ""
        for vd in view.policy_violations[:8]:
            sev = str(vd.get("severity") or "?")
            name = vd.get("policy_name") or vd.get("name") or "policy"
            framework = ", ".join(vd.get("frameworks") or []) or "—"
            evidence = vd.get("evidence") or {}
            trace_url = (
                evidence.get("weave_call_url") if isinstance(evidence, dict) else None
            )
            trace_html = _link_or_text("trace", trace_url) if trace_url else "—"
            viol_rows += (
                f'<tr><td><span class="sev {escape(sev)}">{escape(sev)}</span></td>'
                f"<td><code>{escape(str(name))}</code></td>"
                f"<td>{escape(str(vd.get('message') or '—'))}</td>"
                f"<td>{escape(framework)}</td>"
                f"<td>{trace_html}</td></tr>"
            )
        policy_card = (
            f'<div class="card"><div class="card-hdr">Policy violations · total {pv} · showing {min(8, pv)}</div>'
            f'<div class="card-body"><table><thead><tr><th>Severity</th><th>Policy</th><th>Message</th><th>Frameworks</th><th>Trace</th></tr></thead>'
            f"<tbody>{viol_rows}</tbody></table></div></div>"
        )
    else:
        policy_card = (
            '<div class="card"><div class="card-hdr">Policy violations · total 0</div>'
            '<div class="empty-state"><b>No policy violations.</b> The configured '
            "policies were evaluated against the dataset and produced no critical or "
            "high-severity matches with row-level <code>policy_expectations</code>."
            "</div></div>"
        )

    # Red-team: summary folded into the card header, successful attacks in the
    # table. Both flow off the view's pre-shaped AttackRow list.
    rt_html = ""
    if rt_total:
        sev_counts = []
        for cls, lbl in (
            ("critical", "critical"),
            ("high", "high"),
            ("medium", "medium"),
            ("low", "low"),
        ):
            n = sum(
                1 for a in view.redteam_successful_attacks if a.severity_class == cls
            )
            if n:
                sev_counts.append(f"{n} {lbl}")
        breakdown = f" ({', '.join(sev_counts)})" if sev_counts else ""

        rt_rows = ""
        for atk in view.redteam_successful_attacks[:8]:
            trace_html = (
                _link_or_text("trace", atk.weave_trace_url)
                if atk.weave_trace_url
                else "—"
            )
            rt_rows += (
                f"<tr><td><code>{escape(atk.attack_id)}</code></td>"
                f"<td>{escape(atk.category)}</td>"
                f'<td><span class="sev {atk.severity_class}">{escape(atk.severity_label)}</span></td>'
                f"<td>{trace_html}</td></tr>"
            )
        if not rt_rows:
            rt_rows = '<tr><td colspan="4" class="empty">No successful red-team attacks.</td></tr>'

        header = (
            f"Red-team · {rt_total} attacks · {rt_resisted} resisted · "
            f"{rt_succeeded} succeeded{breakdown}"
        )
        rt_html = (
            f'<div class="card"><div class="card-hdr">{escape(header)}</div>'
            f'<div class="card-body"><table><thead><tr><th>Attack</th><th>Category</th><th>Severity</th><th>Trace</th></tr></thead>'
            f"<tbody>{rt_rows}</tbody></table></div>"
            f'<div class="card-foot">'
            f"A single successful attack at severity ≥ {escape(view.severity_gate_threshold_label)} "
            f"fails the verdict regardless of the aggregate resistance rate. Trace links "
            f"open the full call detail.</div></div>"
        )

    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_WEAVE_VIEW_CSS}</style></head><body>'
        f'<div class="title">{escape(view.title)}</div>'
        f'<div class="subtitle">'
        f"<code>{escape(view.model_name)}</code> · preset <code>{escape(view.preset)}</code> · "
        f"run <code>{escape(view.run_id)}</code> · hash <code>{escape(view.content_hash_short)}</code> · "
        f"{view.duration_seconds:.1f}s"
        f"</div>"
        f'<div class="verdict-band {band_cls}">'
        f'<div class="verdict-line">'
        f'<span class="verdict-badge {view.verdict}">VERDICT: {view.verdict}</span>'
        f"{gate_chips}"
        f"</div>"
        f"{rationale_html}"
        f"</div>"
        f"{scores_card}"
        f"{framework_card}"
        f"{findings_card}"
        f"{unassessed_card}"
        f"{policy_card}"
        f"{rt_html}"
        f"</body></html>"
    )


def _weave_set_view(name: str, body: str, mimetype: str) -> None:
    """Adapter: forward a registered view to ``weave.set_view``."""
    import weave

    weave.set_view(name, body, mimetype=mimetype)


def _compact_result_view(result: Any) -> Any:
    """Surface verdict + top-line scores at the top of the op output pane.

    The full ``AssessmentResult`` is preserved under ``result`` so the
    raw data is one click away. ``cost_estimate`` is stripped from that
    dict before it lands in Weave — Weave already records actual per-op
    LLM spend natively, so the toolkit's static list-price estimate is
    duplicate (and conflicting) noise in the trace UI. Non-Weave consumers
    of ``AssessmentResult.to_dict()`` still see ``cost_estimate``; this
    redaction is local to the Weave view.
    """
    if not isinstance(result, AssessmentResult):
        return result
    try:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result
        if isinstance(result_dict, dict):
            result_dict = {k: v for k, v in result_dict.items() if k != "cost_estimate"}
        return {
            "verdict": "PASS" if result.overall_passed else "FAIL",
            "evaluation_score": result.evaluation_overall_score,
            "composite_score": result.overall_score,
            "result": result_dict,
        }
    except Exception as e:  # pragma: no cover — postprocess must never break op
        logger.debug("compact result view skipped: %s", e)
        return result


def _assessment_call_display_name(call: Any) -> str:
    """Per-call label for the ``rai.assessment`` op trace row.

    Renders e.g. ``rai.assessment · gpt-4o-mini · healthcare`` so the top
    row of the trace tree describes *what* was assessed at a glance.
    The verdict/score isn't available here because the label is computed
    at call-start, before the function body runs; the op's output pane
    has the final verdict.
    """
    try:
        cert = (call.inputs or {}).get("self")
        if cert is None:
            return "rai.assessment"
        model_name = getattr(getattr(cert, "model", None), "name", None) or "model"
        preset = getattr(cert, "preset", None)
        return (
            f"rai.assessment · {model_name} · {preset}"
            if preset
            else f"rai.assessment · {model_name}"
        )
    except Exception:  # pragma: no cover — display name must never raise
        return "rai.assessment"


# Self-register on import. ``rai_toolkit._tracing.init_tracing`` triggers
# this module's import once weave is enabled, so by the time any traced op
# fires ``publish_view`` or runs, the renderer/publisher/op-extensions are
# wired up.
def _flatten_model_response(response: Any) -> Any:
    """Flatten ``ModelResponse`` into a Weave-friendly flat dict.

    ``BaseModel.predict`` returns a dataclass; without postprocessing,
    Weave's serializer renders it under a single ``result`` wrapper in
    the trace UI, which differs from the flat shape ``WeaveModel.predict``
    already emits. Standardising on ``{"output": ..., **metadata}`` for
    every predict op means downstream automation that reads predict
    outputs from Weave traces sees one shape regardless of which path
    (RAI pipeline, chat probe, red-team, Weave-evaluation) produced it.

    Pass-through for any non-ModelResponse value so the postprocess can
    safely apply to ops whose return type drifts in the future.
    """
    if not isinstance(response, ModelResponse):
        return response
    flat: dict[str, Any] = {"output": response.output}
    metadata = response.metadata or {}
    for key, value in metadata.items():
        # Don't let a metadata key shadow ``output``; if a model decides
        # to put its own ``output`` in metadata we keep the canonical one
        # and namespace the conflicting copy so neither is lost.
        if key == "output":
            flat["metadata_output"] = value
        else:
            flat[key] = value
    return flat


_tracing.register_view_renderer("assessment", render_assessment_html)
_tracing.register_view_publisher(_weave_set_view)
_tracing.register_op_extensions(
    "rai.assessment",
    call_display_name=_assessment_call_display_name,
    postprocess_output=_compact_result_view,
)
# Flatten every BaseModel.predict trace (auto-wrapped via
# ``BaseModel.__init_subclass__``) and the GuardedModel variant so both
# paths land on the same trace shape as ``WeaveModel.predict``.
_tracing.register_op_extensions(
    "rai.model.predict",
    postprocess_output=_flatten_model_response,
)
_tracing.register_op_extensions(
    "rai.guardrails.predict",
    postprocess_output=_flatten_model_response,
)
