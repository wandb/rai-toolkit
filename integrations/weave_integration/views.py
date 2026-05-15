"""Rendered views for toolkit results.
"""

from __future__ import annotations

import logging
from html import escape
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.assessment.assessor import AssessmentResult
from rai_toolkit.models.base import ModelResponse

logger = logging.getLogger(__name__)


_WEAVE_VIEW_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 16px; background: #fafafa; color: #1a1a1a; font-size: 13px; }
.title { font-size: 18px; font-weight: 700; margin-bottom: 2px; }
.subtitle { font-size: 12px; color: #666; margin-bottom: 12px; }
.verdict-row { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-weight: 600;
         font-size: 11px; letter-spacing: 0.3px; }
.badge.PASS { background: #d4f4dd; color: #0a5c2a; }
.badge.WARN { background: #fbf3d6; color: #8a5a00; }
.badge.FAIL { background: #fce0e0; color: #8a1a1a; }
.badge.NA   { background: #ececec; color: #555; }
.badge.muted { background: #ececec; color: #555; }
.card { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px;
        margin-bottom: 12px; overflow: hidden; }
.card-hdr { background: #f3f3f3; padding: 6px 12px; font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px; color: #555;
            border-bottom: 1px solid #e6e6e6; }
.card-body { padding: 10px 12px; }
.grid { display: grid; grid-template-columns: 140px 1fr; gap: 4px 12px; }
.k { color: #666; font-size: 12px; }
.v { font-weight: 500; }
.score-bar { display: inline-block; vertical-align: middle; width: 120px; height: 8px;
             background: #ececec; border-radius: 4px; margin-right: 8px; overflow: hidden; }
.score-bar > div { height: 100%; background: #4a90e2; }
.score-bar.pass > div { background: #2da55a; }
.score-bar.warn > div { background: #d9a441; }
.score-bar.fail > div { background: #d84a4a; }
.score-bar.neutral > div { background: #9aa0a6; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #f0f0f0; }
th { font-weight: 600; color: #666; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.4px; }
.sev { font-weight: 600; font-size: 10px; padding: 2px 6px; border-radius: 8px;
       text-transform: uppercase; }
.sev.critical { background: #5b1a1a; color: #fff; }
.sev.high { background: #fce0e0; color: #8a1a1a; }
.sev.medium { background: #fff2cc; color: #7a5a00; }
.sev.low { background: #e8f0ff; color: #1a4a8a; }
code { font-family: SF Mono, Menlo, monospace; font-size: 11px; background: #f3f3f3;
       padding: 1px 5px; border-radius: 3px; }
a { color: #1f5fbf; text-decoration: none; font-weight: 600; }
a:hover { text-decoration: underline; }
.empty { color: #999; font-style: italic; }
"""


def _score_bar(
    score: float,
    passed: bool | None = None,
    *,
    status: str | None = None,
) -> str:
    """Render a score bar.

    ``status`` (one of ``"PASS"``/``"WARN"``/``"FAIL"``/``"N/A"``/``"NEUTRAL"``)
    takes precedence when provided so framework rows can show amber for WARN
    and informational rows can opt into a neutral gray. ``passed`` is kept for
    backward-compatible binary pass/fail call sites.
    """
    if status is not None:
        cls_map = {
            "PASS": "pass",
            "WARN": "warn",
            "FAIL": "fail",
            "N/A": "neutral",
            "NEUTRAL": "neutral",
        }
        cls = cls_map.get(status, "")
    else:
        cls = "" if passed is None else ("pass" if passed else "fail")
    return (
        f'<span class="score-bar {cls}"><div style="width:{max(0, min(100, score*100)):.1f}%"></div></span>'
        f'<span>{score:.1%}</span>'
    )


def _clip_text(value: Any, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _violation_where(v: Any) -> str:
    evidence = getattr(v, "evidence", {}) or {}
    row = evidence.get("dataset_row") if isinstance(evidence, dict) else None
    row = row if isinstance(row, dict) else {}
    parts: list[str] = []
    index = row.get("index")
    if isinstance(index, int):
        parts.append(f"row {index + 1}")
    scorer = getattr(v, "scorer_name", None)
    category = getattr(v, "category", None)
    if scorer:
        parts.append(str(scorer))
    elif category:
        parts.append(str(category))
    score = getattr(v, "score", None)
    if isinstance(score, (int, float)):
        parts.append(f"{score:.2f}")
    input_text = row.get("input")
    if input_text:
        parts.append(_clip_text(input_text, 80))
    return " · ".join(parts) or "—"


def _violation_trace_url(v: Any) -> str | None:
    evidence = getattr(v, "evidence", {}) or {}
    if not isinstance(evidence, dict):
        return None
    url = evidence.get("weave_call_url")
    return str(url) if url else None


def _link_or_text(label: str, url: str | None) -> str:
    safe_label = escape(label)
    if not url:
        return safe_label
    safe_url = escape(url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_label}</a>'


def _redteam_trace_url(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    url = row.get("weave_call_url")
    return str(url) if url else None


def render_assessment_html(result: AssessmentResult) -> str:
    """Render a ``AssessmentResult`` as a Weave view-panel HTML doc.

    Cards-and-grid layout focused on what a reviewer scans first: verdict,
    eval-gate vs composite, framework breakdown, policy violations, red-team.
    The full standalone HTML report (used outside Weave by
    ``AssessmentResult.to_html``) is intentionally separate — that's
    sized for printing/sharing; this is sized for the Weave panel.
    """
    if not isinstance(result, AssessmentResult):
        # Defensive: unknown payload type, render minimal placeholder so the
        # Weave panel still shows *something* rather than failing the op.
        return f"<pre>{escape(repr(result))}</pre>"

    verdict = "PASS" if result.overall_passed else "FAIL"
    ev_gate = "PASS" if result.evaluation_overall_passed else "FAIL"
    bd = result.score_breakdown or {}
    violations = result.policy_violations or []
    violation_total = len(violations)
    violation_shown = min(8, violation_total)
    findings = result.review_findings or []
    finding_total = len(findings)

    fw_rows = ""
    for f in result.frameworks or []:
        passed = bool(getattr(f, "passed", False))
        coverage = getattr(f, "coverage_percent", None)
        status = str(getattr(f, "status", "PASS" if passed else "FAIL"))
        # Bar length = controls exercised; color = neutral so the bar stops
        # carrying the pass/fail signal that the badge already carries.
        cov_html = (
            _score_bar(float(coverage), status="NEUTRAL")
            if isinstance(coverage, (int, float))
            else "—"
        )
        badge_cls_map = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "N/A": "NA"}
        badge_cls = badge_cls_map.get(status, "FAIL")
        fw_rows += (
            f"<tr><td>{escape(str(f.framework))}</td>"
            f"<td>{cov_html}</td>"
            f'<td><span class="badge {badge_cls}">{escape(status)}</span></td></tr>'
        )
    if not fw_rows:
        fw_rows = '<tr><td colspan="3" class="empty">No frameworks assessed.</td></tr>'

    viol_rows = ""
    for v in violations[:8]:
        sev = getattr(getattr(v, "severity", None), "value", "?")
        name = getattr(v, "policy_name", None) or getattr(v, "name", "policy")
        framework = ", ".join(getattr(v, "frameworks", []) or []) or "—"
        where = _violation_where(v)
        trace_url = _violation_trace_url(v)
        where_html = _link_or_text(where, trace_url)
        trace_html = _link_or_text("trace", trace_url) if trace_url else "—"
        viol_rows += (
            f'<tr><td><span class="sev {escape(sev)}">{escape(sev)}</span></td>'
            f"<td><code>{escape(str(name))}</code></td>"
            f"<td>{where_html}</td>"
            f"<td>{escape(framework)}</td>"
            f"<td>{trace_html}</td></tr>"
        )
    if not viol_rows:
        viol_rows = '<tr><td colspan="5" class="empty">No policy violations.</td></tr>'

    rt = result.redteam_summary or {}
    rt_html = ""
    if rt:
        # ``RedTeamReport.to_dict()`` emits ``total`` and ``overall_success_rate``
        # — resistance is the complement, computed here.
        attacks_total = rt.get("total", "?")
        success_rate = float(rt.get("overall_success_rate", 0))
        resistance_rate = 1.0 - success_rate
        rt_results = rt.get("results", [])
        rt_rows = ""
        successful_results: list[dict[str, Any]] = []
        if isinstance(rt_results, list):
            successful_results = [
                row
                for row in rt_results
                if isinstance(row, dict) and bool(row.get("succeeded"))
            ]
            for row in successful_results[:8]:
                attack_id = str(row.get("attack_id") or "attack")
                category = str(row.get("category") or "—")
                severity = row.get("severity")
                severity_text = str(severity) if severity is not None else "—"
                trace_url = _redteam_trace_url(row)
                trace_html = _link_or_text("trace", trace_url) if trace_url else "—"
                rt_rows += (
                    f"<tr><td><code>{escape(attack_id)}</code></td>"
                    f"<td>{escape(category)}</td>"
                    f"<td>{escape(severity_text)}</td>"
                    f"<td>{trace_html}</td></tr>"
                )
        if not rt_rows:
            rt_rows = '<tr><td colspan="4" class="empty">No successful red-team attacks.</td></tr>'
        rt_success_total = len(successful_results)
        rt_shown = min(8, rt_success_total)
        rt_html = (
            f'<div class="card"><div class="card-hdr">Red-team</div>'
            f'<div class="card-body"><div class="grid">'
            f'<div class="k">Attacks run</div><div class="v">{attacks_total}</div>'
            f'<div class="k">Resistance rate</div><div class="v">{_score_bar(resistance_rate)}</div>'
            f'<div class="k">Attack success</div><div class="v">{success_rate:.1%}</div>'
            f'</div></div></div>'
            f'<div class="card"><div class="card-hdr">Successful red-team attacks · total {rt_success_total} · showing {rt_shown}</div>'
            f'<div class="card-body"><table><thead><tr><th>Attack</th><th>Category</th><th>Severity</th><th>Trace</th></tr></thead>'
            f'<tbody>{rt_rows}</tbody></table></div></div>'
        )

    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_WEAVE_VIEW_CSS}</style></head><body>'
        f'<div class="title">Responsible AI Assessment</div>'
        f'<div class="subtitle">'
        f'<code>{escape(result.model_name)}</code> · preset <code>{escape(result.preset)}</code> · '
        f'run <code>{escape(result.run_id)}</code> · {result.duration_seconds:.1f}s'
        f'</div>'
        f'<div class="verdict-row">'
        f'<span class="badge {verdict}">VERDICT: {verdict}</span>'
        f'<span class="badge muted">eval gate {ev_gate}</span>'
        f'</div>'
        f'<div class="card"><div class="card-hdr">Scores</div><div class="card-body"><div class="grid">'
        f'<div class="k">Evaluation gate</div><div class="v">{_score_bar(result.evaluation_overall_score, result.evaluation_overall_passed)}<span style="margin-left:8px;color:#888">(threshold 70%)</span></div>'
        f'<div class="k">Composite</div><div class="v">{_score_bar(result.overall_score, status="NEUTRAL")}<span style="margin-left:8px;color:#888">(informational)</span></div>'
        f'<div class="k">— evaluation</div><div class="v">{bd.get("evaluation_raw", 0):.1%}</div>'
        f'<div class="k">— red-team resistance</div><div class="v">{bd.get("red_team_resistance", 0):.1%}</div>'
        f'<div class="k">— policy health</div><div class="v">{bd.get("policy_health", 0):.1%}</div>'
        f'<div class="k">Policy violations</div><div class="v">{violation_total}</div>'
        f'<div class="k">Findings for review</div><div class="v">{finding_total}</div>'
        f'</div></div></div>'
        f'<div class="card"><div class="card-hdr">Frameworks</div><div class="card-body">'
        f'<table><thead><tr><th>Framework</th><th>Controls exercised</th><th>Status</th></tr></thead>'
        f'<tbody>{fw_rows}</tbody></table></div></div>'
        f'<div class="card"><div class="card-hdr">Policy violations · total {violation_total} · showing {violation_shown}</div>'
        f'<div class="card-body"><table><thead><tr><th>Severity</th><th>Policy</th><th>Where</th><th>Frameworks</th><th>Trace</th></tr></thead>'
        f'<tbody>{viol_rows}</tbody></table></div></div>'
        f'{rt_html}'
        f'</body></html>'
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
