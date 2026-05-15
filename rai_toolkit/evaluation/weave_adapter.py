"""Convert Weave ``get_eval_results`` output into ``EvaluationResults`` for assessment.

Lives under ``rai_toolkit.evaluation`` so importing it does not pull in
``weave`` at import time (unlike ``integrations.weave_integration`` package init).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from rai_toolkit.compliance.frameworks import ComplianceProfile
from rai_toolkit.evaluation.pipeline import EvaluationItem, EvaluationResults, RAIEvaluationPipeline
from rai_toolkit.scorers.base import ScorerResult

logger = logging.getLogger(__name__)


def _list_weave_table_rows(weave_eval_results: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of per-example dicts from Weave ``EvaluationResults``."""
    rows_obj = getattr(weave_eval_results, "rows", None)
    if rows_obj is None:
        return []
    try:
        rows = list(rows_obj)
    except Exception as e:
        logger.debug("Could not iterate weave eval rows: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        elif hasattr(r, "model_dump"):
            out.append(r.model_dump())
        elif hasattr(r, "__dict__"):
            out.append(dict(r.__dict__))
        else:
            out.append({"_raw": r})
    return out


def _extract_output_text(model_output: Any) -> str:
    if model_output is None:
        return ""
    if isinstance(model_output, dict):
        return str(model_output.get("output") or model_output.get("text") or model_output)
    return str(model_output)


def _extract_weave_call_url(wrow: dict[str, Any], model_output: Any) -> str | None:
    """Best-effort row/model trace URL extraction from Weave eval rows."""
    candidates: list[Any] = []
    if isinstance(model_output, dict):
        candidates.extend(
            model_output.get(key)
            for key in ("weave_call_url", "call_url", "ui_url", "trace_url")
        )
        metadata = model_output.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                metadata.get(key)
                for key in ("weave_call_url", "call_url", "ui_url", "trace_url")
            )
    candidates.extend(
        wrow.get(key)
        for key in ("weave_call_url", "call_url", "ui_url", "trace_url")
    )
    for value in candidates:
        if value:
            return str(value)
    return None


def _raw_score_to_scorer_result(scorer_name: str, raw: Any) -> ScorerResult:
    """Map a single Weave scorer cell to :class:`ScorerResult`.

    Honest-on-failure: if the payload doesn't match a known shape, return an
    ``assessed=False`` result rather than a 0.5/passed default. Aggregations
    and policy gates exclude un-assessed rows so we never quietly inflate
    confidence with synthetic numbers.
    """
    if isinstance(raw, ScorerResult):
        return raw
    if not isinstance(raw, dict):
        return _unassessed(
            scorer_name,
            f"{scorer_name}: unparseable scorer payload (type={type(raw).__name__})",
            details={"raw_type": type(raw).__name__},
        )

    # Shape 1: rai_toolkit ScorerResult (our own scorers via WeaveRAIScorer).
    # Both ``score`` and ``passed`` are present; ``assessed`` may be present.
    if "score" in raw and "passed" in raw:
        try:
            score = float(raw["score"])
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        return ScorerResult(
            score=score,
            passed=bool(raw["passed"]),
            category=str(raw.get("category") or ""),
            explanation=str(raw.get("explanation") or ""),
            details=dict(raw.get("details") or {}),
            assessed=bool(raw.get("assessed", True)),
        )

    # Shape 2: Weave's WeaveScorerResult — ``{passed, metadata}``. The
    # binary scorers (toxicity, bias, hallucination, coherence, fluency)
    # all return this shape; the per-category numeric metadata is too
    # scorer-specific to map to a single 0-1 number, so use the binary
    # pass/fail decision the scorer already made.
    if "passed" in raw and "metadata" in raw and "score" not in raw:
        passed = bool(raw["passed"])
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            category="",
            explanation=f"{scorer_name}: {'passed' if passed else 'failed'} (binary scorer)",
            details=dict(raw.get("metadata") or {}),
        )

    # Shape 3: OpenAIModerationScorer — ``{passed, categories}``.
    if "passed" in raw and "categories" in raw and "score" not in raw:
        passed = bool(raw["passed"])
        flagged_cats = list((raw.get("categories") or {}).keys())
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            category="",
            explanation=(
                f"{scorer_name}: passed (no flagged categories)"
                if passed
                else f"{scorer_name}: flagged categories {flagged_cats}"
            ),
            details={"flagged_categories": flagged_cats},
        )

    # Shape 4: legacy numeric-keyed dicts. Only honor these when the value
    # is already in [0,1] — clamping silently turned arbitrary numbers into
    # confident-looking scores.
    for key in ("value", "score", "accuracy", "match"):
        if key in raw and isinstance(raw[key], (int, float)):
            v = float(raw[key])
            if 0.0 <= v <= 1.0:
                return ScorerResult(
                    score=v,
                    passed=v >= 0.5,
                    category="",
                    explanation=f"{scorer_name}: derived from {key}",
                    details=dict(raw),
                )

    return _unassessed(
        scorer_name,
        f"{scorer_name}: unrecognized scorer output shape (keys={sorted(raw.keys())})",
        details=dict(raw),
    )


def _unassessed(scorer_name: str, explanation: str, details: dict[str, Any]) -> ScorerResult:
    """Build an explicitly-un-assessed result. No default 0.5; aggregations skip it."""
    return ScorerResult(
        score=0.0,
        passed=False,
        category="",
        explanation=explanation,
        details=details,
        assessed=False,
    )


def weave_eval_results_to_evaluation_results(
    pipeline: RAIEvaluationPipeline,
    profile: ComplianceProfile,
    model_name: str,
    weave_eval_results: Any,
    weave_summary: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    name: str,
) -> EvaluationResults:
    """Build toolkit :class:`EvaluationResults` from Weave per-row evaluation output."""
    weave_rows = _list_weave_table_rows(weave_eval_results)
    if len(weave_rows) != len(dataset_rows):
        logger.warning(
            "Weave row count (%d) != dataset row count (%d); aligning by minimum",
            len(weave_rows),
            len(dataset_rows),
        )
    n = min(len(weave_rows), len(dataset_rows))
    items: list[EvaluationItem] = []
    for i in range(n):
        wrow = weave_rows[i]
        drow = dataset_rows[i]
        input_text = str(drow.get("input") or drow.get("input_text") or "")
        context = str(drow.get("context") or "")
        expected = str(drow.get("expected") or "")
        raw_out = wrow.get("output")
        if raw_out is None:
            raw_out = wrow.get("model_output")
        model_output = _extract_output_text(raw_out)
        metadata: dict[str, Any] = {"evaluation_backend": "weave", "row_index": i}
        if "policy_expectations" in drow:
            metadata["policy_expectations"] = drow.get("policy_expectations")
        call_url = _extract_weave_call_url(wrow, raw_out)
        if call_url:
            metadata["weave_call_url"] = call_url
        raw_scores = wrow.get("scores") or {}
        scores: dict[str, ScorerResult] = {}
        if isinstance(raw_scores, dict):
            for sname, cell in raw_scores.items():
                scores[str(sname)] = _raw_score_to_scorer_result(str(sname), cell)
        items.append(
            EvaluationItem(
                input=input_text,
                context=context,
                expected=expected,
                model_output=model_output,
                scores=scores,
                metadata=metadata,
            )
        )

    summary = pipeline._compute_summary(items, [])
    overall_score = pipeline._compute_overall_score(summary)
    overall_passed = overall_score >= 0.7

    scorer_names: list[str] = []
    if items and items[0].scores:
        scorer_names = list(items[0].scores.keys())

    return EvaluationResults(
        name=name,
        profile=profile,
        model_name=model_name,
        items=items,
        summary=summary,
        overall_score=overall_score,
        overall_passed=overall_passed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={
            "evaluation_backend": "weave",
            "scorers_used": scorer_names,
            "dataset_size": len(dataset_rows),
            "categories_evaluated": list(summary.keys()),
            "weave_summary": weave_summary,
        },
    )
