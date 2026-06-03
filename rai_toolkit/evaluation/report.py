# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Compliance report generation."""

from __future__ import annotations

import json
from typing import Any

from rai_toolkit.evaluation.pipeline import EvaluationResults


class ComplianceReport:
    """Generates compliance reports from evaluation results.

    Supports multiple output formats: dict, JSON, and summary text.

    Example::

        report = ComplianceReport(results)
        print(report.to_summary())
        report.to_json("compliance_report.json")
    """

    def __init__(self, results: EvaluationResults) -> None:
        self.results = results

    def to_dict(self) -> dict[str, Any]:
        """Generate a structured compliance report as a dict."""
        return {
            "report_metadata": {
                "name": self.results.name,
                "model": self.results.model_name,
                "framework": self.results.profile.framework.value,
                "industry": self.results.profile.industry,
                "timestamp": self.results.timestamp,
                "dataset_size": len(self.results.items),
            },
            "overall": {
                "score": round(self.results.overall_score, 3),
                "passed": self.results.overall_passed,
                "verdict": "COMPLIANT" if self.results.overall_passed else "NON-COMPLIANT",
            },
            "categories": {
                cat_id: {
                    "mean_score": round(stats["mean_score"], 3) if stats.get("mean_score") is not None else None,
                    "pass_rate": round(stats["pass_rate"], 3) if stats.get("pass_rate") is not None else None,
                    "passed_items": stats["passed_items"],
                    "failed_items": stats["failed_items"],
                    "total_items": stats["total_items"],
                    "unassessed_items": stats.get("unassessed_items", 0),
                    "verdict": (
                        "N/A"
                        if stats.get("pass_rate") is None
                        else ("PASS" if stats["pass_rate"] >= 0.7 else "FAIL")
                    ),
                }
                for cat_id, stats in self.results.summary.items()
            },
            "failed_items": [
                {
                    "input": item.input[:200],
                    "output": item.model_output[:200],
                    "failed_scores": {
                        name: {
                            "score": round(result.score, 3),
                            "explanation": result.explanation,
                        }
                        for name, result in item.scores.items()
                        if not result.passed
                    },
                }
                for item in self.results.items
                if any(not r.passed for r in item.scores.values())
            ],
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """Generate JSON report. Optionally write to file."""
        report = self.to_dict()
        json_str = json.dumps(report, indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def to_summary(self) -> str:
        """Generate a human-readable text summary."""
        r = self.results
        lines = [
            f"{'='*60}",
            f"RAI COMPLIANCE REPORT: {r.name}",
            f"{'='*60}",
            f"Model: {r.model_name}",
            f"Framework: {r.profile.framework.value}",
            f"Industry: {r.profile.industry or 'General'}",
            f"Dataset: {len(r.items)} items",
            f"Timestamp: {r.timestamp}",
            f"",
            f"OVERALL: {r.overall_score:.1%} {'PASS' if r.overall_passed else 'FAIL'}",
            f"{'-'*60}",
        ]

        for cat_id, stats in r.summary.items():
            if stats.get("pass_rate") is None:
                lines.append(
                    f"  {cat_id}: N/A "
                    f"({stats.get('unassessed_items', 0)} un-assessed) "
                    f"[N/A]"
                )
                continue
            status = "PASS" if stats["pass_rate"] >= 0.7 else "FAIL"
            lines.append(
                f"  {cat_id}: {stats['mean_score']:.1%} "
                f"({stats['passed_items']}/{stats['total_items']} passed) "
                f"[{status}]"
            )

        failed_count = sum(
            1 for item in r.items
            if any(not result.passed for result in item.scores.values())
        )
        if failed_count:
            lines.append(f"")
            lines.append(f"Items with failures: {failed_count}/{len(r.items)}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)
