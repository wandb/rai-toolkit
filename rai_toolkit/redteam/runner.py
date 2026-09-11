# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Attack orchestrator: runs a catalog of attacks against a model and reports."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.models.base import BaseModel
from rai_toolkit.redteam.attacks import (
    ATTACK_CATALOG,
    Attack,
    AttackCategory,
    AttackTemplate,
    filter_catalog,
)

logger = logging.getLogger(__name__)


class AttackOutcome(str, Enum):
    """Stable outcome values for an attempted attack."""

    SUCCEEDED = "succeeded"
    RESISTED = "resisted"
    UNASSESSED_ERROR = "unassessed_error"


def _attack_display_name(call: Any) -> str:
    """Per-call label for ``rai.redteam.attack``.

    Renders e.g. ``attack[jailbreak/dan-v11 sev=4]`` so the trace tree shows
    which attack family + template each child call ran.
    """
    try:
        template = (call.inputs or {}).get("template")
        if template is None:
            return "attack"
        cat = getattr(getattr(template, "category", None), "value", None) or "?"
        aid = getattr(template, "id", "?")
        sev = getattr(template, "severity", None)
        suffix = f" sev={sev}" if sev is not None else ""
        return f"attack[{cat}/{aid}{suffix}]"
    except Exception:  # pragma: no cover, display-name must never raise
        return "attack"


@dataclass
class AttackResult:
    """Outcome of a single attack.

    Attributes:
        attack_id: Stable identifier from the catalog.
        category: Attack family.
        succeeded: True if the model failed (attack worked).
        model_output: What the model actually said.
        prompt: The adversarial prompt sent.
        severity: 1-5 from the template.
        latency_ms: Round-trip time.
        error: Error message if the model call failed.
        weave_call_url: UI URL for the traced attack call, when Weave is enabled.
    """

    attack_id: str
    category: AttackCategory
    succeeded: bool
    model_output: str
    prompt: str
    severity: int
    latency_ms: float = 0.0
    error: str | None = None
    weave_call_url: str | None = None

    @property
    def assessed(self) -> bool:
        """Whether the model response could be assessed."""
        return self.error is None

    @property
    def outcome(self) -> AttackOutcome:
        """Return the three-state outcome, with execution errors taking priority."""
        if not self.assessed:
            return AttackOutcome.UNASSESSED_ERROR
        if self.succeeded:
            return AttackOutcome.SUCCEEDED
        return AttackOutcome.RESISTED


@dataclass
class FamilyStats:
    """Aggregate stats for one attack family."""

    category: AttackCategory
    total: int = 0
    successes: int = 0
    errors: int = 0

    @property
    def assessed(self) -> int:
        """Number of attacks that produced an assessable model response."""
        return self.total - self.errors

    @property
    def success_rate(self) -> float | None:
        """Successful attacks divided by assessed attacks."""
        return self.successes / self.assessed if self.assessed else None

    @property
    def resistance_rate(self) -> float | None:
        """Resisted attacks divided by assessed attacks."""
        success_rate = self.success_rate
        return 1.0 - success_rate if success_rate is not None else None


@dataclass
class RedTeamReport:
    """Full red-team assessment result."""

    model_name: str
    results: list[AttackResult]
    by_family: dict[AttackCategory, FamilyStats]
    total_duration_s: float
    generated_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def total_successes(self) -> int:
        return sum(1 for r in self.results if r.assessed and r.succeeded)

    @property
    def total_errors(self) -> int:
        """Number of attacks that could not be assessed due to an error."""
        return sum(1 for r in self.results if r.error is not None)

    @property
    def total_assessed(self) -> int:
        """Number of attacks that produced an assessable model response."""
        return self.total - self.total_errors

    @property
    def error_rate(self) -> float:
        """Execution errors divided by all attempted attacks."""
        return self.total_errors / self.total if self.total else 0.0

    @property
    def overall_success_rate(self) -> float | None:
        """Successful attacks divided by assessed attacks."""
        return self.total_successes / self.total_assessed if self.total_assessed else None

    @property
    def overall_resistance_rate(self) -> float | None:
        """Resisted attacks divided by assessed attacks."""
        success_rate = self.overall_success_rate
        return 1.0 - success_rate if success_rate is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "total_duration_s": self.total_duration_s,
            "total": self.total,
            "total_successes": self.total_successes,
            "total_errors": self.total_errors,
            "total_assessed": self.total_assessed,
            "error_rate": self.error_rate,
            "overall_success_rate": self.overall_success_rate,
            "overall_resistance_rate": self.overall_resistance_rate,
            "by_family": {
                cat.value: {
                    "total": s.total,
                    "assessed": s.assessed,
                    "successes": s.successes,
                    "errors": s.errors,
                    "success_rate": s.success_rate,
                    "resistance_rate": s.resistance_rate,
                }
                for cat, s in self.by_family.items()
            },
            "results": [
                {
                    "attack_id": r.attack_id,
                    "category": r.category.value,
                    "succeeded": r.succeeded,
                    "severity": r.severity,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "outcome": r.outcome.value,
                    "weave_call_url": r.weave_call_url,
                    "prompt_preview": r.prompt[:200],
                    "output_preview": (r.model_output or "")[:300],
                }
                for r in self.results
            ],
        }

    def format_summary(self) -> str:
        """Render a terminal-friendly summary."""
        success_rate = self.overall_success_rate
        resistance_rate = self.overall_resistance_rate
        if success_rate is None:
            success_text = "n/a (no attacks assessed)"
            resistance_text = "n/a (no attacks assessed)"
        else:
            success_text = (
                f"{success_rate:.1%} "
                f"({self.total_successes}/{self.total_assessed} assessed)"
            )
            resisted = self.total_assessed - self.total_successes
            resistance_text = (
                f"{resistance_rate:.1%} "
                f"({resisted}/{self.total_assessed} assessed)"
            )

        lines = [
            f"Red-Team Report: {self.model_name}",
            f"  Attacks run:            {self.total}",
            f"  Attacks assessed:       {self.total_assessed}/{self.total}",
            f"  Execution errors:       {self.total_errors}/{self.total} "
            f"({self.error_rate:.1%})",
            f"  Attack success rate:    {success_text}",
            f"  Model resistance rate:  {resistance_text}",
            f"  Duration:               {self.total_duration_s:.1f}s",
            "",
            "By category:",
        ]
        for cat in AttackCategory:
            stats = self.by_family.get(cat)
            if stats is None or stats.total == 0:
                continue
            if stats.success_rate is None:
                detail = (
                    f"n/a (no attacks assessed; {stats.errors}/{stats.total} errors)"
                )
            else:
                detail = (
                    f"{stats.successes}/{stats.assessed} assessed attacks succeeded "
                    f"({stats.success_rate:.0%}); {stats.errors}/{stats.total} errors"
                )
            lines.append(f"  {cat.value:20s}  {detail}")
        return "\n".join(lines)


class AttackRunner:
    """Runs a catalog of attacks against a BaseModel and aggregates results.

    Example::

        runner = AttackRunner(model, max_concurrency=8)
        report = await runner.run_all()
        print(report.format_summary())

    Filter to a specific attack family::

        runner = AttackRunner(
            model,
            categories=[AttackCategory.PROMPT_INJECTION, AttackCategory.JAILBREAK],
        )
    """

    def __init__(
        self,
        model: BaseModel,
        attacks: list[AttackTemplate] | None = None,
        categories: list[AttackCategory] | None = None,
        min_severity: int = 1,
        max_severity: int = 5,
        max_concurrency: int = 4,
    ) -> None:
        if attacks is None:
            attacks = filter_catalog(
                categories=categories,
                min_severity=min_severity,
                max_severity=max_severity,
            )
        if not attacks:
            raise ValueError("No attacks selected; check your filters.")

        self.model = model
        self.attacks = attacks
        self.max_concurrency = max_concurrency

    @_tracing.traced(name="rai.redteam", kind="agent")
    async def run_all(self) -> RedTeamReport:
        """Run every selected attack once against the model."""
        start = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(template: AttackTemplate) -> AttackResult:
            async with semaphore:
                return await self._execute(template)

        tasks = [_run_one(t) for t in self.attacks]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

        by_family = _aggregate(results)
        return RedTeamReport(
            model_name=self.model.name,
            results=results,
            by_family=by_family,
            total_duration_s=duration,
        )

    @_tracing.traced(
        name="rai.redteam.attack",
        kind="tool",
        call_display_name=lambda call: _attack_display_name(call),
    )
    async def _execute(self, template: AttackTemplate) -> AttackResult:
        t0 = time.perf_counter()
        try:
            response = await self.model.predict(input_text=template.template)
            latency = (time.perf_counter() - t0) * 1000
            output = response.output or ""
            succeeded = template.evaluate(output)
            return AttackResult(
                attack_id=template.id,
                category=template.category,
                succeeded=succeeded,
                model_output=output,
                prompt=template.template,
                severity=template.severity,
                latency_ms=latency,
                weave_call_url=_tracing.current_call_url(),
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning("Attack %s failed with exception: %s", template.id, e)
            return AttackResult(
                attack_id=template.id,
                category=template.category,
                succeeded=False,
                model_output="",
                prompt=template.template,
                severity=template.severity,
                latency_ms=latency,
                error=str(e),
                weave_call_url=_tracing.current_call_url(),
            )


def _aggregate(results: list[AttackResult]) -> dict[AttackCategory, FamilyStats]:
    stats: dict[AttackCategory, FamilyStats] = defaultdict(
        lambda: FamilyStats(category=AttackCategory.JAILBREAK)
    )
    for r in results:
        s = stats.setdefault(r.category, FamilyStats(category=r.category))
        s.total += 1
        if r.assessed and r.succeeded:
            s.successes += 1
        if r.error is not None:
            s.errors += 1
    return dict(stats)


__all__ = [
    "Attack",
    "AttackOutcome",
    "AttackResult",
    "AttackRunner",
    "FamilyStats",
    "RedTeamReport",
]
