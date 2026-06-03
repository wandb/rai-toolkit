# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Contracts for multi-turn and agentic evaluation (scaffolding).

The core :class:`rai_toolkit.models.base.BaseModel` API is single-turn
``predict(input_text, context=...) -> ModelResponse``. Production systems are
increasingly multi-turn agents with tools. This module defines **narrow**
protocols so future eval harnesses (Weave traces, Inspect, custom simulators)
can plug in without rewriting the assessment pipeline.

Full orchestration (tool mocks, rollouts, parallel agents) is **not**
implemented here — only types and documentation of the intended seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TurnSpec:
    """One step in a scripted multi-turn dialogue."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticEvalSpec:
    """Description of an agentic scenario to replay or simulate.

    Attributes:
        name: Stable id for reports.
        turns: Ordered messages forming the scenario (may include tool results).
        success_criteria: Optional structured checks (e.g. must-call-tool names).
    """

    name: str
    turns: list[TurnSpec]
    success_criteria: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MultiTurnAgent(Protocol):
    """Agent that consumes a transcript and produces the next assistant message."""

    name: str

    async def step(
        self,
        transcript: list[TurnSpec],
        **kwargs: Any,
    ) -> str:
        """Return the assistant's next message given prior turns."""
        ...


@runtime_checkable
class StreamingAgent(Protocol):
    """Optional: agent that streams tokens (for latency / partial-output guards)."""

    name: str

    async def stream(self, input_text: str, context: str = "", **kwargs: Any) -> Any:
        """Yield or return an async iterator of token chunks; implementation-defined."""
        ...
