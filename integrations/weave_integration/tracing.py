# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weave tracing integration — @weave.op wrapping and initialization."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

import weave

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def weave_init(project: str, entity: str | None = None) -> weave.WeaveClient:
    """Initialize Weave for the project.

    Args:
        project: Weave project name.
        entity: W&B entity (team or user). If None, uses default.

    Returns:
        Weave client instance.
    """
    full_name = f"{entity}/{project}" if entity else project
    client = weave.init(full_name)
    logger.info("Weave initialized: %s", full_name)
    return client


def traced(
    name: str | None = None,
    kind: str | None = None,
    call_display_name: Any = None,
) -> Callable[[F], F]:
    """Decorator to trace a function with @weave.op.

    Wraps any function (including rai_toolkit toolkit functions)
    with Weave tracing for full observability.

    Args:
        name: Custom op name in Weave UI (e.g. ``rai.assessment``).
        kind: Operation kind — Weave uses this to render an icon/color
            in the trace tree. Valid values include ``"llm"``, ``"tool"``,
            ``"agent"``, ``"scorer"``, ``"search"``.
        call_display_name: Optional per-call label. Either a string, or a
            callable ``(Call) -> str`` that returns a label dynamically per
            invocation (useful for showing the model/judge name in the row).

    Example::

        @traced(name="rai_evaluate", kind="tool")
        async def evaluate_model(model, dataset):
            ...
    """
    def decorator(func: F) -> F:
        op_kwargs: dict[str, Any] = {}
        if name:
            op_kwargs["name"] = name
        if kind:
            op_kwargs["kind"] = kind
        if call_display_name is not None:
            op_kwargs["call_display_name"] = call_display_name

        wrapped = weave.op(func, **op_kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return wrapped(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await wrapped(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
