# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Optional Weave tracing shim.

The core toolkit stays platform-agnostic: nothing in `rai_toolkit/` imports
`weave` at module load. This shim gives us a single integration point that is
a no-op when tracing is disabled (or when `weave` isn't installed) and fans
out to `weave.op` when the user opts in via `Assessor(weave_project=...)` or
`rai assess --weave-project ...`.

Design goals:
- Zero overhead when disabled (decorator returns the function unchanged on
  first call, so subsequent calls skip the check entirely).
- Graceful degradation when `weave` isn't installed: we log once and keep
  going with a no-op.
- Per-call UI URLs available inside traced functions via
  :func:`current_call_url`, so the assessment report can deep-link back to
  each evaluation item / red-team run.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ENABLED: bool = False
_CLIENT: Any = None
_PROJECT: str | None = None

# Renderer registry: integrations populate this so the toolkit can fire a
# generic ``publish_view(name, payload)`` without knowing the backend (Weave,
# future tracing UIs, etc.). Each renderer takes the toolkit-side payload
# (e.g. an ``AssessmentResult``) and returns the rendered string body.
_VIEW_RENDERERS: dict[str, Callable[[Any], str]] = {}
_VIEW_PUBLISHER: Callable[[str, str, str], None] | None = None

# Per-op extensions integrations can register (e.g. ``call_display_name``,
# ``postprocess_output``). Keyed by the op name passed to :func:`traced`.
# Lets the toolkit declare ``@traced(name="rai.assessment", kind="agent")``
# and stay free of any weave-specific kwargs while integrations layer on
# UI-only concerns.
_OP_EXTENSIONS: dict[str, dict[str, Any]] = {}


def is_enabled() -> bool:
    return _ENABLED


def project_name() -> str | None:
    return _PROJECT


def init_tracing(project: str, entity: str | None = None) -> Any | None:
    """Initialise Weave tracing for the current process.

    Returns the Weave client handle, or ``None`` if Weave is unavailable.
    Safe to call multiple times; subsequent calls become no-ops unless the
    project changes.
    """
    global _ENABLED, _CLIENT, _PROJECT

    try:
        import weave
    except ImportError:
        logger.warning(
            "Weave tracing requested but `weave` is not installed. "
            "Install with: pip install 'rai-toolkit[weave]'. "
            "Continuing without tracing."
        )
        return None

    full = f"{entity}/{project}" if entity else project
    if _ENABLED and _PROJECT == full:
        return _CLIENT

    try:
        _CLIENT = weave.init(full)
    except Exception as e:
        logger.warning("weave.init(%s) failed: %s. Continuing without tracing.", full, e)
        return None

    _ENABLED = True
    _PROJECT = full
    logger.info("Weave tracing initialised for project %s", full)

    # Eagerly import the Weave integration so its renderers/publishers are
    # registered before any traced op fires ``publish_view``. Soft import:
    # if the integration package isn't on the path (slim install), the
    # toolkit still works; view publishing just becomes a no-op.
    try:
        import integrations.weave_integration  # noqa: F401
    except ImportError as e:
        logger.debug("weave_integration not importable; view publishing disabled: %s", e)

    return _CLIENT


def traced(
    name: str | None = None,
    kind: str | None = None,
    call_display_name: Callable[[Any], str] | None = None,
) -> Callable[[F], F]:
    """Decorator that wraps a function with `weave.op` at call time.

    Applied eagerly (at class definition), but the wrapping only happens on
    the first invocation after tracing is enabled. When tracing stays off the
    wrapper is a transparent pass-through.

    ``call_display_name`` is exposed here for toolkit-side per-call labels
    (e.g. picking the active judge's name for ``rai.judge``). UI-only
    concerns whose implementation lives outside the toolkit (rendering,
    backend-specific kwargs) should instead be layered on by integrations
    via :func:`register_op_extensions` keyed on the op's ``name``.
    """

    def decorator(fn: F) -> F:
        op_cache: dict[str, Any] = {}

        def _get_op() -> Any | None:
            if not _ENABLED:
                return None
            op = op_cache.get("op")
            if op is not None:
                return op
            try:
                import weave

                kwargs: dict[str, Any] = {}
                if name:
                    kwargs["name"] = name
                if kind is not None:
                    kwargs["kind"] = kind
                if call_display_name is not None:
                    kwargs["call_display_name"] = call_display_name
                # Merge integration-supplied extras (postprocess_output,
                # color, …). Looked up by op name so the toolkit never
                # needs to know which extras exist. Integration entries
                # win over the decorator's defaults.
                if name and name in _OP_EXTENSIONS:
                    kwargs.update(_OP_EXTENSIONS[name])
                op = weave.op(fn, **kwargs) if kwargs else weave.op(fn)
            except Exception as e:
                logger.debug("Failed to wrap %s with weave.op: %s", fn.__qualname__, e)
                return None
            op_cache["op"] = op
            return op

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                op = _get_op()
                if op is None:
                    return await fn(*args, **kwargs)
                return await op(*args, **kwargs)

            # Sentinel so callers (e.g. ``BaseModel.__init_subclass__``) can
            # detect "this function is already a tracing wrapper" and avoid
            # double-wrapping. Stores the registered op name to make double-
            # wraps debuggable rather than silent.
            async_wrapper.__rai_traced_op_name__ = name  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            op = _get_op()
            if op is None:
                return fn(*args, **kwargs)
            return op(*args, **kwargs)

        sync_wrapper.__rai_traced_op_name__ = name  # type: ignore[attr-defined]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def register_op_extensions(op_name: str, **op_kwargs: Any) -> None:
    """Layer extra ``weave.op`` kwargs onto a toolkit-defined op.

    Integrations call this at import time for UI-only concerns such as
    ``call_display_name``, ``postprocess_output``, or ``color``. Merged into
    the op kwargs the next time :func:`traced` lazily creates the op.

    Calling twice for the same ``op_name`` is additive; later registrations
    overwrite individual keys but don't drop previously-registered ones.
    """
    _OP_EXTENSIONS.setdefault(op_name, {}).update(op_kwargs)


def register_view_renderer(name: str, fn: Callable[[Any], str]) -> None:
    """Integrations call this to register a view renderer for ``name``.

    """
    _VIEW_RENDERERS[name] = fn


def register_view_publisher(fn: Callable[[str, str, str], None]) -> None:
    """Register the function that actually publishes a rendered view.

    """
    global _VIEW_PUBLISHER
    _VIEW_PUBLISHER = fn


def publish_view(name: str, payload: Any, mimetype: str = "text/html") -> None:
    """Render and publish a view on the current op call.

    No-op when tracing is disabled, when no renderer is registered for
    ``name``, or when no publisher has been wired up.
    """
    if not _ENABLED:
        return
    renderer = _VIEW_RENDERERS.get(name)
    publisher = _VIEW_PUBLISHER
    if renderer is None or publisher is None:
        return
    try:
        body = renderer(payload)
        publisher(name, body, mimetype)
    except Exception as e:  # pragma: no cover, view publishing is cosmetic
        logger.debug("publish_view(%s) failed: %s", name, e)


def current_call_url() -> str | None:
    """Return the UI URL of the currently-executing weave op call.

    """
    if not _ENABLED:
        return None
    try:
        import weave

        call = weave.require_current_call()
        return getattr(call, "ui_url", None)
    except Exception:
        return None


def current_call_id() -> str | None:
    """Return the ID of the currently-executing weave op call.

    The ID is the stable handle integrations use to attach feedback
    (annotations, manual findings, reviewer decisions) back to the same
    trace after the op has returned. Stored on
    ``AssessmentResult.weave_call_id`` so downstream review actions
    can re-target the assessment call without depending on Weave imports in
    the toolkit.
    """
    if not _ENABLED:
        return None
    try:
        import weave

        call = weave.require_current_call()
        return getattr(call, "id", None)
    except Exception:
        return None
