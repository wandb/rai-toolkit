# SPDX-FileCopyrightText: 2026 M4h1m4
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Vendor-neutral adapter wrapping a user-supplied inference function."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from rai_toolkit.models.base import BaseModel, ModelResponse

__all__ = ["CallableModel"]


class CallableModel(BaseModel):
    """Evaluate anything callable without writing a :class:`BaseModel` subclass.

    An in-process model, an internal SDK, or a proprietary endpoint becomes
    gradeable by handing over the function that calls it. No provider
    dependency is added, and no adapter has to be written per platform.

    Example::

        def my_model(prompt: str, context: str = "", **kwargs: Any) -> str:
            return internal_sdk.complete(prompt)

        model = CallableModel(my_model)

    The callable is always invoked as
    ``predict_fn(input_text, context=context, **kwargs)``. Its signature is
    never inspected, so a function shaped differently is adapted by the caller
    rather than guessed at here::

        model = CallableModel(lambda text, context="", **kw: legacy(text))

    Guessing would mean deciding which arguments a user "really" wanted, and
    silently dropping the rest. Passing everything and letting Python raise
    keeps that decision with the person who wrote the function.
    """

    def __init__(
        self,
        predict_fn: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> None:
        # Rejected here rather than at the first call: the failure belongs to
        # the line that built the model, not to an evaluation run started
        # later, where it would surface as a row error instead of a setup one.
        if not callable(predict_fn):
            raise TypeError(
                "predict_fn must be callable, got "
                f"{type(predict_fn).__name__}"
            )
        super().__init__(name)
        self._predict_fn = predict_fn

    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        """Call the wrapped function and normalize what it returns.

        Native coroutine functions are awaited directly. Everything else runs
        on a worker thread, because a synchronous function doing real work -
        a network round trip, a local model - would otherwise hold the event
        loop for its whole duration and stall every other task in the process.

        A synchronous function may still return an awaitable. The thread hop
        happens first and the result is awaited afterwards, so the blocking
        part runs off the loop either way; awaiting on the loop directly would
        leave a blocking factory blocking it. Building a coroutine does not
        run it and coroutines are not bound to a thread, so one created on the
        worker is safe to await here.
        """
        if inspect.iscoroutinefunction(self._predict_fn):
            result = await self._predict_fn(input_text, context=context, **kwargs)
        else:
            result = await asyncio.to_thread(
                self._predict_fn, input_text, context=context, **kwargs
            )
            # Reached by a callable object whose ``__call__`` is async, and by
            # any synchronous wrapper around an async client:
            # ``iscoroutinefunction`` reports False for both.
            if inspect.isawaitable(result):
                result = await result

        # Checked before ``str``: ModelResponse is not a string, but ordering
        # the branches this way keeps the richer type from ever depending on
        # the narrower check.
        if isinstance(result, ModelResponse):
            return result
        if isinstance(result, str):
            # Metadata stays empty rather than being inferred. Provider, token
            # counts and latency are not knowable from a string, and inventing
            # them would put unsourced numbers into an assessment report.
            return ModelResponse(output=result)
        raise TypeError(
            f"{self.name} expected the callable to return str or ModelResponse, "
            f"got {type(result).__name__}"
        )
