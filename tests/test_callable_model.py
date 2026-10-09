# SPDX-FileCopyrightText: 2026 M4h1m4
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import asyncio
import functools
import threading
from typing import Any

import pytest

from rai_toolkit.models import CallableModel, ModelResponse


def _run(model: CallableModel, *args: Any, **kwargs: Any) -> ModelResponse:
    return asyncio.run(model.predict(*args, **kwargs))


# --- the callable receives what it was promised -----------------------------


def test_a_sync_callable_receives_input_context_and_keywords() -> None:
    seen: dict[str, Any] = {}

    def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> str:
        seen.update(
            input_text=input_text,
            context=context,
            kwargs=kwargs,
            thread=threading.current_thread().name,
        )
        return "ok"

    result = _run(CallableModel(predict_fn), "q", context="ctx", temperature=0.5)

    assert result.output == "ok"
    assert seen["input_text"] == "q"
    assert seen["context"] == "ctx"
    assert seen["kwargs"] == {"temperature": 0.5}


def test_a_sync_callable_runs_off_the_event_loop() -> None:
    # A synchronous function doing real work would otherwise hold the loop for
    # its whole duration and stall every other task in the process.
    seen: dict[str, Any] = {}

    def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> str:
        seen["thread"] = threading.current_thread()
        return "ok"

    async def main() -> None:
        seen["loop_thread"] = threading.current_thread()
        await CallableModel(predict_fn).predict("q")

    asyncio.run(main())

    assert seen["thread"] is not seen["loop_thread"]


def test_an_async_callable_receives_and_returns_the_same_values() -> None:
    seen: dict[str, Any] = {}

    async def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> str:
        seen.update(input_text=input_text, context=context, kwargs=kwargs)
        return "async ok"

    result = _run(CallableModel(predict_fn), "q", context="ctx", top_p=0.9)

    assert result.output == "async ok"
    assert seen == {"input_text": "q", "context": "ctx", "kwargs": {"top_p": 0.9}}


def test_context_is_forwarded_even_when_empty() -> None:
    # Always forwarded rather than omitted when blank: a callable that declares
    # context must not see a different call shape depending on the row.
    seen: dict[str, Any] = {}

    def predict_fn(input_text: str, context: str = "sentinel", **kwargs: Any) -> str:
        seen["context"] = context
        return "ok"

    _run(CallableModel(predict_fn), "q")

    assert seen["context"] == ""


def test_a_signature_that_does_not_accept_context_is_not_adapted() -> None:
    # The signature is never inspected, so the mismatch surfaces as Python's
    # own TypeError. Guessing would mean silently dropping arguments the user
    # asked to pass.
    def predict_fn(input_text: str) -> str:
        return "ok"

    with pytest.raises(TypeError):
        _run(CallableModel(predict_fn), "q")


# --- every callable shape ---------------------------------------------------


def test_a_partial_is_treated_as_the_function_it_wraps() -> None:
    async def predict_fn(prefix: str, input_text: str, context: str = "") -> str:
        return f"{prefix}:{input_text}"

    model = CallableModel(functools.partial(predict_fn, "p"))

    assert _run(model, "q").output == "p:q"


def test_a_callable_object_is_accepted() -> None:
    class Predictor:
        def __call__(self, input_text: str, context: str = "", **kwargs: Any) -> str:
            return f"obj:{input_text}"

    assert _run(CallableModel(Predictor()), "q").output == "obj:q"


def test_a_sync_wrapper_returning_an_awaitable_is_awaited() -> None:
    # inspect.iscoroutinefunction reports False for this, so the awaitable is
    # only discovered from what came back.
    async def inner(input_text: str) -> str:
        return f"inner:{input_text}"

    def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> Any:
        return inner(input_text)

    assert _run(CallableModel(predict_fn), "q").output == "inner:q"


def test_an_async_callable_object_is_awaited() -> None:
    class Predictor:
        async def __call__(
            self, input_text: str, context: str = "", **kwargs: Any
        ) -> str:
            return f"aobj:{input_text}"

    assert _run(CallableModel(Predictor()), "q").output == "aobj:q"


def test_the_blocking_part_of_an_awaitable_factory_runs_off_the_loop() -> None:
    # to_thread happens first and the awaitable is awaited afterwards, so work
    # done before the coroutine is built does not block the loop.
    seen: dict[str, Any] = {}

    async def inner() -> str:
        return "ok"

    def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> Any:
        seen["factory_thread"] = threading.current_thread()
        return inner()

    async def main() -> None:
        seen["loop_thread"] = threading.current_thread()
        await CallableModel(predict_fn).predict("q")

    asyncio.run(main())

    assert seen["factory_thread"] is not seen["loop_thread"]


# --- result normalization ---------------------------------------------------


def test_a_string_becomes_a_response_with_empty_metadata() -> None:
    # Provider, token counts and latency are not knowable from a string, and
    # inventing them would put unsourced numbers into an assessment report.
    result = _run(CallableModel(lambda t, context="", **kw: "text"), "q")

    assert isinstance(result, ModelResponse)
    assert result.output == "text"
    assert result.metadata == {}


def test_a_returned_response_is_preserved_unchanged() -> None:
    response = ModelResponse(output="out", metadata={"model": "custom", "tokens": 7})

    result = _run(CallableModel(lambda t, context="", **kw: response), "q")

    assert result is response
    assert result.metadata == {"model": "custom", "tokens": 7}


@pytest.mark.parametrize(
    "value", [None, {"output": "x"}, b"bytes", 42, ["a"], ("a",), 1.5, True]
)
def test_an_unsupported_result_raises_naming_both_types(value: Any) -> None:
    model = CallableModel(lambda t, context="", **kw: value)

    with pytest.raises(TypeError) as excinfo:
        _run(model, "q")

    message = str(excinfo.value)
    assert "str or ModelResponse" in message
    assert type(value).__name__ in message


# --- failures stay the caller's ---------------------------------------------


def test_a_sync_exception_propagates_unwrapped() -> None:
    def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> str:
        raise ValueError("upstream failure")

    with pytest.raises(ValueError, match="upstream failure"):
        _run(CallableModel(predict_fn), "q")


def test_an_async_exception_propagates_unwrapped() -> None:
    async def predict_fn(input_text: str, context: str = "", **kwargs: Any) -> str:
        raise RuntimeError("async upstream failure")

    with pytest.raises(RuntimeError, match="async upstream failure"):
        _run(CallableModel(predict_fn), "q")


@pytest.mark.parametrize("value", [None, "not callable", 42, object()])
def test_a_non_callable_is_rejected_at_construction(value: Any) -> None:
    # Rejected when the model is built, not when a row is scored, so the error
    # points at the line that made the mistake.
    with pytest.raises(TypeError, match="must be callable"):
        CallableModel(value)


# --- naming and tracing -----------------------------------------------------


def test_an_explicit_name_is_used() -> None:
    model = CallableModel(lambda t, context="", **kw: "x", name="internal-sdk-v2")

    assert model.name == "internal-sdk-v2"


def test_the_default_name_falls_back_to_the_class() -> None:
    assert CallableModel(lambda t, context="", **kw: "x").name == "CallableModel"


def test_predict_carries_the_inherited_trace_exactly_once() -> None:
    # BaseModel.__init_subclass__ wraps predict already. Adding a second
    # decorator here would take the op name over and double the span.
    assert (
        getattr(CallableModel.predict, "__rai_traced_op_name__", None)
        == "rai.model.predict"
    )


def test_the_adapter_adds_no_import_time_dependency() -> None:
    import rai_toolkit.models.callable as module

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for third_party in ("import openai", "import anthropic", "import httpx"):
        assert third_party not in text
