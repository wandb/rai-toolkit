# SPDX-FileCopyrightText: 2026 Sanath
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json
from typing import Any

import pytest

import rai_toolkit.examples.registry as example_registry


def test_normalize_row_preserves_falsy_values_without_mutating_input() -> None:
    row: dict[str, Any] = {
        "input_text": 0,
        "question": "fallback question",
        "context": False,
        "passage": "fallback passage",
        "expected": False,
        "answer": "fallback answer",
        "category": 0,
    }
    original = dict(row)

    normalized = example_registry._normalize_row(row, default_category="MIT-3.1")

    assert normalized == {
        "input_text": "0",
        "context": "False",
        "expected": "False",
        "category": "0",
    }
    assert row == original


def test_normalize_row_falls_back_for_missing_none_and_empty_values() -> None:
    row = {
        "input_text": "",
        "question": "fallback question",
        "context": None,
        "passage": "fallback passage",
        "answer": "fallback answer",
        "category": None,
    }

    normalized = example_registry._normalize_row(row, default_category="MIT-3.1")

    assert normalized == {
        "input_text": "fallback question",
        "context": "fallback passage",
        "expected": "fallback answer",
        "category": "MIT-3.1",
    }


def test_load_zero_returns_no_rows_without_invoking_loader(monkeypatch) -> None:
    calls: list[int] = []
    descriptor = example_registry.ExampleDescriptor(
        slug="test-loader",
        name="Test loader",
        description="Test fixture",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="Test fixture",
        loader=lambda limit: calls.append(limit) or [],
    )
    monkeypatch.setitem(example_registry.EXAMPLE_CATALOG, descriptor.slug, descriptor)

    assert example_registry.ExampleRegistry.load(descriptor.slug, limit=0) == []
    assert calls == []


@pytest.mark.parametrize("limit", [-1, True, False])
def test_load_rejects_negative_and_boolean_limits(monkeypatch, limit: object) -> None:
    descriptor = example_registry.ExampleDescriptor(
        slug="test-loader",
        name="Test loader",
        description="Test fixture",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="Test fixture",
        loader=lambda _: [],
    )
    monkeypatch.setitem(example_registry.EXAMPLE_CATALOG, descriptor.slug, descriptor)

    with pytest.raises(ValueError, match="non-negative integer"):
        example_registry.ExampleRegistry.load(descriptor.slug, limit=limit)  # type: ignore[arg-type]


def test_load_checks_limits_before_the_example_file_backend(monkeypatch, tmp_path) -> None:
    filename = "example.json"
    (tmp_path / filename).write_text(
        json.dumps([{"prompt": "File-backed prompt", "answer": "File-backed answer"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(example_registry, "_EXAMPLES_DIR", tmp_path)
    descriptor = example_registry.ExampleDescriptor(
        slug="file-backed",
        name="File-backed fixture",
        description="Test fixture",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="Test fixture",
        example_file=filename,
    )
    monkeypatch.setitem(example_registry.EXAMPLE_CATALOG, descriptor.slug, descriptor)

    assert example_registry.ExampleRegistry.load(descriptor.slug, limit=1) == [
        {
            "input_text": "File-backed prompt",
            "context": "",
            "expected": "File-backed answer",
            "category": "MIT-3.1",
        }
    ]

    def fail_if_file_is_read(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("the file backend should not run")

    monkeypatch.setattr(example_registry, "_load_example_file", fail_if_file_is_read)

    assert example_registry.ExampleRegistry.load(descriptor.slug, limit=0) == []
    for invalid_limit in (-1, True, False):
        with pytest.raises(ValueError, match="non-negative integer"):
            example_registry.ExampleRegistry.load(descriptor.slug, limit=invalid_limit)


def test_load_checks_limits_before_the_generic_huggingface_backend(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
        calls.append((args, kwargs))
        return [{"question": "HF prompt", "answer": "HF answer"}]

    monkeypatch.setattr(example_registry, "_require_datasets", lambda: load_dataset)
    descriptor = example_registry.ExampleDescriptor(
        slug="generic-huggingface",
        name="Generic Hugging Face fixture",
        description="Test fixture",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="Test fixture",
        huggingface_path="example/dataset:config",
    )
    monkeypatch.setitem(example_registry.EXAMPLE_CATALOG, descriptor.slug, descriptor)

    assert example_registry.ExampleRegistry.load(descriptor.slug, limit=1) == [
        {
            "input_text": "HF prompt",
            "context": "",
            "expected": "HF answer",
            "category": "MIT-3.1",
        }
    ]
    assert calls == [
        (("example/dataset", "config"), {"split": "train", "streaming": True})
    ]

    calls.clear()
    assert example_registry.ExampleRegistry.load(descriptor.slug, limit=0) == []
    for invalid_limit in (-1, True, False):
        with pytest.raises(ValueError, match="non-negative integer"):
            example_registry.ExampleRegistry.load(descriptor.slug, limit=invalid_limit)
    assert calls == []


def test_load_checks_limits_before_a_registered_streaming_loader(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append((args, kwargs))
        return [
            {
                "question": "Streaming prompt",
                "best_answer": "Streaming answer",
                "correct_answers": ["Streaming answer"],
            }
        ]

    monkeypatch.setattr(example_registry, "_require_datasets", lambda: load_dataset)

    assert example_registry.ExampleRegistry.load("truthfulqa-gen", limit=1) == [
        {
            "input_text": "Streaming prompt",
            "context": "Accepted truthful answers:\n- Streaming answer",
            "expected": "Streaming answer",
            "category": "MIT-3.1",
        }
    ]
    assert calls == [
        (
            ("truthfulqa/truthful_qa", "generation"),
            {"split": "validation", "streaming": True},
        )
    ]

    calls.clear()
    assert example_registry.ExampleRegistry.load("truthfulqa-gen", limit=0) == []
    for invalid_limit in (-1, True, False):
        with pytest.raises(ValueError, match="non-negative integer"):
            example_registry.ExampleRegistry.load("truthfulqa-gen", limit=invalid_limit)
    assert calls == []
