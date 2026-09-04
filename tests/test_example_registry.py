# SPDX-FileCopyrightText: 2026 Sanath
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

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


@pytest.mark.parametrize("limit", [-1, True])
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
