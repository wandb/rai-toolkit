# SPDX-FileCopyrightText: 2026 Sonike
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json

import pytest

from rai_toolkit.evaluation.datasets import DatasetLoader


@pytest.mark.parametrize("wrapped", [False, True], ids=["array", "data-wrapper"])
@pytest.mark.parametrize("valid_rows", [0, 1], ids=["first-row", "after-valid-row"])
@pytest.mark.parametrize("bad_row", [42, 1.5, "private-dataset-content", True, [], None])
def test_json_non_object_rows_report_path_and_array_position(
    tmp_path, wrapped, valid_rows, bad_row
):
    path = tmp_path / "rows.json"
    rows = [{"prompt": "private-dataset-content"}] * valid_rows + [bad_row]
    data = {"data": rows} if wrapped else rows
    # Pretty printing makes source line numbers differ from array positions.
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert f"row {valid_rows + 1}:" in message
    assert "dataset row must be a JSON object" in message
    assert "line " not in message
    assert "private-dataset-content" not in message


@pytest.mark.parametrize(
    "data",
    [
        None,
        42,
        "private-dataset-content",
        True,
        {},
        {"prompt": "private-dataset-content"},
        {"data": None},
        {"data": 42},
        {"data": "private-dataset-content"},
        {"data": True},
        {"data": {}},
        {"data": {"prompt": "private-dataset-content"}},
    ],
)
def test_json_invalid_top_level_shape_reports_path_and_accepted_shape(tmp_path, data):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "array of objects" in message
    assert "'data' array" in message
    assert "private-dataset-content" not in message


@pytest.mark.parametrize("wrapped", [False, True], ids=["array", "data-wrapper"])
def test_json_preserves_valid_behavior(tmp_path, wrapped):
    path = tmp_path / "rows.json"
    rows = [
        {
            "QUESTION": "你好",
            "ANSWER": "héllo",
            "RETRIEVED_CONTEXT": "context",
            "score": 3,
            "enabled": True,
            "optional": None,
            "policy_expectations": {"allowed": True},
        },
        {"query": "q", "ground_truth": "a", "source": "s"},
        {
            "prompt": "p",
            "reference": "r",
            "documents": ["a", "b"],
            "policy_expectations": [{"allowed": False}],
        },
        {"USER_INPUT": "u", "expected_output": "a"},
        {},
    ]
    data = {"data": rows, "metadata": "ignored"} if wrapped else rows
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert DatasetLoader.from_file(path) == [
        {
            "input": "你好",
            "expected": "héllo",
            "context": "context",
            "score": "3",
            "enabled": "True",
            "optional": "",
            "policy_expectations": {"allowed": True},
        },
        {"input": "q", "expected": "a", "context": "s"},
        {
            "input": "p",
            "expected": "r",
            "context": "['a', 'b']",
            "policy_expectations": [{"allowed": False}],
        },
        {"input": "u", "expected": "a"},
        {},
    ]


@pytest.mark.parametrize("data", [[], {"data": []}])
def test_json_empty_arrays_return_empty_list(tmp_path, data):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert DatasetLoader.from_file(path) == []
