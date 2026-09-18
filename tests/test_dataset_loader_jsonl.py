# SPDX-FileCopyrightText: 2026 Arian Bozorgzad
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json

import pytest

from rai_toolkit.evaluation.datasets import DatasetLoader


def test_jsonl_invalid_json_reports_path_line_and_preserves_cause(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '\n{"prompt":"valid row"}\n\n{"prompt": \n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "line 4" in message
    assert "invalid JSON" in message
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("bad_row", ["[]", '"text"', "1", "true", "null"])
def test_jsonl_non_object_rows_raise_clear_error(tmp_path, bad_row):
    path = tmp_path / "rows.jsonl"
    path.write_text(f'\n{{"prompt":"valid row"}}\n\n{bad_row}\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "line 4" in message
    assert "dataset row must be a JSON object" in message


def test_jsonl_preserves_valid_behavior(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '\n'
        '{"prompt":"héllo","answer":"world","policy_expectations":{"allowed":true}}\n'
        '{}\n'
        '\n',
        encoding="utf-8",
    )

    rows = DatasetLoader.from_file(path)

    assert rows == [
        {
            "input": "héllo",
            "expected": "world",
            "policy_expectations": {"allowed": True},
        },
        {},
    ]


def test_jsonl_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("", encoding="utf-8")

    assert DatasetLoader.from_file(path) == []
