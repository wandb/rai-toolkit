# SPDX-FileCopyrightText: 2026 KodYazicam
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json
from pathlib import Path

import pytest

from rai_toolkit.evaluation.datasets import DatasetLoader


def _write_jsonl(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_invalid_json_reports_file_and_physical_line_with_chained_cause(
    tmp_path,
) -> None:
    path = _write_jsonl(
        tmp_path / "rows.jsonl",
        '\n{"prompt":"valid row"}\n\n{"prompt": \n',
    )

    with pytest.raises(ValueError) as excinfo:
        DatasetLoader.from_file(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 4" in message
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "row",
    ["[]", '["item"]', '"raw-secret-contents"', "42", "true", "null", "3.5"],
)
def test_non_object_row_raises_value_error_with_location(
    tmp_path, row: str
) -> None:
    path = _write_jsonl(
        tmp_path / "rows.jsonl",
        '{"prompt":"valid row"}\n\n' + row + "\n",
    )

    with pytest.raises(ValueError) as excinfo:
        DatasetLoader.from_file(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 3" in message
    assert "JSON object" in message
    assert "raw-secret-contents" not in message


def test_valid_rows_keep_order_and_skip_blank_lines(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path / "rows.jsonl",
        '{"prompt":"first","answer":"one"}\n'
        "\n"
        "  \n"
        '{"question":"second","ground_truth":"two"}\n',
    )

    assert DatasetLoader.from_file(path) == [
        {"input": "first", "expected": "one"},
        {"input": "second", "expected": "two"},
    ]


def test_utf8_text_and_structured_policy_expectations_are_preserved(
    tmp_path,
) -> None:
    path = _write_jsonl(
        tmp_path / "rows.jsonl",
        json.dumps(
            {"prompt": "İstanbul'da giriş ücretli mi?", "policy_expectations": {"max_latency": 1.5}},
            ensure_ascii=False,
        )
        + "\n",
    )

    rows = DatasetLoader.from_file(path)

    assert rows[0]["input"] == "İstanbul'da giriş ücretli mi?"
    assert rows[0]["policy_expectations"] == {"max_latency": 1.5}


def test_empty_object_row_is_valid(tmp_path) -> None:
    path = _write_jsonl(tmp_path / "rows.jsonl", "{}\n")

    assert DatasetLoader.from_file(path) == [{}]


def test_empty_file_returns_empty_list(tmp_path) -> None:
    path = _write_jsonl(tmp_path / "rows.jsonl", "")

    assert DatasetLoader.from_file(path) == []


def test_whitespace_only_file_returns_empty_list(tmp_path) -> None:
    path = _write_jsonl(tmp_path / "rows.jsonl", "\n \n\n")

    assert DatasetLoader.from_file(path) == []
