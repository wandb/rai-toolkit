# SPDX-FileCopyrightText: 2026 Guillermo Lopez Veliz
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import pytest

from rai_toolkit.evaluation.datasets import DatasetLoader


@pytest.mark.parametrize("valid_rows", [0, 1], ids=["first-record", "after-valid-record"])
@pytest.mark.parametrize(
    "bad_record",
    [
        "private-question,private-answer,private-surplus",
        "private-question,private-answer,",
    ],
    ids=["non-empty-surplus", "empty-surplus"],
)
def test_csv_surplus_fields_report_path_and_record_number(
    tmp_path, valid_rows, bad_record
):
    path = tmp_path / "rows.csv"
    lines = ["question,answer"]
    lines.extend(["valid-question,valid-answer"] * valid_rows)
    lines.append(bad_record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert f"record {valid_rows + 1}" in message
    assert "more fields than header" in message
    assert "private-question" not in message
    assert "private-answer" not in message
    assert "private-surplus" not in message
    assert "valid-question" not in message


def test_csv_surplus_fields_after_multiline_and_blank_line_reports_record_number(
    tmp_path,
):
    path = tmp_path / "rows.csv"
    content = (
        "question,answer\n"
        '"Line 1\nLine 2\nLine 3","Multiline\nAnswer"\n'
        "\n"
        "private-question,private-answer,private-extra\n"
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        DatasetLoader.from_file(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "record 2" in message
    assert "line " not in message.lower()
    assert "more fields than header" in message
    assert "private-question" not in message
    assert "private-extra" not in message


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_csv_preserves_valid_behavior(tmp_path, newline):
    path = tmp_path / "rows.csv"
    content = (
        "QUESTION,ANSWER,RETRIEVED_CONTEXT,custom_col\n"
        f'"What is X, Y?",X,"First line{newline}Second line",val1\n'
        "\n"
        "¿Qué es IA?,Inteligencia Artificial,Contexto UTF-8 — test,val2\n"
        "Short row question\n"
    )
    path.write_bytes(content.encode("utf-8"))

    rows = DatasetLoader.from_file(path)

    assert rows == [
        {
            "input": "What is X, Y?",
            "expected": "X",
            "context": f"First line{newline}Second line",
            "custom_col": "val1",
        },
        {
            "input": "¿Qué es IA?",
            "expected": "Inteligencia Artificial",
            "context": "Contexto UTF-8 — test",
            "custom_col": "val2",
        },
        {
            "input": "Short row question",
            "expected": "",
            "context": "",
            "custom_col": "",
        },
    ]


@pytest.mark.parametrize(
    "content",
    ["", "question,answer\n", "question,answer\n\n"],
    ids=["empty-file", "header-only", "header-and-blank-line"],
)
def test_csv_empty_and_header_only_files_return_empty_list(tmp_path, content):
    path = tmp_path / "rows.csv"
    path.write_text(content, encoding="utf-8")

    assert DatasetLoader.from_file(path) == []


def test_csv_utf8_bom_does_not_break_header_normalization(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_bytes("\ufeffquestion,answer\nWhat is 1+1?,2\n".encode())

    assert DatasetLoader.from_file(path) == [{"input": "What is 1+1?", "expected": "2"}]
