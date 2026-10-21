# SPDX-FileCopyrightText: 2026 Sanath
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json
from pathlib import Path

import pytest

from rai_toolkit.cli import main


def test_datasets_list_reports_registered_datasets(capsys) -> None:
    exit_code = main(["datasets", "list"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "bbq" in output
    assert "halueval-qa" in output
    assert "MIT-1.1" in output


def test_demo_datasets_filters_to_requested_preset(capsys) -> None:
    exit_code = main(["datasets", "demo-datasets", "--preset", "hr"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "hr" in output
    assert "bbq" in output
    assert "bold" in output
    assert "healthcare" not in output


def test_redteam_catalog_reports_known_template(capsys) -> None:
    exit_code = main(["redteam", "catalog"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "jb-dan-classic" in output
    assert "jailbreak" in output


def test_policies_lint_accepts_valid_policy_directory(tmp_path: Path, capsys) -> None:
    (tmp_path / "valid.yaml").write_text(
        """\
name: Test policy set
version: "1.0.0"
policies:
  - name: block-secret
    description: Block secret output
    severity: high
    trigger:
      output_contains: [secret]
""",
        encoding="utf-8",
    )

    exit_code = main(["policies", "lint", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "OK: 1 policies loaded" in output
    assert "block-secret" in output


def test_policies_lint_rejects_invalid_policy_directory(tmp_path: Path, capsys) -> None:
    (tmp_path / "invalid.yaml").write_text(
        """\
name: Test policy set
policies:
  - name: invalid policy name
    description: Invalid policy
    trigger: {}
""",
        encoding="utf-8",
    )

    exit_code = main(["policies", "lint", str(tmp_path)])

    error = capsys.readouterr().err
    assert exit_code == 1
    assert error.startswith("FAIL:")


def test_datasets_parser_error_exits_with_code_two(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["datasets", "unknown-command"])

    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [
        ["datasets", "show", "bbqq"],
        ["datasets", "infer-policy-expectations", "bbqq", "--limit", "0"],
    ],
)
def test_unknown_dataset_name_is_an_input_error(command: list[str], capsys) -> None:
    exit_code = main(command)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error: Example dataset 'bbqq' not registered." in captured.err
    assert "Did you mean: bbq?" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_dataset_name_without_a_suggestion(capsys) -> None:
    exit_code = main(["datasets", "show", "zzzzzzzzzz"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Example dataset 'zzzzzzzzzz' not registered." in captured.err
    assert "Did you mean" not in captured.err


def test_datasets_show_reports_a_registered_dataset(capsys) -> None:
    exit_code = main(["datasets", "show", "bbq"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["slug"] == "bbq"
    assert captured.err == ""


def test_infer_policy_expectations_with_limit_zero_prints_an_empty_list(capsys) -> None:
    exit_code = main(["datasets", "infer-policy-expectations", "bbq", "--limit", "0"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == []


def test_loader_key_error_for_a_registered_dataset_is_not_an_unknown_name(monkeypatch) -> None:
    from rai_toolkit.examples import ExampleRegistry

    def broken_load(slug: str, limit: int | None = None) -> list[dict]:
        raise KeyError("question")

    monkeypatch.setattr(ExampleRegistry, "load", staticmethod(broken_load))

    with pytest.raises(KeyError, match="question"):
        main(["datasets", "infer-policy-expectations", "bbq", "--limit", "1"])
