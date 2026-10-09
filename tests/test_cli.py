# SPDX-FileCopyrightText: 2026 Sanath
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

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
