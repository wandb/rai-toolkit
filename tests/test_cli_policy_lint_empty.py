# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from pathlib import Path

from rai_toolkit.cli import main


def test_policies_lint_fails_on_empty_directory(tmp_path: Path, capsys) -> None:
    exit_code = main(["policies", "lint", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("FAIL:")
    assert "*.yaml" in captured.err
    assert str(tmp_path) in captured.err
    assert "OK:" not in captured.out


def test_policies_lint_fails_when_directory_has_no_yaml_files(
    tmp_path: Path, capsys
) -> None:
    (tmp_path / "README.txt").write_text("No policy files here.", encoding="utf-8")

    exit_code = main(["policies", "lint", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("FAIL:")
    assert "*.yaml" in captured.err
    assert str(tmp_path) in captured.err
    assert "OK:" not in captured.out


def test_policies_lint_accepts_valid_empty_policy_set(tmp_path: Path, capsys) -> None:
    (tmp_path / "empty.yaml").write_text(
        """\
name: Empty policy set
version: "1.0.0"
policies: []
""",
        encoding="utf-8",
    )

    exit_code = main(["policies", "lint", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK: 0 policies loaded" in captured.out
    assert captured.err == ""


def test_policies_lint_accepts_disabled_only_policy_set(tmp_path: Path, capsys) -> None:
    (tmp_path / "disabled.yaml").write_text(
        """\
name: Disabled policy set
version: "1.0.0"
policies:
  - name: block-secret
    description: Block secret output
    severity: high
    enabled: false
    trigger:
      output_contains: [secret]
""",
        encoding="utf-8",
    )

    exit_code = main(["policies", "lint", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK: 0 policies loaded" in captured.out
    assert captured.err == ""


def test_policies_lint_accepts_enabled_policy_set(tmp_path: Path, capsys) -> None:
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

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK: 1 policies loaded" in captured.out
    assert "block-secret" in captured.out
    assert captured.err == ""


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

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("FAIL:")
    assert "OK:" not in captured.out
