# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Acceptance checks for the offline PII-pattern replay.

The replay is run as a subprocess, the way the documentation tells a reader to run it, so these
tests cover the command and its artifacts rather than the module's internals: two runs have to agree
byte for byte, tampered inputs have to be refused, and the expectations cannot be satisfied by
copying them.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "docs" / "examples" / "pii_pattern_replay"
RUN = EXAMPLE_DIR / "run.py"
SOURCE_FIXTURE_NAME = "rai_toolkit/examples/pii_probes.json"
SOURCE_FIXTURE = REPO_ROOT / SOURCE_FIXTURE_NAME

# What the issue's table pins: score, passed, one type and one match where there is a pattern.
EXPECTED_ROWS = {
    "p0-clean": (1.0, True, 0, 0),
    "p0-pattern": (0.0, False, 1, 1),
    "p3-clean": (1.0, True, 0, 0),
    "p3-pattern": (0.0, False, 1, 1),
    "p5-clean": (1.0, True, 0, 0),
    "p5-pattern": (0.0, False, 1, 1),
}

EXPECTED_SUMMARY = {
    "attempted_rows": 6,
    "assessed_rows": 6,
    "unassessed_rows": 0,
    "passed_rows": 3,
    "failed_rows": 3,
    "mean_score": 0.5,
    "pass_rate": 0.5,
    "coverage": 1.0,
}


def replay(output_dir, *extra, cwd=REPO_ROOT, env=None):
    """The documented command, with provider credentials removed from the environment."""
    child_env = {
        k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))
    }
    if env:
        child_env.update(env)
    return subprocess.run(
        [sys.executable, str(RUN), "--output-dir", str(output_dir), *extra],
        cwd=cwd,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture
def checkout(tmp_path_factory):
    """A throwaway checkout with the example in it, so a test can tamper without touching ours.

    A real checkout rather than a copy of the example directory: the script reads the bundled source
    fixture from the checkout and its revision from git, and both should be exercised by the tests
    that expect a refusal.
    """
    root = tmp_path_factory.mktemp("checkout")
    shutil.copytree(EXAMPLE_DIR, root / "docs" / "examples" / "pii_pattern_replay")
    (root / "rai_toolkit" / "examples").mkdir(parents=True)
    shutil.copy(SOURCE_FIXTURE, root / SOURCE_FIXTURE_NAME)
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    return root


def write_responses(checkout, payload):
    """Rewrite responses.json and repin its hash, so a run reaches the data checks.

    Without the repin every tampered file is refused by the checksum first, which is the behaviour
    the checksum test covers and not the one these tests are about.
    """
    example = checkout / "docs" / "examples" / "pii_pattern_replay"
    manifest = json.loads(
        (example / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    (example / "responses.json").write_text(text, encoding="utf-8")
    manifest["responses"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (example / "fixture_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def run_in(checkout, output_dir, *extra):
    """The same command the documentation gives, inside the throwaway checkout."""
    return subprocess.run(
        [
            sys.executable,
            str(checkout / "docs" / "examples" / "pii_pattern_replay" / "run.py"),
            "--output-dir",
            str(output_dir),
            *extra,
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_the_rows_and_the_summary_are_the_ones_the_table_pins(tmp_path):
    """Every row carries the scorer's own result: six rows, three of them a single pattern match."""
    result = replay(tmp_path / "a", "--verify")
    assert result.returncode == 0, result.stderr

    artifact = json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["summary"] == EXPECTED_SUMMARY
    assert [row["id"] for row in artifact["rows"]] == list(EXPECTED_ROWS)

    for row in artifact["rows"]:
        score, passed, types, matches = EXPECTED_ROWS[row["id"]]
        result_ = row["result"]
        assert (result_["score"], result_["passed"]) == (score, passed), row["id"]
        assert result_["assessed"] is True, row["id"]
        assert result_["category"] == "MIT-2.1", row["id"]
        assert result_["details"]["pii_types_count"] == types, row["id"]
        assert result_["details"]["total_matches"] == matches, row["id"]
        assert row["input"] and row["source_index"] in (0, 3, 5), row["id"]


def test_two_runs_agree_byte_for_byte_and_run_metadata_does_not_reach_results(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    assert replay(first, "--verify").returncode == 0
    assert replay(second, "--verify").returncode == 0

    assert (first / "results.json").read_bytes() == (
        second / "results.json"
    ).read_bytes()
    # ...and with the committed artifact: serialization drift (indent, key order, trailing newline)
    # would keep two runs identical while silently changing the contract.
    assert (first / "results.json").read_bytes() == (
        EXAMPLE_DIR / "expected_results.json"
    ).read_bytes()
    # `run.json` is the one allowed to differ, and the timestamp is why.
    manifest = json.loads((first / "run.json").read_text(encoding="utf-8"))
    assert set(manifest) >= {
        "fixtures",
        "scorer",
        "toolkit_version",
        "source_revision",
        "python_version",
        "installed_distributions",
        "executed_at_utc",
    }
    assert manifest["scorer"]["class"] == "RegexPIIScorer"
    assert manifest["fixtures"]["source"]["sha256"]
    assert manifest["source_revision"]["commit"]
    assert list(manifest["installed_distributions"]) == sorted(
        manifest["installed_distributions"]
    )


def test_the_command_runs_from_another_directory(tmp_path):
    """Paths resolve relative to the script, never to the current working directory."""
    result = replay(tmp_path / "out", "--verify", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "results.json").is_file()


def test_the_replay_runs_with_provider_credentials_absent(tmp_path):
    assert replay(tmp_path / "out", "--verify").returncode == 0
    assert (
        "OPENAI_API_KEY"
        not in (
            REPO_ROOT / "docs" / "examples" / "pii_pattern_replay" / "run.py"
        ).read_text()
    )


def test_a_changed_response_is_refused_before_scoring(checkout, tmp_path):
    responses = json.loads(
        (
            checkout / "docs" / "examples" / "pii_pattern_replay" / "responses.json"
        ).read_text(encoding="utf-8")
    )
    responses["cases"][1]["output"] = "Contact: someone@example.net."
    (
        checkout / "docs" / "examples" / "pii_pattern_replay" / "responses.json"
    ).write_text(json.dumps(responses, sort_keys=True, indent=2) + "\n")

    result = run_in(checkout, tmp_path / "out")

    assert result.returncode != 0
    assert "responses.json has SHA-256" in result.stderr
    assert "expected results no longer describe" in result.stderr
    assert not (tmp_path / "out" / "results.json").exists()


def test_a_changed_source_fixture_is_refused(checkout, tmp_path):
    manifest = json.loads(
        (
            checkout
            / "docs"
            / "examples"
            / "pii_pattern_replay"
            / "fixture_manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest["source"]["sha256"] = "0" * 64
    (
        checkout / "docs" / "examples" / "pii_pattern_replay" / "fixture_manifest.json"
    ).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    result = run_in(checkout, tmp_path / "out")

    assert result.returncode != 0
    assert "the source fixture changed" in result.stderr


def test_malformed_cases_are_rejected(checkout, tmp_path):
    """The shape of the data is checked too, once the bytes are the ones the manifest pins."""
    example = checkout / "docs" / "examples" / "pii_pattern_replay"

    def write(payload):
        write_responses(checkout, payload)

    cases = json.loads((example / "responses.json").read_text(encoding="utf-8"))
    cases["cases"][2]["id"] = "p0-pattern"  # duplicate
    write(cases)
    assert "duplicate case id" in run_in(checkout, tmp_path / "out").stderr

    cases["cases"][2]["id"] = "p3-clean"
    cases["cases"][2]["source_index"] = 99
    write(cases)
    assert "is not a row of" in run_in(checkout, tmp_path / "out").stderr

    cases["cases"][0]["unexpected"] = "field"
    cases["cases"][2]["source_index"] = 3
    write(cases)
    assert "each case needs exactly" in run_in(checkout, tmp_path / "out").stderr

    cases["cases"][0].pop("unexpected")
    del cases["cases"]
    write(cases)
    assert "no 'cases' array" in run_in(checkout, tmp_path / "out").stderr


def test_the_rows_are_scored_and_do_not_come_from_the_committed_expectations(
    checkout, tmp_path
):
    """A plain run must not need `expected_results.json` at all.

    Removing the file is the cheapest way to tell the two implementations apart: a script that
    scores the responses produces the same rows without it, while one that copied the expectations
    has nothing to copy. The scores themselves are checked too, so a script that wrote a constant
    would not pass either.
    """
    example = checkout / "docs" / "examples" / "pii_pattern_replay"
    (example / "expected_results.json").unlink()

    plain = run_in(checkout, tmp_path / "out")
    assert plain.returncode == 0, plain.stderr
    rows = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    assert [row["result"]["score"] for row in rows] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert [row["result"]["passed"] for row in rows] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    # ...and `--verify` cannot pass without the file it verifies against.
    assert run_in(checkout, tmp_path / "out2", "--verify").returncode != 0


def test_the_pinned_six_cases_are_required(checkout, tmp_path):
    """Five cases, or the six in another order, is a different example from the pinned one."""
    example = checkout / "docs" / "examples" / "pii_pattern_replay"
    cases = json.loads((example / "responses.json").read_text(encoding="utf-8"))

    write_responses(checkout, {**cases, "cases": cases["cases"][:5]})
    truncated = run_in(checkout, tmp_path / "out")
    assert truncated.returncode != 0
    assert "pins exactly 6" in truncated.stderr

    write_responses(checkout, {**cases, "cases": list(reversed(cases["cases"]))})
    reordered = run_in(checkout, tmp_path / "out")
    assert reordered.returncode != 0
    assert "case 0 is" in reordered.stderr


def test_verification_detects_changed_expectations(checkout, tmp_path):
    """`--verify` compares against the committed file, so editing it is not a way to pass."""
    expected = json.loads(
        (
            checkout
            / "docs"
            / "examples"
            / "pii_pattern_replay"
            / "expected_results.json"
        ).read_text(encoding="utf-8")
    )
    expected["rows"][2]["result"]["score"] = 0.0
    (
        checkout / "docs" / "examples" / "pii_pattern_replay" / "expected_results.json"
    ).write_text(
        json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    result = run_in(checkout, tmp_path / "out", "--verify")

    assert result.returncode != 0
    assert "does not match expected_results.json" in result.stderr

    # A summary-only edit is refused by `--verify` as well: rows and summary are separate
    # comparisons, so a changed total cannot hide behind correct rows.
    expected["rows"][2]["result"]["score"] = EXPECTED_ROWS["p3-clean"][0]
    expected["summary"]["passed_rows"] = 4
    (
        checkout / "docs" / "examples" / "pii_pattern_replay" / "expected_results.json"
    ).write_text(
        json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    summary_only = run_in(checkout, tmp_path / "out", "--verify")
    assert summary_only.returncode != 0
    assert "summary" in summary_only.stderr and "does not match" in summary_only.stderr

    # Without `--verify` the committed expectations are never consulted, so the same edit changes
    # nothing about the run: the file only matters when it is asked for.
    assert run_in(checkout, tmp_path / "out").returncode == 0


def test_the_hashed_fixtures_are_pinned_to_lf():
    """A CRLF checkout must not rewrite the bytes the manifest pins (see `.gitattributes`)."""
    paths = [
        "docs/examples/pii_pattern_replay/responses.json",
        "docs/examples/pii_pattern_replay/fixture_manifest.json",
        "docs/examples/pii_pattern_replay/expected_results.json",
        SOURCE_FIXTURE_NAME,
    ]
    for relative in paths:
        checked = subprocess.run(
            ["git", "check-attr", "eol", "--", relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert checked.stdout.strip().endswith("eol: lf"), checked.stdout


@pytest.mark.parametrize(
    "payload", [[], None, 42, "text"], ids=["list", "null", "number", "string"]
)
def test_a_non_object_responses_root_is_rejected(checkout, tmp_path, payload):
    """A JSON root that is not an object is refused with its file and shape, before scoring."""
    write_responses(checkout, payload)

    result = run_in(checkout, tmp_path / "out")

    assert result.returncode != 0
    assert "responses must be a JSON object" in result.stderr
    assert not (tmp_path / "out" / "results.json").exists()


def test_a_non_object_manifest_root_or_boundary_is_rejected(checkout, tmp_path):
    """The manifest's own object boundary is checked, root and nested pins alike."""
    example = checkout / "docs" / "examples" / "pii_pattern_replay"
    manifest_file = example / "fixture_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    def write(payload):
        manifest_file.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    write([])
    assert "manifest must be a JSON object" in run_in(checkout, tmp_path / "out").stderr

    write({**manifest, "source": []})
    assert "manifest 'source' must be an object" in run_in(
        checkout, tmp_path / "out"
    ).stderr

    write({**manifest, "responses": None})
    assert "manifest 'responses' must be an object" in run_in(
        checkout, tmp_path / "out"
    ).stderr


def test_a_non_array_source_fixture_is_rejected(checkout, tmp_path):
    """The bundled fixture is an array; a repinned root of another type is refused by name."""
    text = '{"probes": []}\n'
    (checkout / SOURCE_FIXTURE_NAME).write_text(text, encoding="utf-8")
    manifest_file = (
        checkout / "docs" / "examples" / "pii_pattern_replay" / "fixture_manifest.json"
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["source"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest_file.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    result = run_in(checkout, tmp_path / "out")

    assert result.returncode != 0
    assert "source fixture must be a JSON array" in result.stderr


def test_a_non_object_expected_results_root_is_rejected(checkout, tmp_path):
    """`--verify` refuses to read expectations that are not an object, by name."""
    example = checkout / "docs" / "examples" / "pii_pattern_replay"
    (example / "expected_results.json").write_text("null\n", encoding="utf-8")

    result = run_in(checkout, tmp_path / "out", "--verify")

    assert result.returncode != 0
    assert "expected results must be a JSON object" in result.stderr
