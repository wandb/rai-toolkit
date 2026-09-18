# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""CLI --version reporting from the shared version authority."""

import subprocess
import sys
from pathlib import Path

import pytest

from rai_toolkit._version import __version__
from rai_toolkit.cli import main


def test_version_flag_prints_shared_version(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    captured = capsys.readouterr()
    assert error.value.code == 0
    assert captured.out == f"rai {__version__}\n"
    assert captured.err == ""


def test_help_includes_version(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    captured = capsys.readouterr()
    assert error.value.code == 0
    assert "--version" in captured.out


def test_version_without_distribution_metadata() -> None:
    script = """
import importlib
import importlib.metadata
import runpy
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
authority = runpy.run_path(str(root / "rai_toolkit" / "_version.py"))["__version__"]
original = importlib.metadata.Distribution.from_name


def without_toolkit(name):
    if name.replace("_", "-").lower() == "rai-toolkit":
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)


with patch.object(
    importlib.metadata.Distribution, "from_name", side_effect=without_toolkit
):
    try:
        importlib.metadata.version("rai-toolkit")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("toolkit metadata must be unavailable")
    sys.modules.pop("rai_toolkit", None)
    sys.modules.pop("rai_toolkit.cli", None)
    sys.modules.pop("rai_toolkit._version", None)
    cli = importlib.import_module("rai_toolkit.cli")
    stdout = StringIO()
    stderr = StringIO()
    with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
        try:
            cli.main(["--version"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("expected SystemExit from --version")
    assert stdout.getvalue() == f"rai {authority}\\n"
    assert stderr.getvalue() == ""
"""
    subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1])],
        check=True,
        capture_output=True,
        text=True,
    )
