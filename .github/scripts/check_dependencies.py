# SPDX-FileCopyrightText: 2026 Karan Nisar
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Resolve each declared extra independently and together, without installing."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", default=f"{sys.version_info.major}.{sys.version_info.minor}")
    parser.add_argument("--python-platform")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with project.open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]

    with tempfile.TemporaryDirectory(prefix="rai-dependencies-") as directory:
        for name in ["core", *extras, "combined"]:
            command = [
                "uv", "pip", "compile", str(project),
                "--python-version", args.python_version,
                "--output-file", str(Path(directory) / f"{name}.txt"),
                "--no-header", "--no-annotate",
            ]
            if args.python_platform:
                command.extend(["--python-platform", args.python_platform])
            if name == "combined":
                command.append("--all-extras")
            elif name != "core":
                command.extend(["--extra", name])
            print(f"Resolving {name} on Python {args.python_version}", flush=True)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
