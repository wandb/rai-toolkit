# SPDX-FileCopyrightText: 2026 Adam
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from dataclasses import fields
from importlib.metadata import version

import rai_toolkit
from rai_toolkit.assessment import AssessmentResult


def test_version_is_consistent() -> None:
    assert rai_toolkit.__version__ == version("rai-toolkit")
    toolkit_version = next(
        field for field in fields(AssessmentResult) if field.name == "toolkit_version"
    )
    assert toolkit_version.default == rai_toolkit.__version__
