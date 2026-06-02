# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Assessment workflow — one-call entry point that produces an evidence-based
compliance assessment.

An assessment run combines:
  1. Evaluation against a compliance-aware scorer set (from the toolkit's engine)
  2. Red-team adversarial probing
  3. Policy-as-code violation detection
  4. Framework coverage computation (EU AI Act, NIST, MIT)

The result is a single ``AssessmentResult`` object suitable for rendering as
a CLI summary, JSON export, or signed PDF report. The toolkit deliberately
avoids the binary "certified / not certified" framing — the result is
evidence for a human reviewer's decision, not a stamp.

Example::

    from rai_toolkit.assessment import Assessor

    result = await Assessor(
        model=my_model,
        preset="healthcare",
        datasets=["my-healthcare-eval"],
        run_redteam=True,
    ).run()

    print(result.format_summary())
    result.to_json("assessment-report.json")
"""

from rai_toolkit.assessment.assessor import (
    Assessor,
    AssessmentResult,
    FrameworkAssessment,
)

__all__ = ["Assessor", "AssessmentResult", "FrameworkAssessment"]
