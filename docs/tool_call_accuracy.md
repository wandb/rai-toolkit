<!--
SPDX-FileCopyrightText: 2026 Harsh Raj Singhania
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: rai-toolkit
-->

# Tool-call accuracy

`ToolCallAccuracyScorer` grades a complete, structured invocation trace against
explicit expected and forbidden calls. It is offline: no provider account, API
key, Weave project, or network call is required.

The scorer checks requested tool names and exact arguments. A match does not
prove that a tool executed successfully, that a side effect occurred, or that
the overall task was completed. The current assessment pipeline does not
collect or forward traces. Call the scorer directly with records you already
have.

An unassessed result is not a measured failure. Its `score` is a placeholder
(`0.0`) and `passed` is `False` only because the row could not be graded.
Read `assessed` and `details["skipped"]` before treating the verdict as
evidence.

Each invocation record needs a unique string `id`, a non-empty `name`, and a
JSON-compatible `arguments` object. Criteria use the same argument shape
without an id. `trace_complete` must be exactly `True` for an assessed result.

```python
from rai_toolkit.scorers import ToolCallAccuracyScorer

scorer = ToolCallAccuracyScorer()
criteria = {
    "tool_calls": {
        "expected": [
            {"name": "lookup_record", "arguments": {"record_id": "rec-42"}}
        ],
        "forbidden": ["send_email"],
    }
}

complete_match = scorer.score(
    output="Record rec-42 was found.",
    tool_calls=[
        {
            "id": "call-lookup-1",
            "name": "lookup_record",
            "arguments": {"record_id": "rec-42"},
        }
    ],
    trace_complete=True,
    success_criteria=criteria,
)
print("complete match", complete_match.score, complete_match.passed, complete_match.assessed)
assert complete_match.assessed and complete_match.passed
assert complete_match.score == 1.0
assert complete_match.details["matched_actual_ids"] == ["call-lookup-1"]

wrong_arguments = scorer.score(
    output="Record rec-7 was found.",
    tool_calls=[
        {
            "id": "call-lookup-2",
            "name": "lookup_record",
            "arguments": {"record_id": "rec-7"},
        }
    ],
    trace_complete=True,
    success_criteria=criteria,
)
print(
    "wrong arguments",
    wrong_arguments.score,
    wrong_arguments.passed,
    wrong_arguments.details["argument_mismatches"],
)
assert wrong_arguments.assessed and not wrong_arguments.passed
assert wrong_arguments.score < scorer.threshold
assert wrong_arguments.details["argument_mismatches"][0]["actual_id"] == "call-lookup-2"
assert wrong_arguments.details["argument_mismatches"][0]["actual_arguments"] == {
    "record_id": "rec-7"
}

forbidden_call = scorer.score(
    output="I looked up the record and emailed it.",
    tool_calls=[
        {
            "id": "call-lookup-3",
            "name": "lookup_record",
            "arguments": {"record_id": "rec-42"},
        },
        {
            "id": "call-email-1",
            "name": "send_email",
            "arguments": {"to": "ops@example.com"},
        },
    ],
    trace_complete=True,
    success_criteria=criteria,
)
print(
    "forbidden call",
    forbidden_call.score,
    forbidden_call.passed,
    forbidden_call.details["forbidden_actual_ids"],
)
assert forbidden_call.assessed and not forbidden_call.passed
assert forbidden_call.score == 0.0
assert forbidden_call.details["forbidden_actual_ids"] == ["call-email-1"]

incomplete_trace = scorer.score(
    output="Still collecting tool results.",
    tool_calls=[
        {
            "id": "call-lookup-4",
            "name": "lookup_record",
            "arguments": {"record_id": "rec-42"},
        }
    ],
    trace_complete=False,
    success_criteria=criteria,
)
print(
    "incomplete trace",
    incomplete_trace.score,
    incomplete_trace.passed,
    incomplete_trace.assessed,
    incomplete_trace.details["skipped"],
)
assert not incomplete_trace.assessed
assert incomplete_trace.details["skipped"] == "incomplete_tool_trace"
assert incomplete_trace.score == 0.0  # placeholder, not a measured failure
```

| Case | Meaning |
| --- | --- |
| Complete trace with the expected call and exact arguments | Assessed and passed (`score == 1.0`) |
| Complete trace with the right tool and different arguments | Assessed and failed at the default threshold, with `argument_mismatches` |
| Complete trace containing a forbidden call | Assessed and failed, with that call id in `forbidden_actual_ids` |
| Trace not marked complete | Unassessed, `details["skipped"] == "incomplete_tool_trace"` |
