"""Target tool repair at the real solve, provider-message, and grading seams."""

import json
import time

import pytest
from examples.coding_audit.test_hf import (
    FakeSandbox,
    harness as harness,
    MODEL,
    response,
    terminal,
)

REJECTED_COMMAND = "synthetic-command-that-must-not-run"
REPAIR_COMMAND = "printf answer > /app/answer.txt"


def rejected_call(kind, identifier="rejected-call"):
    call = terminal(REJECTED_COMMAND, identifier)
    if kind in {"unknown-empty", "unknown-command", "provider", "reset"}:
        call["function"]["name"] = {
            "provider": "provider_request",
            "reset": "reset",
        }.get(kind, "command")
    if kind in {"unknown-empty", "empty"}:
        call["function"]["arguments"] = "{}"
    elif kind == "malformed-json":
        call["function"]["arguments"] = '{"command":"unterminated'
    elif kind == "duplicate-json-key":
        call["function"]["arguments"] = (
            '{"command":"synthetic-first","command":"synthetic-second"}'
        )
    elif kind == "argument-object":
        call["function"]["arguments"] = {"command": REJECTED_COMMAND}
    elif kind == "command-number":
        call["function"]["arguments"] = '{"command":7}'
    elif kind == "argument-list":
        call["function"]["arguments"] = "[]"
    elif kind == "oversized-command":
        call["function"]["arguments"] = json.dumps({"command": "x" * 129})
    return call


def assert_feedback(message, identifier, limit):
    assert set(message) == {"role", "tool_call_id", "content"}
    assert message["role"] == "tool" and message["tool_call_id"] == identifier
    assert isinstance(message["content"], str)
    assert len(message["content"].encode()) <= limit
    payload = json.loads(message["content"])
    assert isinstance(payload, dict)
    assert isinstance(payload.get("error"), str) and payload["error"]


def assert_target_wire(requests):
    for request in requests:
        assert "stream" not in request and request["tool_choice"] == "auto"
        assert [tool["function"]["name"] for tool in request["tools"]] == ["terminal"]
        assert "PRIVATE_SOLUTION_SENTINEL" not in repr(request)
        assert "PRIVATE_CANARY_SENTINEL" not in repr(request)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown-empty",
        "unknown-command",
        "empty",
        "malformed-json",
        "duplicate-json-key",
        "command-number",
        "argument-list",
        "oversized-command",
        "provider",
        "reset",
    ],
)
def test_bindable_rejection_reaches_provider_as_feedback_then_repairs_and_grades(
    harness, kind
):
    h = harness
    h.backend.max_output = 256
    h.backend.max_script = 128
    bad = rejected_call(kind)
    h.inference.responses = [
        response(None, calls=[bad]),
        response(None, calls=[terminal(REPAIR_COMMAND, "repair-call")]),
        response("saved"),
    ]
    original = h.inference.chat_completion

    def provider(**request):
        if len(h.inference.requests) == 1:
            assert not h.factory.created[0].commands
            assert request["messages"][-2]["role"] == "assistant"
            assert_feedback(request["messages"][-1], bad["id"], h.backend.max_output)
        return original(**request)

    h.inference.chat_completion = provider
    solved = h.backend.solve({**MODEL, "max_steps": 4}, 0)
    assert solved["error"] is None and solved["cleanup_verified"]
    assert not solved["truncated"] and len(h.inference.requests) == 3
    assert [argv for argv, _ in h.factory.created[0].commands] == [
        ["/bin/bash", "-c", REPAIR_COMMAND]
    ]
    assert solved["files"] == {"/app/answer.txt": b"answer"}
    assert "reward" not in solved
    assert_target_wire(h.inference.requests)
    grade = h.backend.grade(solved)
    assert grade["error"] is None and grade["reward"] == 1
    assert (
        grade["cleanup_verified"] and grade["grade_sandbox_id"] != solved["sandbox_id"]
    )
    assert h.factory.created[1].files.data["/app/answer.txt"] == b"answer"
    assert all(stage == "CANCELED" for stage in h.api.stages.values())


def test_all_invalid_single_step_is_bounded_and_grades_actual_empty_submission(harness):
    h = harness
    h.factory.submitted = None
    h.factory.reward = b"0"
    h.inference.responses = [response(None, calls=[rejected_call("unknown-empty")])]
    solved = h.backend.solve({**MODEL, "max_steps": 1}, 0)
    assert solved["error"] is None and solved["cleanup_verified"]
    assert solved["truncated"] and solved["termination_reason"] == "step_limit"
    assert len(h.inference.requests) == 1 and not h.factory.created[0].commands
    assert_feedback(solved["transcript"][-1], "rejected-call", h.backend.max_output)
    assert solved["files"] == {} and "reward" not in solved
    grade = h.backend.grade(solved)
    assert (
        grade["error"] is None and grade["reward"] == 0 and grade["rawreward"] == b"0"
    )
    assert (
        grade["cleanup_verified"] and grade["grade_sandbox_id"] != solved["sandbox_id"]
    )
    assert "/app/answer.txt" not in h.factory.created[1].files.data


def test_rejected_calls_count_toward_multi_call_budget_before_any_execution(harness):
    h = harness
    h.factory.submitted = None
    h.inference.responses = [
        response(None, calls=[rejected_call("empty", "first")]),
        response(
            None, calls=[rejected_call("empty", "second"), terminal(REPAIR_COMMAND)]
        ),
    ]
    solved = h.backend.solve({**MODEL, "max_steps": 2}, 0)
    assert solved["error"] is None and solved["cleanup_verified"]
    assert solved["truncated"] and solved["termination_reason"] == "tool_call_limit"
    assert len(h.inference.requests) == 2 and solved["files"] == {}
    assert not h.factory.created[0].commands
    assert_feedback(
        h.inference.requests[1]["messages"][-1], "first", h.backend.max_output
    )


def test_rejection_at_call_cap_preserves_saved_files_for_normal_grading(harness):
    h = harness
    h.inference.responses = [
        response(None, calls=[terminal(REPAIR_COMMAND, "saved-call")]),
        response(None, calls=[rejected_call("empty")]),
    ]
    solved = h.backend.solve({**MODEL, "max_steps": 2}, 0)
    assert solved["error"] is None and solved["cleanup_verified"]
    assert solved["truncated"] and solved["termination_reason"] == "step_limit"
    assert len(h.inference.requests) == 2
    assert len(h.factory.created[0].commands) == 1
    assert solved["files"] == {"/app/answer.txt": b"answer"} and "reward" not in solved
    assert_feedback(solved["transcript"][-1], "rejected-call", h.backend.max_output)
    grade = h.backend.grade(solved)
    assert grade["error"] is None and grade["reward"] == 1 and grade["cleanup_verified"]
    assert grade["grade_sandbox_id"] != solved["sandbox_id"]


@pytest.mark.parametrize(
    "malformation",
    [
        "missing-id",
        "empty-id",
        "numeric-id",
        "wrong-type",
        "missing-type",
        "duplicate-id",
        "nonstring-name",
        "argument-object",
    ],
)
def test_unbindable_or_duplicate_identity_is_fatal_before_batch_execution(
    harness, malformation
):
    h = harness
    first = terminal(REPAIR_COMMAND, "first")
    second = terminal(REJECTED_COMMAND, "second")
    if malformation == "missing-id":
        second.pop("id")
    elif malformation == "empty-id":
        second["id"] = ""
    elif malformation == "numeric-id":
        second["id"] = 7
    elif malformation == "wrong-type":
        second["type"] = "reset"
    elif malformation == "missing-type":
        second.pop("type")
    elif malformation == "duplicate-id":
        second["id"] = first["id"]
    elif malformation == "argument-object":
        second["function"]["arguments"] = {"command": REJECTED_COMMAND}
    else:
        second["function"]["name"] = None
    h.inference.responses = [response(None, calls=[first, second])]
    solved = h.backend.solve(MODEL, 0)
    assert solved["error"] and solved["cleanup_verified"]
    assert not h.factory.created[0].commands and len(h.inference.requests) == 1
    assert not any(message.get("role") == "tool" for message in solved["transcript"])
    assert "reward" not in solved


def test_tool_execution_platform_failure_after_feedback_remains_error(
    harness, monkeypatch
):
    h = harness
    h.inference.responses = [
        response(None, calls=[rejected_call("empty")]),
        response(None, calls=[terminal(REPAIR_COMMAND)]),
    ]

    def failed_run(*args, **kwargs):
        raise RuntimeError("synthetic platform failure")

    monkeypatch.setattr(FakeSandbox, "run", failed_run)
    solved = h.backend.solve(MODEL, 0)
    assert solved["error"] == "HF backend operation failed (RuntimeError)"
    assert solved["cleanup_verified"] and len(h.inference.requests) == 2
    assert (
        len(
            [
                message
                for message in solved["transcript"]
                if message.get("role") == "tool"
            ]
        )
        == 1
    )


def test_provider_deadline_after_feedback_remains_error(harness):
    h = harness
    h.backend.timeout = 0.02
    h.inference.responses = [response(None, calls=[rejected_call("empty")])]
    original = h.inference.chat_completion

    def blocked_provider(**request):
        if h.inference.requests:
            time.sleep(1)
        return original(**request)

    h.inference.chat_completion = blocked_provider
    started = time.monotonic()
    solved = h.backend.solve(MODEL, 0)
    assert time.monotonic() - started < 0.5
    assert "deadline" in solved["error"] and solved["cleanup_verified"]
    assert not h.factory.created[0].commands


def test_cleanup_failure_after_tool_repair_remains_error(harness):
    h = harness
    h.api.ignore_cancel = True
    h.inference.responses = [
        response(None, calls=[rejected_call("empty")]),
        response(None, calls=[terminal(REPAIR_COMMAND)]),
        response("saved"),
    ]
    solved = h.backend.solve(MODEL, 0)
    assert solved["error"] == "sandbox cleanup could not be verified"
    assert not solved["cleanup_verified"] and len(h.inference.requests) == 3
    assert len(h.factory.created[0].commands) == 1 and "reward" not in solved
