"""Budget stops preserve valid prefixes; provider violations remain errors.

The shared harness is imported to register its pytest fixture in this module.
"""

import hashlib
import json

import pytest
from examples.coding_audit.hf import HFBackendError
from examples.coding_audit.test_hf import harness as harness, MODEL, response, terminal


def model(**values):
    return {
        **MODEL,
        "max_steps": 48,
        "max_tokens": 4096,
        "max_total_tokens": 32768,
        **values,
    }


def test_next_prompt_is_reserved_before_dispatch_and_saved_files_are_graded(harness):
    h = harness
    (h.root / "instruction.md").write_text("public requirement " * 700)
    h.inference.responses = [
        response(
            None,
            calls=[terminal(identifier="first")],
            usage={
                "prompt_tokens": 15000,
                "completion_tokens": 4000,
                "total_tokens": 19000,
            },
        ),
        response(
            None,
            calls=[terminal(identifier="second")],
            usage={
                "prompt_tokens": 15000,
                "completion_tokens": 4000,
                "total_tokens": 19000,
            },
        ),
    ]
    solved = h.backend.solve(model(), 0)
    assert solved["error"] is None and solved["truncated"]
    assert solved["termination_reason"] == "input_reservation_limit"
    assert len(h.inference.requests) == 1 and len(h.factory.created[0].commands) == 1
    assert solved["usage"] == {"input_tokens": 15000, "output_tokens": 4000}
    assert solved["input_tokens"] + solved["output_tokens"] <= 32768
    preflights = solved["model_metadata"]["preflights"]
    assert preflights[0]["dispatched"] and not preflights[1]["dispatched"]
    assert preflights[1]["output_token_cap"] == 0
    payload = json.dumps(h.inference.requests[0], allow_nan=False).encode()
    assert preflights[0]["request_sha256"] == hashlib.sha256(payload).hexdigest()
    assert preflights[0]["request_bytes"] == len(payload)
    assert solved["files"] == {"/app/answer.txt": b"answer"}
    grade = h.backend.grade(solved)
    assert grade["error"] is None and grade["cleanup_verified"]
    assert grade["grade_sandbox_id"] != solved["sandbox_id"]


def test_insufficient_first_prompt_has_no_model_measurement_or_grade(harness):
    h = harness
    solved = h.backend.solve(model(max_total_tokens=8192), 0)
    assert "initial target prompt" in solved["error"]
    assert not h.inference.requests and not h.factory.created[0].commands
    assert solved["cleanup_verified"] and solved["files"] == {}
    assert len(h.factory.created) == 1 and not solved["truncated"]


@pytest.mark.parametrize("failure", ["output", "total", "input_estimate"])
def test_actual_overruns_are_errors_with_billed_usage_not_truncation(harness, failure):
    h = harness
    prompt, completion = {
        "output": (10, 4097),
        "total": (33000, 5),
        "input_estimate": (12000, 5),
    }[failure]
    h.inference.responses = [
        response(
            None,
            calls=[terminal()],
            usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        )
    ]
    solved = h.backend.solve(model(), 0)
    assert "provider exceeded token budget" in solved["error"]
    assert not solved["truncated"] and not h.factory.created[0].commands
    assert solved["usage"] == {"input_tokens": prompt, "output_tokens": completion}
    receipt = solved["model_metadata"]["responses"][0]
    assert receipt["usage"]["total_tokens"] == prompt + completion
    expected = {
        "output": "response_output_cap",
        "total": "cumulative_total_cap",
        "input_estimate": "input_reservation_contract",
    }[failure]
    assert expected in receipt["budget_violations"] and receipt["discarded"]
    assert len(h.factory.created) == 1 and solved["cleanup_verified"]


def test_nested_reasoning_usage_is_retained_without_double_charging(harness):
    h = harness
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 4},
    }
    h.inference.responses = [response("saved", usage=usage)]
    solved = h.backend.solve(model(), 0)
    assert solved["error"] is None
    assert solved["usage"] == {"input_tokens": 10, "output_tokens": 5}
    receipt = solved["model_metadata"]["responses"][0]
    assert receipt["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4
    assert receipt["response_validated"] and not receipt["discarded"]


@pytest.mark.parametrize(
    "problem", ["missing_total", "bad_total", "nonfinite_detail", "oversized_detail"]
)
def test_invalid_usage_is_never_a_normal_budget_stop(harness, problem):
    h = harness
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if problem == "missing_total":
        usage.pop("total_tokens")
    elif problem == "bad_total":
        usage["total_tokens"] = 999
    elif problem == "nonfinite_detail":
        usage["detail"] = float("nan")
    else:
        usage["detail"] = "x" * 65537
    h.inference.responses = [response(None, calls=[terminal()], usage=usage)]
    solved = h.backend.solve(model(), 0)
    assert solved["error"] and not solved["truncated"]
    assert not h.factory.created[0].commands and solved["cleanup_verified"]
    assert len(h.factory.created) == 1
    json.dumps(solved, allow_nan=False)


@pytest.mark.parametrize("prefix", [False, True])
def test_funded_length_discards_partial_calls_and_grades_existing_state(
    harness, prefix
):
    h = harness
    h.factory.submitted = None if not prefix else b"answer"
    h.factory.reward = b"0" if not prefix else b"1"
    partial = terminal("synthetic-must-not-execute", "partial")
    partial["function"]["arguments"] = '{"command":"unterminated'
    ended = response(None, calls=[partial])
    ended["choices"][0]["finish_reason"] = "length"
    h.inference.responses = (
        [response(None, calls=[terminal(identifier="saved")])] if prefix else []
    ) + [ended]
    solved = h.backend.solve(model(), 0)
    assert solved["error"] is None and solved["truncated"]
    assert solved["termination_reason"] == "response_output_limit"
    assert len(h.factory.created[0].commands) == int(prefix)
    assert solved["files"] == ({"/app/answer.txt": b"answer"} if prefix else {})
    assert solved["model_metadata"]["responses"][-1]["finish_reason"] == "length"
    assert solved["model_metadata"]["responses"][-1]["discarded"]
    assert "reward" not in solved
    grade = h.backend.grade(solved)
    assert grade["error"] is None and grade["reward"] == int(prefix)
    assert (
        grade["grade_sandbox_id"] != solved["sandbox_id"] and grade["cleanup_verified"]
    )


@pytest.mark.parametrize("violation", ["model", "role", "usage", "bytes"])
def test_length_does_not_mask_invalid_response(harness, violation):
    h = harness
    ended = response(None, calls=[terminal()])
    ended["choices"][0]["finish_reason"] = "length"
    if violation == "model":
        ended["model"] = "wrong/model"
    elif violation == "role":
        ended["choices"][0]["message"]["role"] = "user"
    elif violation == "usage":
        ended = response(
            None,
            calls=[terminal()],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4097,
                "total_tokens": 4107,
            },
        )
        ended["choices"][0]["finish_reason"] = "length"
    else:
        ended["choices"][0]["message"]["content"] = "x" * 16385
    h.inference.responses = [ended]
    solved = h.backend.solve(model(), 0)
    assert solved["error"] and not solved["truncated"]
    assert not h.factory.created[0].commands and len(h.factory.created) == 1
    assert solved["cleanup_verified"]


def test_provider_exception_with_old_budget_message_is_not_recovered(harness):
    h = harness

    def failure(**kwargs):
        raise HFBackendError("provider exceeded token budget")

    h.inference.chat_completion = failure
    solved = h.backend.solve(model(), 0)
    assert (
        solved["error"] == "provider exceeded token budget" and not solved["truncated"]
    )
    assert solved["cleanup_verified"] and len(h.factory.created) == 1
