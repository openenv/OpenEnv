# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Negotiating with a provider that rejects our parameters, and authenticating to it.

Every error payload in this file is a verbatim copy of one a live endpoint returned. That matters
more than usual here: the same rejection of the same parameter is phrased two different ways by two
models from the same vendor, and one of them does not populate `error.param` at all, so a matcher
written against a single shape passes its tests and fails in production.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
compat = pytest.importorskip("openenv.core.harness.capture.compat")
upstream = pytest.importorskip("openenv.core.harness.capture.upstream")

diagnose = compat.diagnose


def error(message: str, param=None, code=None) -> dict:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }


# --- reading a 400 ----------------------------------------------------------
def test_unrecognized_argument_names_the_param_in_the_message_only():
    """gpt-4o-mini's phrasing. `error.param` is null, so the name must come from the prose."""
    fix = diagnose(error("Unrecognized request argument supplied: return_token_ids"))
    assert (fix.param, fix.action) == ("return_token_ids", "drop")


def test_unknown_parameter_is_the_same_rejection_phrased_differently():
    """gpt-5.6's phrasing for the identical problem, with `param` populated and a different code."""
    fix = diagnose(
        error(
            "Unknown parameter: 'return_token_ids'.",
            param="return_token_ids",
            code="unknown_parameter",
        )
    )
    assert (fix.param, fix.action) == ("return_token_ids", "drop")


def test_a_named_replacement_becomes_a_rename():
    fix = diagnose(
        error(
            "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead.",
            param="max_tokens",
            code="unsupported_parameter",
        )
    )
    assert (fix.param, fix.replacement) == ("max_tokens", "max_completion_tokens")


def test_an_unsupported_value_drops_the_param_so_the_default_applies():
    fix = diagnose(
        error(
            "Unsupported value: 'temperature' does not support 0 with this model. Only the "
            "default (1) value is supported.",
            param="temperature",
            code="unsupported_value",
        )
    )
    assert (fix.param, fix.action) == ("temperature", "drop")


def test_an_unsupported_parameter_with_no_alternative_is_dropped():
    fix = diagnose(
        error(
            "Unsupported parameter: 'logprobs' is not supported with this model.",
            param="logprobs",
            code="unsupported_parameter",
        )
    )
    assert (fix.param, fix.action) == ("logprobs", "drop")


def test_an_error_that_names_no_parameter_yields_no_fix():
    """A context-length error is real. Guessing a param to drop would hide it."""
    assert diagnose(error("This model's maximum context length is 8192 tokens")) is None


@pytest.mark.parametrize("field", ["model", "messages", "tools"])
def test_a_protected_field_is_never_dropped(field):
    """Dropping one of these turns a clear 400 into a silently different request."""
    assert (
        diagnose(
            error(
                f"Unsupported parameter: '{field}' is not supported with this model.",
                param=field,
                code="unsupported_parameter",
            )
        )
        is None
    )


def test_a_rename_to_itself_is_refused():
    """Otherwise "use 'temperature' instead" — meaning use another VALUE — loops forever."""
    fix = diagnose(
        error(
            "Unsupported value: 'temperature' does not support 0. Use 'temperature' instead.",
            param="temperature",
            code="unsupported_value",
        )
    )
    assert fix.replacement == ""


def test_apply_reports_whether_it_changed_anything():
    """A fix naming an absent param means retrying would send the identical request."""
    fix = compat.ParamFix(param="logprobs")
    assert fix.apply({"logprobs": True}) is True
    assert fix.apply({"model": "m"}) is False


def test_a_rename_carries_the_value_across():
    body = {"max_tokens": 512}
    compat.ParamFix(param="max_tokens", replacement="max_completion_tokens").apply(body)
    assert body == {"max_completion_tokens": 512}, (
        "losing the value would silently uncap the completion length"
    )


# --- authenticating ---------------------------------------------------------
def test_authorization_gets_a_bearer_prefix():
    assert upstream.auth_headers("sk-abc") == {"Authorization": "Bearer sk-abc"}


def test_a_custom_header_gets_the_raw_key():
    """Anthropic's native route wants `x-api-key` with no prefix."""
    assert upstream.auth_headers("sk-abc", "x-api-key") == {"x-api-key": "sk-abc"}


def test_no_key_means_no_header():
    """A local vLLM needs no credential, and sending an empty Bearer is worse than sending none."""
    assert upstream.auth_headers(None) == {}
    assert upstream.auth_headers("") == {}


# --- the live path: negotiation and retries through the client --------------
class FakeTransport:
    """An httpx transport that replies from a script, recording every request body it saw."""

    def __init__(self, responses):
        import json as _json

        self._json = _json
        self.responses = list(responses)
        self.seen: list[dict] = []
        self.headers: list[dict] = []

    def handle_request(self, request):
        import httpx

        self.seen.append(self._json.loads(request.content or b"{}"))
        self.headers.append(dict(request.headers))
        status, payload = self.responses.pop(0)
        return httpx.Response(
            status,
            json=payload,
            headers={"content-type": "application/json"},
            request=request,
        )


def client_with(responses, **kwargs):
    import httpx

    client = upstream.InferenceClient("http://engine", **kwargs)
    transport = FakeTransport(responses)
    client._client = httpx.AsyncClient(
        base_url="http://engine",
        transport=httpx.MockTransport(transport.handle_request),
        headers=upstream.auth_headers(
            kwargs.get("api_key"), kwargs.get("auth_header", "Authorization")
        ),
    )
    return client, transport


def run(coro):
    import asyncio

    return asyncio.run(coro)


OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def test_a_rejected_param_is_dropped_and_the_call_retried():
    client, transport = client_with(
        [
            (400, error("Unrecognized request argument supplied: return_token_ids")),
            (200, OK),
        ]
    )
    result = run(client.completion({"model": "m", "messages": []}))
    assert result["choices"][0]["message"]["content"] == "ok"
    assert "return_token_ids" in transport.seen[0]
    assert "return_token_ids" not in transport.seen[1]


def test_the_fix_is_cached_so_the_next_call_costs_no_extra_round_trip():
    client, transport = client_with(
        [
            (400, error("Unrecognized request argument supplied: return_token_ids")),
            (200, OK),
            (200, OK),
        ]
    )
    run(client.completion({"model": "m", "messages": []}))
    run(client.completion({"model": "m", "messages": []}))
    assert len(transport.seen) == 3, "the second call must not rediscover the fix"
    assert "return_token_ids" not in transport.seen[2]
    assert [str(f) for f in client.param_fixes] == ["dropped return_token_ids"]


def test_a_400_that_names_nothing_actionable_is_raised():
    client, _ = client_with([(400, error("context length exceeded"))])
    with pytest.raises(upstream.UpstreamHTTPError):
        run(client.completion({"model": "m", "messages": []}))


def test_a_429_is_retried_rather_than_becoming_a_502(monkeypatch):
    """A single 429 used to truncate the agent's trajectory while leaving a well-formed graph."""
    monkeypatch.setattr(upstream.asyncio, "sleep", _no_sleep)
    client, transport = client_with(
        [(429, {"error": {"message": "slow down"}}), (200, OK)]
    )
    result = run(client.completion({"model": "m", "messages": []}))
    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(transport.seen) == 2


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(upstream.asyncio, "sleep", _no_sleep)
    client, transport = client_with([(503, {"error": {"message": "down"}})] * 3)
    with pytest.raises(upstream.UpstreamHTTPError):
        run(client.completion({"model": "m", "messages": []}))
    assert len(transport.seen) == client._MAX_ATTEMPTS


def test_a_404_is_not_retried():
    """Only transient statuses are worth repeating; a wrong route never becomes right."""
    client, transport = client_with([(404, {"error": {"message": "no such route"}})])
    with pytest.raises(upstream.UpstreamHTTPError):
        run(client.completion({"model": "m", "messages": []}))
    assert len(transport.seen) == 1


def test_the_api_key_reaches_the_upstream_request():
    client, transport = client_with([(200, OK)], api_key="sk-secret")
    run(client.completion({"model": "m", "messages": []}))
    assert transport.headers[0]["authorization"] == "Bearer sk-secret"


def test_the_capture_level_governs_what_is_sent():
    client, transport = client_with([(200, OK)], capture_level="text")
    run(client.completion({"model": "m", "messages": []}))
    assert "logprobs" not in transport.seen[0]
    assert "return_token_ids" not in transport.seen[0]


async def _no_sleep(_seconds):
    return None


# --- a provider that names the value it wants ------------------------------
def test_a_named_value_becomes_a_set():
    """gpt-5.6 refuses function tools on the chat route unless reasoning_effort is 'none'.

    Verbatim, and the reason a `set` action exists at all: every tools-bearing call 400s otherwise,
    which for a coding agent is every call that matters. Observed as a rollout that made one model
    call and then sat idle for the full 900s agent timeout with an empty agent log.
    """
    fix = diagnose(
        error(
            "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
            "/v1/chat/completions. To use function tools, use /v1/responses or set "
            "reasoning_effort to 'none'."
        )
    )
    assert (fix.param, fix.action, fix.value) == ("reasoning_effort", "set", "none")


def test_a_set_applies_to_a_body_that_lacks_the_param():
    """The only fix that ADDS a key. The request was rejected for lacking a value, not carrying one."""
    body = {"model": "gpt-5.6-sol", "tools": [{}]}
    fix = compat.ParamFix(param="reasoning_effort", value="none")
    assert fix.apply(body) is True
    assert body["reasoning_effort"] == "none"


def test_a_set_that_is_already_satisfied_reports_no_change():
    """Otherwise the retry loop would resend an identical body and spin on the same 400."""
    fix = compat.ParamFix(param="reasoning_effort", value="none")
    assert fix.apply({"reasoning_effort": "none"}) is False


def test_a_set_never_targets_a_protected_field():
    assert diagnose(error("To fix this, set messages to 'none'.")) is None


def test_the_set_fix_is_applied_and_cached_on_the_live_path():
    client, transport = client_with(
        [
            (
                400,
                error(
                    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
                    "/v1/chat/completions. To use function tools, use /v1/responses or set "
                    "reasoning_effort to 'none'."
                ),
            ),
            (200, OK),
            (200, OK),
        ]
    )
    run(client.completion({"model": "gpt-5.6-sol", "messages": [], "tools": [{}]}))
    run(client.completion({"model": "gpt-5.6-sol", "messages": [], "tools": [{}]}))
    assert "reasoning_effort" not in transport.seen[0]
    assert transport.seen[1]["reasoning_effort"] == "none"
    assert transport.seen[2]["reasoning_effort"] == "none"
    assert [str(f) for f in client.param_fixes] == ["set reasoning_effort=none"]


# --- a provider that quotes differently and uses different vocabulary -------
#
# Anthropic's OpenAI-compat route rejects `temperature` with a GENERIC code, a null `param`, and the
# name in BACKTICKS. Nothing structured identifies the field, so the wording is the only signal —
# and every earlier pattern here was written against OpenAI's punctuation.
ANTHROPIC_DEPRECATED = {
    "error": {
        "code": "invalid_request_error",
        "message": "`temperature` is deprecated for this model.",
        "type": "invalid_request_error",
        "param": None,
    }
}


def test_a_backtick_quoted_deprecated_param_is_dropped():
    fix = diagnose(ANTHROPIC_DEPRECATED)
    assert (fix.param, fix.action) == ("temperature", "drop")


def test_a_generic_error_naming_a_protected_field_is_still_refused():
    """`messages` must not be empty is a real error to surface, not a param to delete."""
    assert (
        diagnose(
            {
                "error": {
                    "code": "invalid_request_error",
                    "message": "`messages` is not supported here",
                    "param": None,
                }
            }
        )
        is None
    )


def test_a_generic_error_with_no_rejection_wording_yields_no_fix():
    """The wording gate is what stops an arbitrary 400 becoming a silently different request."""
    assert (
        diagnose(
            {
                "error": {
                    "code": "invalid_request_error",
                    "message": "The request body is malformed",
                    "param": None,
                }
            }
        )
        is None
    )


def test_backticks_also_work_for_a_rename_and_a_set():
    """Punctuation is a provider's habit, not a semantic difference."""
    renamed = diagnose(
        {
            "error": {
                "message": "Unsupported parameter: `max_tokens` is not supported. "
                "Use `max_completion_tokens` instead.",
                "param": None,
                "code": "unsupported_parameter",
            }
        }
    )
    assert (renamed.param, renamed.replacement) == (
        "max_tokens",
        "max_completion_tokens",
    )
    was_set = diagnose({"error": {"message": "or set reasoning_effort to `none`."}})
    assert (was_set.param, was_set.value) == ("reasoning_effort", "none")


# --- mutually exclusive parameters ------------------------------------------
#
# Found by the compatibility matrix: openclaw and openhands-sdk send BOTH `max_tokens` and
# `max_completion_tokens`, which Anthropic's compat route rejects. It failed every call of those
# rollouts, on that provider only.
CONFLICT = "Setting 'max_tokens' and 'max_completion_tokens' at the same time is not supported."


def test_a_known_conflicting_pair_drops_the_legacy_spelling():
    fix = diagnose(error(CONFLICT, code="invalid_request_error"))
    assert (fix.param, fix.action) == ("max_tokens", "drop")


def test_an_unknown_conflicting_pair_is_refused_rather_than_guessed():
    """Dropping the wrong half of a conflict is a silently different request; the 400 is honest."""
    assert (
        diagnose(error("Setting 'foo' and 'bar' at the same time is not supported."))
        is None
    )


def test_a_parameter_name_is_never_invented_from_prose():
    """The regression this pair of tests exists for.

    With the quotes optional, the fallback matched the bare word before "is not supported" and pulled
    the parameter name **"time"** out of "at the same time is not supported". A provider that means a
    parameter always quotes it, so requiring the quotes costs nothing and stops the compat layer
    deleting fields nobody named.
    """
    fix = diagnose(error(CONFLICT, code="invalid_request_error"))
    assert fix.param != "time"
    assert diagnose(error("the request is not supported")) is None
    assert diagnose(error("this model is deprecated")) is None
