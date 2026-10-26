# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Request shapes that vLLM rejects outright, normalised before they reach it.

Each case here is a real harness sending something an OpenAI-spec engine 400s on. A 400 does not
degrade a rollout, it truncates it: the agent loses the call, and the captured trajectory ends early
while still looking structurally valid. These are cheap to assert and expensive to rediscover.
"""

from __future__ import annotations

import pytest

server = pytest.importorskip("openenv.core.harness.capture.server")

normalise_for_capture = server.normalise_for_capture


def test_stream_is_forced_off():
    """Capture needs one whole response; reassembling ids from SSE deltas corrupts silently."""
    request = {"messages": [], "stream": True}
    normalise_for_capture(request)
    assert request["stream"] is False


def test_stream_options_is_dropped():
    """vLLM 400s on `stream_options` once `stream` is False. opencode sends it on every call."""
    request = {
        "messages": [],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    normalise_for_capture(request)
    assert "stream_options" not in request


def test_empty_tools_is_dropped():
    """kimi-cli sends `tools: []`, which vLLM rejects: 'must not be an empty array'."""
    request = {"messages": [], "tools": []}
    normalise_for_capture(request)
    assert "tools" not in request


def test_empty_functions_is_dropped():
    """The legacy spelling fails the same way."""
    request = {"messages": [], "functions": []}
    normalise_for_capture(request)
    assert "functions" not in request


def test_tool_choice_is_dropped_with_the_tools_it_referenced():
    """`tool_choice` without `tools` is invalid, and means nothing once the list is gone."""
    request = {"messages": [], "tools": [], "tool_choice": "auto"}
    normalise_for_capture(request)
    assert "tools" not in request
    assert "tool_choice" not in request


def test_populated_tools_are_left_alone():
    """The guard must be narrow: dropping real tools would change what the model can do."""
    tools = [{"type": "function", "function": {"name": "bash"}}]
    request = {"messages": [], "tools": tools, "tool_choice": "auto"}
    normalise_for_capture(request)
    assert request["tools"] == tools
    assert request["tool_choice"] == "auto"


def test_absent_tool_keys_are_not_invented():
    """A request with no tool keys must stay that way rather than gain empty ones."""
    request = {"messages": []}
    normalise_for_capture(request)
    assert "tools" not in request
    assert "functions" not in request
    assert "tool_choice" not in request


# --- engine shape differences -----------------------------------------------
upstream = pytest.importorskip("openenv.core.harness.capture.upstream")
normalize_response = upstream.normalize_response


def test_sglang_per_choice_prompt_ids_are_hoisted():
    """SGLang returns the prompt ids on the choice; vLLM returns them at the top level.

    Every reader downstream (`check_upstream_response`, the capture server's `_ingest`, the UI) looks
    only at the top level, so an un-hoisted SGLang response reads as "no prompt ids" and fails
    validation even though the rollout path handles it perfectly well.
    """
    response = {
        "choices": [
            {
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [4, 5],
                "message": {"content": "hi"},
            }
        ]
    }

    out = normalize_response(response)

    assert out["prompt_token_ids"] == [1, 2, 3]


def test_an_engines_own_top_level_prompt_ids_win():
    """vLLM's value must not be overwritten by a choice that also carries one."""
    response = {
        "prompt_token_ids": [9, 9, 9],
        "choices": [{"prompt_token_ids": [1, 2, 3], "message": {"content": "hi"}}],
    }

    assert normalize_response(response)["prompt_token_ids"] == [9, 9, 9]


def test_nothing_is_invented_when_no_choice_carries_prompt_ids():
    response = {"choices": [{"message": {"content": "hi"}}]}
    assert "prompt_token_ids" not in normalize_response(response)


def test_hoisting_survives_a_response_with_no_choices():
    assert normalize_response({}) == {}


def test_parallel_tool_calls_goes_with_an_empty_tools_array():
    """Found by the compatibility matrix: codex against OpenAI failed EVERY call with

        Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only allowed when
        'tools' are specified.

    It sends `tools: []` plus `parallel_tool_calls`; stripping only the empty list left the orphan.
    vLLM ignores the orphan, which is why this survived until a hosted provider was tried.
    """
    server = pytest.importorskip("openenv.core.harness.capture.server")
    body = {
        "model": "m",
        "tools": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }
    server.normalise_for_capture(body)
    assert "tools" not in body
    assert "parallel_tool_calls" not in body
    assert "tool_choice" not in body


def test_parallel_tool_calls_survives_when_tools_are_real():
    """It is only invalid without tools; a genuine manifest must keep its companions."""
    server = pytest.importorskip("openenv.core.harness.capture.server")
    body = {
        "model": "m",
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "parallel_tool_calls": True,
    }
    server.normalise_for_capture(body)
    assert body["parallel_tool_calls"] is True
    assert len(body["tools"]) == 1
