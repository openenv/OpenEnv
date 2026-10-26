# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What reaches the engine, and what must not.

Every case here produces a normal-looking response containing nothing trainable, which is the
failure this whole layer exists to prevent. None of them raise, and none of them are visible in a
rollout summary.
"""

from __future__ import annotations

import httpx
import pytest

upstream = pytest.importorskip("openenv.core.harness.capture.upstream")
validate_llm = pytest.importorskip("openenv.core.harness.capture.validate_llm")
dialects = pytest.importorskip("openenv.core.harness.capture.dialects")
detection = pytest.importorskip("openenv.core.harness.capture.detection")

prepare_request = upstream.prepare_request


def chat_transformer():
    return dialects.TransformManager().get(detection.APIType.OPENAI_CHAT)


# --- capture parameters -----------------------------------------------------
def test_capture_parameters_are_forced_on():
    out = prepare_request({"messages": []})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True


@pytest.mark.parametrize(
    "given,expected",
    [({}, 0), ({"top_logprobs": None}, 0), ({"top_logprobs": 5}, 5)],
)
def test_top_logprobs_is_never_left_as_none(given, expected):
    """`setdefault` keeps an explicit `None`, and vLLM then returns no logprobs at all.

    A harness sending `"top_logprobs": null` would get a perfectly normal response whose
    `logprobs.content[]` is empty, so every turn it produced would be silently untrainable.
    """
    assert prepare_request({"messages": [], **given})["top_logprobs"] == expected


def test_prior_thinking_is_renamed_for_vllm():
    """vLLM reads a prior turn's thinking from `reasoning`; the dialects emit `reasoning_content`.

    Without the rename the earlier turn renders as an empty `<think></think>`, so the prompt differs
    from what the model actually produced and prefix matching breaks for the turn after it.
    """
    out = prepare_request(
        {"messages": [{"role": "assistant", "reasoning_content": "thought"}]}
    )
    message = out["messages"][0]
    assert message["reasoning"] == "thought"
    assert "reasoning_content" not in message


# --- the internal marker ----------------------------------------------------
def test_served_model_marker_is_read_during_transform():
    """The marker has to be on the body before the transformer runs, because that is when it reads it.

    It used to be attached afterwards, by the upstream client, so every transformer saw `None`: the
    Qwen3.5 fix below never applied and the marker travelled on to the engine unused.
    """
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(
        {**body, "_served_model": "Qwen/Qwen3.5-9B"}
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_without_the_marker_the_model_specific_fix_cannot_apply():
    """Pins the old behaviour, so the test above cannot pass for the wrong reason."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(dict(body))
    assert "chat_template_kwargs" not in out


def test_the_fix_is_model_specific():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(
        {**body, "_served_model": "meta-llama/Llama-3-8B"}
    )
    assert "enable_thinking" not in (out.get("chat_template_kwargs") or {})


def test_the_marker_never_reaches_the_engine():
    """It is an internal field. An unknown key on the wire is at best noise and at worst a 400."""
    assert "_served_model" not in chat_transformer().transform_request(
        {"model": "m", "messages": [], "_served_model": "Qwen/Qwen3.5-9B"}
    )
    assert "_served_model" not in prepare_request(
        {"messages": [], "_served_model": "leaked"}, served_model="x"
    )


# --- the caller's messages are the ones the graph stores ---------------------
def test_prepare_request_does_not_rewrite_the_callers_messages():
    """`reasoning_content` -> `reasoning` is for the engine only.

    The message dicts handed in here are the same objects the graph stores as `request_messages`.
    Renaming a field in place means the captured turn no longer records what the harness sent, so
    conversation export and any re-tokenisation downstream see the wrong shape.
    """
    message = {"role": "assistant", "content": "hi", "reasoning_content": "thinking"}
    request = {"messages": [message]}

    out = prepare_request(dict(request), served_model="m")

    assert message["reasoning_content"] == "thinking", "caller's message was mutated"
    assert "reasoning" not in message
    # The engine still gets the rename it needs.
    assert out["messages"][0]["reasoning"] == "thinking"
    assert "reasoning_content" not in out["messages"][0]


def test_messages_without_reasoning_are_passed_through_untouched():
    message = {"role": "user", "content": "go"}
    out = prepare_request({"messages": [message]})
    assert out["messages"][0] is message


# --- engine base URL --------------------------------------------------------
#
# The OpenAI SDK rule: a base with no path gets `/v1`, a base with a path is already the API root.
# `http://host:8000/v1` used to become `/v1/v1/models`; a gateway such as Gemini's
# `/v1beta/openai` used to become `/v1beta/openai/v1/chat/completions` and 404.
API_BASES = [
    ("http://localhost:8000", "http://localhost:8000/v1"),
    ("http://localhost:8000/", "http://localhost:8000/v1"),
    ("http://host:8000/v1", "http://host:8000/v1"),
    ("http://host:8000/v1/", "http://host:8000/v1"),
    ("https://router.huggingface.co/v1", "https://router.huggingface.co/v1"),
    ("https://gw.example/openai/v1", "https://gw.example/openai/v1"),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    ("https://api.z.ai/api/paas/v4", "https://api.z.ai/api/paas/v4"),
]


def _mock_client(client, seen):
    def respond(request):
        seen.append(str(request.url))
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(respond)
    )


@pytest.mark.parametrize("given,api_base", API_BASES)
async def test_upstream_routes_keep_the_provider_base_path(given, api_base):
    client = upstream.InferenceClient(given, capture_level="text")
    seen: list[str] = []
    _mock_client(client, seen)
    try:
        await client.list_models()
        await client.completion({"messages": [{"role": "user", "content": "go"}]})
    finally:
        await client.aclose()
    assert seen == [f"{api_base}/models", f"{api_base}/chat/completions"]


@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        # Anthropic's SDK convention: the base excludes `/v1`, so a gateway path is kept and
        # `/v1/messages` goes after it.
        (
            "https://gw.example/api/anthropic",
            "https://gw.example/api/anthropic/v1/messages",
        ),
    ],
)
async def test_anthropic_route_keeps_the_anthropic_base_convention(given, expected):
    client = upstream.InferenceClient(given, provider="anthropic")
    seen: list[str] = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(respond)
    )
    try:
        await client.completion(
            {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        )
    finally:
        await client.aclose()
    assert seen == [expected]


@pytest.mark.parametrize("given,api_base", API_BASES)
def test_validate_llm_probes_keep_the_provider_base_path(monkeypatch, given, api_base):
    listed: list[str] = []
    posted: list[str] = []

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "m"}]}'

    def urlopen(request, timeout):
        listed.append(request.full_url)
        return Listing()

    def fake_post(url, body, timeout, api_key=None, auth_header="Authorization"):
        posted.append(url)
        return {
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                    "token_ids": [7],
                    "logprobs": {"content": [{"logprob": -0.1, "top_logprobs": []}]},
                }
            ],
        }

    monkeypatch.setattr(validate_llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(validate_llm, "_post", fake_post)
    validate_llm.validate_llm(given, "m")
    assert listed == [f"{api_base}/models"]
    # The completion, tool and logprobs-mode probes all go to the same route.
    assert len(posted) > 1
    assert set(posted) == {f"{api_base}/chat/completions"}


# --- capture level: what a provider will actually accept --------------------
#
# Every case below was observed against a live endpoint. The `text` level exists because current
# OpenAI models answer `logprobs` with a 400, and the `tokens` level cannot be sent to them at all:
# `return_token_ids` is rejected outright, so injecting it unconditionally meant every single call
# failed rather than merely losing its token ids.
def test_tokens_level_is_the_existing_behaviour():
    out = prepare_request({"messages": []}, capture_level="tokens")
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_logprobs_level_asks_for_logprobs_but_not_token_ids():
    out = prepare_request({"messages": []}, capture_level="logprobs")
    assert out["logprobs"] is True
    assert out["top_logprobs"] == 0
    assert "return_token_ids" not in out


def test_text_level_injects_nothing():
    out = prepare_request({"messages": [], "model": "m"}, capture_level="text")
    assert out == {"messages": [], "model": "m"}


def test_text_level_strips_prior_thinking_instead_of_renaming_it():
    """`reasoning`/`reasoning_content` is a vLLM accommodation, and a non-standard message field is
    the same 400 hazard as a non-standard top-level param."""
    out = prepare_request(
        {"messages": [{"role": "assistant", "reasoning_content": "thinking"}]},
        capture_level="text",
    )
    message = out["messages"][0]
    assert "reasoning_content" not in message
    assert "reasoning" not in message


def test_text_level_still_does_not_rewrite_the_callers_messages():
    original = {"role": "assistant", "reasoning_content": "thinking"}
    messages = [original]
    prepare_request({"messages": messages}, capture_level="text")
    assert original["reasoning_content"] == "thinking", (
        "the caller's message objects are the ones the graph stores; rewriting them in place "
        "means a captured turn no longer records what the harness sent"
    )


# --- sampling knobs that bias a processed logprob -------------------------------------------------
#
# vLLM masks the top-p/top-k tail to -inf and takes the log-softmax afterwards, so under
# `--logprobs-mode processed_logprobs` a captured logprob is renormalised over the surviving set:
# `log p_full(t) - log(kept_mass)`. A trainer recomputing over the full vocabulary gets
# `log p_full(t)`, so GRPO's step-0 importance ratio lands on `kept_mass` instead of 1 — a silent,
# systematic bias that grows with how aggressive the truncation is. Penalties are worse still, since
# they reorder rather than merely shift.
#
# At the train tier these are therefore neutralised: an on-policy rollout has to sample from the
# distribution being trained. At the eval tiers they are left alone — eval should score exactly the
# configuration the harness asked for.


def test_the_train_tier_neutralises_truncation():
    body = upstream.prepare_request(
        {"model": "m", "messages": [], "top_p": 0.95, "top_k": 40, "min_p": 0.05},
        capture_level="tokens",
    )
    assert body["top_p"] == 1.0
    assert body["top_k"] == -1
    assert body["min_p"] == 0.0


def test_the_train_tier_neutralises_penalties():
    body = upstream.prepare_request(
        {
            "model": "m",
            "messages": [],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "repetition_penalty": 1.1,
        },
        capture_level="tokens",
    )
    assert body["frequency_penalty"] == 0.0
    assert body["presence_penalty"] == 0.0
    assert body["repetition_penalty"] == 1.0


def test_temperature_survives_the_train_tier():
    """Temperature scales the whole distribution without truncating it, so a processed logprob under
    it is still the sampling distribution's own — and it is the policy the trainer means to learn."""
    body = upstream.prepare_request(
        {"model": "m", "messages": [], "temperature": 0.7}, capture_level="tokens"
    )
    assert body["temperature"] == 0.7


@pytest.mark.parametrize("level", ["logprobs", "text"])
def test_the_eval_tiers_honour_what_the_harness_asked_for(level):
    body = upstream.prepare_request(
        {"model": "m", "messages": [], "top_p": 0.8, "frequency_penalty": 0.5},
        capture_level=level,
    )
    assert body["top_p"] == 0.8, (
        "an eval rollout must score the requested configuration"
    )
    assert body["frequency_penalty"] == 0.5


def test_the_callers_dict_is_not_edited():
    """The graph stores the harness's own request objects; overriding a copy is what keeps the trace
    honest about what was asked for."""
    original = {"model": "m", "messages": [], "top_p": 0.95}
    upstream.prepare_request(dict(original), capture_level="tokens")
    assert original["top_p"] == 0.95


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"top_p": 1.0, "top_k": -1, "min_p": 0.0},
        {"top_k": 0},  # 0 and -1 both mean "off", depending on engine
        {"top_p": None, "frequency_penalty": None},
        {
            "top_p": "0.9"
        },  # a string cannot be compared numerically; leave it to the engine
    ],
)
def test_nothing_is_flagged_when_nothing_truncates(body):
    assert upstream.truncating_params(body) == {}


def test_what_was_requested_is_reported_verbatim():
    found = upstream.truncating_params({"top_p": 0.95, "temperature": 0.7, "top_k": 40})
    assert found == {"top_p": 0.95, "top_k": 40}
