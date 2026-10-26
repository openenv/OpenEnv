"""Native provider semantics and honest capability classification."""

import copy
import json

import httpx
import pytest
from openenv.core.harness.capture.providers import (
    anthropic_request,
    anthropic_response,
    replay_anthropic,
)
from openenv.core.harness.capture.upstream import InferenceClient, UpstreamRequestError
from openenv.core.harness.capture.validate_llm import validate_llm


def native_message():
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [
            {
                "type": "thinking",
                "thinking": "Check first",
                "signature": "signed-original",
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run",
                "input": {"cmd": "pwd"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 7},
    }


def test_native_history_and_signatures_are_preserved_without_mutation():
    original = {
        "model": "alias",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "assistant", "content": native_message()["content"]}],
    }
    before = copy.deepcopy(original)
    body = anthropic_request(
        {"_openenv_native_request": original, "max_tokens": 100}, "pinned"
    )
    assert original == before
    assert body["messages"] == before["messages"]
    assert (
        body["model"] == "pinned"
        and body["max_tokens"] == 100
        and body["stream"] is False
    )


def test_chat_tool_roundtrip_keeps_ids_and_json():
    response = anthropic_response(native_message())
    assistant = response["choices"][0]["message"]
    assistant.pop("reasoning_content")
    body = anthropic_request(
        {
            "messages": [
                assistant,
                {"role": "tool", "tool_call_id": "toolu_1", "content": "/tmp"},
            ]
        },
        "pinned",
    )
    call = body["messages"][0]["content"][0]
    result = body["messages"][1]["content"][0]
    assert call["id"] == result["tool_use_id"] == "toolu_1"
    assert call["input"] == {"cmd": "pwd"}
    assert response["usage"] == {
        "prompt_tokens": 17,
        "completion_tokens": 5,
        "total_tokens": 22,
    }
    assert "prompt_token_ids" not in response
    assert response["choices"][0]["logprobs"] is None


def test_unsigned_reasoning_is_rejected():
    with pytest.raises(UpstreamRequestError, match="signed"):
        anthropic_request(
            {"messages": [{"role": "assistant", "reasoning_content": "thought"}]}, "m"
        )


@pytest.mark.asyncio
async def test_native_wire_uses_messages_and_native_headers():
    def respond(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["anthropic-beta"] == "context-management-2025-06-27"
        body = json.loads(request.content)
        assert body["model"] == "pinned"
        assert "return_token_ids" not in body and "logprobs" not in body
        return httpx.Response(200, json=native_message())

    client = InferenceClient(
        "https://test/v1",
        served_model="pinned",
        api_key="test-secret",
        provider="anthropic",
    )
    http_client = await client._get_client()
    await http_client.aclose()
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(respond),
        headers={"x-api-key": "test-secret", "anthropic-version": "2023-06-01"},
    )
    try:
        response = await client.completion(
            {
                "messages": [{"role": "user", "content": "go"}],
                "_openenv_native_headers": {
                    "anthropic-beta": "context-management-2025-06-27",
                    "x-api-key": "must-not-replace-real-key",
                },
            }
        )
        assert client.capture_level == "text"
        assert response["_openenv_native_response"] == native_message()
    finally:
        await client.aclose()


def test_sse_preserves_real_signature():
    events = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in replay_anthropic(native_message())
    ]
    signatures = [
        event["delta"]["signature"]
        for event in events
        if event.get("delta", {}).get("type") == "signature_delta"
    ]
    assert signatures == ["signed-original"]
    assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"


def test_probe_native_tools_never_certifies_training(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            payload = native_message()
            payload["content"] = [
                {
                    "type": "tool_use",
                    "id": "toolu_probe",
                    "name": "report_ok",
                    "input": {"value": "ok"},
                }
            ]
            return json.dumps(payload).encode()

    def urlopen(request, timeout):
        assert request.full_url == "https://test/v1/messages"
        assert request.get_header("X-api-key") == "key"
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    report = validate_llm(
        "https://test/v1", "claude-test", provider="anthropic", api_key="key"
    )
    assert report.reachable and report.tool_support == "ok"
    assert report.capture_level == "text" and not report.trainable and not report.ok


def test_logprob_permission_fix_removes_dependent_top_logprobs():
    from openenv.core.harness.capture.compat import diagnose

    body = {
        "logprobs": True,
        "top_logprobs": 0,
        "messages": [{"role": "user", "content": "hi"}],
    }
    fix = diagnose(
        {
            "error": {
                "message": "You are not allowed to request logprobs from this model"
            }
        }
    )
    assert fix.apply(body)
    assert body == {"messages": [{"role": "user", "content": "hi"}]}
    assert not fix.apply(body)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frequency_penalty", 1),
        ("presence_penalty", 1),
        ("repetition_penalty", 1.1),
        ("min_p", 0.1),
    ],
)
def test_native_conversion_rejects_sampling_semantics_it_cannot_preserve(key, value):
    with pytest.raises(UpstreamRequestError, match=key):
        anthropic_request({"messages": [], key: value}, "model")


def test_explicit_eval_keeps_capability_but_disables_supervision():
    from openenv.core.harness.capture.export import export_session
    from openenv.core.harness.capture.sessions import rollout_type_for, SessionRegistry

    registry = SessionRegistry()
    session = registry.create(capture_level="tokens", purpose="eval")
    result = export_session(session, capture_level="tokens")
    assert result["capture_level"] == "tokens" and result["rollout_type"] == "eval"
    assert not result["trainable"]
    assert rollout_type_for("auto", "tokens") == "train"
    with pytest.raises(ValueError, match="exact engine"):
        registry.create(capture_level="text", purpose="train")
    with pytest.raises(ValueError, match="sampling"):
        registry.create(
            capture_level="tokens", purpose="eval", sampling={"temperature": 0.8}
        )


@pytest.mark.asyncio
async def test_native_stream_is_consumable_by_anthropic_sdk():
    anthropic = pytest.importorskip("anthropic")
    message = native_message()

    def respond(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(replay_anthropic(message)).encode(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = anthropic.AsyncAnthropic(api_key="test", http_client=http_client)
        async with client.messages.stream(
            model="claude-test",
            max_tokens=64,
            messages=[{"role": "user", "content": "go"}],
        ) as stream:
            reconstructed = await stream.get_final_message()
        assert reconstructed.content[0].signature == "signed-original"
        assert reconstructed.content[1].id == "toolu_1"
        assert reconstructed.content[1].input == {"cmd": "pwd"}
        assert reconstructed.stop_reason == "tool_use"


def test_eval_sampling_requires_explicit_eval_and_preserves_requested_policy():
    from openenv.core.harness.capture.sessions import SessionRegistry

    registry = SessionRegistry()
    policy = {"temperature": 0.8, "top_p": 1, "top_k": -1}
    session = registry.create(
        purpose="eval", capture_level="text", eval_sampling=policy
    )
    assert session.eval_sampling == policy and not session.sampling
    with pytest.raises(ValueError, match="explicit eval"):
        registry.create(purpose="auto", eval_sampling=policy)
    native = anthropic_request(
        {"_openenv_native_request": {"messages": [], **policy}}, "model"
    )
    assert native["temperature"] == 0.8
    assert "top_p" not in native and "top_k" not in native


def test_strict_tools_preserve_constraint_on_native_anthropic():
    request = {
        "messages": [],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    assert anthropic_request(request, "model")["tools"][0]["strict"] is True


@pytest.mark.parametrize(
    "block",
    [
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "server_tool_use", "id": "srv", "name": "web_search", "input": {}},
        {
            "type": "text",
            "text": "cited answer",
            "citations": [{"type": "web_search_result_location"}],
        },
    ],
)
def test_response_semantics_are_not_silently_dropped_across_protocols(block):
    from openenv.core.harness.capture.providers import ProviderConversionError

    native = native_message()
    native["content"].append(block)
    with pytest.raises(ProviderConversionError, match="cannot be preserved"):
        anthropic_response(native)
    response = anthropic_response(native, native_passthrough=True)
    assert response["_openenv_native_response"] == native
    response["_openenv_native_response"]["content"].clear()
    assert native["content"]


def test_native_pause_is_not_translated_to_successful_stop():
    from openenv.core.harness.capture.providers import ProviderConversionError

    native = native_message()
    native["stop_reason"] = "pause_turn"
    with pytest.raises(ProviderConversionError, match="stop reason"):
        anthropic_response(native)
    assert (
        anthropic_response(native, native_passthrough=True)["_openenv_native_response"][
            "stop_reason"
        ]
        == "pause_turn"
    )
