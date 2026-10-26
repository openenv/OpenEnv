"""Native upstream conversion; independent of the harness-facing wire protocol."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from .dialects.images import openai_chat_content_to_anthropic_blocks
from .upstream import UpstreamRequestError


class ProviderConversionError(UpstreamRequestError):
    """A request cannot be translated without losing semantics; retrying cannot fix it."""


def _content_blocks(content):
    if content is None:
        return []
    if not isinstance(content, (str, list)):
        raise ProviderConversionError(
            "Anthropic conversion requires string or block-list content"
        )
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict) or part.get("type") not in {
                "text",
                "image_url",
            }:
                raise ProviderConversionError(
                    "Anthropic conversion cannot preserve this content block"
                )
            if part["type"] == "text" and not isinstance(part.get("text"), str):
                raise ProviderConversionError("text content must be a string")
            if part["type"] == "image_url":
                from .dialects.images import openai_chat_image_to_anthropic

                if openai_chat_image_to_anthropic(part) is None:
                    raise ProviderConversionError("image content cannot be converted")
    return openai_chat_content_to_anthropic_blocks(content)


def anthropic_request(request: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert canonical chat input, or preserve an original native Messages request."""
    original = request.get("_openenv_native_request")
    if original is not None:
        body = copy.deepcopy(original)
        body.pop("_served_model", None)
        body["model"] = model
        body["stream"] = False
        if "temperature" in body and body.get("top_p") == 1:
            body.pop("top_p")
        if body.get("top_k") == -1:
            body.pop("top_k")
        # The proxy's cap remains authoritative for native requests too.
        body["max_tokens"] = min(
            body.get("max_tokens", 4096), request.get("max_tokens", 4096)
        )
        return body
    unsupported = [
        k
        for k in ("response_format", "logit_bias", "audio", "modalities")
        if request.get(k)
    ]
    for key, neutral in (
        ("frequency_penalty", 0),
        ("presence_penalty", 0),
        ("repetition_penalty", 1),
        ("min_p", 0),
    ):
        if request.get(key) is not None and request[key] != neutral:
            unsupported.append(key)
    if unsupported:
        raise ProviderConversionError(
            "native Anthropic conversion cannot preserve: " + ", ".join(unsupported)
        )
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": request.get(
            "max_tokens", request.get("max_completion_tokens", 4096)
        ),
        "stream": False,
    }
    for key in ("temperature", "top_p", "top_k"):
        if key in request and request[key] is not None:
            if (key == "top_k" and request[key] == -1) or (
                key == "top_p" and request[key] == 1
            ):
                continue
            body[key] = request[key]
    if "stop" in request:
        stop = request["stop"]
        body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    messages = []
    system = []
    for message in request.get("messages", []):
        role = message["role"]
        content = message.get("content")
        if role in ("system", "developer"):
            if messages:
                raise ProviderConversionError(
                    "Anthropic cannot preserve a system instruction inserted after conversation turns"
                )
            system.extend(_content_blocks(content))
            continue
        if role == "tool":
            native = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": _content_blocks(content),
                    }
                ],
            }
        elif role in ("user", "assistant"):
            blocks = _content_blocks(content) if content else []
            if message.get("reasoning_content") or message.get("reasoning"):
                raise ProviderConversionError(
                    "Anthropic thinking history requires original signed native blocks"
                )
            for call in message.get("tool_calls") or []:
                function = call["function"]
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError as exc:
                        raise ProviderConversionError(
                            "tool arguments must be a JSON object"
                        ) from exc
                if not isinstance(args, dict):
                    raise ProviderConversionError(
                        "tool arguments must be a JSON object"
                    )
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": args,
                    }
                )
            native = {"role": role, "content": blocks}
        else:
            raise ProviderConversionError(f"unsupported role for Anthropic: {role}")
        if messages and messages[-1]["role"] == native["role"]:
            messages[-1]["content"].extend(native["content"])
        else:
            messages.append(native)
    body["messages"] = messages
    if system:
        body["system"] = system
    if request.get("tools"):
        tools = []
        for tool in request["tools"]:
            if tool.get("type") != "function":
                raise ProviderConversionError(
                    "native Anthropic conversion requires function tools"
                )
            function = tool["function"]
            tools.append(
                {
                    "name": function["name"],
                    **({"strict": function["strict"]} if "strict" in function else {}),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        body["tools"] = tools
    choice = request.get("tool_choice")
    if choice in ("auto", "none", "required"):
        body["tool_choice"] = {"type": "any" if choice == "required" else choice}
    elif isinstance(choice, dict):
        body["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    if request.get("parallel_tool_calls") is False and body.get("tools"):
        body.setdefault("tool_choice", {"type": "auto"})[
            "disable_parallel_tool_use"
        ] = True
    return body


def anthropic_response(
    native: dict[str, Any], *, native_passthrough: bool = False
) -> dict[str, Any]:
    """Keep the original response while normalizing text/tool calls for capture."""
    text, reasoning, calls = [], [], []
    if not native_passthrough and native.get("stop_reason") not in {
        None,
        "end_turn",
        "stop_sequence",
        "tool_use",
        "max_tokens",
    }:
        raise ProviderConversionError(
            "Anthropic stop reason cannot be represented by this harness protocol"
        )
    for block in native.get("content", []):
        if not native_passthrough and (
            block.get("type") not in {"text", "thinking", "tool_use"}
            or block.get("citations")
        ):
            raise ProviderConversionError(
                "Anthropic response block cannot be preserved by this harness protocol: "
                + str(block.get("type"))
            )
        if block["type"] == "text":
            text.append(block["text"])
        elif block["type"] == "thinking":
            reasoning.append(block["thinking"])
        elif block["type"] == "tool_use":
            calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"], ensure_ascii=False),
                    },
                }
            )
    usage = native.get("usage", {})
    prompt = sum(
        usage.get(k, 0)
        for k in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )
    output = usage.get("output_tokens", 0)
    return {
        "id": native["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": native["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(text),
                    "reasoning_content": "".join(reasoning) or None,
                    "tool_calls": calls or None,
                },
                "finish_reason": {"tool_use": "tool_calls", "max_tokens": "length"}.get(
                    native.get("stop_reason"), "stop"
                ),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
        },
        "_openenv_native_response": copy.deepcopy(native),
    }


def replay_anthropic(native: dict[str, Any]):
    """Emit SDK-compatible events without replacing native thinking signatures."""

    def frame(event):
        return (
            "event: "
            + event["type"]
            + "\ndata: "
            + json.dumps(event, ensure_ascii=False)
            + "\n\n"
        )

    message = {
        **native,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**native.get("usage", {}), "output_tokens": 0},
    }
    yield frame({"type": "message_start", "message": message})
    for index, block in enumerate(native.get("content", [])):
        initial = dict(block)
        deltas = []
        if block["type"] == "text":
            initial["text"] = ""
            deltas.append({"type": "text_delta", "text": block["text"]})
        elif block["type"] == "tool_use":
            initial["input"] = {}
            deltas.append(
                {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"], ensure_ascii=False),
                }
            )
        elif block["type"] == "thinking":
            initial.update(thinking="", signature="")
            deltas.extend(
                [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            )
        yield frame(
            {"type": "content_block_start", "index": index, "content_block": initial}
        )
        for delta in deltas:
            yield frame({"type": "content_block_delta", "index": index, "delta": delta})
        yield frame({"type": "content_block_stop", "index": index})
    yield frame(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": native.get("stop_reason"),
                "stop_sequence": native.get("stop_sequence"),
            },
            "usage": {"output_tokens": native.get("usage", {}).get("output_tokens", 0)},
        }
    )
    yield frame({"type": "message_stop"})
