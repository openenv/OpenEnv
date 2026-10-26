"""Anthropic Messages API transformer.

Transforms between Anthropic Messages API and OpenAI Chat Completions API.
Aligned with agent-harness-proxy/src/harness_proxy/transform/anthropic.py.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ..upstream import UpstreamRequestError
from .base import BaseTransformer
from .images import (
    anthropic_content_to_openai_chat,
    openai_chat_content_to_anthropic_blocks,
)
from .reasoning import extract_reasoning_from_anthropic_content, make_signature

# Claude Code SDK leaks `x-anthropic-billing-header: ...cch=<hash>;` as the
# first line of the system prompt. The cch= hash changes per request, so
# rendered prompt tokens drift every turn and prefix_merging can't chain
# multi-turn traces. Strip the line before forwarding to SGLang.
_CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    r"^\s*x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE
)


@dataclass
class _AnthropicToolCallState:
    id: str
    name: str = ""
    anthropic_index: int | None = None
    buffered_arguments: str = ""
    started: bool = False


class AnthropicStreamState:
    """Per-request Anthropic streaming state.

    Anthropic SSE blocks are stateful across chunks: content blocks must be
    explicitly started, optionally receive multiple deltas, and then be closed
    before the final message delta. This helper tracks those open blocks for a
    single upstream OpenAI/SGLang stream.
    """

    def __init__(self, model: str, finish_to_stop_reason: dict[str, str]):
        self.model = model
        self.finish_to_stop_reason = finish_to_stop_reason
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.next_block_index = 0
        self.text_block_index: int | None = None
        self.text_block_started = False
        self.thinking_block_index: int | None = None
        self.thinking_block_started = False
        self.thinking_buffer = ""
        self.tool_calls: dict[int, _AnthropicToolCallState] = {}
        self.stop_reason = "end_turn"
        self.output_tokens = 0
        self.any_block_started = False
        self.completed = False

    def process_chunk(
        self, chunk: dict[str, Any], is_first: bool = False
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if is_first:
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )

        usage = chunk.get("usage", {})
        if usage:
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.stop_reason = self.finish_to_stop_reason.get(finish_reason, "end_turn")

        # Thinking blocks must precede text and tool_use per Anthropic spec.
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self.thinking_block_started:
                events.append(self._open_thinking_block())
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.thinking_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                }
            )
            self.thinking_buffer += reasoning

        content = delta.get("content")
        if content:
            thinking_stop = self._close_thinking_block()
            if thinking_stop:
                events.extend(thinking_stop)
            if not self.text_block_started:
                events.append(self._open_text_block())
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.text_block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
            )

        tool_call_deltas = delta.get("tool_calls") or []
        if not isinstance(tool_call_deltas, list):
            tool_call_deltas = [tool_call_deltas]
        for tool_call_delta in tool_call_deltas:
            if isinstance(tool_call_delta, dict):
                events.extend(self._process_tool_call(tool_call_delta))

        return events

    def finalize(self) -> list[dict[str, Any]]:
        if self.completed:
            return []

        events: list[dict[str, Any]] = []

        thinking_stop = self._close_thinking_block()
        if thinking_stop:
            events.extend(thinking_stop)

        text_stop = self._close_text_block()
        if text_stop:
            events.append(text_stop)

        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if tool_state.started and tool_state.anthropic_index is not None:
                events.append(
                    {
                        "type": "content_block_stop",
                        "index": tool_state.anthropic_index,
                    }
                )

        if not self.any_block_started:
            empty_index = self.next_block_index
            events.append(
                {
                    "type": "content_block_start",
                    "index": empty_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            events.append({"type": "content_block_stop", "index": empty_index})

        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            }
        )
        events.append({"type": "message_stop"})

        self.completed = True
        return events

    def _open_text_block(self) -> dict[str, Any]:
        self.text_block_started = True
        self.text_block_index = self.next_block_index
        self.next_block_index += 1
        self.any_block_started = True
        return {
            "type": "content_block_start",
            "index": self.text_block_index,
            "content_block": {"type": "text", "text": ""},
        }

    def _close_text_block(self) -> dict[str, Any] | None:
        if not self.text_block_started or self.text_block_index is None:
            return None

        event = {"type": "content_block_stop", "index": self.text_block_index}
        self.text_block_started = False
        self.text_block_index = None
        return event

    def _open_thinking_block(self) -> dict[str, Any]:
        self.thinking_block_started = True
        self.thinking_block_index = self.next_block_index
        self.next_block_index += 1
        self.any_block_started = True
        return {
            "type": "content_block_start",
            "index": self.thinking_block_index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }

    def _close_thinking_block(self) -> list[dict[str, Any]] | None:
        if not self.thinking_block_started or self.thinking_block_index is None:
            return None
        idx = self.thinking_block_index
        events = [
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "signature_delta",
                    "signature": make_signature(self.thinking_buffer),
                },
            },
            {"type": "content_block_stop", "index": idx},
        ]
        self.thinking_block_started = False
        self.thinking_block_index = None
        return events

    def _process_tool_call(
        self, tool_call_delta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        tool_index = tool_call_delta.get("index", 0)
        if not isinstance(tool_index, int):
            tool_index = 0

        tool_state = self.tool_calls.get(tool_index)
        if tool_state is None:
            tool_state = _AnthropicToolCallState(
                # `or`, not a get() default: a present-but-null id returns None and defeats the fallback.
                id=tool_call_delta.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            )
            self.tool_calls[tool_index] = tool_state
        elif tool_call_delta.get("id"):
            tool_state.id = tool_call_delta["id"]

        function = tool_call_delta.get("function", {})
        name = function.get("name")
        if isinstance(name, str) and name:
            tool_state.name += name

        args = function.get("arguments")
        args_str = ""
        if isinstance(args, str) and args:
            args_str = args
        elif args not in (None, ""):
            args_str = json.dumps(args)

        if args_str:
            tool_state.buffered_arguments += args_str

        if tool_state.name and not tool_state.started:
            thinking_stop = self._close_thinking_block()
            if thinking_stop:
                events.extend(thinking_stop)

            text_stop = self._close_text_block()
            if text_stop:
                events.append(text_stop)

            tool_state.started = True
            tool_state.anthropic_index = self.next_block_index
            self.next_block_index += 1
            self.any_block_started = True

            events.append(
                {
                    "type": "content_block_start",
                    "index": tool_state.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_state.id,
                        "name": tool_state.name,
                        "input": {},
                    },
                }
            )

            if tool_state.buffered_arguments:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": tool_state.anthropic_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_state.buffered_arguments,
                        },
                    }
                )
                tool_state.buffered_arguments = ""
        elif tool_state.started and args_str and tool_state.anthropic_index is not None:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": tool_state.anthropic_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_str,
                    },
                }
            )

        return events


class AnthropicTransformer(BaseTransformer):
    """Transform between Anthropic and OpenAI API formats."""

    FINISH_TO_STOP_REASON: dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "stop_sequence": "stop_sequence",
    }

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = []

        # Handle system message
        system = body.get("system")
        if system:
            system_content = self._flatten_content(system)
            # Drop Claude Code's per-request billing header line (breaks
            # prefix_merging because cch= changes every turn).
            system_content = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", system_content)
            if system_content:
                messages.append({"role": "system", "content": system_content})

        # Transform messages
        request_messages = body.get("messages", [])
        if not isinstance(request_messages, list):
            raise UpstreamRequestError("Anthropic messages must be an array")
        if any(not isinstance(message, dict) for message in request_messages):
            raise UpstreamRequestError("Anthropic messages must be objects")
        for msg in request_messages:
            transformed = self._transform_message(msg)
            if transformed:
                if isinstance(transformed, list):
                    messages.extend(transformed)
                else:
                    messages.append(transformed)

        result: dict[str, Any] = {
            "messages": messages,
            "max_tokens": body.get("max_tokens", 4096),
        }
        if "model" in body:
            result["model"] = body["model"]

        if "temperature" in body:
            result["temperature"] = body["temperature"]
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "top_k" in body:
            result["top_k"] = body["top_k"]
        if "stop_sequences" in body:
            result["stop"] = body["stop_sequences"]
        if body.get("stream", False):
            result["stream"] = True

        # Anthropic `thinking` request param → enable_thinking on chat template.
        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") in {
            "enabled",
            "adaptive",
        }:
            chat_template_kwargs = dict(result.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            result["chat_template_kwargs"] = chat_template_kwargs

        # Tools. Claude Code sometimes sends tools=[] on compaction/summary
        # turns; forwarding tool_choice without a non-empty tools list makes
        # SGLang reject with "tool_choice only allowed when tools specified".
        if "tools" in body:
            declared_tools = body["tools"]
            if not isinstance(declared_tools, list):
                raise UpstreamRequestError("Anthropic tools must be an array")
            if any(not isinstance(tool, dict) for tool in declared_tools):
                raise UpstreamRequestError("Anthropic tools must be objects")
            tools = self._transform_tools_to_openai(declared_tools)
            if tools:
                result["tools"] = tools
                result["tool_choice"] = self._transform_tool_choice_to_openai(
                    body.get("tool_choice", {"type": "auto"})
                )

        return self._normalize_request(
            result,
            body.get("_served_model"),
        )

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        choices = response.get("choices", [])
        if not choices:
            return self._error_response("No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        content = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    "signature": make_signature(reasoning),
                }
            )

        text = message.get("content")
        if text or (isinstance(text, list) and text):
            content.extend(openai_chat_content_to_anthropic_blocks(text))

        for tool_call in message.get("tool_calls") or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "input": self._parse_json_safe(
                        tool_call.get("function", {}).get("arguments", "{}")
                    ),
                }
            )

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = self.FINISH_TO_STOP_REASON.get(finish_reason, "end_turn")
        usage = response.get("usage", {})
        anthropic_usage = self._usage_to_anthropic(usage)

        if not content:
            content.append({"type": "text", "text": ""})

        return {
            "id": f"msg_{response.get('id', uuid.uuid4().hex)}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_request.get("model", "claude-3"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": anthropic_usage,
        }

    def create_stream_state(
        self, original_request: dict[str, Any]
    ) -> AnthropicStreamState:
        return AnthropicStreamState(
            model=original_request.get("model", "claude-3"),
            finish_to_stop_reason=self.FINISH_TO_STOP_REASON,
        )

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Best-effort single-chunk Anthropic transform.

        The server uses `create_stream_state()` for request-scoped streaming.
        This fallback keeps direct callers working for simple single-chunk cases.
        """
        state = self.create_stream_state(original_request)
        events = state.process_chunk(chunk, is_first=is_first)
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason"):
            events.extend(state.finalize())
        return events

    def _transform_message(self, msg: dict[str, Any]) -> Optional[dict | list]:
        """Transform a single Anthropic message to OpenAI format."""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            return {"role": role, "content": content}

        if not isinstance(content, list):
            return {"role": role, "content": str(content)}

        # Check for mixed content: tool_result blocks + other content
        tool_results = [
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_result"
        ]
        tool_uses = [
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"
        ]
        text_blocks = [
            c for c in content if isinstance(c, dict) and c.get("type") == "text"
        ]

        messages = []

        # Assistant `thinking` blocks → reasoning_content (kept for replay).
        reasoning_text = ""
        if role == "assistant":
            reasoning_text = extract_reasoning_from_anthropic_content(content)

        # Handle assistant messages with tool_use blocks
        if role == "assistant" and tool_uses:
            tool_calls = []
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if reasoning_text:
                msg_dict["reasoning_content"] = reasoning_text
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            return msg_dict

        # Handle user messages with tool_result blocks
        if role == "user" and tool_results:
            # Each tool_result becomes a tool message
            for tr in tool_results:
                tool_content = tr.get("content", "")
                converted_content = anthropic_content_to_openai_chat(tool_content)
                text_content = self._flatten_content(converted_content)
                # Anthropic marks failed tool results with is_error=true.
                # Surface this to the model so it can see the call failed
                # rather than treating the payload as normal output.
                if tr.get("is_error"):
                    text_content = (
                        f"[Tool Error] {text_content}"
                        if text_content
                        else "[Tool Error]"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": text_content,
                    }
                )
                # OpenAI tool messages stay text-only; images are sent as a
                # follow-up user message, so mixed text/image order is not preserved.
                image_parts = self._image_parts(converted_content)
                if image_parts:
                    messages.append({"role": "user", "content": image_parts})

            # Any extra user text should come after the tool results.
            text_parts = [b.get("text", "") for b in text_blocks if b.get("text")]
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
            return messages if messages else None

        # Regular content blocks — keep images when present.
        result: dict[str, Any] = {
            "role": role,
            "content": anthropic_content_to_openai_chat(content),
        }
        if role == "assistant" and reasoning_text:
            result["reasoning_content"] = reasoning_text
        return result

    def _flatten_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(self._flatten_content(block.get("content", "")))
            return "\n".join(parts)
        return str(content)

    def _image_parts(self, content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return []
        return [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]

    def _transform_tools_to_openai(self, tools: list[dict]) -> list[dict]:
        result = []
        for tool in tools:
            # Anthropic server tools (web_search_*, code_execution_*) carry an
            # explicit `type` and have no `input_schema`. SGLang can't dispatch
            # them, so drop rather than forwarding a stub function tool.
            tool_type = tool.get("type")
            if (
                tool_type
                and tool_type not in ("custom", "function")
                and "input_schema" not in tool
            ):
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return result

    def _transform_tool_choice_to_openai(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "auto":
                return "auto"
            elif tc_type == "any":
                return "required"
            elif tc_type == "none":
                return "none"
            elif tc_type == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")},
                }
        return "auto"

    def _parse_json_safe(self, s: str) -> dict:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _usage_to_anthropic(self, usage: dict[str, Any]) -> dict[str, Any]:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cache_read = self._cached_prompt_tokens(usage)
        input_tokens = (
            max(prompt_tokens - cache_read, 0) if cache_read else prompt_tokens
        )

        result: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens,
        }
        if cache_read:
            result["cache_read_input_tokens"] = cache_read
        cache_creation = usage.get("cache_creation_input_tokens")
        if isinstance(cache_creation, int) and cache_creation:
            result["cache_creation_input_tokens"] = cache_creation
        return result

    def _cached_prompt_tokens(self, usage: dict[str, Any]) -> int:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                return cached
        cached = usage.get("cached_tokens")
        return cached if isinstance(cached, int) else 0

    def _error_response(self, message: str) -> dict[str, Any]:
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
