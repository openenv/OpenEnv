"""OpenAI Responses API transformer.

Transforms between OpenAI Responses API (Codex CLI) and OpenAI Chat Completions.
Aligned with agent-harness-proxy/src/harness_proxy/transform/openai_responses.py.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ..upstream import UpstreamRequestError
from .base import BaseTransformer
from .images import openai_responses_input_content_to_chat
from .reasoning import encrypt_reasoning, extract_reasoning_from_responses_item


@dataclass
class _ResponsesToolCallState:
    name: str = ""
    call_id: str = ""
    arguments: str = ""
    started: bool = False
    fc_id: str = ""


class ResponsesStreamState:
    """Per-request Responses API streaming state."""

    def __init__(self, model: str):
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.text_started = False
        self.text_content = ""
        self.message_output_index = 0
        self.output_index_offset = 0
        self.tool_calls: dict[int, _ResponsesToolCallState] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.reasoning_started = False
        self.reasoning_closed = False
        self.reasoning_content = ""
        self.reasoning_id = ""
        self.completed = False

    def process_chunk(
        self, chunk: dict[str, Any], is_first: bool = False
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if is_first:
            events.append(
                {
                    "type": "response.created",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": self.model,
                        "output": [],
                        "usage": self.usage.copy(),
                    },
                }
            )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage["input_tokens"] = usage.get(
                "prompt_tokens", self.usage["input_tokens"]
            )
            self.usage["output_tokens"] = usage.get(
                "completion_tokens", self.usage["output_tokens"]
            )
            self.usage["total_tokens"] = usage.get(
                "total_tokens", self.usage["total_tokens"]
            )

        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        # Reasoning item must come first so the harness sees the chain-of-thought
        # before any output_text or function_call items.
        reasoning_delta = delta.get("reasoning_content")
        if isinstance(reasoning_delta, str) and reasoning_delta:
            if not self.reasoning_started:
                self.reasoning_started = True
                self.reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                events.append(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": self.reasoning_id,
                            "summary": [],
                            "content": [],
                            "status": "in_progress",
                        },
                    }
                )
                events.append(
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": self.reasoning_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    }
                )
            self.reasoning_content += reasoning_delta
            events.append(
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": self.reasoning_id,
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": reasoning_delta,
                }
            )

        content = delta.get("content")
        if content:
            # Close reasoning before opening message.
            events.extend(self._close_reasoning())
            if not self.text_started:
                self.text_started = True
                self.message_output_index = 1 if self.reasoning_started else 0
                self.output_index_offset = self.message_output_index + 1
                message_id = f"msg_{uuid.uuid4().hex[:24]}"
                events.append(
                    {
                        "type": "response.output_item.added",
                        "output_index": self.message_output_index,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                )
                events.append(
                    {
                        "type": "response.content_part.added",
                        "output_index": self.message_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    }
                )

            self.text_content += content
            events.append(
                {
                    "type": "response.output_text.delta",
                    "output_index": self.message_output_index,
                    "content_index": 0,
                    "delta": content,
                }
            )

        tool_calls_delta = delta.get("tool_calls") or []
        if not isinstance(tool_calls_delta, list):
            tool_calls_delta = [tool_calls_delta]

        if tool_calls_delta and self.reasoning_started and not self.reasoning_closed:
            events.extend(self._close_reasoning())
            if not self.text_started:
                # No text — tools come immediately after reasoning.
                self.output_index_offset = 1

        for tool_call in tool_calls_delta:
            if not isinstance(tool_call, dict):
                continue

            tool_index = tool_call.get("index", 0)
            if not isinstance(tool_index, int):
                tool_index = 0

            tool_state = self.tool_calls.get(tool_index)
            if tool_state is None:
                tool_state = _ResponsesToolCallState(
                    # `or`, not a get() default: a tool call whose "id" is present but null returns None from
                    # get(), which defeats the fallback and emits an item with no call_id.
                    call_id=tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                )
                self.tool_calls[tool_index] = tool_state
            elif tool_call.get("id"):
                tool_state.call_id = tool_call["id"]

            function = tool_call.get("function", {})
            name = function.get("name")
            if isinstance(name, str) and name:
                tool_state.name += name

            arguments = function.get("arguments")
            arguments_str = ""
            if isinstance(arguments, str) and arguments:
                arguments_str = arguments
            elif arguments not in (None, ""):
                arguments_str = json.dumps(arguments)
            if arguments_str:
                tool_state.arguments += arguments_str

            output_index = self.output_index_offset + tool_index
            if tool_state.name and not tool_state.started:
                tool_state.started = True
                tool_state.fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                events.append(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": tool_state.fc_id,
                            "call_id": tool_state.call_id,
                            "name": tool_state.name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    }
                )

                if tool_state.arguments:
                    events.append(
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": output_index,
                            "delta": tool_state.arguments,
                        }
                    )
            elif tool_state.started and arguments_str:
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "delta": arguments_str,
                    }
                )

        return events

    def finalize(self) -> list[dict[str, Any]]:
        if self.completed:
            return []

        events: list[dict[str, Any]] = []

        # Close reasoning if it never got closed by content/tools.
        events.extend(self._close_reasoning())

        if self.text_started:
            events.append(
                {
                    "type": "response.content_part.done",
                    "output_index": self.message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.text_content},
                }
            )
            events.append(
                {
                    "type": "response.output_item.done",
                    "output_index": self.message_output_index,
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": self.text_content}],
                    },
                }
            )

        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if not tool_state.started:
                continue

            output_index = self.output_index_offset + tool_index
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "arguments": tool_state.arguments,
                }
            )
            events.append(
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": tool_state.fc_id or f"fc_{uuid.uuid4().hex[:24]}",
                        "call_id": tool_state.call_id,
                        "name": tool_state.name,
                        "arguments": tool_state.arguments,
                        "status": "completed",
                    },
                }
            )

        output: list[dict[str, Any]] = []
        if self.reasoning_started:
            output.append(
                {
                    "type": "reasoning",
                    "id": self.reasoning_id,
                    "summary": [
                        {"type": "summary_text", "text": self.reasoning_content}
                    ],
                    "content": [
                        {"type": "reasoning_text", "text": self.reasoning_content}
                    ],
                    "encrypted_content": encrypt_reasoning(self.reasoning_content),
                    "status": "completed",
                }
            )
        if self.text_started:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": self.text_content}],
                }
            )
        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if tool_state.started:
                output.append(
                    {
                        "type": "function_call",
                        "id": tool_state.fc_id or f"fc_{uuid.uuid4().hex[:24]}",
                        "call_id": tool_state.call_id,
                        "name": tool_state.name,
                        "arguments": tool_state.arguments,
                        "status": "completed",
                    }
                )

        events.append(
            {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "completed",
                    "model": self.model,
                    "output": output,
                    "usage": self.usage.copy(),
                },
            }
        )
        self.completed = True
        return events

    def _close_reasoning(self) -> list[dict[str, Any]]:
        if not self.reasoning_started or self.reasoning_closed:
            return []
        self.reasoning_closed = True
        return [
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": self.reasoning_id,
                "output_index": 0,
                "summary_index": 0,
                "text": self.reasoning_content,
            },
            {
                "type": "response.reasoning_summary_part.done",
                "item_id": self.reasoning_id,
                "output_index": 0,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": self.reasoning_content},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": self.reasoning_id,
                    "summary": [
                        {"type": "summary_text", "text": self.reasoning_content}
                    ],
                    "content": [
                        {"type": "reasoning_text", "text": self.reasoning_content}
                    ],
                    "encrypted_content": encrypt_reasoning(self.reasoning_content),
                    "status": "completed",
                },
            },
        ]


class OpenAIResponsesTransformer(BaseTransformer):
    """Transform OpenAI Responses API to/from SGLang chat completions."""

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        instructions = body.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_data = body.get("input", "")
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            if any(not isinstance(item, dict) for item in input_data):
                raise UpstreamRequestError("Responses input items must be objects")
            messages.extend(self._convert_input_items_to_messages(input_data))
        else:
            raise UpstreamRequestError(
                "Responses input must be a string or an array of objects"
            )

        result: dict[str, Any] = {"messages": messages}
        if "model" in body:
            result["model"] = body["model"]

        if "max_tokens" in body:
            result["max_tokens"] = body["max_tokens"]
        if "max_output_tokens" in body:
            result["max_tokens"] = body["max_output_tokens"]
        if "temperature" in body:
            result["temperature"] = body["temperature"]
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "top_logprobs" in body:
            result["top_logprobs"] = body["top_logprobs"]
        if "parallel_tool_calls" in body:
            result["parallel_tool_calls"] = body["parallel_tool_calls"]
        if "stream" in body:
            result["stream"] = body["stream"]

        text_cfg = body.get("text")
        if isinstance(text_cfg, dict):
            response_format = self._response_format_from_text_config(text_cfg)
            if response_format is not None:
                result["response_format"] = response_format

        # Responses `reasoning` request param → enable_thinking.
        reasoning_cfg = body.get("reasoning")
        if isinstance(reasoning_cfg, dict) and self._reasoning_config_enables_thinking(
            reasoning_cfg
        ):
            chat_template_kwargs = dict(result.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            result["chat_template_kwargs"] = chat_template_kwargs

        # SGLang rejects tool_choice without a non-empty tools list; bind
        # the pair so one can't be forwarded without the other.
        declared_tools = body.get("tools", [])
        if not isinstance(declared_tools, list):
            raise UpstreamRequestError("Responses tools must be an array")
        if any(not isinstance(tool, dict) for tool in declared_tools):
            raise UpstreamRequestError("Responses tools must be objects")
        tools = self._convert_tools(declared_tools)
        if tools:
            result["tools"] = tools
            if "tool_choice" in body:
                result["tool_choice"] = self._tool_choice_to_openai_chat(
                    body["tool_choice"]
                )

        return self._normalize_request(
            result,
            body.get("_served_model"),
        )

    def _response_format_from_text_config(
        self,
        text_cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        format_cfg = text_cfg.get("format")
        if not isinstance(format_cfg, dict):
            return None

        format_type = format_cfg.get("type")
        if format_type == "text":
            return None
        if format_type == "json_object":
            return {"type": "json_object"}
        if format_type != "json_schema":
            return None

        json_schema = format_cfg.get("json_schema")
        if isinstance(json_schema, dict):
            return {"type": "json_schema", "json_schema": json_schema}

        converted = {
            key: format_cfg[key]
            for key in ("name", "description", "schema", "strict")
            if key in format_cfg
        }
        if not converted:
            return None
        return {"type": "json_schema", "json_schema": converted}

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        choices = response.get("choices", [])
        if not choices:
            return self._make_error_response("No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        output_items: list[dict[str, Any]] = []

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            output_items.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                    "content": [{"type": "reasoning_text", "text": reasoning}],
                    "encrypted_content": encrypt_reasoning(reasoning),
                    "status": "completed",
                }
            )

        content = message.get("content")
        if content:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content}],
                }
            )

        # Only replay a call as `local_shell_call` when the REQUEST actually declared a local_shell
        # tool. The name alone is not evidence: a user-defined function called `execute` or
        # `run_command` was being rewritten to a shell call, which diverged the replay from what was
        # sampled — turn k recorded `name="execute"`, turn k+1 replayed `name="shell"`, the exact
        # token prefix no longer matched, and the turn was orphaned into a new root. codex would also
        # have run the caller's own function as a shell command.
        shell_declared = self._declares_local_shell(original_request)
        for tc in message.get("tool_calls") or []:
            func = tc.get("function", {})
            name = func.get("name", "")
            if shell_declared and name in ("shell", "execute", "run_command"):
                output_items.append(self._local_shell_call_from_tool_call(tc))
            else:
                output_items.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:24]}",
                        "call_id": tc.get("id", ""),
                        "name": name,
                        "arguments": func.get("arguments", "{}"),
                        "status": "completed",
                    }
                )

        usage = response.get("usage", {})
        response_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        cached_tokens = self._cached_prompt_tokens(usage)
        if cached_tokens:
            response_usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex}"),
            "object": "response",
            "created_at": response.get("created", int(time.time())),
            "status": "completed",
            "model": original_request.get("model", response.get("model", "unknown")),
            "output": output_items,
            "usage": response_usage,
        }

    def create_stream_state(
        self, original_request: dict[str, Any]
    ) -> ResponsesStreamState:
        return ResponsesStreamState(
            model=original_request.get("model", "unknown"),
        )

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Best-effort single-chunk Responses transform."""
        state = self.create_stream_state(original_request)
        events = state.process_chunk(chunk, is_first=is_first)
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason"):
            events.extend(state.finalize())
        return events

    def _convert_input_items_to_messages(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []
        pending_tool_outputs: list[dict[str, Any]] = []
        pending_input_content: list[dict[str, Any]] = []
        pending_reasoning: str = ""

        for item in items:
            item_type = item.get("type")

            if item_type == "reasoning":
                # A new reasoning item starts a new turn block. If the prior
                # block already has its function_call_output, flush it now so
                # this reasoning attaches to the NEXT function_call, not the
                # previous one. (Otherwise codex's per-fc reasoning gets
                # accumulated and dumped onto the wrong assistant message,
                # breaking the prefix_merging chain.)
                if pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                            pending_reasoning,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""
                reasoning_text = extract_reasoning_from_responses_item(item)
                if reasoning_text:
                    pending_reasoning = (
                        f"{pending_reasoning}\n{reasoning_text}"
                        if pending_reasoning
                        else reasoning_text
                    )
                continue

            if item_type in {"input_text", "input_image"}:
                if pending_tool_calls or pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls, pending_tool_outputs, pending_reasoning
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""
                pending_input_content.append(item)
                continue

            if item_type == "message":
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                if pending_tool_calls or pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls, pending_tool_outputs, pending_reasoning
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""

                role = item.get("role", "user")
                content = openai_responses_input_content_to_chat(
                    item.get("content", "")
                )
                msg: dict[str, Any] = {"role": role, "content": content}
                if role == "assistant" and pending_reasoning:
                    msg["reasoning_content"] = pending_reasoning
                    pending_reasoning = ""
                messages.append(msg)

            elif item_type == "function_call":
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                if pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                            pending_reasoning,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""
                pending_tool_calls.append(
                    {
                        "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )

            elif item_type in {"local_shell_call", "shell_call"}:
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                if pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                            pending_reasoning,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""
                pending_tool_calls.append(self._local_shell_call_to_tool_call(item))

            elif item_type == "function_call_output":
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                pending_tool_outputs.extend(self._function_call_output_messages(item))

            elif item_type in {"local_shell_call_output", "shell_call_output"}:
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                pending_tool_outputs.extend(self._local_shell_output_messages(item))

            else:
                if pending_input_content:
                    messages.extend(self._flush_input_content(pending_input_content))
                    pending_input_content = []
                if pending_tool_calls or pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                            pending_reasoning,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                    pending_reasoning = ""
                converted = self._convert_response_item_to_message(item)
                if isinstance(converted, list):
                    messages.extend(converted)
                elif converted:
                    messages.append(converted)

        if pending_input_content:
            messages.extend(self._flush_input_content(pending_input_content))

        if pending_tool_calls or pending_tool_outputs:
            messages.extend(
                self._flush_tool_block(
                    pending_tool_calls, pending_tool_outputs, pending_reasoning
                )
            )
            pending_reasoning = ""

        # Trailing reasoning with no following assistant message: synthesize one.
        if pending_reasoning:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": pending_reasoning,
                }
            )

        return messages

    def _function_call_output_messages(
        self, item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        output = self._function_call_output_content(item.get("output", ""))
        converted_content = openai_responses_input_content_to_chat(output)
        messages = [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": self._flatten_function_call_output(output),
            }
        ]

        image_parts = self._image_parts(converted_content)
        if image_parts:
            messages.append({"role": "user", "content": image_parts})
        return messages

    def _local_shell_call_to_tool_call(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("call_id")
            or item.get("id")
            or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": self._local_shell_action_to_arguments(item.get("action")),
            },
        }

    def _local_shell_output_messages(
        self, item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        call_id = item.get("call_id") or item.get("id") or ""
        return self._function_call_output_messages(
            {"call_id": call_id, "output": item.get("output", "")}
        )

    def _local_shell_action_to_arguments(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        if not isinstance(action, dict):
            return "{}"

        command = action.get("command")
        if isinstance(command, str):
            stripped = command.strip()
            if stripped.startswith(("{", "[")):
                try:
                    json.loads(stripped)
                    return stripped
                except json.JSONDecodeError:
                    pass
            return json.dumps({"cmd": command})

        commands = action.get("commands")
        if isinstance(commands, list):
            command_values = [cmd for cmd in commands if isinstance(cmd, str)]
            args: dict[str, Any]
            if len(command_values) == 1:
                args = {"cmd": command_values[0]}
            else:
                args = {"commands": command_values}
            for key in ("timeout_ms", "max_output_length"):
                if key in action:
                    args[key] = action[key]
            return json.dumps(args)

        args = {key: value for key, value in action.items() if key != "type"}
        return json.dumps(args) if args else "{}"

    @staticmethod
    def _declares_local_shell(original_request: dict[str, Any]) -> bool:
        """Whether the request offered a local_shell tool, in either spelling.

        codex declares `{"type": "local_shell"}` among its Responses tools. Absent that, a shell-named
        function is an ordinary function and has to be replayed as one.
        """
        for tool in original_request.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") in {"local_shell", "shell"}:
                return True
        return False

    def _local_shell_call_from_tool_call(
        self, tool_call: dict[str, Any]
    ) -> dict[str, Any]:
        function = tool_call.get("function", {})
        arguments = (
            function.get("arguments", "{}") if isinstance(function, dict) else "{}"
        )
        call_id = tool_call.get("id", "")
        return {
            "type": "local_shell_call",
            "id": f"lsh_{uuid.uuid4().hex[:24]}",
            "call_id": call_id,
            "status": "completed",
            "action": self._local_shell_action_from_arguments(arguments),
        }

    def _local_shell_action_from_arguments(self, arguments: Any) -> dict[str, Any]:
        parsed: Any = None
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
        elif isinstance(arguments, dict):
            parsed = arguments

        if isinstance(parsed, dict):
            commands = parsed.get("commands")
            if isinstance(commands, list):
                action = {"commands": [cmd for cmd in commands if isinstance(cmd, str)]}
            else:
                command = parsed.get("cmd") or parsed.get("command")
                action = {"commands": [command]} if isinstance(command, str) else {}
            for key in ("timeout_ms", "max_output_length"):
                if key in parsed:
                    action[key] = parsed[key]
            if action.get("commands"):
                return action

        if isinstance(arguments, str) and arguments:
            return {"commands": [arguments]}
        return {"commands": []}

    def _function_call_output_content(self, output: Any) -> Any:
        if isinstance(output, dict):
            if self._is_responses_content_block(output):
                return [output]
            for key in ("output", "body", "content"):
                if key in output:
                    return self._function_call_output_content(output[key])
        return output

    def _flatten_function_call_output(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            parts = []
            for block in output:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") in {"input_text", "output_text", "text"}:
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
            return "\n".join(parts)
        if isinstance(output, dict):
            return json.dumps(output)
        return str(output) if output is not None else ""

    def _image_parts(self, content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return []
        return [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]

    def _is_responses_content_block(self, block: dict[str, Any]) -> bool:
        return block.get("type") in {
            "input_text",
            "output_text",
            "text",
            "input_image",
            "image_url",
        }

    def _flush_input_content(
        self, content_parts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        content = openai_responses_input_content_to_chat(content_parts)
        return [{"role": "user", "content": content}] if content else []

    def _flush_tool_block(
        self,
        tool_calls: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
        reasoning: str = "",
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": list(tool_calls),
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)
        messages.extend(tool_outputs)
        return messages

    def _convert_response_item_to_message(
        self,
        item: dict[str, Any],
    ) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            content = openai_responses_input_content_to_chat(item.get("content", []))
            if content:
                return {"role": role, "content": content}

        elif item_type == "function_call_output":
            return self._function_call_output_messages(item)

        # Fallback: plain {role, content} dict
        if not item_type and "role" in item and "content" in item:
            role = item["role"]
            content = item["content"]
            if isinstance(content, str):
                return {"role": role, "content": content}
            if isinstance(content, list):
                converted = openai_responses_input_content_to_chat(content)
                if converted:
                    return {"role": role, "content": converted}

        return None

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                converted.append({"type": "function", "function": tool["function"]})
                continue

            tool_type = tool.get("type")
            if tool_type in {"shell", "local_shell"}:
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "description": tool.get(
                                "description",
                                "Run shell commands in the local workspace.",
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "cmd": {"type": "string"},
                                    "commands": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "timeout_ms": {"type": "number"},
                                    "max_output_length": {"type": "number"},
                                },
                            },
                        },
                    }
                )
                continue

            # Drop server-side tool types Polar can't dispatch (web_search,
            # file_search, computer_use, mcp, code_interpreter, image_generation,
            # custom, etc.). Only client-side functions/shell are convertible.
            if tool_type and tool_type != "function":
                continue

            name = tool.get("name") or tool.get("id", "")
            if not name:
                continue

            parameters = tool.get("parameters")
            if parameters is None:
                input_schema = tool.get("inputSchema") or tool.get("input_schema")
                if isinstance(input_schema, dict):
                    json_schema = input_schema.get("jsonSchema")
                    parameters = (
                        json_schema if isinstance(json_schema, dict) else input_schema
                    )
                else:
                    parameters = {}

            func_def: dict[str, Any] = {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": parameters,
            }
            if "strict" in tool:
                func_def["strict"] = tool["strict"]
            converted.append({"type": "function", "function": func_def})

        return converted

    def _tool_choice_to_openai_chat(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, str):
            if tool_choice == "shell":
                return {"type": "function", "function": {"name": "shell"}}
            return tool_choice

        if not isinstance(tool_choice, dict):
            return tool_choice

        choice_type = tool_choice.get("type")
        if choice_type == "function":
            function = tool_choice.get("function")
            if isinstance(function, dict):
                return tool_choice
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "function": {"name": name}}
        if choice_type in {"shell", "local_shell"}:
            return {"type": "function", "function": {"name": "shell"}}
        return tool_choice

    def _reasoning_config_enables_thinking(self, reasoning_cfg: dict[str, Any]) -> bool:
        if not reasoning_cfg:
            return False
        effort = reasoning_cfg.get("effort")
        if isinstance(effort, str) and effort.lower() == "none":
            return False
        return True

    def _cached_prompt_tokens(self, usage: dict[str, Any]) -> int:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                return cached
        cached = usage.get("cached_tokens")
        return cached if isinstance(cached, int) else 0

    def _make_error_response(self, message: str) -> dict[str, Any]:
        return {
            "type": "response.failed",
            "response": {
                "id": "resp_error",
                "object": "response",
                "status": "failed",
                "error": {"code": "internal_error", "message": message},
            },
        }
