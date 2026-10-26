"""Google Generative AI API transformer with tool-call support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .base import BaseTransformer
from .images import (
    google_content_parts_to_openai_chat,
    google_part_to_openai_chat,
    openai_chat_content_to_google_parts,
)
from .reasoning import extract_reasoning_from_gemini_parts, make_google_signature


@dataclass
class _GoogleToolCallState:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class _GoogleStreamState:
    """Accumulate streamed OpenAI tool-call deltas into Google response parts."""

    def __init__(self, transformer: "GoogleTransformer") -> None:
        self._transformer = transformer
        self._tool_calls: dict[int, _GoogleToolCallState] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._emitted_tool_calls = False
        self._emitted_finish_reason = False

    def process_chunk(
        self,
        chunk: dict[str, Any],
        *,
        is_first: bool = False,
    ) -> list[dict[str, Any]]:
        del is_first

        choices = chunk.get("choices", [])
        if not choices:
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                self._usage = usage
            return []

        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self._usage = usage

        parts: list[dict[str, Any]] = []
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            parts.append(
                {
                    "thought": True,
                    "text": reasoning,
                    "thoughtSignature": make_google_signature(reasoning),
                }
            )
        content = delta.get("content")
        if isinstance(content, str) and content:
            parts.append({"text": content})

        for tool_call in self._transformer._normalize_tool_call_deltas(
            delta.get("tool_calls")
        ):
            tool_index = tool_call.get("index", 0)
            if not isinstance(tool_index, int):
                tool_index = 0
            state = self._tool_calls.setdefault(tool_index, _GoogleToolCallState())

            if isinstance(tool_call.get("id"), str) and tool_call["id"]:
                state.call_id = tool_call["id"]

            function = tool_call.get("function", {})
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    state.name += name

                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    state.arguments += arguments
                elif arguments not in (None, ""):
                    state.arguments += json.dumps(arguments)

        finish_reason = choice.get("finish_reason")
        current_finish_reason = (
            finish_reason if isinstance(finish_reason, str) and finish_reason else None
        )
        if current_finish_reason:
            self._finish_reason = current_finish_reason

        if parts:
            if current_finish_reason:
                self._emitted_finish_reason = True
            return [
                self._transformer._build_stream_response(
                    parts,
                    finish_reason=current_finish_reason,
                    usage=self._usage,
                )
            ]

        if self._finish_reason and self._tool_calls and not self._emitted_tool_calls:
            self._emitted_tool_calls = True
            tool_parts = [
                self._transformer._tool_call_part(
                    name=state.name,
                    arguments=state.arguments,
                    call_id=state.call_id,
                )
                for _, state in sorted(self._tool_calls.items())
                if state.name
            ]
            if tool_parts:
                if self._finish_reason:
                    self._emitted_finish_reason = True
                return [
                    self._transformer._build_stream_response(
                        tool_parts,
                        finish_reason=self._finish_reason,
                        usage=self._usage,
                    )
                ]

        if current_finish_reason and not self._emitted_finish_reason:
            self._emitted_finish_reason = True
            return [
                self._transformer._build_stream_response(
                    [],
                    finish_reason=current_finish_reason,
                    usage=self._usage,
                )
            ]

        return []

    def finalize(self) -> list[dict[str, Any]]:
        if self._tool_calls and not self._emitted_tool_calls:
            self._emitted_tool_calls = True
            tool_parts = [
                self._transformer._tool_call_part(
                    name=state.name,
                    arguments=state.arguments,
                    call_id=state.call_id,
                )
                for _, state in sorted(self._tool_calls.items())
                if state.name
            ]
            if tool_parts:
                if self._finish_reason:
                    self._emitted_finish_reason = True
                return [
                    self._transformer._build_stream_response(
                        tool_parts,
                        finish_reason=self._finish_reason,
                        usage=self._usage,
                    )
                ]
        return []


class GoogleTransformer(BaseTransformer):
    """Transform between Google Generative AI and OpenAI API formats."""

    ROLE_MAP = {
        "user": "user",
        "model": "assistant",
        "system": "system",
        "developer": "system",
    }
    FINISH_REASON_MAP_REVERSE = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "content_filter": "SAFETY",
        "tool_calls": "STOP",
        "stop_sequence": "STOP",
    }

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        config = body.get("config")
        config_section = config if isinstance(config, dict) else {}

        system_instruction = (
            body.get("systemInstruction")
            or body.get("system_instruction")
            or config_section.get("systemInstruction")
            or config_section.get("system_instruction")
        )
        system_text = self._extract_system_instruction_text(system_instruction)
        if system_text:
            messages.append({"role": "system", "content": system_text})

        for content in body.get("contents", []):
            messages.extend(self._convert_content_to_messages(content))

        result: dict[str, Any] = {"messages": messages}
        if "model" in body:
            result["model"] = body["model"]

        gen_config: dict[str, Any] = {}
        for source in (
            config_section.get("generationConfig"),
            body.get("generationConfig"),
        ):
            if isinstance(source, dict):
                gen_config.update(source)
        if not gen_config and config_section:
            gen_config = config_section

        if "maxOutputTokens" in gen_config:
            result["max_tokens"] = gen_config["maxOutputTokens"]
        if "temperature" in gen_config:
            result["temperature"] = gen_config["temperature"]
        if "topP" in gen_config:
            result["top_p"] = gen_config["topP"]
        if "topK" in gen_config:
            result["top_k"] = gen_config["topK"]
        if "stopSequences" in gen_config:
            result["stop"] = gen_config["stopSequences"]
        if "candidateCount" in gen_config:
            result["n"] = gen_config["candidateCount"]
        if "presencePenalty" in gen_config:
            result["presence_penalty"] = gen_config["presencePenalty"]
        if "frequencyPenalty" in gen_config:
            result["frequency_penalty"] = gen_config["frequencyPenalty"]
        if "seed" in gen_config:
            result["seed"] = gen_config["seed"]
        if "logprobs" in gen_config:
            result["top_logprobs"] = gen_config["logprobs"]

        response_format = self._convert_response_format(gen_config)
        if response_format is not None:
            result["response_format"] = response_format

        # Gemini `thinkingConfig.includeThoughts: true` → enable_thinking.
        thinking_cfg = gen_config.get("thinkingConfig") or config_section.get(
            "thinkingConfig"
        )
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("includeThoughts"):
            chat_template_kwargs = dict(result.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            result["chat_template_kwargs"] = chat_template_kwargs

        # SGLang rejects tool_choice without a non-empty tools list; bind
        # the pair so one can't be forwarded without the other.
        tools = self._convert_tools(
            body.get("tools") or config_section.get("tools") or []
        )
        if tools:
            result["tools"] = tools
            tool_choice = self._convert_tool_choice(
                body.get("toolConfig") or config_section.get("toolConfig") or {}
            )
            if tool_choice is not None:
                result["tool_choice"] = tool_choice

        return self._normalize_request(
            result,
            body.get("_served_model"),
        )

    def _convert_response_format(
        self, gen_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        response_format_cfg = gen_config.get("responseFormat")
        if isinstance(response_format_cfg, dict):
            text_format = response_format_cfg.get("text")
            if isinstance(text_format, dict):
                mime_type = text_format.get("mimeType")
                if self._is_json_mime_type(mime_type):
                    schema = text_format.get("schema")
                    return self._chat_response_format_from_schema(schema)

        mime_type = gen_config.get("responseMimeType")
        if not self._is_json_mime_type(mime_type):
            return None

        schema = (
            gen_config.get("responseJsonSchema")
            or gen_config.get("_responseJsonSchema")
            or gen_config.get("responseSchema")
        )
        return self._chat_response_format_from_schema(schema)

    @staticmethod
    def _is_json_mime_type(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return (
            value.lower() == "application/json" or value.upper() == "APPLICATION_JSON"
        )

    def _chat_response_format_from_schema(self, schema: Any) -> dict[str, Any]:
        if isinstance(schema, dict) and schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "google_response",
                    "schema": self._normalize_google_schema(schema),
                },
            }
        return {"type": "json_object"}

    def _normalize_google_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                normalized[key] = value.lower()
            elif key == "properties" and isinstance(value, dict):
                normalized[key] = {
                    prop_name: self._normalize_google_schema(prop_schema)
                    if isinstance(prop_schema, dict)
                    else prop_schema
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                normalized[key] = self._normalize_google_schema(value)
            elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
                normalized[key] = [
                    self._normalize_google_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                normalized[key] = value
        return normalized

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        del original_request

        candidates = []
        for i, choice in enumerate(response.get("choices", [])):
            message = choice.get("message", {})
            parts = []
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                parts.append(
                    {
                        "thought": True,
                        "text": reasoning,
                        "thoughtSignature": make_google_signature(reasoning),
                    }
                )
            content = message.get("content")
            if content or isinstance(content, list):
                parts.extend(openai_chat_content_to_google_parts(content))
            parts.extend(self._tool_call_parts_from_message(message))

            finish_reason = choice.get("finish_reason", "stop")
            google_finish = self.FINISH_REASON_MAP_REVERSE.get(finish_reason, "STOP")

            candidates.append(
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": google_finish,
                    "index": i,
                    "safetyRatings": [],
                }
            )

        usage = response.get("usage", {})
        usage_metadata = {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }
        cached_tokens = self._cached_prompt_tokens(usage)
        if cached_tokens:
            usage_metadata["cachedContentTokenCount"] = cached_tokens
        result = {
            "candidates": candidates,
            "usageMetadata": usage_metadata,
        }
        function_calls = self._response_function_calls(candidates)
        if function_calls:
            result["functionCalls"] = function_calls
        return result

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any]:
        del original_request, is_first

        candidates = []
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {}) or {}
            parts = []
            reasoning_chunk = delta.get("reasoning_content")
            if isinstance(reasoning_chunk, str) and reasoning_chunk:
                parts.append(
                    {
                        "thought": True,
                        "text": reasoning_chunk,
                        "thoughtSignature": make_google_signature(reasoning_chunk),
                    }
                )
            content = delta.get("content")
            if content:
                parts.append({"text": content})
            for tool_call in self._normalize_tool_call_deltas(delta.get("tool_calls")):
                function = tool_call.get("function", {})
                parts.append(
                    self._tool_call_part(
                        name=str(function.get("name") or ""),
                        arguments=function.get("arguments", ""),
                        call_id=str(tool_call.get("id") or ""),
                    )
                )

            candidate: dict[str, Any] = {
                "content": {"parts": parts, "role": "model"},
                "index": choice.get("index", 0),
            }
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                candidate["finishReason"] = self.FINISH_REASON_MAP_REVERSE.get(
                    finish_reason, "STOP"
                )
            candidates.append(candidate)

        result: dict[str, Any] = {"candidates": candidates}
        usage = chunk.get("usage")
        if usage:
            result["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }
        function_calls = self._response_function_calls(candidates)
        if function_calls:
            result["functionCalls"] = function_calls
        return result

    def create_stream_state(
        self, original_request: dict[str, Any]
    ) -> _GoogleStreamState:
        del original_request
        return _GoogleStreamState(self)

    def is_streaming_request(self, body: dict[str, Any]) -> bool:
        """Part of the transformer interface, but NOT how the proxy decides.

        Google signals streaming in the URL (`:streamGenerateContent`, `?alt=sse`) rather than the
        body, so `server.wants_stream` inspects the target path and this never sees the information it
        would need. It returns False rather than guessing, and the dead `_streaming` body flag it used
        to read was removed: nothing set it, and `normalise_for_capture` forces `stream=False` on every
        upstream call regardless, so the branch could not have taken effect even if something had.
        """
        return False

    def _convert_content_to_messages(self, content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, dict):
            return []

        parts = content.get("parts", [])
        role = content.get("role", "user")
        openai_role = self.ROLE_MAP.get(role, "user")
        messages: list[dict[str, Any]] = []
        user_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []

        for part in parts:
            if isinstance(part, str):
                user_parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                continue
            if "text" in part and isinstance(part["text"], str):
                user_parts.append({"type": "text", "text": part["text"]})
                continue
            image_part = google_part_to_openai_chat(part)
            if image_part:
                user_parts.append(image_part)
                continue
            if "functionCall" in part and isinstance(part["functionCall"], dict):
                function_call = part["functionCall"]
                tool_calls.append(
                    {
                        "id": function_call.get("id")
                        or function_call.get("call_id")
                        or "",
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", ""),
                            "arguments": json.dumps(function_call.get("args", {})),
                        },
                    }
                )
                continue
            if "functionResponse" in part and isinstance(
                part["functionResponse"], dict
            ):
                function_response = part["functionResponse"]
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_response.get("id")
                        or function_response.get("call_id")
                        or function_response.get("name", ""),
                        "content": json.dumps(function_response.get("response", {})),
                    }
                )

        if openai_role == "assistant":
            message_content = google_content_parts_to_openai_chat(parts)
            reasoning_text = extract_reasoning_from_gemini_parts(parts)
            if message_content or tool_calls or reasoning_text:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message_content,
                }
                if reasoning_text:
                    assistant_message["reasoning_content"] = reasoning_text
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
        elif openai_role == "system":
            system_text = self._extract_text_from_parts(parts)
            if system_text:
                messages.append({"role": "system", "content": system_text})
        else:
            if user_parts:
                messages.append(
                    {
                        "role": "user",
                        "content": google_content_parts_to_openai_chat(parts),
                    }
                )
            messages.extend(tool_messages)

        return messages

    def _convert_tools(self, google_tools: list[Any]) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in google_tools:
            if not isinstance(tool, dict):
                continue
            declarations = tool.get("functionDeclarations") or tool.get(
                "function_declarations"
            )
            if not isinstance(declarations, list):
                continue
            for declaration in declarations:
                if not isinstance(declaration, dict):
                    continue
                name = declaration.get("name")
                if not isinstance(name, str) or not name:
                    continue
                function: dict[str, Any] = {"name": name}
                description = declaration.get("description")
                if isinstance(description, str) and description:
                    function["description"] = description
                parameters = (
                    declaration.get("parameters")
                    or declaration.get("parametersJsonSchema")
                    or declaration.get("parameters_json_schema")
                )
                if not isinstance(parameters, dict):
                    parameters = {"type": "object", "properties": {}}
                function["parameters"] = parameters
                openai_tools.append({"type": "function", "function": function})
        return openai_tools

    def _convert_tool_choice(self, tool_config: Any) -> Any | None:
        if not isinstance(tool_config, dict):
            return None
        function_calling_config = tool_config.get(
            "functionCallingConfig"
        ) or tool_config.get("function_calling_config")
        if not isinstance(function_calling_config, dict):
            return None

        mode = str(function_calling_config.get("mode", "")).upper()
        allowed_names = function_calling_config.get(
            "allowedFunctionNames"
        ) or function_calling_config.get("allowed_function_names")
        if mode == "NONE":
            return "none"
        if mode in {"ANY", "VALIDATED"}:
            if isinstance(allowed_names, list) and len(allowed_names) == 1:
                allowed_name = allowed_names[0]
                if isinstance(allowed_name, str) and allowed_name:
                    return {"type": "function", "function": {"name": allowed_name}}
            return "required"
        return None

    def _tool_call_parts_from_message(
        self, message: dict[str, Any]
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for tool_call in message.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            parts.append(
                self._tool_call_part(
                    name=name,
                    arguments=function.get("arguments", ""),
                    call_id=str(tool_call.get("id") or ""),
                )
            )
        return parts

    def _tool_call_part(
        self, *, name: str, arguments: Any, call_id: str
    ) -> dict[str, Any]:
        function_call: dict[str, Any] = {
            "name": name,
            "args": self._parse_arguments(arguments),
        }
        if call_id:
            function_call["id"] = call_id
        return {"functionCall": function_call}

    def _parse_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, (dict, list, int, float, bool)) or arguments is None:
            return arguments if arguments is not None else {}
        if not isinstance(arguments, str):
            return {"value": arguments}
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": stripped}

    def _normalize_tool_call_deltas(self, tool_calls: Any) -> list[dict[str, Any]]:
        if isinstance(tool_calls, list):
            return [
                tool_call for tool_call in tool_calls if isinstance(tool_call, dict)
            ]
        if isinstance(tool_calls, dict):
            return [tool_calls]
        return []

    def _build_stream_response(
        self,
        parts: list[dict[str, Any]],
        *,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "content": {"parts": parts, "role": "model"},
            "index": 0,
        }
        if finish_reason:
            candidate["finishReason"] = self.FINISH_REASON_MAP_REVERSE.get(
                finish_reason, "STOP"
            )
        result: dict[str, Any] = {"candidates": [candidate]}
        if usage:
            result["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }
        function_calls = self._response_function_calls(result["candidates"])
        if function_calls:
            result["functionCalls"] = function_calls
        return result

    def _response_function_calls(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        function_calls: list[dict[str, Any]] = []
        for candidate in candidates:
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []) or []:
                if not isinstance(part, dict):
                    continue
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    function_calls.append(function_call)
        return function_calls

    def _extract_text_from_parts(self, parts: list) -> str:
        texts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)

    def _extract_system_instruction_text(self, system_instruction: Any) -> str:
        if isinstance(system_instruction, str):
            return system_instruction
        if isinstance(system_instruction, dict):
            return self._extract_text_from_parts(system_instruction.get("parts", []))
        if isinstance(system_instruction, list):
            return self._extract_text_from_parts(system_instruction)
        return ""

    def _cached_prompt_tokens(self, usage: dict[str, Any]) -> int:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                return cached
        cached = usage.get("cached_tokens")
        return cached if isinstance(cached, int) else 0
