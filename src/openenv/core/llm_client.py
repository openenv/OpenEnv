# SPDX-License-Identifier: BSD-3-Clause

"""LLM client abstraction for calling LLM endpoints.

Provides a generic RPC abstraction: point it at an endpoint/port, tell it the
protocol, and it works. OpenAI-compatible API is the first implementation,
covering OpenAI, vLLM, TGI, Ollama, HuggingFace Inference API, etc.
Anthropic's native API is supported via `AnthropicClient`.

Examples:

    ```python
    client = OpenAIClient("http://localhost", 8000, model="meta-llama/...")
    response = await client.complete("What is 2+2?")

    # The endpoint may carry its own port and path prefix:
    client = OpenAIClient("http://localhost:8000/v1", port=None, model="meta-llama/...")

    # Or use the factory for hosted APIs:
    client = create_llm_client("openai", model="gpt-4", api_key="sk-...")
    response = await client.complete_with_tools(messages, tools)
    ```
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI

_OPENAI_API_PREFIX = "/v1"
_MIN_PORT, _MAX_PORT = 1, 65535


def _redact_userinfo(endpoint: str) -> str:
    """Drop `user:password@` from a URL so it is safe to echo in an error."""
    scheme, sep, rest = endpoint.partition("://")
    authority, slash, tail = rest.partition("/")
    if not sep or "@" not in authority:
        return endpoint
    return f"{scheme}{sep}***@{authority.rsplit('@', 1)[-1]}{slash}{tail}"


def _join_endpoint_port(endpoint: str, port: int | None) -> str:
    """Validate an endpoint URL and combine it with an optional port.

    Only `http`/`https` URLs with a host are accepted. Credentials, query
    strings and fragments are rejected: the endpoint is logged and persisted in
    rollout metadata, and the SDKs build request URLs by appending to the path,
    which corrupts a query string. `port` is appended only when the URL does
    not name one; an explicit port that differs from the one in the URL is
    rejected rather than silently overridden. A trailing slash on the path is
    dropped. Invalid endpoints raise `ValueError`.
    """
    safe = _redact_userinfo(endpoint)
    try:
        parts = urlsplit(endpoint)
        if parts.scheme not in ("http", "https"):
            raise ValueError("expected an http:// or https:// URL")
        if not parts.hostname:
            raise ValueError("missing host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("credentials in the URL are not supported, use api_key")
        if parts.query or parts.fragment:
            raise ValueError("query strings and fragments are not supported")
        if parts.netloc.endswith(":"):
            raise ValueError("empty port")
        url_port = parts.port
        for candidate in (url_port, port):
            if candidate is not None and not _MIN_PORT <= candidate <= _MAX_PORT:
                raise ValueError(
                    f"port {candidate} is out of range {_MIN_PORT}-{_MAX_PORT}"
                )
    except ValueError as exc:
        raise ValueError(f"Invalid endpoint URL {safe!r}: {exc}") from exc

    if url_port is None:
        if port is not None:
            parts = parts._replace(netloc=f"{parts.netloc}:{port}")
    elif port is not None and port != url_port:
        raise ValueError(
            f"Endpoint URL {safe!r} already specifies port {url_port}, "
            f"which conflicts with port={port}"
        )
    return urlunsplit(parts._replace(path=parts.path.rstrip("/")))


def _openai_base_url(base_url: str) -> str:
    """Append the OpenAI `/v1` prefix when `base_url` has no path.

    A URL that already names a path (`/v1`, or a gateway prefix such as
    `/openai/v1`) is used as-is, matching the OpenAI SDK convention that
    `base_url` includes the API prefix.
    """
    parts = urlsplit(base_url)
    if parts.path in ("", "/"):
        return urlunsplit(parts._replace(path=_OPENAI_API_PREFIX))
    return base_url


@dataclass
class ToolCall:
    """A single tool/function call returned by the model."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from an LLM, with optional tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message_dict(self) -> dict[str, Any]:
        """Convert to an OpenAI-format assistant message dict."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(ABC):
    """Abstract base for LLM endpoint clients.

    Subclass and implement `complete()` for your protocol.

    Args:
        endpoint (`str`):
            The `http(s)` base URL of the LLM service (e.g. "http://localhost").
            May include a port and a path (e.g. "http://localhost:8000/v1").
            Credentials, query strings and fragments are rejected.
        port (`int` or `None`):
            The port the service listens on. Appended to `endpoint` when the
            URL does not name one; must match the URL's port when both are given.
    """

    def __init__(self, endpoint: str, port: int | None):
        self.endpoint = endpoint
        self.port = port

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt, return the text response.

        Args:
            prompt (`str`):
                The user prompt to send.
            **kwargs:
                Override default parameters (temperature, max_tokens, etc.).

        Returns:
            The model's text response.
        """
        ...

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages with tool definitions, return a normalized response.

        Messages use OpenAI-format dicts (`{"role": "...", "content": "..."}`).
        Tools use MCP tool definitions; they are converted internally.

        Args:
            messages (`list[dict[str, Any]]`):
                Conversation history as OpenAI-format message dicts.
            tools (`list[dict[str, Any]]`):
                MCP tool definitions.
            **kwargs:
                Override default parameters (temperature, max_tokens, etc.).

        Returns:
            An [`LLMResponse`] with the model's text and any tool calls.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling"
        )

    @property
    def base_url(self) -> str:
        """Base URL of the service: `endpoint` plus `port` when the URL names none."""
        return _join_endpoint_port(self.endpoint, self.port)


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs.

    Works with: OpenAI, vLLM, TGI, Ollama, HuggingFace Inference API,
    or any endpoint that speaks the OpenAI chat completions format.

    Args:
        endpoint (`str`):
            The base URL (e.g. "http://localhost"). May include a port and a
            path (e.g. "http://localhost:8000/v1"). The `/v1` API prefix is
            appended when the URL has no path; a URL with a path is used as-is.
        port (`int` or `None`):
            The port number, appended when `endpoint` does not name one; must
            match the URL's port when both are given.
        model (`str`):
            Model name to pass to the API.
        api_key (`str`, *optional*):
            API key. Defaults to "not-needed" for local endpoints.
        system_prompt (`str`, *optional*):
            System message prepended to every request.
        temperature (`float`, *optional*, defaults to `0.0`):
            Default sampling temperature.
        max_tokens (`int`, *optional*, defaults to `256`):
            Default max tokens in the response.
        use_max_completion_tokens (`bool`, *optional*, defaults to `False`):
            Use max_completion_tokens instead of max_tokens. Required for newer OpenAI models
            (gpt-5-mini, o1, o3). Not supported by self-hosted OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        endpoint: str,
        port: int | None,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        use_max_completion_tokens: bool = False,
    ):
        super().__init__(endpoint, port)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tokens_param = (
            "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
        )
        self._omit_temperature = use_max_completion_tokens

        self._client = AsyncOpenAI(
            base_url=_openai_base_url(self.base_url),
            api_key=api_key if api_key is not None else "not-needed",
        )

    def _chat_completion_kwargs(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._tokens_param: kwargs.get("max_tokens", self.max_tokens),
        }
        if not self._omit_temperature:
            create_kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        return create_kwargs

    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a chat completion request.

        Args:
            prompt (`str`):
                The user message.
            **kwargs:
                Overrides for temperature, max_tokens.

        Returns:
            The assistant's response text.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = self._chat_completion_kwargs(messages, **kwargs)
        response = await self._client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        create_kwargs = self._chat_completion_kwargs(messages, **kwargs)
        openai_tools = _mcp_tools_to_openai(tools)
        if openai_tools:
            create_kwargs["tools"] = openai_tools

        response = await self._client.chat.completions.create(**create_kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)


class AnthropicClient(LLMClient):
    """Client for Anthropic's Messages API.

    Requires the `anthropic` package (lazy-imported at construction time).

    Args:
        endpoint (`str`):
            The base URL (e.g. `https://api.anthropic.com`). May include a port.
        port (`int` or `None`):
            The port number, appended when `endpoint` does not name one; must
            match the URL's port when both are given.
        model (`str`):
            Model name (e.g. "claude-sonnet-4-20250514").
        api_key (`str`, *optional*):
            Anthropic API key.
        system_prompt (`str`, *optional*):
            System message prepended to every request.
        temperature (`float`, *optional*, defaults to `0.0`):
            Default sampling temperature.
        max_tokens (`int`, *optional*, defaults to `256`):
            Default max tokens in the response.
    """

    def __init__(
        self,
        endpoint: str,
        port: int | None,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        super().__init__(endpoint, port)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = AsyncAnthropic(
            base_url=self.base_url,
            api_key=api_key if api_key is not None else "not-needed",
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if self.system_prompt:
            create_kwargs["system"] = self.system_prompt

        response = await self._client.messages.create(**create_kwargs)
        return "".join(block.text for block in response.content if block.type == "text")

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        system, anthropic_msgs = _openai_msgs_to_anthropic(messages)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_msgs,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        system_text = system or self.system_prompt
        if system_text:
            create_kwargs["system"] = system_text
        anthropic_tools = _mcp_tools_to_anthropic(tools)
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = await self._client.messages.create(**create_kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, args=block.input)
                )

        return LLMResponse(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HOSTED_PROVIDERS: dict[str, tuple[str, int, type[LLMClient]]] = {
    "openai": ("https://api.openai.com", 443, OpenAIClient),
    "anthropic": ("https://api.anthropic.com", 443, AnthropicClient),
}

# Models that require max_completion_tokens instead of max_tokens and do not
# accept an explicit temperature parameter. Checked by prefix to cover versioned
# names such as "o1-2024-12-17" or "gpt-5-mini-2026-01-15".
_MAX_COMPLETION_TOKENS_PREFIXES: frozenset[str] = frozenset(
    {"gpt-5-mini", "o1", "o3", "o4-mini"}
)


def create_llm_client(
    provider: str,
    model: str,
    api_key: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> LLMClient:
    """Create an LLM client for a hosted provider.

    Args:
        provider (`str`):
            Provider name ("openai" or "anthropic").
        model (`str`):
            Model identifier.
        api_key (`str`):
            API key for the provider.
        system_prompt (`str`, *optional*):
            System message prepended to every request.
        temperature (`float`, *optional*, defaults to `0.0`):
            Sampling temperature.
        max_tokens (`int`, *optional*, defaults to `4096`):
            Maximum tokens in the response.

    Returns:
        A configured [`LLMClient`] instance.
    """
    key = provider.lower()
    if key not in _HOSTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Supported: {sorted(_HOSTED_PROVIDERS)}"
        )
    endpoint, port, cls = _HOSTED_PROVIDERS[key]
    extra: dict[str, Any] = {}
    if cls is OpenAIClient and any(
        model.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES
    ):
        extra["use_max_completion_tokens"] = True
    return cls(
        endpoint,
        port,
        model,
        api_key=api_key,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )


# ---------------------------------------------------------------------------
# MCP tool-schema helpers
# ---------------------------------------------------------------------------


def _clean_mcp_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize an MCP tool `inputSchema` for LLM function-calling APIs."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    # Shallow copy to avoid mutating the caller's schema dict.
    schema = dict(schema)

    if "oneOf" in schema:
        for option in schema["oneOf"]:
            if isinstance(option, dict) and option.get("type") == "object":
                schema = option
                break
        else:
            return {"type": "object", "properties": {}, "required": []}

    if "allOf" in schema:
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for sub in schema["allOf"]:
            if isinstance(sub, dict):
                if "properties" in sub:
                    merged["properties"].update(sub["properties"])
                if "required" in sub:
                    merged["required"].extend(sub["required"])
        schema = merged

    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if isinstance(option, dict) and option.get("type") == "object":
                schema = option
                break
        else:
            return {"type": "object", "properties": {}, "required": []}

    schema.setdefault("type", "object")
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema


def _mcp_tools_to_openai(
    mcp_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    result = []
    for tool in mcp_tools:
        input_schema = tool.get(
            "inputSchema", {"type": "object", "properties": {}, "required": []}
        )
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": _clean_mcp_schema(input_schema),
                },
            }
        )
    return result


def _mcp_tools_to_anthropic(
    mcp_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool format."""
    result = []
    for tool in mcp_tools:
        input_schema = tool.get(
            "inputSchema", {"type": "object", "properties": {}, "required": []}
        )
        result.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": _clean_mcp_schema(input_schema),
            }
        )
    return result


def _openai_msgs_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns `(system_text, anthropic_messages)`. System-role messages are
    extracted and concatenated; tool-result messages are converted to
    Anthropic's `tool_result` content blocks inside user turns.
    """
    system_parts: list[str] = []
    anthropic_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append(msg["content"])

        elif role == "user":
            anthropic_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            if msg.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args,
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": content})
            else:
                anthropic_msgs.append(
                    {"role": "assistant", "content": msg.get("content", "")}
                )

        elif role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg["content"],
            }
            # Anthropic requires tool results in user turns; merge if possible.
            if (
                anthropic_msgs
                and anthropic_msgs[-1]["role"] == "user"
                and isinstance(anthropic_msgs[-1]["content"], list)
            ):
                anthropic_msgs[-1]["content"].append(tool_result)
            else:
                anthropic_msgs.append({"role": "user", "content": [tool_result]})

    system = "\n\n".join(system_parts)
    return system, anthropic_msgs
