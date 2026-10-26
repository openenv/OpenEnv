"""Which wire dialect is this request written in?

Four dialects reach the intercept, because coding agents did not agree on one:

    openai_chat        opencode, qwen-coder, goose, swe-agent, mini-swe-agent, terminus-2, ...
    openai_responses   codex, trae-agent
    anthropic          claude-code
    google             gemini-cli, antigravity-sdk

Detection is by path first, then headers, then body shape — strongest signal to weakest. Path is
unambiguous when present; a header is a deliberate client declaration; body shape is a last resort
and can be coincidental, so it is only consulted when nothing better exists.

Getting this wrong is not subtle: a Google request parsed as chat-completions produces a 400 and the
agent silently does nothing, which reads as "captured nothing" rather than as a routing bug. That is
why trae-agent looked like a chat harness for a full night — its config says `provider: openai`, but
the access log showed exactly one `POST /v1/responses` against 465 chat calls.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class APIType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    GOOGLE = "google"


def detect(path: str, headers: dict[str, str], body: dict[str, Any]) -> APIType:
    """Classify one request. Defaults to chat-completions, the most common dialect."""
    if "/v1/messages" in path:
        return APIType.ANTHROPIC
    if "/v1/chat/completions" in path:
        return APIType.OPENAI_CHAT
    if "/v1/responses" in path:
        return APIType.OPENAI_RESPONSES
    # Google puts the method in the path: `/v1beta/models/{model}:generateContent`, and its
    # streaming variant `:streamGenerateContent`. The comparison must be case-insensitive: the
    # streaming form capitalises the G, so a literal `"generateContent" in path` matches the
    # non-streaming route and misses every streaming one. A missed Google request is then handed to
    # the chat-completions transformer, which finds no `messages` and produces a valid-looking
    # response in the wrong envelope, and gemini-cli reports that as nothing at all.
    if "generatecontent" in path.lower():
        return APIType.GOOGLE

    if "anthropic-version" in {k.lower() for k in headers}:
        return APIType.ANTHROPIC

    if "contents" in body:
        return APIType.GOOGLE
    if "input" in body and "instructions" in body:
        return APIType.OPENAI_RESPONSES

    return APIType.OPENAI_CHAT


def extract_model(api_type: APIType, body: dict[str, Any]) -> str:
    """The model name the client asked for, whatever the dialect calls it."""
    if api_type is APIType.GOOGLE:
        return body.get("model", "gemini-pro")
    return body.get("model", "unknown")
