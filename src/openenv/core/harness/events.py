# SPDX-License-Identifier: BSD-3-Clause

"""Event and response types for agentic harness turns (RFC 005)."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HarnessEventType(str, Enum):
    """Types of events emitted while a harness processes one turn."""

    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_CHUNK = "llm_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT_OUTPUT = "text_output"
    ERROR = "error"
    TURN_COMPLETE = "turn_complete"


class HarnessEvent(BaseModel):
    """
    A single event from a harness turn.

    Concrete adapters map their native event streams to this schema so the
    orchestrator can inspect what happened during a turn regardless of which
    harness produced it.

    Args:
        type (`HarnessEventType`):
            The event type.
        timestamp (`float`, *optional*, defaults to the current Unix time):
            When the event occurred.
        data (`dict[str, Any]`, *optional*):
            Event-type-specific payload. Expected keys by type:
            `LLM_REQUEST`: `messages`, `model`; `LLM_RESPONSE`: `content`,
            `usage`; `LLM_CHUNK`: `content`, `index`; `TOOL_CALL`:
            `tool_name`, `arguments`; `TOOL_RESULT`: `tool_name`, `result`,
            `error`; `TEXT_OUTPUT`: `text`; `ERROR`: `message`,
            `recoverable`; `TURN_COMPLETE`: `response` and optionally `done`.
    """

    type: HarnessEventType
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class HarnessResponse(BaseModel):
    """
    Complete response from a single conversational turn.

    Args:
        response (`str`):
            The harness's text response for this turn.
        events (`list[HarnessEvent]`, *optional*):
            All events emitted during this turn.
        done (`bool`, *optional*, defaults to `False`):
            Whether the harness considers the task complete.
    """

    response: str
    events: list[HarnessEvent] = Field(default_factory=list)
    done: bool = False


class HarnessClientMessage(BaseModel):
    """
    One client frame on the production `/harness` WebSocket.

    Args:
        type (`str`):
            Must be `"message"`.
        content (`str`):
            The user message for this conversational turn.
    """

    type: Literal["message"]
    content: str


class HarnessProtocolError(BaseModel):
    """
    A recoverable rejection of a client frame on `/harness`.

    No turn was started. The client can correct its message and retry on the
    same connection. The distinct `protocol_error` discriminator keeps this
    response separate from terminal harness `error` events.

    Args:
        data (`dict[str, Any]`):
            Error details including `message` and `code`.
    """

    type: Literal["protocol_error"] = "protocol_error"
    data: dict[str, Any]


def events_to_metadata(events: list[HarnessEvent]) -> list[dict[str, Any]]:
    """
    Convert harness events to JSON-serializable dicts for observation metadata.

    This is the sanctioned way to place events into
    [`~openenv.core.env_server.types.Observation`] metadata, guaranteeing the
    payload survives wire serialization.

    Args:
        events (`list[HarnessEvent]`):
            Events to convert.

    Returns:
        `list[dict]` of JSON-serializable event payloads.
    """
    return [event.model_dump(mode="json") for event in events]


__all__ = [
    "HarnessClientMessage",
    "HarnessEvent",
    "HarnessEventType",
    "HarnessResponse",
    "HarnessProtocolError",
    "events_to_metadata",
]
