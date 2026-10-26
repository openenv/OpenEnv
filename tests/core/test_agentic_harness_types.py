# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RFC 005 agentic harness foundation types."""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import pytest
from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    AgenticHarnessAdapter,
    events_to_metadata,
    HarnessConfig,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
    HarnessTransport,
    resolve_tool_conflicts,
)
from pydantic import ValidationError


def make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


class TestHarnessConfig:
    def test_defaults(self):
        config = HarnessConfig(name="openclaw", command=["openclaw", "run"])
        assert config.working_directory == "/workspace"
        assert config.env_vars == {}
        assert config.transport is HarnessTransport.STDIO
        assert config.mcp_config_path is None
        assert config.startup_timeout_s == 30.0
        assert config.session_timeout_s == 600.0
        assert config.model is None
        assert config.api_key_env_var is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            HarnessConfig(name="x", command=["x"], unexpected="nope")

    def test_empty_command_rejected(self):
        with pytest.raises(ValidationError):
            HarnessConfig(name="x", command=[])

    def test_transport_coerces_from_string(self):
        config = HarnessConfig(name="x", command=["x"], transport="http")
        assert config.transport is HarnessTransport.STREAMABLE_HTTP


class TestHarnessEvents:
    def test_timestamp_defaults_to_now(self):
        event = HarnessEvent(type=HarnessEventType.TEXT_OUTPUT)
        assert event.timestamp > 0

    def test_json_round_trip(self):
        event = HarnessEvent(
            type=HarnessEventType.TOOL_CALL,
            data={"tool_name": "read_file", "arguments": {"path": "a.py"}},
        )
        restored = HarnessEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_events_to_metadata_is_json_serializable(self):
        events = [
            HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "t"}),
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE, data={"response": "done"}
            ),
        ]
        payload = events_to_metadata(events)
        assert json.loads(json.dumps(payload)) == payload
        assert payload[0]["type"] == "tool_call"

    def test_response_defaults(self):
        response = HarnessResponse(response="hi")
        assert response.events == []
        assert response.done is False


class TestResolveToolConflicts:
    def test_no_conflicts_passthrough(self):
        tools = [make_tool("query_db"), make_tool("lint")]
        resolved = resolve_tool_conflicts(tools, frozenset({"read_file"}))
        assert [t.name for t in resolved] == ["query_db", "lint"]

    def test_conflicting_names_prefixed_and_schema_preserved(self):
        tool = Tool(
            name="read_file",
            description="env reader",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        resolved = resolve_tool_conflicts([tool], frozenset({"read_file"}))
        assert resolved[0].name == "env_read_file"
        assert resolved[0].description == "env reader"
        assert resolved[0].input_schema == tool.input_schema

    def test_reserved_name_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            resolve_tool_conflicts([make_tool("reset")], frozenset())

    def test_duplicate_env_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            resolve_tool_conflicts([make_tool("lint"), make_tool("lint")], frozenset())

    def test_prefixed_name_ambiguity_rejected(self):
        tools = [make_tool("read_file"), make_tool("env_read_file")]
        with pytest.raises(ValueError, match="also taken"):
            resolve_tool_conflicts(tools, frozenset({"read_file"}))

    def test_prefixed_name_collides_with_builtin_rejected(self):
        with pytest.raises(ValueError, match="also taken"):
            resolve_tool_conflicts(
                [make_tool("read_file")],
                frozenset({"read_file", "env_read_file"}),
            )

    def test_input_not_mutated(self):
        tool = make_tool("read_file")
        resolve_tool_conflicts([tool], frozenset({"read_file"}))
        assert tool.name == "read_file"


class StreamingAdapter(AgenticHarnessAdapter):
    """Fake adapter yielding a scripted event stream."""

    def __init__(self, config: HarnessConfig, events: list[HarnessEvent]):
        super().__init__(config)
        self._events = events

    async def start(self, working_directory: str) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        pass

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        for event in self._events:
            yield event

    async def is_alive(self) -> bool:
        return True


class TestSendMessageDefault:
    @staticmethod
    def _adapter(events: list[HarnessEvent]) -> StreamingAdapter:
        return StreamingAdapter(HarnessConfig(name="fake", command=["fake"]), events)

    async def test_collects_events_and_terminal_turn_complete(self):
        events = [
            HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "t"}),
            HarnessEvent(type=HarnessEventType.TEXT_OUTPUT, data={"text": "..."}),
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE,
                data={"response": "all fixed", "done": True},
            ),
        ]
        response = await self._adapter(events).send_message("go")
        assert response.response == "all fixed"
        assert response.done is True
        assert response.events == events

    async def test_done_defaults_false(self):
        events = [
            HarnessEvent(type=HarnessEventType.TURN_COMPLETE, data={"response": "wip"})
        ]
        response = await self._adapter(events).send_message("go")
        assert response.done is False

    async def test_missing_turn_complete_raises(self):
        events = [HarnessEvent(type=HarnessEventType.TEXT_OUTPUT, data={"text": "..."})]
        with pytest.raises(HarnessError, match="TURN_COMPLETE"):
            await self._adapter(events).send_message("go")

    async def test_empty_stream_raises(self):
        with pytest.raises(HarnessError, match="TURN_COMPLETE"):
            await self._adapter([]).send_message("go")
