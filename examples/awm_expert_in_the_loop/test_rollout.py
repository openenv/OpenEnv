"""Unit tests for the pure helpers in the AWM expert-in-the-loop rollout.

These avoid any network or model access so they run in CI.

```bash
PYTHONPATH=src:envs uv run pytest examples/awm_expert_in_the_loop/test_rollout.py -v
```
"""

from __future__ import annotations

from dataclasses import dataclass

from rollout import (
    format_tools,
    has_tool_call_tags,
    is_content_filter_error,
    parse_tool_call,
    safe_parse_arguments,
    TaskResult,
)


def test_parse_tool_call_valid():
    out = parse_tool_call('thinking <tool_call>{"name": "list_tools"}</tool_call> done')
    assert out == {"name": "list_tools"}


def test_parse_tool_call_list_takes_first():
    out = parse_tool_call('<tool_call>[{"name": "a"}, {"name": "b"}]</tool_call>')
    assert out == {"name": "a"}


def test_parse_tool_call_missing_tags():
    assert parse_tool_call("no tool call here") is None


def test_parse_tool_call_bad_json():
    assert parse_tool_call("<tool_call>{not json}</tool_call>") is None
    assert has_tool_call_tags("<tool_call>{not json}</tool_call>")


def test_parse_tool_call_requires_name():
    assert parse_tool_call('<tool_call>{"arguments": {}}</tool_call>') is None


def test_safe_parse_arguments_from_string():
    assert safe_parse_arguments('{"a": 1}') == {"a": 1}


def test_safe_parse_arguments_passthrough_dict():
    assert safe_parse_arguments({"a": 1}) == {"a": 1}


def test_safe_parse_arguments_bad_returns_empty():
    assert safe_parse_arguments("not json") == {}
    assert safe_parse_arguments(42) == {}


def test_is_content_filter_error():
    assert is_content_filter_error(Exception("triggered content_filter"))
    assert is_content_filter_error(Exception("violates content management policy"))
    assert not is_content_filter_error(Exception("timeout"))


@dataclass
class _FakeTool:
    name: str
    description: str
    input_schema: dict


def test_format_tools_includes_required_params():
    tools = [
        _FakeTool(
            name="create_skill",
            description="Create a skill",
            input_schema={
                "properties": {"skill_name": {"type": "string", "description": "name"}},
                "required": ["skill_name"],
            },
        )
    ]
    text = format_tools(tools, verbose=True)
    assert "create_skill" in text
    assert "skill_name" in text
    assert "(required)" in text


def test_task_result_fields():
    result = TaskResult("s", 0, "t", 3, 1, 2, "complete", 1.0)
    assert result.reward == 1.0
    assert result.filtered is False


def test_normalize_scenario_name():
    import awm_data

    assert awm_data.normalize_scenario_name("Marketplace 1") == "marketplace_1"
    assert awm_data.normalize_scenario_name("e-Commerce/33") == "e_commerce_33"
    assert awm_data.normalize_scenario_name("  Foo__Bar  ") == "foo_bar"


class _FakeCompletions:
    def __init__(self, fail_on_temperature: bool, finish_reason: str = "stop"):
        self.fail_on_temperature = fail_on_temperature
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_temperature and "temperature" in kwargs:
            raise Exception("Unsupported value: 'temperature' does not support 0.1")

        class _Msg:
            content = "hello"

        class _Choice:
            message = _Msg()
            finish_reason = self.finish_reason

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, fail_on_temperature: bool = False, finish_reason: str = "stop"):
        self.chat = type(
            "C",
            (),
            {"completions": _FakeCompletions(fail_on_temperature, finish_reason)},
        )()


def test_chat_complete_omits_temperature_when_none():
    import asyncio

    from llm import chat_complete

    client = _FakeClient()
    out = asyncio.run(
        chat_complete(client, "m", [{"role": "user", "content": "x"}], max_tokens=8)
    )
    assert out == "hello"
    assert "temperature" not in client.chat.completions.calls[0]


def test_chat_complete_passes_temperature():
    import asyncio

    from llm import chat_complete

    client = _FakeClient()
    asyncio.run(
        chat_complete(
            client,
            "m",
            [{"role": "user", "content": "x"}],
            max_tokens=8,
            temperature=0.3,
        )
    )
    assert client.chat.completions.calls[0]["temperature"] == 0.3


def test_chat_complete_retries_without_temperature_on_error():
    import asyncio

    from llm import chat_complete

    client = _FakeClient(fail_on_temperature=True)
    out = asyncio.run(
        chat_complete(
            client,
            "m",
            [{"role": "user", "content": "x"}],
            max_tokens=8,
            temperature=0.0,
        )
    )
    assert out == "hello"
    # first call had temperature and failed; retry omitted it
    assert "temperature" in client.chat.completions.calls[0]
    assert "temperature" not in client.chat.completions.calls[1]


def test_chat_complete_raises_on_empty_length_response():
    import asyncio

    from llm import chat_complete

    class _EmptyCompletions:
        async def create(self, **kwargs):
            class _Msg:
                content = ""

            class _Choice:
                message = _Msg()
                finish_reason = "length"

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _EmptyClient:
        chat = type("C", (), {"completions": _EmptyCompletions()})()

    try:
        asyncio.run(
            chat_complete(
                _EmptyClient(), "m", [{"role": "user", "content": "x"}], max_tokens=8
            )
        )
    except RuntimeError as exc:
        assert "exhausted max_completion_tokens" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
