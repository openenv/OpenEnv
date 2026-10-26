"""Exercise the ThinkingBox adapter without requiring live benchmark services.

The suite covers wire contracts, native-runtime fidelity, provenance, data
integrity, evaluator output, and failure isolation.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "thinkingbox",
    reason="ThinkingBox is installed by the thinkingbox_env project.",
)

from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect
from thinkingbox.common.chat_types import (
    Conversation,
    DecodeResult,
    ParallelToolCall,
    TestResult as NativeTestResult,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    AgentConfig,
    HydratedTestCase,
    ScenarioConfig,
    SessionProxyConfig,
    ToolDefOverride,
)
from thinkingbox.common.usage_types import Usage


_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "envs"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from openenv.core import EnvClient, ListToolsAction
from thinkingbox_env import (
    benchmark_data as data_loader,
    evaluation as thinkingbox_eval,
    runtime as thinkingbox_runtime,
)
from thinkingbox_env.benchmark_data import (
    _extract_archive,
    DataBundle,
    DATASET_NAME,
    DatasetError,
    ensure_data,
    load_test_uids,
    resolve_data_bundle,
)
from thinkingbox_env.client import DEFAULT_MESSAGE_TIMEOUT_S, ThinkingBoxEnv
from thinkingbox_env.models import (
    _FinishAction,
    CallToolAction,
    CANONICAL_TEST_UIDS,
    DATA_COMMIT,
    DATA_MANIFEST_PATH,
    DATA_RELEASE_NAME,
    SubmitMessageAction,
    SubmittedToolCall,
    ThinkingBoxAction,
    ThinkingBoxExecutionProvenance,
    ThinkingBoxObservation,
)
from thinkingbox_env.runtime import (
    load_thinkingbox_config,
    load_thinkingbox_runtime_provenance,
    require_canonical_thinkingbox_runtime,
)
from thinkingbox_env.server import (
    app as app_module,
    config as server_config,
    ThinkingBoxEnvironment,
)
from thinkingbox_env.server.app import create_thinkingbox_app
from thinkingbox_env.server.config import load_runtime_settings, RuntimeSettings


UID = "demo.py:test_case"


class FakeProxyClient:
    def __init__(
        self,
        *,
        fail_call: bool = False,
        fail_effects: bool = False,
        fail_destroy: bool = False,
        tools: list[ToolDef] | None = None,
        results: dict[str, str] | None = None,
    ) -> None:
        self.session_id = "proxy-secret-session"
        self.fail_call = fail_call
        self.fail_effects = fail_effects
        self.fail_destroy = fail_destroy
        self.created = 0
        self.destroyed = 0
        self.effects_calls = 0
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.available_tools: list[str] = []
        self.server_config: dict[str, Any] = {}
        self.tools = tools
        self.results = results or {}

    async def session_create(
        self, server_config: dict[str, Any], available_tools: list[str]
    ) -> dict[str, Any]:
        self.created += 1
        self.server_config = server_config
        self.available_tools = available_tools
        return {"tenant": "Contoso", "private": "private-secret"}

    async def list_tools(self) -> list[ToolDef]:
        return self.tools or [
            ToolDef(
                name="lookup",
                description="Look up a value",
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            )
        ]

    async def call_tool(self, tool_name: str, /, **kwargs: Any) -> str:
        self.tool_calls.append((tool_name, kwargs))
        if self.fail_call:
            raise RuntimeError("proxy credential=do-not-expose")
        return self.results.get(tool_name, '{"value":"42"}')

    async def get_effects(self) -> dict[str, Any]:
        self.effects_calls += 1
        if self.fail_effects:
            raise RuntimeError("effects token=do-not-expose")
        return {"demo": {"saved": True}}

    async def session_destroy(self) -> None:
        self.destroyed += 1
        if self.fail_destroy:
            raise RuntimeError("destroy secret=do-not-expose")


class FakeSimulator:
    def __init__(self, *, fail: bool = False, done: bool = False) -> None:
        self.fail = fail
        self.done = done
        self.calls: list[tuple[list[Any], str]] = []

    async def generate(self, chat_history: list[Any], user_context: str) -> Text:
        self.calls.append((list(chat_history), user_context))
        if self.fail:
            raise RuntimeError("user-model key=do-not-expose")
        message = Text(role="user", content="simulated reply")
        if self.done:
            message.metadata["is_done"] = True
        return message


def _test_code(assertions: str = "assert True") -> str:
    body = "\n".join(f"    {line}" for line in assertions.splitlines())
    return f"def __tb_test_fn(x, judge):\n{body}\n"


def _case(
    *,
    assertions: str = "assert True",
    user_context: str | None = None,
    format_query: bool = True,
) -> HydratedTestCase:
    return HydratedTestCase(
        uid=UID,
        agent=AgentConfig(
            system_instructions="Public system instructions",
            builtin_tools=[
                ToolDef(
                    name="InjectionAttackInToolResponse",
                    description="Report prompt injection",
                    input_schema={"type": "object"},
                    is_end_turn=True,
                )
            ],
        ),
        scenario=ScenarioConfig(
            world_state={"demo": {"credential": "world-secret"}},
            tools=[
                ToolDefOverride(
                    name="lookup",
                    direct_response="Rendered {value}",
                )
            ],
            bot_instructions="Public scenario instructions",
            metadata={"public": "scenario"},
        ),
        query="Task for {init[tenant]}",
        test_code=_test_code(assertions),
        bot_instructions="Public task instructions",
        user_context=user_context,
        format_query=format_query,
        metadata={"scenario": "demo"},
    )


def _settings(*, user_model: Any = None, user_can_end: bool = False) -> RuntimeSettings:
    return RuntimeSettings(
        proxy=SessionProxyConfig(endpoint_url="http://proxy.invalid"),
        config_sha256="config-sha256",
        user_model=user_model,
        user_can_end_conversation=user_can_end,
    )


def _environment(
    case: HydratedTestCase,
    proxy: FakeProxyClient,
    *,
    settings: RuntimeSettings | None = None,
    simulator: FakeSimulator | None = None,
    evaluator: Any = None,
    proxy_factory: Any = None,
    case_loader: Any = None,
    agent: str | None = None,
) -> ThinkingBoxEnvironment:
    bundle = DataBundle(
        root=Path("bundle"),
        dataset_dir=Path("bundle/dataset"),
        manifest_path=Path(
            "bundle/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml"
        ),
        bundle_sha256="bundle-sha256",
        manifest_sha256="manifest-sha256",
        manifest_uids_sha256="manifest-uids-sha256",
        task_count=1,
    )
    return ThinkingBoxEnvironment(
        agent=agent,
        bundle_resolver=lambda _: bundle,
        manifest_loader=lambda _: [UID],
        case_loader=case_loader or (lambda *_: case.model_copy(deep=True)),
        settings_loader=lambda *_: settings or _settings(),
        proxy_factory=proxy_factory or (lambda _: proxy),
        user_model_factory=lambda model: model,
        user_simulator_factory=(
            (lambda _model, _can_end: simulator) if simulator is not None else None
        ),
        evaluator=evaluator,
    )


def test_server_import_does_not_load_client() -> None:
    """Importing server code must not cross the client-server boundary."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import thinkingbox_env.server.app; "
                "assert 'thinkingbox_env.client' not in sys.modules"
            ),
        ],
        cwd=_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_ROOT / "envs"))),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_client_and_evaluator_imports_do_not_load_server_modules() -> None:
    """Client-side package entry points must not cross into trusted server code."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import thinkingbox_env.client; "
                "import thinkingbox_env.evaluation; "
                "assert not any(name.startswith('thinkingbox_env.server') "
                "for name in sys.modules)"
            ),
        ],
        cwd=_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_ROOT / "envs"))),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_examples_and_client_have_no_server_imports() -> None:
    paths = (
        _ROOT / "envs/thinkingbox_env/client.py",
        _ROOT / "envs/thinkingbox_env/evaluation.py",
        _ROOT / "examples/thinkingbox/example_usage.py",
        _ROOT / "examples/thinkingbox/eval_testlist.py",
    )
    for path in paths:
        assert "thinkingbox_env.server" not in path.read_text(encoding="utf-8")


def test_runtime_provenance_uses_installed_pep610_identity() -> None:
    """Runtime identity must reflect the actual VCS or local package source."""
    provenance = load_thinkingbox_runtime_provenance()

    assert len(provenance.source_sha256) == 64
    if provenance.source_type == "vcs":
        assert provenance.identity == provenance.commit_id
        assert provenance.canonical is True
        assert require_canonical_thinkingbox_runtime() == provenance
    else:
        assert provenance.identity == f"source-sha256:{provenance.source_sha256}"
        assert provenance.commit_id is None
        assert provenance.canonical is False
        with pytest.raises(RuntimeError, match="does not match"):
            require_canonical_thinkingbox_runtime()
    for obsolete in (
        "THINKINGBOX_REVISION",
        "DATA_REVISION",
        "DATA_ARCHIVE_SHA256",
        "DATASET_SHA256",
        "DATA_MANIFEST_SHA256",
        "DATA_MANIFEST_UIDS_SHA256",
    ):
        assert not hasattr(data_loader, obsolete)
        assert not hasattr(thinkingbox_runtime, obsolete)


def test_editable_runtime_records_deterministic_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editable installs use content identity without leaking or faking a commit."""
    package_root = tmp_path / "thinkingbox"
    package_root.mkdir()
    source = package_root / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    class EditableDistribution:
        version = "0.1.0"

        @staticmethod
        def read_text(name: str) -> str | None:
            if name != "direct_url.json":
                return None
            return json.dumps(
                {
                    "url": tmp_path.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    monkeypatch.setattr(
        thinkingbox_runtime.metadata,
        "distribution",
        lambda name: (
            EditableDistribution()
            if name == "thinkingbox"
            else pytest.fail(f"unexpected distribution lookup: {name}")
        ),
    )
    monkeypatch.setattr(
        thinkingbox_runtime,
        "_thinkingbox_source_root",
        lambda: package_root,
    )
    load_thinkingbox_runtime_provenance.cache_clear()
    first = load_thinkingbox_runtime_provenance()
    load_thinkingbox_runtime_provenance.cache_clear()
    second = load_thinkingbox_runtime_provenance()

    assert first == second
    assert first.source_type == "editable"
    assert first.identity == f"source-sha256:{first.source_sha256}"
    assert first.commit_id is None
    assert first.source_url is None
    assert str(tmp_path) not in str(first)
    with pytest.raises(RuntimeError, match="does not match"):
        require_canonical_thinkingbox_runtime()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    load_thinkingbox_runtime_provenance.cache_clear()
    changed = load_thinkingbox_runtime_provenance()
    assert changed.identity != first.identity
    load_thinkingbox_runtime_provenance.cache_clear()


def test_action_and_observation_schema_is_public_and_strict() -> None:
    assert isinstance(
        ThinkingBoxAction.model_validate({"type": "list_tools"}), ListToolsAction
    )
    action = ThinkingBoxAction.model_validate(
        {
            "type": "call_tool",
            "tool_name": "lookup",
            "arguments": {"key": "x"},
            "call_id": "call-1",
        }
    )
    assert isinstance(action, CallToolAction)
    assert action.call_id == "call-1"
    batch = ThinkingBoxAction.model_validate(
        {
            "type": "call_tool",
            "parallel_tool_calls": [
                {
                    "name": "lookup",
                    "arguments": {"key": "x"},
                    "call_id": "batch-1",
                },
                {
                    "name": "lookup",
                    "arguments": {"key": "y"},
                    "call_id": "batch-2",
                },
            ],
        }
    )
    assert isinstance(batch, CallToolAction)
    assert len(batch.parallel_tool_calls) == 2
    assert isinstance(
        ThinkingBoxAction.model_validate(
            {"type": "submit_message", "content": "hello"}
        ),
        SubmitMessageAction,
    )
    with pytest.raises(ValidationError):
        ThinkingBoxAction.model_validate(
            {"type": "call_tool", "tool_name": "lookup", "arguments": []}
        )
    with pytest.raises(ValidationError):
        ThinkingBoxAction.model_validate(
            {
                "type": "call_tool",
                "tool_name": "lookup",
                "parallel_tool_calls": [
                    {"name": "lookup", "call_id": "duplicate-mode"}
                ],
            }
        )

    schema = ThinkingBoxAction.model_json_schema()
    assert len(schema["oneOf"]) == 3
    assert "_finish" not in str(schema)
    assert "user_context" not in str(ThinkingBoxObservation.model_json_schema())


def test_client_serializes_actions_and_terminal_envelopes() -> None:
    client = ThinkingBoxEnv(base_url="http://127.0.0.1:8000")
    assert client._message_timeout == DEFAULT_MESSAGE_TIMEOUT_S
    payload = client._step_payload(
        CallToolAction(
            tool_name="lookup",
            arguments={"key": "x"},
            call_id="call-2",
        )
    )
    assert payload == {
        "metadata": {},
        "type": "call_tool",
        "tool_name": "lookup",
        "arguments": {"key": "x"},
        "call_id": "call-2",
        "parallel_tool_calls": [],
    }

    result = client._parse_result(
        {
            "observation": {
                "kind": "terminal",
                "reward_type": "pass",
                "system_error": False,
                "steps_taken": 2,
            },
            "reward": 1.0,
            "done": True,
        }
    )
    assert result.done is True
    assert result.reward == 1.0
    assert result.observation.kind == "terminal"


def test_client_reset_omits_none_agent_and_other_server_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def capture_reset(_: EnvClient[Any, Any, Any], **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(EnvClient, "reset", capture_reset)
    client = ThinkingBoxEnv(base_url="http://127.0.0.1:8000")

    client.reset(UID, seed=7)
    client.reset(
        UID,
        dataset="/server/data",
        agent="custom-agent",
        config="/server/config.yaml",
    )

    assert calls[0] == {"test_uid": UID, "seed": 7}
    assert calls[1] == {
        "test_uid": UID,
        "dataset": "/server/data",
        "agent": "custom-agent",
        "config": "/server/config.yaml",
    }


def test_standard_app_schema_does_not_advertise_finish() -> None:
    from thinkingbox_env.server.app import app

    response = TestClient(app).get("/schema")

    assert response.status_code == 200
    schema = response.json()
    assert "_finish" not in str(schema["action"])
    assert "submit_message" in str(schema["action"])
    route_paths = {route.path for route in app.routes}
    assert "/ws" in route_paths
    assert "/reset" not in route_paths
    assert "/step" not in route_paths
    assert "/state" not in route_paths


def test_supported_websocket_path_retains_one_episode() -> None:
    proxy = FakeProxyClient()
    test_app = create_thinkingbox_app(
        lambda: _environment(_case(), proxy),
        max_concurrent_envs=1,
    )

    with (
        TestClient(test_app) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.send_json({"type": "reset", "data": {"test_uid": UID}})
        reset = websocket.receive_json()
        websocket.send_json({"type": "step", "data": {"type": "list_tools"}})
        listed = websocket.receive_json()

        assert reset["data"]["observation"]["kind"] == "reset"
        assert listed["data"]["observation"]["kind"] == "tools"
        assert listed["data"]["observation"]["task_uid"] == UID
        websocket.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert proxy.created == 1
    assert proxy.destroyed == 1


@pytest.mark.parametrize(
    ("configured", "user_model", "judge_model", "expected_status"),
    [
        (False, object(), object(), 503),
        (True, None, object(), 503),
        (True, object(), None, 503),
        (True, object(), object(), 200),
    ],
)
def test_readiness_reports_required_components_and_typesense_limitation(
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
    user_model: Any,
    judge_model: Any,
    expected_status: int,
) -> None:
    class FakeResponse:
        is_success = True

    class FakeAsyncClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def get(self, _: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        app_module.config,
        "THINKINGBOX_CONFIG",
        "config.yaml" if configured else "",
    )
    monkeypatch.setattr(app_module.benchmark_data, "data_ready", lambda: True)
    monkeypatch.setattr(app_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        app_module,
        "load_runtime_settings",
        lambda _path, proxy: RuntimeSettings(
            proxy=proxy,
            user_model=user_model,
            judge_model=judge_model,
        ),
    )

    response = TestClient(app_module.app).get("/ready")
    payload = response.json()

    assert response.status_code == expected_status
    assert payload["data"] is True
    assert payload["session_proxy"] is True
    assert payload["runtime_config"] is configured
    assert payload["user_model"] is (configured and user_model is not None)
    assert payload["judge_model"] is (configured and judge_model is not None)
    assert payload["typesense"]["observable"] is False
    assert payload["typesense"]["ready"] is None
    assert "not observable" in payload["typesense"]["limitation"]


@pytest.mark.asyncio
async def test_reset_is_private_and_proxy_allowlist_is_scenario_only() -> None:
    case = _case(user_context=None)
    proxy = FakeProxyClient()
    env = _environment(case, proxy)

    observation = await env.reset_async(test_uid=UID)
    dumped = str(observation.model_dump(mode="json"))

    assert observation.kind == "reset"
    assert observation.task == "Task for Contoso"
    assert [tool.name for tool in observation.tools or []] == [
        "InjectionAttackInToolResponse",
        "lookup",
    ]
    assert proxy.available_tools == ["lookup"]
    assert "private-secret" not in dumped
    assert "world-secret" not in dumped
    assert "proxy-secret-session" not in dumped
    assert "__tb_test_fn" not in dumped
    assert "lookup" in dumped

    listed = await env.step_async(ListToolsAction())
    assert [tool.name for tool in listed.tools or []] == [
        "InjectionAttackInToolResponse",
        "lookup",
    ]


@pytest.mark.asyncio
async def test_reset_none_agent_uses_server_default_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_agents: list[str] = []
    monkeypatch.setattr(server_config, "AGENT", "server-agent")
    case = _case()

    def load_case(_uid: str, _base_dir: str, agent: str) -> HydratedTestCase:
        selected_agents.append(agent)
        return case.model_copy(deep=True)

    env = _environment(
        case,
        FakeProxyClient(),
        case_loader=load_case,
    )
    default_reset = await env.reset_async(test_uid=UID, agent=None)
    explicit_reset = await env.reset_async(test_uid=UID, agent="explicit-agent")

    assert selected_agents == ["server-agent", "explicit-agent"]
    assert default_reset.metadata["agent"] == "server-agent"
    assert explicit_reset.metadata["agent"] == "explicit-agent"


@pytest.mark.asyncio
async def test_allowlist_unknown_tool_and_native_direct_response_context() -> None:
    assertions = """
assert x.response == "Finished <DONE>"
assert x.tool_direct_responses == ["Rendered 42"]
assert x.effects == {"demo": {"saved": True}}
lookup = next(call for call in x.tool_calls if call.tool_call.name == "lookup")
assert lookup.tool_call.id == "call-7"
assert lookup.tool_response.content == '{"value":"42"}'
assert x.init_result["tenant"] == "Contoso"
assert x.session_id == "proxy-secret-session"
assert x.metadata["uid"] == "demo.py:test_case"
"""
    proxy = FakeProxyClient()
    env = _environment(_case(assertions=assertions), proxy)
    await env.reset_async(test_uid=UID)

    unknown = await env.step_async(
        CallToolAction(tool_name="verify", arguments={}, call_id="bad-call")
    )
    assert unknown.done is False
    assert unknown.tool_error == "tool_not_found"
    assert proxy.tool_calls == []

    tool = await env.step_async(
        CallToolAction(
            tool_name="lookup",
            arguments={"key": "answer"},
            call_id="call-7",
        )
    )
    assert tool.call_id == "call-7"
    assert tool.tool_result == '{"value":"42"}'
    assert tool.direct_response == "Rendered 42"

    terminal = await env.step_async(SubmitMessageAction(content="Finished <DONE>"))
    assert terminal.done is True
    assert terminal.reward == 1.0
    assert terminal.system_error is False
    assert proxy.effects_calls == 1
    assert proxy.destroyed == 1


@pytest.mark.asyncio
async def test_direct_response_counts_toward_native_agent_limit() -> None:
    case = _case()
    case.max_agent_sim_turns = 2
    proxy = FakeProxyClient()
    env = _environment(case, proxy)
    await env.reset_async(test_uid=UID)

    tool = await env.step_async(
        CallToolAction(tool_name="lookup", arguments={"key": "answer"})
    )
    terminal = await env.step_async(SubmitMessageAction(content="must not be recorded"))

    assert tool.done is False
    assert tool.direct_response == "Rendered 42"
    assert terminal.done is True
    assert terminal.finish_reason == "agent_limit"
    assert terminal.steps_taken == 1


@pytest.mark.asyncio
async def test_parallel_batch_preserves_all_direct_responses_and_agent_turns() -> None:
    captured: dict[str, Any] = {}

    async def evaluator(_case: Any, context: Any, _settings: Any) -> NativeTestResult:
        captured["context"] = context
        return NativeTestResult(result=True, reward=1.0)

    case = _case()
    case.max_agent_sim_turns = 3
    case.scenario.tools.extend(
        [
            ToolDefOverride(name="lookup_two", direct_response="Second {value}"),
            ToolDefOverride(name="lookup_three"),
        ]
    )
    proxy = FakeProxyClient(
        tools=[
            ToolDef(
                name="lookup",
                description="First lookup",
                input_schema={"type": "object"},
            ),
            ToolDef(
                name="lookup_two",
                description="Second lookup",
                input_schema={"type": "object"},
            ),
            ToolDef(
                name="lookup_three",
                description="Third lookup",
                input_schema={"type": "object"},
            ),
        ],
        results={
            "lookup": '{"value":"first"}',
            "lookup_two": '{"value":"second"}',
            "lookup_three": '{"value":"third"}',
        },
    )
    env = _environment(case, proxy, evaluator=evaluator)
    await env.reset_async(test_uid=UID)

    batch = await env.step_async(
        CallToolAction(
            parallel_tool_calls=[
                SubmittedToolCall(
                    name="lookup",
                    arguments={"key": "one"},
                    call_id="parallel-1",
                ),
                SubmittedToolCall(
                    name="lookup_two",
                    arguments={"key": "two"},
                    call_id="parallel-2",
                ),
                SubmittedToolCall(
                    name="lookup_three",
                    arguments={"key": "three"},
                    call_id="parallel-3",
                ),
            ]
        )
    )
    terminal = await env.step_async(SubmitMessageAction(content="too late"))

    assert batch.kind == "tool_batch"
    assert [result.call_id for result in batch.tool_results or []] == [
        "parallel-1",
        "parallel-2",
        "parallel-3",
    ]
    assert [result.direct_response for result in batch.tool_results or []] == [
        "Rendered first",
        "Second second",
        None,
    ]
    assert proxy.tool_calls == [
        ("lookup", {"key": "one"}),
        ("lookup_two", {"key": "two"}),
        ("lookup_three", {"key": "three"}),
    ]
    assert terminal.finish_reason == "agent_limit"
    assert terminal.steps_taken == 1
    context = captured["context"]
    parallel_messages = [
        message for message in context.messages if isinstance(message, ParallelToolCall)
    ]
    assert len(parallel_messages) == 1
    assert [call.id for call in parallel_messages[0].tool_calls] == [
        "parallel-1",
        "parallel-2",
        "parallel-3",
    ]
    assert [entry.tool_call.id for entry in context.tool_calls] == [
        "parallel-1",
        "parallel-2",
        "parallel-3",
    ]
    tool_turn = [
        message
        for message in context.messages
        if isinstance(message, (ParallelToolCall, ToolResponse))
        or isinstance(message, Text)
        and message.tag == "direct"
    ]
    assert [
        message.T if not isinstance(message, Text) else f"{message.T}:{message.tag}"
        for message in tool_turn
    ] == [
        "ParallelToolCall",
        "ToolResponse",
        "ToolResponse",
        "ToolResponse",
        "Text:direct",
        "Text:direct",
    ]
    assert [message.content for message in tool_turn[-2:]] == [
        "Rendered first",
        "Second second",
    ]
    assert context.tool_direct_responses == ["Rendered first", "Second second"]


@pytest.mark.asyncio
async def test_parallel_batch_records_parse_error_without_executing_call() -> None:
    case = _case()
    case.scenario.tools.append(ToolDefOverride(name="lookup_two"))
    proxy = FakeProxyClient(
        tools=[
            ToolDef(
                name="lookup",
                description="Malformed lookup",
                input_schema={"type": "object"},
            ),
            ToolDef(
                name="lookup_two",
                description="Valid lookup",
                input_schema={"type": "object"},
            ),
        ],
        results={"lookup_two": '{"value":"second"}'},
    )
    env = _environment(case, proxy)
    await env.reset_async(test_uid=UID)

    batch = await env.step_async(
        CallToolAction(
            parallel_tool_calls=[
                SubmittedToolCall(
                    name="lookup",
                    arguments={},
                    call_id="malformed",
                    parse_error="Error: invalid JSON arguments",
                ),
                SubmittedToolCall(
                    name="lookup_two",
                    arguments={"key": "two"},
                    call_id="valid",
                ),
            ]
        )
    )

    assert batch.done is False
    assert proxy.tool_calls == [("lookup_two", {"key": "two"})]
    assert [
        (result.call_id, result.content, result.tool_error)
        for result in batch.tool_results or []
    ] == [
        ("malformed", "Error: invalid JSON arguments", "invalid_args"),
        ("valid", '{"value":"second"}', None),
    ]
    tool_messages = [
        message
        for message in batch.messages or []
        if message["T"] in {"ParallelToolCall", "ToolResponse"}
    ]
    assert [message["T"] for message in tool_messages] == [
        "ParallelToolCall",
        "ToolResponse",
        "ToolResponse",
    ]
    assert tool_messages[0]["tool_calls"][0]["metadata"]["error"] == (
        "Error: invalid JSON arguments"
    )
    assert [message["id"] for message in tool_messages[1:]] == [
        "malformed",
        "valid",
    ]
    await env.aclose()


@pytest.mark.asyncio
async def test_parallel_batch_stops_siblings_after_proxy_failure() -> None:
    class FailSecondProxy(FakeProxyClient):
        async def call_tool(self, tool_name: str, /, **kwargs: Any) -> str:
            self.tool_calls.append((tool_name, kwargs))
            if tool_name == "lookup_two":
                raise RuntimeError("proxy failed")
            return '{"value":"ok"}'

    case = _case()
    case.scenario.tools.extend(
        [
            ToolDefOverride(name="lookup_two"),
            ToolDefOverride(name="lookup_three"),
        ]
    )
    proxy = FailSecondProxy(
        tools=[
            ToolDef(
                name=name,
                description=name,
                input_schema={"type": "object"},
            )
            for name in ("lookup", "lookup_two", "lookup_three")
        ]
    )
    env = _environment(case, proxy)
    await env.reset_async(test_uid=UID)

    terminal = await env.step_async(
        CallToolAction(
            parallel_tool_calls=[
                SubmittedToolCall(
                    name="lookup",
                    arguments={"key": "one"},
                    call_id="first",
                ),
                SubmittedToolCall(
                    name="lookup_two",
                    arguments={"key": "two"},
                    call_id="failed",
                ),
                SubmittedToolCall(
                    name="lookup_three",
                    arguments={"key": "three"},
                    call_id="must-not-run",
                ),
            ]
        )
    )

    assert terminal.done is True
    assert terminal.system_error is True
    assert proxy.tool_calls == [
        ("lookup", {"key": "one"}),
        ("lookup_two", {"key": "two"}),
    ]
    tool_messages = [
        message
        for message in terminal.messages or []
        if message["T"] in {"ParallelToolCall", "ToolResponse"}
    ]
    assert [message["T"] for message in tool_messages] == ["ParallelToolCall"]
    assert [call["id"] for call in tool_messages[0]["tool_calls"]] == [
        "first",
        "failed",
        "must-not-run",
    ]


@pytest.mark.asyncio
async def test_terminal_parallel_call_action_executes_no_siblings() -> None:
    proxy = FakeProxyClient()
    env = _environment(_case(), proxy)
    await env.reset_async(test_uid=UID)

    terminal = await env.step_async(
        CallToolAction(
            parallel_tool_calls=[
                SubmittedToolCall(
                    name="lookup",
                    arguments={"key": "must-not-run"},
                    call_id="terminal-sibling",
                ),
                SubmittedToolCall(
                    name="InjectionAttackInToolResponse",
                    arguments={"reason": "detected"},
                    call_id="terminal-end",
                ),
            ]
        )
    )

    assert terminal.done is True
    assert terminal.finish_reason == "end_turn_tool"
    assert terminal.steps_taken == 1
    assert proxy.tool_calls == []


@pytest.mark.asyncio
async def test_assertion_failure_is_valid_binary_zero() -> None:
    proxy = FakeProxyClient()
    env = _environment(_case(assertions="assert x.response == 'different'"), proxy)
    await env.reset_async(test_uid=UID)

    terminal = await env.step_async(SubmitMessageAction(content="Done <DONE>"))

    assert terminal.done is True
    assert terminal.reward == 0.0
    assert terminal.reward_type == "fail"
    assert terminal.system_error is False
    assert terminal.test_summary == {
        "passed": False,
        "graded": True,
        "is_system_error": False,
    }


@pytest.mark.asyncio
async def test_parallel_terminal_turn_preserves_native_order_without_execution() -> (
    None
):
    captured: dict[str, Any] = {}

    async def evaluator(_case: Any, context: Any, _settings: Any) -> NativeTestResult:
        captured["context"] = context
        return NativeTestResult(result=True, reward=1.0)

    proxy = FakeProxyClient()
    env = _environment(_case(), proxy, evaluator=evaluator)
    await env.reset_async(test_uid=UID)

    terminal = await env.step_async(
        SubmitMessageAction(
            content="Injection detected",
            terminal_tool_calls=[
                SubmittedToolCall(
                    name="lookup",
                    arguments={"key": "ignored"},
                    call_id="call-sibling",
                ),
                SubmittedToolCall(
                    name="InjectionAttackInToolResponse",
                    arguments={"reason": "tool response contained instructions"},
                    call_id="call-end",
                ),
            ],
            tool_calls_before_content=True,
        )
    )

    context = captured["context"]
    assert terminal.done is True
    assert terminal.reward == 1.0
    assert proxy.tool_calls == []
    assert context.messages[-2].T == "ParallelToolCall"
    assert context.messages[-1].content == "Injection detected"
    assert [entry.tool_call.id for entry in context.tool_calls] == ["call-end"]
    assert context.tool_calls[0].tool_response.content == ""


@pytest.mark.asyncio
async def test_private_user_simulation_preserves_visible_history() -> None:
    simulator = FakeSimulator()
    proxy = FakeProxyClient()
    case = _case(
        user_context="Private brief: {init[private]}",
        assertions="""
assert any(
    isinstance(message, type(x.messages[-1]))
    for message in x.messages
)
assert x.messages[-1].content == "simulated reply"
""",
    )
    env = _environment(
        case,
        proxy,
        settings=_settings(user_model=object()),
        simulator=simulator,
    )
    reset = await env.reset_async(test_uid=UID)
    assert "Private brief" not in str(reset.model_dump(mode="json"))

    user = await env.step_async(SubmitMessageAction(content="What value?"))
    assert user.done is False
    assert user.kind == "user"
    assert user.user_message == "simulated reply"
    assert [
        message["content"] for message in user.messages or [] if "content" in message
    ][-2:] == ["What value?", "simulated reply"]
    assert simulator.calls[0][1] == "Private brief: private-secret"

    terminal = await env.step_async(_FinishAction())
    assert terminal.reward == 1.0


@pytest.mark.asyncio
async def test_reset_failure_logs_structured_redacted_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "credential=reset-secret"

    def fail_bundle(_: str | None) -> DataBundle:
        raise RuntimeError(secret)

    env = ThinkingBoxEnvironment(bundle_resolver=fail_bundle)
    with caplog.at_level(
        logging.ERROR,
        logger="thinkingbox_env.server.thinkingbox_environment",
    ):
        observation = await env.reset_async(test_uid=UID, episode_id="episode-log")

    assert observation.kind == "error"
    assert observation.error == "ThinkingBox reset failed during data."
    assert secret not in str(observation.model_dump(mode="json"))
    assert secret not in caplog.text
    record = next(
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "thinkingbox.infrastructure_failure"
    )
    assert record.tb_stage == "data"
    assert record.tb_primary is True
    assert record.tb_task_uid == UID
    assert record.tb_episode_id == "episode-log"
    assert record.exception_type == "RuntimeError"
    assert record.exc_info is not None
    assert "details redacted" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["proxy", "effects", "simulator"])
async def test_infrastructure_failures_latch_and_skip_grader(failure: str) -> None:
    grader_calls = 0

    async def evaluator(*_: Any) -> NativeTestResult:
        nonlocal grader_calls
        grader_calls += 1
        return NativeTestResult(result=True, reward=1.0)

    simulator = FakeSimulator(fail=failure == "simulator")
    proxy = FakeProxyClient(
        fail_call=failure == "proxy",
        fail_effects=failure == "effects",
    )
    user_context = "Private context" if failure == "simulator" else None
    settings = _settings(user_model=object()) if user_context else _settings()
    env = _environment(
        _case(user_context=user_context),
        proxy,
        settings=settings,
        simulator=simulator if user_context else None,
        evaluator=evaluator,
    )
    await env.reset_async(test_uid=UID)

    if failure == "proxy":
        terminal = await env.step_async(
            CallToolAction(tool_name="lookup", arguments={"key": "x"})
        )
    else:
        terminal = await env.step_async(
            SubmitMessageAction(
                content="Continue" if failure == "simulator" else "Done <DONE>"
            )
        )

    assert terminal.done is True
    assert terminal.reward == 0.0
    assert terminal.reward_type == "system_error"
    assert terminal.system_error is True
    assert terminal.test_summary["infrastructure_stage"] in {
        "proxy_call",
        "effects_or_evaluator",
        "user_simulator",
    }
    assert "do-not-expose" not in str(terminal.model_dump(mode="json"))
    assert grader_calls == 0
    assert proxy.destroyed == 1


@pytest.mark.asyncio
async def test_finalize_once_and_teardown_failure_cannot_return_success() -> None:
    grader_calls = 0

    async def evaluator(*_: Any) -> NativeTestResult:
        nonlocal grader_calls
        grader_calls += 1
        return NativeTestResult(result=True, reward=1.0)

    proxy = FakeProxyClient(fail_destroy=True)
    env = _environment(_case(), proxy, evaluator=evaluator)
    await env.reset_async(test_uid=UID)

    first, second = await asyncio.gather(
        env.step_async(_FinishAction(reason="limit")),
        env.step_async(_FinishAction(reason="limit")),
    )

    assert first is second
    assert first.reward == 0.0
    assert first.system_error is True
    assert grader_calls == 1
    assert proxy.effects_calls == 1
    assert proxy.destroyed == 1

    again = await env.step_async(_FinishAction())
    assert again is first
    assert grader_calls == 1


@pytest.mark.asyncio
async def test_reset_and_close_teardown_are_idempotent() -> None:
    proxies = [FakeProxyClient(), FakeProxyClient()]

    def factory(_: RuntimeSettings) -> FakeProxyClient:
        return proxies.pop(0)

    first, second = proxies
    env = _environment(_case(), first, proxy_factory=factory)
    await env.reset_async(test_uid=UID)
    await env.reset_async(test_uid=UID)

    assert first.destroyed == 1
    assert second.created == 1

    await env.aclose()
    await env.aclose()
    assert second.destroyed == 1


def _write_local_bundle(root: Path, marker: str) -> list[str]:
    for relative in (
        "dataset/agent",
        "dataset/scenario",
        "dataset/test_case",
        "servers",
        "support",
        "releases/thinkingbox_bench_v1",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "LICENSE.txt").write_text("license\n", encoding="utf-8")
    (root / "dataset/scenario/content.txt").write_text(marker, encoding="utf-8")
    uids = [UID, "retail.py:test_case", "booking.py:test_case"]
    manifest = (
        root
        / "releases"
        / "thinkingbox_bench_v1"
        / "testlist_thinkingbox_bench_v1.yaml"
    )
    manifest.write_text(
        "".join(f"- {uid}\n" for uid in uids),
        encoding="utf-8",
    )
    return uids


def test_canonical_uid_identity_is_complete_ordered_and_stable() -> None:
    assert len(CANONICAL_TEST_UIDS) == 507
    assert len(CANONICAL_TEST_UIDS) == len(set(CANONICAL_TEST_UIDS))
    assert CANONICAL_TEST_UIDS[0] == (
        "sandbox_external_retail_group1.py:test_case_ST002_001"
    )
    assert CANONICAL_TEST_UIDS[-1] == "sandbox_consulting_group1.py:test_trv_013"
    assert thinkingbox_eval._sha256_json(list(CANONICAL_TEST_UIDS)) == (
        "63b1f9fae0ee0465bfafd83815bd97a03d151029947e5197e4f328babce75490"
    )


def _write_runtime_config(path: Path, model: str) -> None:
    path.write_text(
        f"""
mcp_proxy:
  endpoint_url: http://proxy.invalid
orchestrator:
  type: thinkingbox
  agent_model:
    type: custom
    factory: test_models.AgentSession
    model: {model}
judge_model:
  type: custom
  factory: test_models.JudgeSession
  model: judge-{model}
""".lstrip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_reset_exposes_exact_selected_config_and_data_fingerprints(
    tmp_path: Path,
) -> None:
    default_root = tmp_path / "default-data"
    selected_root = tmp_path / "selected-data"
    _write_local_bundle(default_root, "default")
    _write_local_bundle(selected_root, "selected")
    default_config = tmp_path / "default.yaml"
    selected_config = tmp_path / "selected.yaml"
    _write_runtime_config(default_config, "default-model")
    _write_runtime_config(selected_config, "selected-model")

    proxy = FakeProxyClient()
    env = ThinkingBoxEnvironment(
        dataset=str(default_root),
        config_path=str(default_config),
        case_loader=lambda *_: _case(),
        proxy_factory=lambda _: proxy,
    )
    reset = await env.reset_async(
        test_uid=UID,
        dataset=str(selected_root),
        config=str(selected_config),
    )

    selected_bundle = resolve_data_bundle(str(selected_root))
    default_bundle = resolve_data_bundle(str(default_root))
    execution = reset.metadata["execution_provenance"]
    assert (
        ThinkingBoxExecutionProvenance.model_validate(execution).model_dump(mode="json")
        == execution
    )
    runtime = load_thinkingbox_runtime_provenance()
    assert execution == {
        "thinkingbox_revision": runtime.identity,
        "thinkingbox_source_sha256": runtime.source_sha256,
        "thinkingbox_source_type": runtime.source_type,
        "data_release": "local",
        "data_revision": "local",
        "config_sha256": hashlib.sha256(selected_config.read_bytes()).hexdigest(),
        "data_bundle_sha256": selected_bundle.bundle_sha256,
        "manifest_path": DATA_MANIFEST_PATH,
        "manifest_sha256": selected_bundle.manifest_sha256,
        "manifest_uids_sha256": selected_bundle.manifest_uids_sha256,
        "task_count": len(load_test_uids(selected_bundle)),
    }
    assert (
        execution["config_sha256"]
        != hashlib.sha256(default_config.read_bytes()).hexdigest()
    )
    assert execution["data_bundle_sha256"] != default_bundle.bundle_sha256
    assert str(selected_root) not in str(reset.metadata)
    assert "selected-model" not in str(reset.metadata)

    await env.aclose()


def test_local_bundle_requires_exact_manifest_and_derives_task_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / DATASET_NAME
    for relative in (
        "dataset/agent",
        "dataset/scenario",
        "dataset/test_case",
        "servers",
        "support",
        "releases/thinkingbox_bench_v1",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "LICENSE.txt").write_text("license\n", encoding="utf-8")
    uids = ["case.py:test_one", "case.py:test_two", "case.py:test_three"]
    manifest = (
        root
        / "releases"
        / "thinkingbox_bench_v1"
        / "testlist_thinkingbox_bench_v1.yaml"
    )
    manifest.write_text(
        "".join(f"- {uid}\n" for uid in uids),
        encoding="utf-8",
    )

    bundle = resolve_data_bundle(str(root))

    assert bundle.dataset_dir == root / "dataset"
    assert load_test_uids(bundle) == uids
    assert bundle.task_count == len(uids)
    assert bundle.revision == "local"
    assert DATASET_NAME == "thinkingbox_bench"

    (root / "servers/changed.py").write_text(
        "changed: true\n",
        encoding="utf-8",
    )
    changed = resolve_data_bundle(str(root))
    assert changed.bundle_sha256 != bundle.bundle_sha256
    assert changed.manifest_sha256 == bundle.manifest_sha256


@pytest.mark.parametrize("pinned", [False, True], ids=["explicit", "published"])
def test_concurrent_bundle_validation_ignores_bytecode_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pinned: bool,
) -> None:
    root = tmp_path / DATASET_NAME
    _write_local_bundle(root, "source")
    module = root / "servers/generated_module.py"
    module.write_text("VALUE = 42\n", encoding="utf-8")
    if pinned:
        _pin_fixture_bundle(root, monkeypatch)
    baseline = data_loader._validate_root(root, require_stamp=pinned)

    env = {
        key: value
        for key, value in os.environ.items()
        if key != "PYTHONDONTWRITEBYTECODE"
    }
    subprocess.run(
        [sys.executable, "-c", "import generated_module"],
        cwd=module.parent,
        env=env,
        check=True,
    )
    cache = module.parent / "__pycache__"
    bytecode = next(cache.glob("generated_module.*.pyc"))
    companion = cache / "unexpected.py"
    companion.write_text("raise RuntimeError\n", encoding="utf-8")
    legacy_bytecode = root / "servers/generated.pyo"
    legacy_bytecode.write_bytes(b"legacy bytecode")
    generated = {
        path.relative_to(root): path.read_bytes()
        for path in (bytecode, companion, legacy_bytecode)
    }

    def reject_deletion(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("bundle validation must not delete generated bytecode")

    monkeypatch.setattr(data_loader.shutil, "rmtree", reject_deletion)
    with ThreadPoolExecutor(max_workers=8) as executor:
        bundles = list(
            executor.map(
                lambda _: data_loader._validate_root(root, require_stamp=pinned),
                range(16),
            )
        )

    assert {bundle.bundle_sha256 for bundle in bundles} == {baseline.bundle_sha256}
    assert {path: (root / path).read_bytes() for path in generated} == generated


def test_local_bundle_rejects_obsolete_manifest_fallback(tmp_path: Path) -> None:
    """Only the pinned release manifest path may define adapter task identity."""
    root = tmp_path / DATASET_NAME
    for relative in (
        "dataset/agent",
        "dataset/scenario",
        "dataset/test_case",
        "servers",
        "support",
        "releases/thinkingbox_bench",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "LICENSE.txt").write_text("license\n", encoding="utf-8")
    (root / "releases/thinkingbox_bench/testlist_thinkingbox_bench.yaml").write_text(
        "- case.py:test_one\n", encoding="utf-8"
    )

    with pytest.raises(DatasetError, match=DATA_MANIFEST_PATH):
        resolve_data_bundle(str(root))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]\n", "non-empty"),
        ("- case.py:test_one\n- case.py:test_one\n", "unique"),
        ("- case.py:test_one\n- 7\n", "non-empty"),
    ],
)
def test_manifest_requires_nonempty_unique_string_uids(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    """Manifest validation derives count only from valid ordered string UIDs."""
    root = tmp_path / DATASET_NAME
    _write_valid_cached_bundle(root)
    (
        root / "releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml"
    ).write_text(payload, encoding="utf-8")

    with pytest.raises(DatasetError, match=message):
        resolve_data_bundle(str(root))


def _write_valid_cached_bundle(root: Path) -> None:
    for relative in (
        "dataset/agent",
        "dataset/scenario",
        "dataset/test_case",
        "servers",
        "support",
        "releases/thinkingbox_bench_v1",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "LICENSE.txt").write_text("license\n", encoding="utf-8")
    manifest = (
        root
        / "releases"
        / "thinkingbox_bench_v1"
        / "testlist_thinkingbox_bench_v1.yaml"
    )
    manifest.write_text(
        "- case.py:test_one\n- case.py:test_two\n- case.py:test_three\n",
        encoding="utf-8",
    )
    (root / "dataset/scenario/content.txt").write_text(
        "original\n",
        encoding="utf-8",
    )


def _pin_fixture_bundle(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = data_loader._validate_root(root, require_stamp=False)
    monkeypatch.setattr(
        data_loader,
        "DATA_BUNDLE_SHA256",
        bundle.bundle_sha256,
    )
    (root / ".openenv-thinkingbox-data.json").write_text(
        json.dumps(data_loader._canonical_stamp(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_explicit_pinned_cache_preserves_data_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / DATASET_NAME
    _write_valid_cached_bundle(root)
    _pin_fixture_bundle(root, monkeypatch)

    bundle = resolve_data_bundle(str(root))

    assert bundle.release_name == DATA_RELEASE_NAME
    assert bundle.revision == DATA_COMMIT


def test_pinned_cache_rejects_mutated_content_with_unchanged_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / DATASET_NAME
    _write_valid_cached_bundle(root)
    _pin_fixture_bundle(root, monkeypatch)
    resolve_data_bundle(str(root))
    content = root / "dataset/scenario/content.txt"
    original_stat = content.stat()
    content.write_text("modified\n", encoding="utf-8")
    os.utime(content, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(DatasetError, match="pinned release"):
        resolve_data_bundle(str(root))


def test_cache_publication_revalidates_after_interprocess_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    template = tmp_path / "template"
    _write_valid_cached_bundle(template)
    _pin_fixture_bundle(template, monkeypatch)
    destination = data_loader._cache_destination(cache_root)
    lock_paths: list[str] = []

    class PublishingLock:
        def __init__(self, path: str) -> None:
            lock_paths.append(path)

        def __enter__(self) -> Any:
            shutil.copytree(template, destination)
            return self

        def __exit__(self, *_: Any) -> None:
            pass

    monkeypatch.setattr(data_loader, "FileLock", PublishingLock)
    monkeypatch.setattr(
        data_loader,
        "_download",
        lambda *_args, **_kwargs: pytest.fail(
            "valid cache published while waiting must be reused"
        ),
    )

    bundle = ensure_data(cache_root=cache_root, archive_url="unused")

    assert bundle.root == destination
    assert destination.is_dir()
    assert lock_paths == [str(data_loader._publication_lock(destination))]


def test_first_download_validates_content_before_stamping_or_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    destination = data_loader._cache_destination(cache_root)
    validations: list[tuple[Path, bool]] = []

    def fake_download(_url: str, archive: Path, _timeout: float) -> None:
        archive.write_bytes(b"archive")

    def fake_extract(_archive: Path, extracted: Path) -> None:
        _write_valid_cached_bundle(extracted)

    def reject_content(bundle: DataBundle) -> None:
        validations.append((bundle.root, (bundle.root / data_loader._STAMP).exists()))
        raise DatasetError("noncanonical fixture")

    monkeypatch.setattr(data_loader, "_download", fake_download)
    monkeypatch.setattr(data_loader, "_extract_archive", fake_extract)
    monkeypatch.setattr(data_loader, "_require_canonical_content", reject_content)

    with pytest.raises(DatasetError, match="noncanonical fixture"):
        ensure_data(cache_root=cache_root, archive_url="unused")

    assert len(validations) == 1
    assert validations[0][1] is False
    assert not destination.exists()


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = b"bad"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("root/../../outside")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(DatasetError, match="unsafe path"):
        _extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "root/dataset/C:/outside.txt",
        r"root/dataset/C:\outside.txt",
        r"root/dataset/\\server\share\outside.txt",
        r"root/dataset/..\outside.txt",
    ],
    ids=["drive-forward-slash", "drive", "unc", "backslash-traversal"],
)
def test_archive_extraction_rejects_windows_paths_portably(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive = tmp_path / "windows-path.tar.gz"
    payload = b"bad"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(DatasetError, match="unsafe path"):
        _extract_archive(archive, tmp_path / "out")


def test_archive_extraction_keeps_safe_members_beneath_destination(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "safe.tar.gz"
    payload = b"safe"
    with tarfile.open(archive, "w:gz") as output:
        directory = tarfile.TarInfo("root/dataset/agent")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
        member = tarfile.TarInfo("root/dataset/agent/case.yaml")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
        ignored = tarfile.TarInfo("root/README.md")
        ignored.size = len(payload)
        output.addfile(ignored, io.BytesIO(payload))

    destination = tmp_path / "out"
    _extract_archive(archive, destination)

    assert (destination / "dataset/agent/case.yaml").read_bytes() == payload
    assert not (destination / "README.md").exists()


def test_archive_extraction_rejects_resolved_target_escape(tmp_path: Path) -> None:
    archive = tmp_path / "symlink-parent.tar.gz"
    payload = b"bad"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("root/dataset/outside.txt")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "out"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    try:
        (destination / "dataset").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(DatasetError, match="unsafe path"):
        _extract_archive(archive, destination)
    assert not (outside / "outside.txt").exists()


def test_dockerfile_uses_canonical_python311_base_and_frozen_sync() -> None:
    dockerfile = (_ROOT / "envs/thinkingbox_env/server/Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest" in dockerfile
    assert dockerfile.count("FROM ${BASE_IMAGE}") == 2
    assert "python:3.12-slim" not in dockerfile
    assert "curl -LsSf https://astral.sh/uv/install.sh" in dockerfile
    assert "if [ -f uv.lock ]; then" in dockerfile
    assert "uv sync --frozen --no-install-project --no-editable" in dockerfile
    assert "uv sync --frozen --no-editable" in dockerfile
    assert "COPY --from=builder /app/env /app/env" in dockerfile
    assert 'ENV PATH="/app/env/.venv/bin:$PATH"' in dockerfile
    assert "COPY --from=builder /app/env/.venv /app/.venv" not in dockerfile
    assert "apt-get install -y --no-install-recommends git" in dockerfile


def test_docker_context_excludes_local_and_result_artifacts_but_keeps_lock() -> None:
    """The image context should remain reproducible and omit local run artifacts."""
    env_root = _ROOT / "envs/thinkingbox_env"
    patterns = (env_root / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for required in (
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "build",
        "dist",
        "*.egg-info",
        "*.pyc",
        "*.pyo",
        "*.log",
        "output*",
        "result*",
        "error*",
    ):
        assert required in patterns
    assert "uv.lock" not in patterns
    assert (env_root / "SKIP_HF_DEPLOYMENT").read_text(encoding="utf-8") == (
        "disabled now, coming soon\n"
    )


def test_python311_metadata_pin_and_packaged_cli_are_canonical() -> None:
    env_root = _ROOT / "envs/thinkingbox_env"
    project = tomllib.loads((env_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = (env_root / "uv.lock").read_text(encoding="utf-8")

    assert project["project"]["requires-python"] == ">=3.11"
    assert project["project"]["scripts"]["thinkingbox-eval"] == (
        "thinkingbox_env.evaluation:main"
    )
    thinkingbox_dependency = next(
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("thinkingbox @ ")
    )
    assert thinkingbox_dependency.endswith("@40c1212f9582ca90175079bc313e530e9e9a4981")
    assert 'requires-python = ">=3.11"' in lock
    assert "40c1212f9582ca90175079bc313e530e9e9a4981" in lock
    assert "file://" not in thinkingbox_dependency
    assert "file://" not in lock
    runtime_source = (env_root / "runtime.py").read_text(encoding="utf-8")
    assert "import copy" not in runtime_source
    assert "reasoning_effort" not in runtime_source


def test_examples_and_packaged_cli_run_from_repository_root() -> None:
    wrapper = _ROOT / "examples/thinkingbox/eval_testlist.py"
    wrapper_lines = [
        line
        for line in wrapper.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith('"""')
    ]
    assert len(wrapper_lines) <= 4
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_ROOT / "envs"))),
    }
    commands = (
        [sys.executable, "examples/thinkingbox/example_usage.py", "--help"],
        [sys.executable, "examples/thinkingbox/eval_testlist.py", "--help"],
        [sys.executable, "-m", "thinkingbox_env.evaluation", "--help"],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_native_config_forwards_agent_user_and_judge_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "thinkingbox.yaml"
    path.write_text(
        """
mcp_proxy:
  endpoint_url: https://proxy.invalid
  headers:
    authorization: proxy-secret
judge_model:
  type: custom
  factory: test_models.JudgeSession
  model: judge-model
  api_key: judge-secret
user_model:
  type: custom
  factory: test_models.UserSession
  model: user-model
  api_key: user-secret
orchestrator:
  type: thinkingbox
  agent_model:
    type: custom
    factory: test_models.AgentSession
    model: gpt-5.4
    reasoning_effort: xhigh
    api_key: agent-secret
judge_type: motivation
user_can_end_conversation: true
""".lstrip(),
        encoding="utf-8",
    )

    config = thinkingbox_eval.load_thinkingbox_config(path)
    captured: list[Any] = []
    expected_factory = object()
    monkeypatch.setattr(
        thinkingbox_eval,
        "get_agent_session_factory",
        lambda orchestrator: captured.append(orchestrator) or expected_factory,
    )

    assert thinkingbox_eval.configured_agent_session_factory(config) is expected_factory
    assert captured == [config.orchestrator]
    assert config.orchestrator.agent_model.model_dump()["model"] == "gpt-5.4"
    assert config.orchestrator.agent_model.model_dump()["reasoning_effort"] == "xhigh"

    settings = load_runtime_settings(
        str(path),
        SessionProxyConfig(endpoint_url="https://default.invalid"),
    )
    assert settings.user_model.model_dump() == config.user_model.model_dump()
    assert settings.judge_model.model_dump() == config.judge_model.model_dump()
    assert settings.judge_type == "motivation"
    assert settings.user_can_end_conversation is True
    assert settings.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = thinkingbox_eval.config_provenance(config, path)
    assert provenance["agent_model"]["model"] == "gpt-5.4"
    assert provenance["agent_model"]["reasoning_effort"] == "xhigh"
    assert provenance["user_model"]["model"] == "user-model"
    assert provenance["judge_model"]["model"] == "judge-model"
    assert "secret" not in str(provenance)
    assert "proxy.invalid" not in str(provenance)


def test_native_runtime_accepts_aoai_responses_xhigh_directly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "thinkingbox-xhigh.yaml"
    path.write_text(
        """
mcp_proxy:
  endpoint_url: https://proxy.invalid
agent_model:
  type: aoai_responses
  credential:
    type: az-cli
  account_name: aoai-example
  deployment: gpt-5.4
  is_reasoning: true
  reasoning_effort: xhigh
  reasoning_source: none
judge_model:
  type: aoai
  credential:
    type: az-cli
  account_name: aoai-example
  deployment: judge
user_model:
  type: aoai
  credential:
    type: az-cli
  account_name: aoai-example
  deployment: user
""".lstrip(),
        encoding="utf-8",
    )

    config = load_thinkingbox_config(path)

    assert config.orchestrator.agent_model.type == "aoai_responses"
    assert config.orchestrator.agent_model.reasoning_effort == "xhigh"


def test_framework_hash_uses_only_tracked_runtime_sources_and_lockfiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = [
        Path("examples/thinkingbox/eval_testlist.py"),
        Path("envs/thinkingbox_env/evaluation.py"),
        Path("envs/thinkingbox_env/server/thinkingbox_environment.py"),
        Path("envs/thinkingbox_env/pyproject.toml"),
        Path("envs/thinkingbox_env/uv.lock"),
        Path("src/openenv/core/env_client.py"),
        Path("envs/thinkingbox_env/.venv/lib/generated.py"),
        Path("envs/thinkingbox_env/build/generated.py"),
        Path("src/openenv/core/generated/client.py"),
    ]
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    monkeypatch.setattr(
        thinkingbox_eval,
        "_git_tracked_files",
        lambda _: tracked,
    )

    initial = thinkingbox_eval._framework_sha256(tmp_path)
    (tmp_path / "envs/thinkingbox_env/.venv/lib/generated.py").write_text(
        "changed virtualenv\n",
        encoding="utf-8",
    )
    (tmp_path / "envs/thinkingbox_env/build/generated.py").write_text(
        "changed build output\n",
        encoding="utf-8",
    )
    (tmp_path / "src/openenv/core/generated/client.py").write_text(
        "changed generated output\n",
        encoding="utf-8",
    )
    (tmp_path / "envs/thinkingbox_env/untracked.py").write_text(
        "untracked runtime-looking file\n",
        encoding="utf-8",
    )
    assert thinkingbox_eval._framework_sha256(tmp_path) == initial

    (tmp_path / "envs/thinkingbox_env/uv.lock").write_text(
        "changed lockfile\n",
        encoding="utf-8",
    )
    assert thinkingbox_eval._framework_sha256(tmp_path) != initial


def test_runtime_framework_hash_uses_actual_imported_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "installed/thinkingbox_env"
    openenv_root = tmp_path / "installed/openenv/core"
    checkout_root = tmp_path / "checkout"
    package_root.mkdir(parents=True)
    openenv_root.mkdir(parents=True)
    (package_root / "evaluation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (openenv_root / "env_client.py").write_text("VALUE = 1\n", encoding="utf-8")
    false_openenv = checkout_root / "src/openenv/core/env_client.py"
    false_openenv.parent.mkdir(parents=True)
    false_openenv.write_text("FALSE = 1\n", encoding="utf-8")

    roots = {
        "thinkingbox_env": package_root,
        "openenv.core": openenv_root,
    }
    monkeypatch.setattr(
        thinkingbox_eval,
        "_module_source_root",
        lambda name: roots[name],
    )
    monkeypatch.setattr(thinkingbox_eval, "_git_root", lambda _: checkout_root)
    assert thinkingbox_eval._framework_checkout_root(package_root) is None

    source_package = checkout_root / "envs/thinkingbox_env"
    source_package.mkdir(parents=True)
    assert thinkingbox_eval._framework_checkout_root(source_package) == checkout_root

    initial = thinkingbox_eval._runtime_framework_sha256(checkout_root)
    false_openenv.write_text("FALSE = 2\n", encoding="utf-8")
    assert thinkingbox_eval._runtime_framework_sha256(checkout_root) == initial

    (openenv_root / "env_client.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert thinkingbox_eval._runtime_framework_sha256(checkout_root) != initial


def _eval_provenance() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "thinkingbox_revision": "runtime-revision",
        "thinkingbox_source_sha256": "runtime-source-sha256",
        "thinkingbox_source_type": "vcs",
        "data_release": "release-name",
        "data_revision": "data-revision",
        "data_bundle_sha256": "bundle-sha256",
        "config": {
            "sha256": "config-sha256",
            "agent_model": {"type": "custom", "model": "agent-model"},
            "user_model": None,
            "judge_model": {"type": "custom", "model": "judge-model"},
            "judge_type": "legacy",
            "user_can_end_conversation": False,
        },
        "manifest_path": "releases/release/testlist.yaml",
        "manifest_sha256": "manifest-sha256",
        "manifest_uids_sha256": "manifest-uids-sha256",
        "task_count": 1,
        "requested_test_list_sha256": "test-list-sha256",
        "framework": {
            "revision": "pr-revision",
            "sha256": "framework-sha256",
        },
        "policy": None,
        "python": "3.13.0",
        "run": {
            "agent": "think",
            "repeat": 3,
            "repetition_start": 0,
            "shard_index": 0,
            "shard_count": 1,
            "limit": None,
            "message_timeout_s": 1800.0,
            "base_url_sha256": "base-url-sha256",
            "selected_uids_sha256": "selected-uids-sha256",
        },
    }


def _eval_outcome(passed: bool, provenance: dict[str, Any]) -> Any:
    messages = [
        Text(role="user", content="public task"),
        Text(role="assistant", content="public answer"),
    ]
    observation = SimpleNamespace(
        kind="terminal",
        task_uid=UID,
        finish_reason="done",
        reward_type="pass" if passed else "fail",
        system_error=False,
        error=None,
        test_summary={
            "passed": passed,
            "graded": True,
            "is_system_error": False,
        },
        steps_taken=2,
        messages=[message.model_dump(mode="json") for message in messages],
        metadata={
            "thinkingbox_revision": provenance["thinkingbox_revision"],
            "data_revision": provenance["data_revision"],
            "execution_provenance": thinkingbox_eval._expected_execution_provenance(
                provenance
            ),
        },
    )
    return thinkingbox_eval._EpisodeOutcome(
        result=SimpleNamespace(
            done=True,
            reward=1.0 if passed else 0.0,
            observation=observation,
            metadata=None,
        ),
        error=None,
        trace=thinkingbox_eval._ExecutionTrace(
            messages=messages,
            usage=[Usage(input_tokens=4, output_tokens=2, total_tokens=6)],
            steps_taken=2,
        ),
    )


def _fixture_decode_result(
    uid: str,
    repetition: int,
    passed: bool,
    provenance: dict[str, Any],
) -> DecodeResult:
    return DecodeResult(
        uid=uid,
        messages=[],
        test_result=NativeTestResult(
            result=passed,
            reward=1.0 if passed else 0.0,
        ),
        test_tags=_case().tags,
        usage=[],
        metadata={
            "schema_version": 3,
            "repetition": repetition,
            "repetition_id": thinkingbox_eval._repetition_id(uid, repetition),
            "attempt": 1,
            "steps_taken": 0,
            "step_counts": {
                "openenv_steps": 0,
                "native_messages": 0,
                "assistant_messages": 0,
                "user_messages": 0,
                "tool_batches": 0,
                "tool_responses": 0,
            },
            "execution_provenance": (
                thinkingbox_eval._expected_execution_provenance(provenance)
            ),
            "provenance": provenance,
        },
        is_system_error=False,
        finish_reason="done",
    )


def _write_fixture_results(path: Path, results: list[DecodeResult]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            thinkingbox_eval._write_canonical(stream, result)


def test_eval_canonical_fixture_validates_and_native_tb_agg_is_exact(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    results = [
        thinkingbox_eval._canonical_decode_result(
            uid=UID,
            repetition=repetition,
            attempt=1,
            provenance=provenance,
            test_case=_case(),
            outcome=_eval_outcome(passed, provenance),
        )
        for repetition, passed in enumerate((True, True, False))
    ]
    path = tmp_path / "results.jsonl"
    _write_fixture_results(path, results)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    serialized = "\n".join(lines)
    assert "world-secret" not in serialized
    assert "assert True" not in serialized
    assert "user_context" not in serialized
    assert "test_context" not in serialized
    for line in lines:
        decoded = DecodeResult.model_validate_json(line)
        raw = json.loads(line)
        assert decoded.test_result is not None
        assert set(raw["test_result"]) == {
            "result",
            "reward",
            "is_system_error",
        }
        assert raw["usage"][0]["total_tokens"] == 6
        assert raw["metadata"]["steps_taken"] == 2
        assert "test_context" not in raw

    report = thinkingbox_eval.validate_coverage(
        path,
        [UID],
        [0, 1, 2],
        expected_provenance=provenance,
    )
    assert report.total_results == 3

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "thinkingbox.cli.main",
            "agg",
            str(path),
            "-f",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    aggregated = json.loads(completed.stdout)
    assert aggregated["per_test"][0]["uid"] == UID
    assert aggregated["per_test"][0]["runs"] == 3
    assert aggregated["per_test"][0]["passed"] == 2
    assert aggregated["per_test"][0]["failed"] == 1
    assert aggregated["per_test"][0]["error"] == 0
    assert aggregated["metrics"] == {
        "num_tests": 1,
        "runs_per_test": 3,
        "total_runs": 3,
        "mean_pass": 2 / 3,
        "mean_pass_ci_low": pytest.approx(0.1767360971),
        "mean_pass_ci_high": pytest.approx(0.9612523822),
        "unbiased_pass_at_k": [[1, 2 / 3]],
        "pass_power_k": [[1, 2 / 3]],
    }

    leaked = results[0].model_copy(deep=True)
    leaked.metadata["user_context"] = "private"
    with pytest.raises(thinkingbox_eval.CoverageError, match="forbidden"):
        thinkingbox_eval._canonical_payload(leaked)


def test_eval_canonical_payload_allows_agent_tool_argument_field_names() -> None:
    provenance = _eval_provenance()
    result = _fixture_decode_result(UID, 0, True, provenance)
    result.messages = [
        ParallelToolCall(
            tool_calls=[
                ToolCall(
                    name="authenticate",
                    arguments={
                        "token": "agent-supplied-placeholder",
                        "request": {
                            "headers": {"authorization": "placeholder"},
                            "session_id": "agent-session",
                        },
                    },
                    id="credential-shaped-arguments",
                )
            ]
        )
    ]

    payload = thinkingbox_eval._canonical_payload(result)

    assert payload["messages"][0]["tool_calls"][0]["arguments"] == {
        "token": "agent-supplied-placeholder",
        "request": {
            "headers": {"authorization": "placeholder"},
            "session_id": "agent-session",
        },
    }


def test_eval_rejects_mismatched_or_mutating_server_execution_provenance() -> None:
    provenance = _eval_provenance()
    outcome = _eval_outcome(True, provenance)
    outcome.result.observation.metadata["execution_provenance"]["config_sha256"] = (
        "different-config"
    )
    with pytest.raises(
        thinkingbox_eval._OperationalResultError,
        match="did not exactly match",
    ):
        thinkingbox_eval._canonical_decode_result(
            uid=UID,
            repetition=0,
            attempt=1,
            provenance=provenance,
            test_case=_case(),
            outcome=outcome,
        )

    changing = _eval_outcome(True, provenance)
    changing.trace.execution_provenance = (
        thinkingbox_eval._expected_execution_provenance(provenance)
    )
    changing.result.observation.metadata["execution_provenance"][
        "data_bundle_sha256"
    ] = "mutated-bundle"
    with pytest.raises(
        thinkingbox_eval._OperationalResultError,
        match="did not exactly match",
    ):
        thinkingbox_eval._canonical_decode_result(
            uid=UID,
            repetition=0,
            attempt=1,
            provenance=provenance,
            test_case=_case(),
            outcome=changing,
        )


def test_eval_strict_raw_validation_rejects_discarded_and_forbidden_fields(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    canonical = thinkingbox_eval._canonical_decode_result(
        uid=UID,
        repetition=0,
        attempt=1,
        provenance=provenance,
        test_case=_case(),
        outcome=_eval_outcome(True, provenance),
    )
    payload = thinkingbox_eval._canonical_payload(canonical)
    output = tmp_path / "results.jsonl"

    discarded = copy.deepcopy(payload)
    discarded["messages"][0]["benign_extra"] = "pydantic would discard this"
    output.write_text(json.dumps(discarded) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="raw shape"):
        thinkingbox_eval.validate_coverage(
            output,
            [UID],
            [0],
            expected_provenance=provenance,
        )

    forbidden = copy.deepcopy(payload)
    forbidden["usage"][0]["effects"] = {"private": True}
    output.write_text(json.dumps(forbidden) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="forbidden"):
        thinkingbox_eval.validate_coverage(
            output,
            [UID],
            [0],
            expected_provenance=provenance,
        )

    credential = copy.deepcopy(payload)
    credential["messages"][0]["credential"] = "secret"
    output.write_text(json.dumps(credential) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="forbidden"):
        thinkingbox_eval.validate_coverage(
            output,
            [UID],
            [0],
            expected_provenance=provenance,
        )


def test_eval_botdesigner_failure_is_canonical_but_system_errors_are_quarantined(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    botdesigner_error = type(
        "BotDesignerActivityError",
        (RuntimeError,),
        {},
    )("benchmark activity failed")
    botdesigner_outcome = thinkingbox_eval._EpisodeOutcome(
        result=None,
        error=botdesigner_error,
        trace=thinkingbox_eval._ExecutionTrace(
            messages=[Text(role="assistant", content="partial public answer")],
            usage=[],
            steps_taken=1,
            execution_provenance=(
                thinkingbox_eval._expected_execution_provenance(provenance)
            ),
        ),
    )
    canonical = thinkingbox_eval._canonical_decode_result(
        uid=UID,
        repetition=0,
        attempt=1,
        provenance=provenance,
        test_case=_case(),
        outcome=botdesigner_outcome,
    )
    assert canonical.test_result is not None
    assert canonical.test_result.result is False
    assert canonical.is_system_error is False
    assert canonical.finish_reason == "agent_error"
    assert canonical.metadata["benchmark_failure_type"] == "BotDesignerActivityError"

    operational = RuntimeError("credential=trusted-sidecar-only")
    operational_outcome = thinkingbox_eval._EpisodeOutcome(
        result=None,
        error=operational,
        trace=thinkingbox_eval._ExecutionTrace(),
    )
    with pytest.raises(RuntimeError, match="trusted-sidecar-only"):
        thinkingbox_eval._canonical_decode_result(
            uid=UID,
            repetition=1,
            attempt=2,
            provenance=provenance,
            test_case=_case(),
            outcome=operational_outcome,
        )

    error_record = thinkingbox_eval._operational_error_record(
        uid=UID,
        repetition=1,
        attempt=2,
        provenance=provenance,
        exc=operational,
        outcome=operational_outcome,
    )
    errors_path = tmp_path / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as stream:
        thinkingbox_eval._write_error(stream, error_record)
    attempts = thinkingbox_eval._error_attempts(
        errors_path,
        expected=thinkingbox_eval._expected_repetition_ids([UID], [1]),
        provenance=provenance,
    )
    assert attempts == {thinkingbox_eval._repetition_id(UID, 1): 2}
    error_payload = json.loads(errors_path.read_text(encoding="utf-8"))
    assert error_payload["error"] == {
        "type": "RuntimeError",
        "message": "details redacted",
        "chain": [
            {
                "type": "RuntimeError",
                "message": "details redacted",
            }
        ],
        "traceback": "details redacted",
    }
    assert "trusted-sidecar-only" not in errors_path.read_text(encoding="utf-8")
    assert "trusted-sidecar-only" not in json.dumps(
        thinkingbox_eval._canonical_payload(canonical)
    )

    try:
        try:
            raise botdesigner_error
        except Exception:
            raise RuntimeError("teardown failed")
    except RuntimeError as teardown_error:
        assert not thinkingbox_eval._is_botdesigner_exception(teardown_error)


@pytest.mark.asyncio
async def test_eval_run_retries_to_dual_outputs_and_resume_skips_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    errors = tmp_path / "errors.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text("unused: true\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"- {UID}\n", encoding="utf-8")
    provenance = _eval_provenance()
    base_provenance = {key: value for key, value in provenance.items() if key != "run"}
    bundle = SimpleNamespace(
        root=tmp_path,
        dataset_dir=tmp_path,
        manifest_path=manifest,
        bundle_sha256=provenance["data_bundle_sha256"],
        manifest_sha256=provenance["manifest_sha256"],
        manifest_uids_sha256=provenance["manifest_uids_sha256"],
        task_count=1,
        release_name=provenance["data_release"],
        revision=provenance["data_revision"],
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "load_thinkingbox_config_with_sha256",
        lambda _: (object(), provenance["config"]["sha256"]),
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "configured_agent_session_factory",
        lambda _: object(),
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "resolve_data_bundle",
        lambda _: bundle,
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "load_test_uids",
        lambda _: [UID],
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "get_dataset_case_by_name",
        lambda *_args, **_kwargs: _case(),
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "_base_provenance",
        lambda **_: copy.deepcopy(base_provenance),
    )
    outcomes = [
        thinkingbox_eval._EpisodeOutcome(
            result=None,
            error=RuntimeError("transport credential=trusted-sidecar"),
            trace=thinkingbox_eval._ExecutionTrace(),
        ),
        _eval_outcome(True, provenance),
        _eval_outcome(True, provenance),
    ]

    async def fake_run_episode(*_args: Any, **_kwargs: Any) -> Any:
        return outcomes.pop(0)

    monkeypatch.setattr(thinkingbox_eval, "_run_episode", fake_run_episode)
    encode_canonical = thinkingbox_eval._encode_canonical
    encode_attempts = 0

    def fail_first_canonical_encoding(result: DecodeResult) -> str:
        nonlocal encode_attempts
        encode_attempts += 1
        if encode_attempts == 1:
            raise thinkingbox_eval.CoverageError(
                "synthetic canonical validation failure"
            )
        return encode_canonical(result)

    monkeypatch.setattr(
        thinkingbox_eval,
        "_encode_canonical",
        fail_first_canonical_encoding,
    )
    args = SimpleNamespace(
        repeat=1,
        repetition_start=0,
        attempts_per_repetition=3,
        message_timeout=1800.0,
        shard_count=1,
        shard_index=0,
        limit=None,
        aggregate_output=None,
        coverage_profile=None,
        coverage_input=[],
        output=str(output),
        errors_output=str(errors),
        validate_only=False,
        config=str(config),
        policy=None,
        dataset=None,
        test_list=None,
        agent="think",
        base_url="http://env.invalid",
        env_config=None,
        env_dataset=None,
        resume=False,
    )
    await thinkingbox_eval._run(args)

    canonical_lines = output.read_text(encoding="utf-8").splitlines()
    error_lines = errors.read_text(encoding="utf-8").splitlines()
    assert len(canonical_lines) == 1
    assert len(error_lines) == 2
    canonical = DecodeResult.model_validate_json(canonical_lines[0])
    operational_error = json.loads(error_lines[0])
    write_error = json.loads(error_lines[1])
    assert canonical.test_result is not None
    assert canonical.test_result.result is True
    assert canonical.metadata["attempt"] == 3
    assert operational_error["attempt"] == 1
    assert operational_error["retryable"] is True
    assert operational_error["error"]["message"] == "details redacted"
    assert write_error["attempt"] == 2
    assert write_error["retryable"] is True
    assert write_error["error"]["message"] == "details redacted"
    assert "trusted-sidecar" not in "\n".join(error_lines)
    assert "synthetic canonical validation failure" not in "\n".join(error_lines)
    assert "trusted-sidecar" not in canonical_lines[0]

    async def should_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("valid canonical repetition should be resumed")

    monkeypatch.setattr(thinkingbox_eval, "_run_episode", should_not_run)
    args.resume = True
    await thinkingbox_eval._run(args)
    assert output.read_text(encoding="utf-8").splitlines() == canonical_lines
    assert errors.read_text(encoding="utf-8").splitlines() == error_lines

    outcomes.append(_eval_outcome(True, provenance))
    monkeypatch.setattr(thinkingbox_eval, "_run_episode", fake_run_episode)
    args.resume = False

    def fail_result_write(_stream: Any, _encoded: str) -> None:
        raise OSError("result stream unavailable")

    monkeypatch.setattr(
        thinkingbox_eval,
        "_write_encoded_canonical",
        fail_result_write,
    )
    with pytest.raises(OSError, match="result stream unavailable"):
        await thinkingbox_eval._run(args)
    assert errors.read_text(encoding="utf-8") == ""


def test_eval_resume_rejects_malformed_duplicate_unexpected_and_stale_records(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    expected = thinkingbox_eval._expected_repetition_ids([UID], [0])
    valid = _fixture_decode_result(UID, 0, True, provenance)
    valid_payload = thinkingbox_eval._canonical_payload(valid)
    output = tmp_path / "results.jsonl"
    _write_fixture_results(output, [valid])
    assert thinkingbox_eval._load_resume_state(
        output,
        expected=expected,
        provenance=provenance,
    ) == {thinkingbox_eval._repetition_id(UID, 0)}

    output.write_text('{"uid":', encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="malformed"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )

    encoded = json.dumps(valid_payload, sort_keys=True)
    output.write_text(f"{encoded}\n{encoded}\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="duplicate"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )

    unexpected_uid = copy.deepcopy(valid_payload)
    unexpected_uid["uid"] = "unexpected.py:test_case"
    output.write_text(json.dumps(unexpected_uid) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="does not match"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )

    unexpected_index = copy.deepcopy(valid_payload)
    unexpected_index["metadata"]["repetition"] = 1
    output.write_text(json.dumps(unexpected_index) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="does not match"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )

    mutations = (
        ("runtime", lambda value: value.__setitem__("thinkingbox_revision", "old")),
        ("data", lambda value: value.__setitem__("data_revision", "old")),
        (
            "data bundle",
            lambda value: value.__setitem__("data_bundle_sha256", "old"),
        ),
        (
            "config",
            lambda value: value["config"].__setitem__("sha256", "old"),
        ),
        (
            "framework",
            lambda value: value["framework"].__setitem__("revision", "old"),
        ),
        (
            "shard",
            lambda value: value["run"].__setitem__("shard_index", 1),
        ),
    )
    for _, mutate in mutations:
        stale = copy.deepcopy(valid_payload)
        mutate(stale["metadata"]["provenance"])
        output.write_text(json.dumps(stale) + "\n", encoding="utf-8")
        with pytest.raises(
            thinkingbox_eval.CoverageError,
            match="stale/mixed provenance",
        ):
            thinkingbox_eval._load_resume_state(
                output,
                expected=expected,
                provenance=provenance,
            )

    stale_execution = copy.deepcopy(valid_payload)
    stale_execution["metadata"]["execution_provenance"]["manifest_sha256"] = (
        "different-server-manifest"
    )
    output.write_text(json.dumps(stale_execution) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="stale/mixed provenance"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )


def test_eval_resume_rejects_non_newline_terminated_canonical_and_error_jsonl(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    expected = thinkingbox_eval._expected_repetition_ids([UID], [0])
    canonical = thinkingbox_eval._canonical_payload(
        _fixture_decode_result(UID, 0, True, provenance)
    )
    output = tmp_path / "results.jsonl"
    encoded = json.dumps(canonical, sort_keys=True)
    output.write_text(encoded, encoding="utf-8")

    with pytest.raises(thinkingbox_eval.CoverageError, match="terminal newline"):
        thinkingbox_eval._load_resume_state(
            output,
            expected=expected,
            provenance=provenance,
        )
    assert output.read_text(encoding="utf-8") == encoded

    errors = tmp_path / "errors.jsonl"
    error_record = thinkingbox_eval._operational_error_record(
        uid=UID,
        repetition=0,
        attempt=1,
        provenance=provenance,
        exc=RuntimeError("transport failed"),
        outcome=None,
    )
    encoded_error = json.dumps(error_record, sort_keys=True)
    errors.write_text(encoded_error, encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="terminal newline"):
        thinkingbox_eval._error_attempts(
            errors,
            expected=expected,
            provenance=provenance,
        )
    assert errors.read_text(encoding="utf-8") == encoded_error


def test_eval_coverage_rejects_missing_duplicate_and_system_error(
    tmp_path: Path,
) -> None:
    provenance = _eval_provenance()
    uids = [UID, "demo.py:test_case_2"]
    repetitions = [0, 1]
    results = [
        _fixture_decode_result(uid, repetition, True, provenance)
        for uid in uids
        for repetition in repetitions
    ]
    output = tmp_path / "results.jsonl"
    _write_fixture_results(output, results)
    report = thinkingbox_eval.validate_coverage(
        output,
        uids,
        repetitions,
        expected_provenance=provenance,
    )
    assert report.total_results == 4

    _write_fixture_results(output, results[:-1])
    with pytest.raises(thinkingbox_eval.CoverageError, match="incomplete"):
        thinkingbox_eval.validate_coverage(
            output,
            uids,
            repetitions,
            expected_provenance=provenance,
        )

    _write_fixture_results(output, [*results, results[0]])
    with pytest.raises(thinkingbox_eval.CoverageError, match="duplicate"):
        thinkingbox_eval.validate_coverage(
            output,
            uids,
            repetitions,
            expected_provenance=provenance,
        )

    system_error = thinkingbox_eval._canonical_payload(results[0])
    system_error["is_system_error"] = True
    output.write_text(json.dumps(system_error) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="system error"):
        thinkingbox_eval.validate_coverage(
            output,
            [UID],
            [0],
            expected_provenance=provenance,
        )


def test_eval_coverage_combines_matching_shards_and_rejects_stale_assignment(
    tmp_path: Path,
) -> None:
    uids = [f"demo.py:test_case_{index}" for index in range(4)]
    paths: list[Path] = []
    for shard_index in range(2):
        shard_uids = [uid for index, uid in enumerate(uids) if index % 2 == shard_index]
        provenance = _eval_provenance()
        provenance["run"].update(
            {
                "repeat": 1,
                "shard_index": shard_index,
                "shard_count": 2,
                "selected_uids_sha256": thinkingbox_eval._sha256_json(shard_uids),
            }
        )
        path = tmp_path / f"shard-{shard_index}.jsonl"
        _write_fixture_results(
            path,
            [_fixture_decode_result(uid, 0, True, provenance) for uid in shard_uids],
        )
        paths.append(path)

    report = thinkingbox_eval.validate_coverage(paths, uids, [0])
    assert report.total_results == 4

    stale = json.loads(paths[1].read_text(encoding="utf-8").splitlines()[0])
    stale["metadata"]["provenance"]["run"]["shard_index"] = 0
    paths[1].write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(thinkingbox_eval.CoverageError, match="run parameters|shard"):
        thinkingbox_eval.validate_coverage(paths, uids, [0])


def test_eval_limited_shards_reject_modulo_match_outside_post_limit_uids(
    tmp_path: Path,
) -> None:
    uids = [f"demo.py:test_case_{index}" for index in range(4)]
    paths: list[Path] = []
    for shard_index in range(2):
        modulo_uids = [
            uid for index, uid in enumerate(uids) if index % 2 == shard_index
        ]
        shard_uids = modulo_uids[:1]
        provenance = _eval_provenance()
        provenance["run"].update(
            {
                "repeat": 1,
                "shard_index": shard_index,
                "shard_count": 2,
                "limit": 1,
                "selected_uids_sha256": thinkingbox_eval._sha256_json(shard_uids),
            }
        )
        path = tmp_path / f"limited-shard-{shard_index}.jsonl"
        _write_fixture_results(
            path,
            [_fixture_decode_result(uid, 0, True, provenance) for uid in modulo_uids],
        )
        paths.append(path)

    with pytest.raises(thinkingbox_eval.CoverageError, match="stale shard"):
        thinkingbox_eval.validate_coverage(paths, uids, [0])


@pytest.mark.parametrize(
    ("profile", "repetitions", "expected_count"),
    [
        ("canary", (0,), 3),
        ("full", tuple(range(20)), 3 * 20),
    ],
)
def test_eval_coverage_profiles_use_manifest_count_and_required_repetitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    repetitions: tuple[int, ...],
    expected_count: int,
) -> None:
    uids = [f"fixture.py:test_case_{index:03d}" for index in range(3)]
    provenance = _eval_provenance()
    manifest_uids_sha256 = thinkingbox_eval._sha256_json(uids)
    provenance["manifest_uids_sha256"] = manifest_uids_sha256
    provenance["task_count"] = len(uids)
    canonical_keys = {
        "thinkingbox_revision",
        "thinkingbox_source_sha256",
        "thinkingbox_source_type",
        "data_release",
        "data_revision",
        "data_bundle_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_uids_sha256",
        "task_count",
    }
    monkeypatch.setattr(
        thinkingbox_eval,
        "_canonical_release_identity",
        lambda: {key: provenance[key] for key in canonical_keys},
    )
    provenance["run"]["repeat"] = len(repetitions)
    provenance["run"]["selected_uids_sha256"] = thinkingbox_eval._sha256_json(uids)
    output = tmp_path / f"{profile}.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for uid in uids:
            for repetition in repetitions:
                payload = thinkingbox_eval._canonical_payload(
                    _fixture_decode_result(
                        uid,
                        repetition,
                        repetition % 2 == 0,
                        provenance,
                    )
                )
                stream.write(json.dumps(payload, sort_keys=True) + "\n")

    report = thinkingbox_eval.validate_coverage(
        output,
        uids,
        repetitions,
        profile=profile,
    )
    assert report.uid_count == len(uids)
    assert report.total_results == expected_count


def test_eval_coverage_profiles_reject_unrelated_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_uids = ["canonical.py:test_one", "canonical.py:test_two"]
    uids = ["unrelated.py:test_one", "unrelated.py:test_two"]
    identity = {
        key: _eval_provenance()[key]
        for key in (
            "thinkingbox_revision",
            "thinkingbox_source_sha256",
            "thinkingbox_source_type",
            "data_release",
            "data_revision",
            "data_bundle_sha256",
            "manifest_path",
            "manifest_sha256",
        )
    }
    identity.update(
        {
            "manifest_uids_sha256": thinkingbox_eval._sha256_json(canonical_uids),
            "task_count": len(canonical_uids),
        }
    )
    monkeypatch.setattr(
        thinkingbox_eval,
        "_canonical_release_identity",
        lambda: identity,
    )
    output = tmp_path / "unrelated.jsonl"
    output.write_text("", encoding="utf-8")

    with pytest.raises(
        thinkingbox_eval.CoverageError,
        match="authoritative benchmark manifest",
    ):
        thinkingbox_eval.validate_coverage(
            output,
            uids,
            (0,),
            profile="canary",
        )


def test_eval_strict_profile_rejects_noncanonical_expected_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uids = [f"fixture.py:test_case_{index:03d}" for index in range(3)]
    provenance = _eval_provenance()
    provenance["task_count"] = len(uids)
    provenance["manifest_uids_sha256"] = thinkingbox_eval._sha256_json(uids)
    provenance["run"]["repeat"] = 1
    provenance["run"]["selected_uids_sha256"] = thinkingbox_eval._sha256_json(uids)
    canonical = {
        key: provenance[key]
        for key in (
            "thinkingbox_revision",
            "thinkingbox_source_sha256",
            "thinkingbox_source_type",
            "data_release",
            "data_revision",
            "data_bundle_sha256",
            "manifest_path",
            "manifest_sha256",
            "manifest_uids_sha256",
            "task_count",
        )
    }
    canonical["data_bundle_sha256"] = "canonical-bundle-sha256"
    monkeypatch.setattr(
        thinkingbox_eval,
        "_canonical_release_identity",
        lambda: canonical,
    )
    output = tmp_path / "noncanonical-expected-provenance.jsonl"
    _write_fixture_results(
        output,
        [_fixture_decode_result(uid, 0, True, provenance) for uid in uids],
    )

    with pytest.raises(
        thinkingbox_eval.CoverageError,
        match="noncanonical benchmark content",
    ):
        thinkingbox_eval.validate_coverage(
            output,
            uids,
            (0,),
            expected_provenance=provenance,
            profile="canary",
        )


@pytest.mark.asyncio
async def test_eval_episode_discards_timed_out_client_before_next_repetition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[Any] = []

    class FakeClient:
        def __init__(
            self,
            *,
            base_url: str,
            message_timeout_s: float,
        ) -> None:
            self.base_url = base_url
            self.message_timeout_s = message_timeout_s
            self.index = len(clients)
            self.closed = False
            clients.append(self)

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_: Any) -> None:
            self.closed = True

        async def reset(self, *_: Any, **__: Any) -> Any:
            if self.index == 0:
                raise TimeoutError("delayed response")
            return SimpleNamespace(done=True, observation=SimpleNamespace())

    monkeypatch.setattr(thinkingbox_eval, "ThinkingBoxEnv", FakeClient)
    args = SimpleNamespace(
        base_url="http://env.invalid",
        message_timeout=DEFAULT_MESSAGE_TIMEOUT_S,
        agent="think",
    )
    kwargs = {
        "test_case": object(),
        "agent_session_factory": object(),
        "policy": None,
        "env_config": "config.yaml",
        "env_dataset": None,
    }

    first = await thinkingbox_eval._run_episode(
        args,
        UID,
        "repeat-0",
        **kwargs,
    )
    second = await thinkingbox_eval._run_episode(
        args,
        UID,
        "repeat-1",
        **kwargs,
    )

    assert isinstance(first.error, TimeoutError)
    assert first.result is None
    assert second.error is None
    assert second.result.done is True
    assert len(clients) == 2
    assert all(client.closed for client in clients)
    assert all(
        client.message_timeout_s == DEFAULT_MESSAGE_TIMEOUT_S for client in clients
    )
    assert DEFAULT_MESSAGE_TIMEOUT_S >= 1800


@pytest.mark.asyncio
async def test_custom_policy_receives_only_public_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_result = SimpleNamespace(
        done=False,
        observation=SimpleNamespace(metadata={}),
    )
    policy_result = SimpleNamespace(
        done=True,
        observation=SimpleNamespace(metadata={}),
    )
    private_config = SimpleNamespace(api_key="config-credential-secret")
    private_test_case = SimpleNamespace(
        test_code="assert answer_key_secret",
        user_context="private-user-context-secret",
        init={"tenant": "private-init-secret"},
        world_state={"balance": "private-world-secret"},
    )
    received: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def reset(self, *_: Any, **__: Any) -> Any:
            return reset_result

    def policy(
        env: Any,
        visible_reset_result: Any,
        repetition_id: str,
        **kwargs: Any,
    ) -> Any:
        received.update(
            {
                "env": env,
                "reset_result": visible_reset_result,
                "repetition_id": repetition_id,
                "kwargs": kwargs,
            }
        )
        return policy_result

    monkeypatch.setattr(thinkingbox_eval, "ThinkingBoxEnv", FakeClient)
    args = SimpleNamespace(
        base_url="http://env.invalid",
        message_timeout=DEFAULT_MESSAGE_TIMEOUT_S,
        agent="think",
    )

    outcome = await thinkingbox_eval._run_episode(
        args,
        UID,
        "repeat-public",
        test_case=private_test_case,
        agent_session_factory=private_config,
        policy=policy,
        env_config="config-with-credentials.yaml",
        env_dataset=None,
    )

    assert outcome.error is None
    assert outcome.result is policy_result
    assert received["reset_result"] is reset_result
    assert received["repetition_id"] == "repeat-public"
    assert received["kwargs"] == {}
    assert private_config not in received.values()
    assert private_test_case not in received.values()
    assert all(
        secret not in repr(received)
        for secret in (
            "config-credential-secret",
            "answer_key_secret",
            "private-user-context-secret",
            "private-init-secret",
            "private-world-secret",
        )
    )


@pytest.mark.asyncio
async def test_custom_policy_supports_async_callbacks() -> None:
    env = object()
    reset_result = object()
    policy_result = object()

    async def policy(
        received_env: Any,
        received_reset_result: Any,
        repetition_id: str,
    ) -> Any:
        assert received_env is env
        assert received_reset_result is reset_result
        assert repetition_id == "repeat-async"
        return policy_result

    assert (
        await thinkingbox_eval._invoke_policy(
            policy,
            env,
            reset_result,
            "repeat-async",
        )
        is policy_result
    )


@pytest.mark.asyncio
async def test_custom_policy_none_result_remains_an_episode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def reset(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(metadata={}),
            )

    monkeypatch.setattr(thinkingbox_eval, "ThinkingBoxEnv", FakeClient)
    args = SimpleNamespace(
        base_url="http://env.invalid",
        message_timeout=DEFAULT_MESSAGE_TIMEOUT_S,
        agent="think",
    )

    outcome = await thinkingbox_eval._run_episode(
        args,
        UID,
        "repeat-none",
        test_case=object(),
        agent_session_factory=object(),
        policy=lambda *_: None,
        env_config="config.yaml",
        env_dataset=None,
    )

    assert outcome.result is None
    assert isinstance(outcome.error, RuntimeError)
    assert str(outcome.error) == "configured policy returned no episode result"


@pytest.mark.asyncio
async def test_native_agent_proxy_submits_one_provider_parallel_batch() -> None:
    batches: list[list[SubmittedToolCall]] = []

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            batches.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=call.call_id,
                            content=f"result:{call.call_id}",
                        )
                        for call in calls
                    ]
                ),
            )

    calls = [
        ToolCall(
            name="lookup",
            arguments={"key": "one"},
            id="provider-call-1",
        ),
        ToolCall(
            name="lookup",
            arguments={"key": "two"},
            id="provider-call-2",
        ),
    ]
    agent = SimpleNamespace(
        conversation=Conversation(messages=[ParallelToolCall(tool_calls=calls)])
    )
    proxy = thinkingbox_eval._OpenEnvProxy(FakeEnv(), [])
    proxy.bind(agent)

    assert await proxy.call_tool("lookup", key="one") == "result:provider-call-1"
    assert await proxy.call_tool("lookup", key="two") == "result:provider-call-2"
    assert [[call.call_id for call in batch] for batch in batches] == [
        ["provider-call-1", "provider-call-2"]
    ]


@pytest.mark.asyncio
async def test_native_agent_proxy_preserves_duplicate_parallel_call_ids() -> None:
    batches: list[list[SubmittedToolCall]] = []

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            batches.append(calls)
            await asyncio.sleep(0)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=call.call_id,
                            content=f"result:{call.call_id}",
                        )
                        for call in calls
                    ]
                ),
            )

    calls = [
        ToolCall(
            name="lookup",
            arguments={"key": "same"},
            id="duplicate-call-1",
        ),
        ToolCall(
            name="lookup",
            arguments={"key": "same"},
            id="duplicate-call-2",
        ),
    ]
    agent = SimpleNamespace(
        conversation=Conversation(messages=[ParallelToolCall(tool_calls=calls)])
    )
    proxy = thinkingbox_eval._OpenEnvProxy(FakeEnv(), [])
    proxy.bind(agent)

    results = await asyncio.gather(
        proxy.call_tool("lookup", key="same"),
        proxy.call_tool("lookup", key="same"),
    )

    assert results == ["result:duplicate-call-1", "result:duplicate-call-2"]
    assert [[call.call_id for call in batch] for batch in batches] == [
        ["duplicate-call-1", "duplicate-call-2"]
    ]


@pytest.mark.asyncio
async def test_native_agent_proxy_forwards_provider_parse_errors() -> None:
    batches: list[list[SubmittedToolCall]] = []

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            batches.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(call_id=call.call_id, content="ok")
                        for call in calls
                    ]
                ),
            )

    calls = [
        ToolCall(
            name="lookup",
            arguments={},
            id="malformed",
            metadata={"error": "Error: invalid JSON arguments"},
        ),
        ToolCall(
            name="lookup",
            arguments={"key": "valid"},
            id="valid",
        ),
    ]
    agent = SimpleNamespace(
        conversation=Conversation(messages=[ParallelToolCall(tool_calls=calls)])
    )
    proxy = thinkingbox_eval._OpenEnvProxy(FakeEnv(), [])
    proxy.bind(agent)

    assert await proxy.call_tool("lookup") == "ok"
    assert await proxy.call_tool("lookup", key="valid") == "ok"
    assert len(batches) == 1
    assert batches[0][0].parse_error == "Error: invalid JSON arguments"
    assert batches[0][1].parse_error is None


@pytest.mark.asyncio
async def test_native_agent_proxy_allows_tool_argument_named_name() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    class FakeEnv:
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            call_id: str | None = None,
        ) -> Any:
            captured.append((tool_name, arguments))
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(tool_result="ok"),
            )

    proxy = thinkingbox_eval._OpenEnvProxy(FakeEnv(), [])

    assert await proxy.call_tool("find_driver", name="Taylor") == "ok"
    assert captured == [("find_driver", {"name": "Taylor"})]


@pytest.mark.parametrize(
    ("call", "response_content"),
    [
        (
            ToolCall(
                name="lookup",
                arguments={},
                id="malformed-only",
                metadata={"error": "Error: invalid JSON arguments"},
            ),
            "Error: invalid JSON arguments",
        ),
        (
            ToolCall(name="unknown", arguments={}, id="unknown-only"),
            "Error: function 'unknown' does not exist",
        ),
    ],
    ids=["malformed", "unknown"],
)
@pytest.mark.asyncio
async def test_configured_agent_submits_locally_handled_batch_and_continues(
    call: ToolCall,
    response_content: str,
) -> None:
    events: list[str] = []
    batches: list[list[SubmittedToolCall]] = []
    trace = thinkingbox_eval._ExecutionTrace()

    class FakeAgent:
        def __init__(self) -> None:
            self.conversation = Conversation()

        def add_messages(self, added: list[Any]) -> None:
            self.conversation.messages.extend(added)

        async def decode_turn_iter(self, user_message: Text | None) -> Any:
            if user_message is not None:
                self.add_messages([user_message])
            batch = ParallelToolCall(tool_calls=[call])
            self.add_messages([batch])
            yield batch
            response = ToolResponse(
                name=call.name,
                content=response_content,
                id=call.id,
            )
            self.add_messages([response])
            yield response
            answer = Text(role="assistant", content="Recovered and continued")
            self.add_messages([answer])
            yield answer

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            events.append("call_tools")
            batches.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=child.call_id,
                            content=response_content,
                        )
                        for child in calls
                    ],
                    steps_taken=1,
                    metadata={},
                ),
            )

        async def submit_message(self, content: str) -> Any:
            events.append("submit_message")
            assert content == "Recovered and continued"
            return SimpleNamespace(
                done=True,
                observation=SimpleNamespace(
                    error=None,
                    user_message=None,
                    steps_taken=2,
                    metadata={},
                ),
            )

        async def finish(self, reason: str) -> Any:
            raise AssertionError(f"unexpected finish: {reason}")

    reset_result = SimpleNamespace(
        observation=SimpleNamespace(
            task="task",
            bot_instructions=None,
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description="Lookup",
                    input_schema={"type": "object"},
                )
            ],
        )
    )
    result = await thinkingbox_eval.run_configured_agent(
        FakeEnv(),
        reset_result,
        _case(),
        lambda **_: FakeAgent(),
        trace,
    )

    assert result.done is True
    assert events == ["call_tools", "submit_message"]
    assert len(batches) == 1
    assert [(child.name, child.call_id, child.parse_error) for child in batches[0]] == [
        (call.name, call.id, call.metadata.get("error"))
    ]
    assert trace.steps_taken == 2
    assert [message.T for message in trace.messages] == [
        "Text",
        "ParallelToolCall",
        "ToolResponse",
        "Text",
    ]


@pytest.mark.asyncio
async def test_configured_agent_submits_mixed_batch_exactly_once() -> None:
    batches: list[list[SubmittedToolCall]] = []
    returned: list[str] = []

    class FakeAgent:
        def __init__(self, proxy: Any) -> None:
            self.proxy = proxy
            self.conversation = Conversation()

        def add_messages(self, added: list[Any]) -> None:
            self.conversation.messages.extend(added)

        async def decode_turn_iter(self, user_message: Text | None) -> Any:
            if user_message is not None:
                self.add_messages([user_message])
            malformed = ToolCall(
                name="lookup",
                arguments={},
                id="mixed-malformed",
                metadata={"error": "Error: invalid JSON arguments"},
            )
            valid = ToolCall(
                name="lookup",
                arguments={},
                id="mixed-valid",
            )
            batch = ParallelToolCall(tool_calls=[malformed, valid])
            self.add_messages([batch])
            yield batch
            valid_result = await self.proxy.call_tool("lookup")
            returned.append(valid_result)
            responses = [
                ToolResponse(
                    name=malformed.name,
                    content=malformed.metadata["error"],
                    id=malformed.id,
                ),
                ToolResponse(
                    name=valid.name,
                    content=valid_result,
                    id=valid.id,
                ),
            ]
            for response in responses:
                self.add_messages([response])
                yield response
            answer = Text(role="assistant", content="Batch handled")
            self.add_messages([answer])
            yield answer

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            batches.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=call.call_id,
                            content=(call.parse_error or f"result:{call.call_id}"),
                        )
                        for call in calls
                    ]
                ),
            )

        async def submit_message(self, content: str) -> Any:
            assert content == "Batch handled"
            return SimpleNamespace(
                done=True,
                observation=SimpleNamespace(error=None, user_message=None),
            )

        async def finish(self, reason: str) -> Any:
            raise AssertionError(f"unexpected finish: {reason}")

    reset_result = SimpleNamespace(
        observation=SimpleNamespace(
            task="task",
            bot_instructions=None,
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description="Lookup",
                    input_schema={"type": "object"},
                )
            ],
        )
    )

    result = await thinkingbox_eval.run_configured_agent(
        FakeEnv(),
        reset_result,
        _case(),
        lambda mcp_proxy, **_: FakeAgent(mcp_proxy),
    )

    assert result.done is True
    assert len(batches) == 1
    assert [(call.call_id, call.parse_error) for call in batches[0]] == [
        ("mixed-malformed", "Error: invalid JSON arguments"),
        ("mixed-valid", None),
    ]
    assert returned == ["result:mixed-valid"]


@pytest.mark.asyncio
async def test_configured_agent_reports_exact_tool_only_limit() -> None:
    batches: list[list[SubmittedToolCall]] = []
    finish_reasons: list[str] = []

    class FakeAgent:
        def __init__(self) -> None:
            self.conversation = Conversation()

        def add_messages(self, added: list[Any]) -> None:
            self.conversation.messages.extend(added)

        async def decode_turn_iter(self, user_message: Text | None) -> Any:
            if user_message is not None:
                self.add_messages([user_message])
            for index in range(2):
                batch = ParallelToolCall(
                    tool_calls=[
                        ToolCall(
                            name="lookup",
                            arguments={"index": index},
                            id=f"tool-only-{index}",
                        )
                    ]
                )
                self.add_messages([batch])
                yield batch

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            batches.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=call.call_id,
                            content=f"result:{call.call_id}",
                        )
                        for call in calls
                    ]
                ),
            )

        async def submit_message(self, content: str) -> Any:
            raise AssertionError(f"unexpected assistant message: {content}")

        async def finish(self, reason: str) -> Any:
            finish_reasons.append(reason)
            return SimpleNamespace(
                done=True,
                observation=SimpleNamespace(
                    finish_reason=reason,
                    metadata={},
                ),
            )

    case = _case()
    case.max_agent_sim_turns = 2
    reset_result = SimpleNamespace(
        observation=SimpleNamespace(
            task="task",
            bot_instructions=None,
            tools=[],
        )
    )

    result = await thinkingbox_eval.run_configured_agent(
        FakeEnv(),
        reset_result,
        case,
        lambda **_: FakeAgent(),
    )

    assert result.done is True
    assert finish_reasons == ["agent_limit"]
    assert [[call.call_id for call in batch] for batch in batches] == [
        ["tool-only-0"],
        ["tool-only-1"],
    ]


@pytest.mark.asyncio
async def test_configured_agent_counts_only_submitted_visible_text() -> None:
    submitted: list[str] = []

    class FakeAgent:
        def __init__(self) -> None:
            self.conversation = Conversation()
            self.turn = 0

        def add_messages(self, added: list[Any]) -> None:
            self.conversation.messages.extend(added)

        async def decode_turn_iter(self, user_message: Text | None) -> Any:
            if user_message is not None:
                self.add_messages([user_message])
            self.turn += 1
            contents = (
                ["intermediate text", "first submitted text"]
                if self.turn == 1
                else ["second submitted text"]
            )
            for content in contents:
                message = Text(role="assistant", content=content)
                self.add_messages([message])
                yield message

    class FakeEnv:
        async def submit_message(self, content: str) -> Any:
            submitted.append(content)
            return SimpleNamespace(
                done=len(submitted) == 2,
                observation=SimpleNamespace(
                    error=None,
                    user_message="continue" if len(submitted) == 1 else None,
                    metadata={},
                ),
            )

        async def finish(self, reason: str) -> Any:
            raise AssertionError(f"unexpected finish: {reason}")

    case = _case()
    case.max_agent_sim_turns = 2
    reset_result = SimpleNamespace(
        observation=SimpleNamespace(
            task="task",
            bot_instructions=None,
            tools=[],
        )
    )

    result = await thinkingbox_eval.run_configured_agent(
        FakeEnv(),
        reset_result,
        case,
        lambda **_: FakeAgent(),
    )

    assert result.done is True
    assert submitted == ["first submitted text", "second submitted text"]


async def _run_fake_agent_messages(
    messages: list[Any],
    *,
    batch_submissions: list[list[SubmittedToolCall]] | None = None,
) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []

    class FakeAgent:
        def __init__(self) -> None:
            self.conversation = Conversation()

        def add_messages(self, added: list[Any]) -> None:
            self.conversation.messages.extend(added)

        async def decode_turn_iter(self, user_message: Text | None) -> Any:
            if user_message is not None:
                self.conversation.messages.append(user_message)
            for message in messages:
                self.conversation.messages.append(message)
                yield message

    class FakeEnv:
        async def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
            if batch_submissions is not None:
                batch_submissions.append(calls)
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(
                    tool_results=[
                        SimpleNamespace(
                            call_id=call.call_id,
                            content=f"result:{call.call_id}",
                        )
                        for call in calls
                    ]
                ),
            )

        async def submit_message(
            self,
            content: str | None,
            *,
            terminal_tool_calls: list[SubmittedToolCall] | None = None,
            tool_calls_before_content: bool = False,
        ) -> Any:
            submissions.append(
                {
                    "content": content,
                    "terminal_tool_calls": terminal_tool_calls or [],
                    "tool_calls_before_content": tool_calls_before_content,
                }
            )
            return SimpleNamespace(
                done=True,
                observation=SimpleNamespace(
                    error=None,
                    user_message=None,
                ),
            )

        async def finish(self, reason: str) -> Any:
            raise AssertionError(f"unexpected finish: {reason}")

    reset_result = SimpleNamespace(
        observation=SimpleNamespace(
            task="task",
            bot_instructions=None,
            tools=[],
        )
    )
    await thinkingbox_eval.run_configured_agent(
        FakeEnv(),
        reset_result,
        _case(),
        lambda **_: FakeAgent(),
    )
    return submissions


@pytest.mark.asyncio
async def test_ordered_tail_ignores_earlier_end_turn_before_later_text() -> None:
    end_call = ToolCall(
        name="InjectionAttackInToolResponse",
        arguments={},
        id="early-end",
        metadata={"is_end_turn_tool": True},
    )
    early_terminal = ParallelToolCall(
        tool_calls=[end_call],
        metadata={"is_end_turn_tool": True},
    )

    batches: list[list[SubmittedToolCall]] = []
    submissions = await _run_fake_agent_messages(
        [early_terminal, Text(role="assistant", content="Please clarify")],
        batch_submissions=batches,
    )

    assert batches == []
    assert submissions == [
        {
            "content": "Please clarify",
            "terminal_tool_calls": [],
            "tool_calls_before_content": False,
        }
    ]


@pytest.mark.asyncio
async def test_ordered_tail_submits_only_actual_terminal_batch_and_adjacent_text() -> (
    None
):
    earlier = ParallelToolCall(
        tool_calls=[
            ToolCall(
                name="lookup",
                arguments={"key": "earlier"},
                id="earlier-batch",
            )
        ]
    )
    terminal_call = ToolCall(
        name="InjectionAttackInToolResponse",
        arguments={"reason": "detected"},
        id="actual-terminal",
        metadata={"is_end_turn_tool": True},
    )
    terminal = ParallelToolCall(
        tool_calls=[terminal_call],
        metadata={"is_end_turn_tool": True},
    )

    batches: list[list[SubmittedToolCall]] = []
    submissions = await _run_fake_agent_messages(
        [
            earlier,
            Text(role="assistant", content="Adjacent explanation"),
            terminal,
        ],
        batch_submissions=batches,
    )

    assert [[call.call_id for call in batch] for batch in batches] == [
        ["earlier-batch"]
    ]
    assert len(submissions) == 1
    assert submissions[0]["content"] == "Adjacent explanation"
    assert submissions[0]["tool_calls_before_content"] is False
    assert [call.call_id for call in submissions[0]["terminal_tool_calls"]] == [
        "actual-terminal"
    ]
