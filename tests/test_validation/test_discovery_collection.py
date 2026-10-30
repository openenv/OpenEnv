"""Opt-in discovery uses the measured socket and never retains its credentials."""

import json
from dataclasses import asdict

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan
from test_runtime_transport import EpisodeConnection

TOKEN = "test-validation-token-" * 2
CAPABILITY = "test-session-capability-" * 2


class DiscoveryConnection(EpisodeConnection):
    def __init__(self, response=None):
        super().__init__()
        self.requests = []
        self.response = response

    def send(self, payload):
        super().send(payload)
        self.requests.append(json.loads(payload))

    def recv(self, timeout):
        if self.operation == "validation_open":
            return json.dumps(
                {
                    "type": "validation_open",
                    "data": {"schema_version": 1, "capability": CAPABILITY},
                }
            )
        if self.operation == "validation_read":
            return json.dumps(
                {"type": "validation", "data": {"schema_version": 1, "rubric": []}}
            )
        if self.operation == "mcp":
            if isinstance(self.response, Exception):
                raise self.response
            if isinstance(self.response, str):
                return self.response
            return json.dumps(self.response or tool_response())
        return super().recv(timeout)


def tool_response(tools=None):
    return {
        "type": "mcp",
        "data": {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [] if tools is None else tools},
            "error": None,
        },
    }


def collect(monkeypatch, connection, *, task_fault=None, **kwargs):
    original_client = httpx.Client
    urls = []

    def respond(request):
        urls.append(request.url.path)
        if request.url.path == "/schema":
            return httpx.Response(200, json={"observation": {"type": "object"}})
        if task_fault == "unsupported":
            return httpx.Response(501, text="private server details")
        if request.url.path == "/list_environments":
            return httpx.Response(200, json=["probe"])
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json=[{"name": "train"}])
        if request.url.path.endswith("/num_tasks"):
            return httpx.Response(200, json={"num_tasks": 100})
        assert request.url.path.endswith("/task")
        return httpx.Response(200, json={"task": {"id": task_fault or "public-task"}})

    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **options: original_client(
            transport=httpx.MockTransport(respond), **options
        ),
    )
    monkeypatch.setattr(collector, "connect", lambda *args, **options: connection)
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "bounded", "seed": 7},
            "actions": [{"increment": 1}],
        }
    )
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000",
        plan,
        episode_timeout_s=2,
        validation_token=TOKEN,
        **kwargs,
    )
    return evidence, urls


def test_default_collection_does_not_discover_tools_or_tasks(monkeypatch):
    connection = DiscoveryConnection()
    evidence, urls = collect(monkeypatch, connection)
    assert "mcp" not in [request["type"] for request in connection.requests]
    assert urls == ["/schema"]
    assert (
        evidence.tools_json
        is evidence.tools_error
        is evidence.tasks_json
        is evidence.tasks_error
        is None
    )


def test_discovery_stays_on_measured_socket_outside_trajectory(monkeypatch):
    connection = DiscoveryConnection(
        tool_response([{"name": "echo", "inputSchema": {"type": "object"}}])
    )
    evidence, urls = collect(
        monkeypatch, connection, collect_tools=True, collect_tasks=True
    )
    assert [request["type"] for request in connection.requests] == [
        "validation_open",
        "mcp",
        "reset",
        "state",
        "step",
        "state",
        "validation_read",
        "close",
    ]
    assert connection.requests[1] == {
        "type": "mcp",
        "data": {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    }
    assert [row.operation for row in evidence.exchanges] == [
        "reset",
        "state",
        "step",
        "state",
    ]
    assert (
        evidence.failure_reason is evidence.tools_error is evidence.tasks_error is None
    )
    assert json.loads(evidence.tools_json) == {
        "tools": [{"name": "echo", "inputSchema": {"type": "object"}}]
    }
    assert json.loads(evidence.tasks_json)["counts"] == {"train": 100}
    assert urls == [
        "/schema",
        "/list_environments",
        "/probe/splits",
        "/probe/num_tasks",
        "/probe/task",
        "/probe/task",
    ]
    assert TOKEN not in json.dumps(asdict(evidence)) and CAPABILITY not in json.dumps(
        asdict(evidence)
    )


@pytest.mark.parametrize(
    "fault",
    [
        "unsupported",
        "wrong_id",
        "boolean_id",
        "missing_tools",
        "bad_json",
        "oversize",
        "transport",
    ],
)
def test_failed_tool_discovery_is_not_empty_success_and_preserves_episode(
    monkeypatch, fault
):
    response = tool_response()
    if fault == "unsupported":
        response["data"].update(
            result=None, error={"code": -32601, "message": "private error"}
        )
    elif fault == "wrong_id":
        response["data"]["id"] = 2
    elif fault == "boolean_id":
        response["data"]["id"] = True
    elif fault == "missing_tools":
        response["data"]["result"] = {}
    elif fault == "bad_json":
        response = "{private invalid json"
    elif fault == "oversize":
        response = "x" * (collector.MAX_MESSAGE_BYTES + 1)
    else:
        response = ValueError("private transport error")
    evidence, _ = collect(
        monkeypatch, DiscoveryConnection(response), collect_tools=True
    )
    assert evidence.tools_json is None
    if fault == "unsupported":
        assert evidence.tools_error is None
        assert evidence.tools_unavailable == "unsupported capability: tool_discovery"
    else:
        assert evidence.tools_error
        assert evidence.tools_unavailable is None
    assert "private" not in repr(evidence)
    assert evidence.failure_reason is None
    assert len(evidence.exchanges) == 4


@pytest.mark.parametrize("secret", [TOKEN, CAPABILITY])
@pytest.mark.parametrize("escaped", [False, True])
def test_discovery_credential_echo_is_discarded_even_with_unicode_escapes(
    monkeypatch, secret, escaped
):
    response = json.dumps(tool_response([{"name": "echo", "description": secret}]))
    if escaped:
        response = response.replace(
            secret, "".join(f"\\u{ord(char):04x}" for char in secret)
        )
    evidence, _ = collect(
        monkeypatch,
        DiscoveryConnection(response),
        collect_tools=True,
        collect_tasks=True,
        task_fault=secret,
    )
    assert evidence.tools_json is evidence.tasks_json is None
    assert evidence.tools_error and evidence.tasks_error
    assert evidence.failure_reason is None
    assert secret not in json.dumps(asdict(evidence))


def test_task_failure_does_not_poison_successful_tools_or_episode(monkeypatch):
    evidence, _ = collect(
        monkeypatch,
        DiscoveryConnection(),
        collect_tools=True,
        collect_tasks=True,
        task_fault="unsupported",
    )
    assert json.loads(evidence.tools_json) == {"tools": []}
    assert evidence.tools_error is None
    assert evidence.tasks_json is None and evidence.tasks_error
    assert evidence.failure_reason is None and len(evidence.exchanges) == 4


def test_tool_pagination_remains_visible_to_grader(monkeypatch):
    response = tool_response([{"name": "echo"}])
    response["data"]["result"]["nextCursor"] = "next-page"
    evidence, _ = collect(
        monkeypatch, DiscoveryConnection(response), collect_tools=True
    )
    assert json.loads(evidence.tools_json)["nextCursor"] == "next-page"
    assert evidence.tools_error is evidence.failure_reason is None
    assert len(evidence.exchanges) == 4


def test_socket_failure_keeps_discovery_error_and_fails_replay(monkeypatch):
    connection = DiscoveryConnection(ConnectionError("private socket error"))
    send = connection.send

    def broken_send(payload):
        if json.loads(payload)["type"] == "reset":
            raise ConnectionError("closed socket")
        send(payload)

    connection.send = broken_send
    evidence, _ = collect(monkeypatch, connection, collect_tools=True)
    assert evidence.tools_error == "tool discovery failed (ConnectionError)"
    assert evidence.failure_phase == "reset"
    assert evidence.failure_reason == "reset failed (ConnectionError)"
    assert evidence.exchanges == ()


@pytest.mark.parametrize(
    "reply,status",
    [
        ({"type": "error", "data": {"code": "UNKNOWN_TYPE"}}, "skip"),
        (
            {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32601, "message": "unsupported"},
                },
            },
            "skip",
        ),
        (
            {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {
                        "code": -32603,
                        "message": "Environment does not support MCP",
                        "data": None,
                    },
                },
            },
            "skip",
        ),
        (
            {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "error": {"code": -32601, "message": "unsupported"},
                },
            },
            "fail",
        ),
        ("malformed", "fail"),
        (
            {
                "type": "mcp",
                "data": {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            },
            "pass",
        ),
    ],
)
def test_tool_discovery_distinguishes_unsupported_from_failures(
    monkeypatch, tmp_path, reply, status
):
    from dataclasses import replace

    from openenv.validation.graders.runtime.discovery import (
        ToolDeclarationAccuracyGrader,
    )
    from test_runtime_discovery import subject

    evidence, _ = collect(monkeypatch, DiscoveryConnection(reply), collect_tools=True)
    measured = subject(tmp_path)
    measured.manifest.capabilities.declared_tools = []
    result = ToolDeclarationAccuracyGrader().run(
        replace(measured, runtime_evidence=evidence)
    )
    assert result.status.value == status
    assert evidence.failure_reason is None
    if status == "skip":
        assert evidence.tools_error is None
        assert result.evidence == ["unsupported capability: tool_discovery"]


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "internal",
        "near_message",
        "auth",
        "wrong_code",
        "string_code",
        "wrong_id",
        "boolean_id",
        "version",
        "envelope",
        "result",
        "error_data",
    ],
)
def test_legacy_non_mcp_reply_is_an_exact_exception(monkeypatch, tmp_path, mutation):
    from dataclasses import replace

    from openenv.core.env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse
    from openenv.validation.graders.runtime.discovery import (
        ToolDeclarationAccuracyGrader,
    )
    from test_runtime_discovery import subject

    # Non-MCP tools/list in 7ab2ceaa^ uses this exact response constructor.
    rpc = JsonRpcResponse.error_response(
        JsonRpcErrorCode.INTERNAL_ERROR,
        "Environment does not support MCP",
        request_id=1,
    ).model_dump()
    reply = {"type": "mcp", "data": rpc}
    if mutation == "internal":
        rpc["error"]["message"] = "Internal error"
    elif mutation == "near_message":
        rpc["error"]["message"] += " temporarily"
    elif mutation == "auth":
        rpc["error"]["message"] = "Unauthorized"
    elif mutation == "wrong_code":
        rpc["error"]["code"] = -32000
    elif mutation == "string_code":
        rpc["error"]["code"] = "-32603"
    elif mutation == "wrong_id":
        rpc["id"] = 2
    elif mutation == "boolean_id":
        rpc["id"] = True
    elif mutation == "version":
        rpc["jsonrpc"] = "1.0"
    elif mutation == "envelope":
        reply["type"] = "error"
    elif mutation == "result":
        rpc["result"] = {"tools": []}
    elif mutation == "error_data":
        rpc["error"]["data"] = {"reason": "internal failure"}
    evidence, _ = collect(monkeypatch, DiscoveryConnection(reply), collect_tools=True)
    measured = subject(tmp_path)
    measured.manifest.capabilities.declared_tools = []
    result = ToolDeclarationAccuracyGrader().run(
        replace(measured, runtime_evidence=evidence)
    )
    assert result.status.value == ("skip" if mutation == "none" else "fail")
    assert evidence.tools_json is None
    assert evidence.failure_reason is None and len(evidence.exchanges) == 4
    if mutation == "none":
        assert evidence.tools_error is None
        assert result.evidence == ["unsupported capability: tool_discovery"]
    else:
        assert evidence.tools_error and evidence.tools_unavailable is None
