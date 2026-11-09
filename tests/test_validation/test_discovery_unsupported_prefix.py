"""A late JSON limit must retain independently measured discovery findings."""

import json
from dataclasses import asdict
from decimal import Decimal
from types import SimpleNamespace

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders.runtime.discovery import (
    TaskDeclarationAccuracyGrader,
    ToolDeclarationAccuracyGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.collector import (
    RuntimeCollectionInterrupted,
    RuntimeCollectionUnsupported,
)
from openenv.validation.runtime.json_values import loads
from openenv.validation.types import CheckStatus
from test_discovery_collection import collect, DiscoveryConnection, TOKEN, tool_response

DISCOVERY_FIELDS = (
    "tools_json",
    "tools_error",
    "tools_unavailable",
    "tasks_json",
    "tasks_error",
)
SCENARIOS = (
    "valid",
    "tool_mismatch",
    "task_mismatch",
    "tool_error",
    "task_error",
    "unavailable",
    "credentials",
)


class PrefixConnection(DiscoveryConnection):
    def __init__(self, response, failure):
        super().__init__(response)
        self.failure = failure

    def recv(self, timeout):
        if self.operation == "validation_open":
            # This fixture only implements discovery and the episode protocol.
            return json.dumps({"type": "error", "data": {"code": "UNKNOWN_TYPE"}})
        if self.operation == "step":
            if self.failure == "depth":
                return (
                    '{"type":"observation","data":{"observation":'
                    + "[" * 20_000
                    + "0"
                    + "]" * 20_000
                    + "}}"
                )
            if self.failure == "numeric":
                # Exercise the arithmetic recovery branch after real discovery;
                # stdlib episode decoding itself accepts extreme exponents.
                raise ArithmeticError(TOKEN)
            if self.failure == "interrupt":
                raise KeyboardInterrupt(TOKEN)
            if self.failure == "error":
                raise ValueError(TOKEN)
        return super().recv(timeout)


def collect_prefix(monkeypatch, scenario, failure, **options):
    tools = [{"name": "echo", "inputSchema": {"minimum": "exact-number"}}]
    if scenario == "tool_mismatch":
        tools[0]["name"] = "undeclared"
    response = tool_response(tools)
    task_fault = None
    if scenario == "tool_error":
        response["data"]["result"] = {}
    elif scenario == "task_error":
        task_fault = "unsupported"
    elif scenario == "unavailable":
        response = {"type": "error", "data": {"code": "UNKNOWN_TYPE"}}
    elif scenario == "credentials":
        tools[0]["description"] = TOKEN
        task_fault = TOKEN
    raw = json.dumps(response).replace('"exact-number"', "1.00000000000000001")
    if scenario == "credentials":
        raw = raw.replace(TOKEN, "".join(f"\\u{ord(char):04x}" for char in TOKEN))
    connection = PrefixConnection(raw, failure)
    # Failed teardown must not replace the original failure or its prefix.
    connection.close_failure = failure is not None
    try:
        if failure in {"depth", "numeric", "interrupt"}:
            expected = (
                RuntimeCollectionInterrupted
                if failure == "interrupt"
                else RuntimeCollectionUnsupported
            )
            with pytest.raises(expected) as caught:
                collect(monkeypatch, connection, task_fault=task_fault, **options)
            evidence = caught.value.evidence
            if failure != "interrupt":
                capability = (
                    "json_depth" if failure == "depth" else "json_numeric_range"
                )
                assert str(caught.value) == (
                    f"{capability}: runtime JSON exceeds validator support"
                )
                assert evidence.failure_reason == (
                    f"step JSON exceeds validator support ({capability})"
                )
        else:
            evidence, _ = collect(
                monkeypatch, connection, task_fault=task_fault, **options
            )
    finally:
        assert connection.closed
    return evidence


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("failure", ["depth", "numeric", "interrupt", "error", None])
def test_late_failures_preserve_discovery_prefix(monkeypatch, scenario, failure):
    with monkeypatch.context() as case_patch:
        evidence = collect_prefix(
            case_patch, scenario, failure, collect_tools=True, collect_tasks=True
        )
    with monkeypatch.context() as control_patch:
        control = collect_prefix(
            control_patch, scenario, None, collect_tools=True, collect_tasks=True
        )
    assert {field: getattr(evidence, field) for field in DISCOVERY_FIELDS} == {
        field: getattr(control, field) for field in DISCOVERY_FIELDS
    }
    assert evidence.schema_response_json == control.schema_response_json
    assert evidence.observation_schema_json == control.observation_schema_json
    assert evidence.telemetry_unavailable == "unsupported capability: session_telemetry"
    assert evidence.telemetry_json is evidence.telemetry_error is None
    assert TOKEN not in json.dumps(asdict(evidence))
    assert evidence.failure_phase == (None if failure is None else "step")
    assert evidence.exchanges[:2] == control.exchanges[:2]
    # With authorization enabled, unparseable replies cannot be retained safely.
    assert len(evidence.exchanges) == (4 if failure is None else 2)
    if failure in {"interrupt", "error"}:
        name = "KeyboardInterrupt" if failure == "interrupt" else "ValueError"
        assert evidence.failure_reason == f"step failed ({name})"
    elif failure is None:
        assert evidence.failure_reason is None

    manifest = load_fixture_manifest("served_min_pass")
    # Only advertise capabilities actually exercised by this protocol fixture.
    manifest["capabilities"] = {
        "verifier": {"kind": "reward_channel", "entry": None},
        "declared_tools": ["echo"],
        "task_api": True,
        "declared_task_count": {"train": 99 if scenario == "task_mismatch" else 100},
    }
    subject = SimpleNamespace(
        manifest=NormalizedManifest.model_validate(manifest), runtime_evidence=evidence
    )
    tool = ToolDeclarationAccuracyGrader().run(subject)
    task = TaskDeclarationAccuracyGrader().run(subject)
    assert tool.status is (
        CheckStatus.FAIL
        if scenario in {"tool_mismatch", "tool_error", "credentials"}
        else CheckStatus.SKIP
        if scenario == "unavailable"
        else CheckStatus.PASS
    ), tool.evidence
    assert task.status is (
        CheckStatus.FAIL
        if scenario in {"task_mismatch", "task_error", "credentials"}
        else CheckStatus.PASS
    ), task.evidence
    if scenario == "unavailable":
        assert tool.evidence == ["unsupported capability: tool_discovery"]
    if scenario == "valid":
        minimum = loads(evidence.tools_json)["tools"][0]["inputSchema"]["minimum"]
        assert type(minimum) is Decimal
        assert minimum == Decimal("1.00000000000000001")


@pytest.mark.parametrize("failure", ["depth", "numeric"])
@pytest.mark.parametrize("tools,tasks", [(False, False), (True, False), (False, True)])
def test_late_json_limit_does_not_invent_omitted_discovery(
    monkeypatch, failure, tools, tasks
):
    evidence = collect_prefix(
        monkeypatch, "valid", failure, collect_tools=tools, collect_tasks=tasks
    )
    assert (evidence.tools_json is not None) is tools
    assert (evidence.tasks_json is not None) is tasks
    assert (
        evidence.tools_error
        is evidence.tools_unavailable
        is evidence.tasks_error
        is None
    )
