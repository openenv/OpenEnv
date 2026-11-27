"""Run-scoped credentials must not survive primary, isolation or replay collection."""

import hashlib
import json
import shutil
from decimal import Decimal

import pytest
import yaml
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from openenv.core.env_server.types import WSErrorCode
from openenv.validation import runner
from openenv.validation.runtime import collector, replay
from openenv.validation.runtime.json_values import loads
from openenv.validation.types import CheckStatus, Level
from support.runtime import FakeRuntimeProvider
from test_late_collection_protocol import served
from test_runtime_execution import FIXTURE


PUBLIC_SCHEMA = '{ "observation": {"type":"object","minimum":1.0000000000000001}, "state": {"description":"public α"} }'


def protocol_app(provider, attack_call, surface):
    app = FastAPI()
    schemas, requests = [], []

    def control_token():
        # Read the actual runner-minted launch credential, never a separate oracle.
        return provider.launches[0].env_vars["OPENENV_VALIDATION_TOKEN"]

    @app.get("/schema")
    def schema():
        call = len(schemas) + 1
        raw = PUBLIC_SCHEMA
        if call == attack_call and surface in {"schema", "raw_schema"}:
            suffix = (
                ',"note":'
                + json.dumps(control_token())
                + (',"note":"public"' if surface == "raw_schema" else "")
            )
            raw = raw[:-1] + suffix + "}"
        schemas.append(raw)
        return Response(raw, media_type="application/json")

    @app.get("/oracle/solve.sh")
    def oracle():
        return Response("denied", status_code=403)

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        call = len(schemas)
        steps = 0
        await connection.accept()
        try:
            while (request := await connection.receive_json())["type"] != "close":
                operation = request["type"]
                requests.append((call, operation))
                if operation not in {"reset", "step", "state"}:
                    if operation == "validation_open":
                        assert request["data"]["token"] == control_token()
                    await connection.send_json(
                        {
                            "type": "error",
                            "data": {
                                "code": WSErrorCode.UNKNOWN_TYPE,
                                "message": f"Unknown message type: {operation}",
                            },
                        }
                    )
                    continue
                if operation == "reset":
                    steps = 0
                elif operation == "step":
                    steps += 1
                if operation == "state":
                    response = {
                        "type": "state",
                        "data": {"episode_id": "validation-probe", "step_count": steps},
                    }
                else:
                    observation = {"counter": steps}
                    if call == attack_call and surface == "wire":
                        observation["message"] = control_token()
                    response = {
                        "type": "observation",
                        "data": {
                            "observation": observation,
                            "reward": 0,
                            "done": steps == 2,
                        },
                    }
                await connection.send_json(response)
        except WebSocketDisconnect:
            # An intentionally rejected credential response closes this session.
            pass

    return app, schemas, requests


@pytest.mark.parametrize(
    "attack_call,surface",
    [(None, None)]
    + [
        (call, surface)
        for call in (1, 2, 3)
        for surface in ("schema", "raw_schema", "wire")
    ],
    ids=["public-control"]
    + [
        f"{phase}-{surface}"
        for phase in ("primary", "sequential", "fresh")
        for surface in ("schema", "raw_schema", "wire")
    ],
)
def test_control_token_omission_across_real_collections(
    tmp_path, monkeypatch, attack_call, surface
):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("__pycache__"))
    path = root / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    # This server declares only the episode protocol and oracle HTTP probe it serves.
    for field in ("declared_tools", "rubric_tree", "task_api", "declared_task_count"):
        document["validation"]["capabilities"].pop(field, None)
    path.write_text(yaml.safe_dump(document))
    provider = FakeRuntimeProvider()
    app, schemas, requests = protocol_app(provider, attack_call, surface)
    collected, connections = [], []
    real_collect, real_connect = collector.collect_runtime_evidence, collector.connect

    def collect(*args, **kwargs):
        evidence = real_collect(*args, **kwargs)
        collected.append((kwargs, evidence))
        return evidence

    def connect(*args, **kwargs):
        connections.append(kwargs)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(runner, "collect_runtime_evidence", collect)
    monkeypatch.setattr(replay, "collect_runtime_evidence", collect)
    monkeypatch.setattr(collector, "connect", connect)
    bundle = tmp_path / "bundle"
    with served(app) as base_url:
        provider.subject.base_url = base_url
        report = runner.run_validation(
            root, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
        )

    marker = provider.launches[0].env_vars["OPENENV_VALIDATION_TOKEN"]
    api_contains_marker = any(marker in repr(evidence) for _, evidence in collected)
    bundle_marker_files = [
        p.name for p in bundle.iterdir() if marker.encode() in p.read_bytes()
    ]
    # Keep reproduction diagnostics separate from the credential-filtered bundle.
    (tmp_path / "privacy-observation.json").write_text(
        json.dumps(
            {
                "api_contains_marker": api_contains_marker,
                "bundle_marker_files": bundle_marker_files,
                "schema_raw_contains_marker": [marker in raw for raw in schemas],
                "schema_decoded_contains_marker": [
                    marker in json.dumps(json.loads(raw)) for raw in schemas
                ],
                "verdict": report.verdict.value,
                "launch_token_sha256": hashlib.sha256(marker.encode()).hexdigest(),
                "collector_token_sha256": [
                    hashlib.sha256(kwargs["validation_token"].encode()).hexdigest()
                    if kwargs.get("validation_token")
                    else None
                    for kwargs, _ in collected
                ],
                "websocket_max_sizes": [c["max_size"] for c in connections],
                "operations": requests,
            }
        )
    )
    assert not api_contains_marker
    assert not bundle_marker_files
    assert provider.subject.stopped
    assert len(provider.launches) == len(provider.builds) == 1
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is True

    expected_calls = 1 if attack_call == 1 else 5
    assert len(schemas) == len(collected) == expected_calls
    assert [kwargs.get("episodes", 1) for kwargs, _ in collected] == [1, 2, 1, 1, 1][
        :expected_calls
    ]
    for index, (kwargs, _) in enumerate(collected):
        assert kwargs["episode_timeout_s"] == 30.0
        assert kwargs.get("request_timeout_s") is None
        assert kwargs["validation_token"] == marker
        if index in (1, 2):
            assert kwargs["request_telemetry"] is False
        else:
            assert kwargs.get("request_telemetry", True) is True
    assert [call for call, operation in requests if operation == "validation_open"] == (
        ([] if surface != "wire" else [1]) if attack_call == 1 else [1, 4, 5]
    )
    assert not any(operation == "validation_read" for _, operation in requests)
    for connection in connections:
        assert 0 < connection["open_timeout"] <= 30.0
        assert connection["max_queue"] == 1
        assert connection["compression"] is None
    expected_sizes = [
        1024 * 1024 if i in (1, 2) else 8 * 1024 * 1024
        for i in range(expected_calls)
        if not (attack_call == i + 1 and surface != "wire")
    ]
    assert [c["max_size"] for c in connections] == expected_sizes

    results = {r.check_id: r for r in report.results}
    assert results["runtime.startup"].status is (
        CheckStatus.FAIL if attack_call == 1 else CheckStatus.PASS
    )
    assert report.verdict.value == ("fail" if attack_call == 1 else "warn")
    names = ["collector", "isolation-sequential", "isolation-fresh"]
    for call, (_, evidence) in enumerate(collected[:3], 1):
        metadata = loads((bundle / f"{names[call - 1]}-evidence.json").read_text())
        if call > 1:
            assert evidence.telemetry_json is None
            assert evidence.telemetry_error is None
            assert not (bundle / f"{names[call - 1]}-telemetry.json").exists()
        if call == attack_call:
            phase = "reset" if surface == "wire" else "schema"
            assert evidence.failure_phase == metadata["failure_phase"] == phase
            assert (
                evidence.failure_reason
                == metadata["failure_reason"]
                == f"{phase} failed (ValueError)"
            )
            assert metadata["complete"] is False
            assert evidence.exchanges == ()
            if surface != "wire":
                assert evidence.schema_response_json is None
                assert metadata["schema_response_available"] is False
        else:
            assert evidence.failure_reason is None
            assert metadata["complete"] is True
            assert len(evidence.exchanges) == (12 if call == 2 else 6)
        if not (call == attack_call and surface != "wire"):
            assert evidence.schema_response_json == PUBLIC_SCHEMA
            assert metadata["schema_response_available"] is True
            assert metadata["schema_response_parse_failed"] is False
            assert metadata["schema_response"]["state"] == {"description": "public α"}
            assert metadata["schema_response"]["observation"]["minimum"] == Decimal(
                "1.0000000000000001"
            )
            assert metadata["observation_schema"]["minimum"] == Decimal(
                "1.0000000000000001"
            )
        assert metadata["omitted_trace_fields"] == []
        assert metadata["redacted"] is False
    if attack_call == 1:
        assert not (bundle / "replays.json").exists()
        samples = []
    else:
        samples = json.loads((bundle / "replays.json").read_text())["samples"]
        assert [sample["scope"] for sample in samples] == ["session", "seed"]
    for (_, evidence), sample in zip(collected[3:], samples, strict=True):
        assert evidence.failure_reason is sample["failure_reason"] is None
        assert len(evidence.exchanges) == len(sample["trace"]) == 6
        assert sample["omitted_trace_fields"] == []
        assert evidence.schema_response_json == PUBLIC_SCHEMA
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
