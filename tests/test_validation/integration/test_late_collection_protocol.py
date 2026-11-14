"""Late JSON limits must not discard completed evidence or skip oracle HTTP probes."""

import json
import shutil
import socket
import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn
import yaml
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from openenv.core.env_server.types import WSErrorCode
from openenv.validation.runner import run_validation
from openenv.validation.types import CheckStatus, Level
from support.runtime import FakeRuntimeProvider
from test_runtime_execution import FIXTURE


@contextmanager
def served(app):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


def protocol_app(root, unsupported_call, limit, fault):
    app = FastAPI()
    schema_calls = 0
    http_probes = []
    oracle = (root / "oracle/solve.sh").read_text()

    @app.get("/schema")
    def schema():
        nonlocal schema_calls
        schema_calls += 1
        if schema_calls == unsupported_call and limit != "ws_depth":
            raw = (
                "[" * 20000 + "0" + "]" * 20000
                if limit == "schema_depth"
                else "1e9999999999999999999"
            )
            return Response(
                '{"observation":' + raw + "}", media_type="application/json"
            )
        return {"observation": {"type": "object"}}

    @app.get("/oracle/solve.sh")
    def oracle_file():
        http_probes.append("/oracle/solve.sh")
        # Status is not evidence of containment: denied bodies can still leak.
        return Response(oracle if fault == "oracle_body" else "denied", status_code=403)

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        call = schema_calls
        resets = steps = 0
        await connection.accept()
        try:
            while (request := await connection.receive_json())["type"] != "close":
                operation = request["type"]
                if operation not in {"reset", "step", "state"}:
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
                    resets += 1
                    steps = 0
                elif operation == "step":
                    steps += 1
                if operation == "state":
                    response = {
                        "type": "state",
                        "data": {"episode_id": "validation-probe", "step_count": steps},
                    }
                else:
                    data = {
                        "observation": {"counter": steps},
                        "reward": 0,
                        "done": steps == 2,
                    }
                    if call == 1 and fault == "primary_reward":
                        data["reward"] = 2
                    if call == 1 and fault == "primary_oracle":
                        data["observation"]["private"] = oracle
                    # A supported oracle leak preceding the later unsupported response.
                    if call == unsupported_call and fault == "prefix_oracle":
                        data["observation"]["private"] = oracle
                    response = {"type": "observation", "data": data}
                raw = json.dumps(response)
                if (
                    limit == "ws_depth"
                    and call == unsupported_call
                    and operation == "step"
                    and (unsupported_call == 3 or resets == 2)
                ):
                    raw = raw.replace(
                        '"counter": ' + str(steps),
                        '"counter": ' + "[" * 20000 + "0" + "]" * 20000,
                    )
                await connection.send_text(raw)
        except WebSocketDisconnect:
            # The collector closes sessions after complete or unsupported evidence.
            pass

    return app, http_probes


@pytest.mark.parametrize("unsupported_call", [2, 3], ids=["sequential", "fresh"])
@pytest.mark.parametrize("limit", ["schema_depth", "schema_numeric", "ws_depth"])
@pytest.mark.parametrize(
    "fault", [None, "primary_reward", "primary_oracle", "oracle_body", "prefix_oracle"]
)
def test_late_collection_limits_grade_completed_evidence(
    tmp_path, unsupported_call, limit, fault
):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("__pycache__"))
    # This small server implements the episode protocol and oracle HTTP probe.
    path = root / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    capabilities = document["validation"]["capabilities"]
    for field in ("declared_tools", "rubric_tree", "task_api", "declared_task_count"):
        capabilities.pop(field, None)
    path.write_text(yaml.safe_dump(document))
    app, probes = protocol_app(root, unsupported_call, limit, fault)
    provider = FakeRuntimeProvider()
    bundle = tmp_path / "bundle"
    with served(app) as base_url:
        provider.subject.base_url = base_url
        report = run_validation(
            root, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
        )
    results = {r.check_id: r for r in report.results}
    assert results["runtime.startup"].status is CheckStatus.PASS
    assert provider.subject.stopped
    assert probes == ["/oracle/solve.sh"]
    assert results["runtime.reward_well_formed"].status is (
        CheckStatus.FAIL if fault == "primary_reward" else CheckStatus.PASS
    )
    for name in ("observation_schema", "state_contract"):
        assert results[f"runtime.{name}"].status is CheckStatus.PASS
    oracle_leak = fault in {"primary_oracle", "oracle_body"} or (
        fault == "prefix_oracle" and limit == "ws_depth"
    )
    oracle_result = results["runtime.oracle_containment"]
    assert oracle_result.status is (
        CheckStatus.FAIL if oracle_leak else CheckStatus.SKIP
    )
    if fault == "oracle_body":
        assert "GET /oracle/solve.sh" in " ".join(oracle_result.evidence)
    assert results["runtime.episode_isolation"].status is CheckStatus.SKIP
    assert report.verdict.value == (
        "fail" if oracle_leak or fault == "primary_reward" else "warn"
    )
    phase = "sequential" if unsupported_call == 2 else "fresh"
    metadata = json.loads((bundle / f"isolation-{phase}-evidence.json").read_text())
    assert metadata["complete"] is False
    assert metadata["failure_phase"] == ("step" if limit == "ws_depth" else "schema")
    capability = "json_numeric_range" if limit == "schema_numeric" else "json_depth"
    assert capability in metadata["failure_reason"]
    other = "fresh" if unsupported_call == 2 else "sequential"
    assert (
        json.loads((bundle / f"isolation-{other}-evidence.json").read_text())[
            "complete"
        ]
        is True
    )
    if limit == "ws_depth":
        rows = json.loads((bundle / f"isolation-{phase}-trace.json").read_text())
        # Credential-aware collection cannot retain an unscannable response.
        # The complete reset/state prefix still reaches grading and the bundle.
        assert len(rows) == (8 if unsupported_call == 2 else 2)
        assert rows[-1]["operation"] == "state"
        assert metadata["omitted_trace_fields"] == []
    else:
        assert metadata["schema_response_available"] is False
        assert metadata["schema_response_parse_failed"] is False
        assert metadata["schema_available"] is False
        assert metadata["reset_schema_available"] is False
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is True
