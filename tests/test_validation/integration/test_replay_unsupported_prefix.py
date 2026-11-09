"""Real collector limits retain bounded replay prefixes for grading and bundles."""

import hashlib
import json
from dataclasses import replace
from decimal import Decimal

import pytest
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from openenv.core.env_server.types import WSErrorCode
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.basic import RewardWellFormedGrader
from openenv.validation.graders.runtime.repeatability import EpisodeDeterminismGrader
from openenv.validation.runtime import replay
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.collector import collect_runtime_evidence
from openenv.validation.runtime.contracts import (
    ReplayEvidence,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.runtime.json_values import dumps, loads
from openenv.validation.types import CheckStatus
from test_late_collection_protocol import served
from test_runtime_artifacts import report
from test_runtime_replay import inputs, run


def protocol_app(unsupported_call, limit):
    app = FastAPI()
    schema_calls = 0
    schema_raw = (
        '{"observation":{"type":"object","x-exact":1.0000000000000001},'
        '"reset_observation":{"type":"object","x-integer":9007199254740993},'
        '"action":{"description":"é"}}'
    )
    operations = []

    @app.get("/schema")
    def schema():
        nonlocal schema_calls
        schema_calls += 1
        raw = schema_raw
        if schema_calls == unsupported_call + 1 and limit != "ws_depth":
            value = (
                "1e9999999999999999999"
                if limit == "schema_numeric"
                else "[" * 20000 + "0" + "]" * 20000
            )
            raw = schema_raw[:-1] + ',"limit":' + value + "}"
        return Response(raw, media_type="application/json")

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        call = schema_calls
        steps = 0
        await connection.accept()
        try:
            while (request := await connection.receive_json())["type"] != "close":
                operation = request["type"]
                operations.append((call, operation))
                if operation not in {"reset", "step", "state"}:
                    await connection.send_json(
                        {"type": "error", "data": {"code": WSErrorCode.UNKNOWN_TYPE}}
                    )
                    continue
                if operation == "step":
                    steps += 1
                if operation == "state":
                    raw = dumps(
                        {
                            "type": "state",
                            "data": {"episode_id": "same-episode", "step_count": steps},
                        }
                    )
                else:
                    raw = (
                        '{"type":"observation","data":{"observation":{"counter":'
                        + str(steps)
                        + ',"precise":1.0000000000000001,"integer":9007199254740993},'
                        '"reward":0.5,"done":false}}'
                    )
                    if (
                        call == unsupported_call + 1
                        and limit == "ws_depth"
                        and operation == "step"
                    ):
                        raw = raw.replace(
                            '"counter":1',
                            '"counter":' + "[" * 20000 + "0" + "]" * 20000,
                        )
                await connection.send_text(raw)
        except WebSocketDisconnect:
            # The collector closes after rejecting the unsupported response.
            pass

    return app, schema_raw, operations


def rebuild_replays(directory, primary):
    saved = loads((directory / "replays.json").read_text())
    samples = []
    for row in saved["samples"]:
        sample = RuntimeEvidence(
            exchanges=tuple(
                WireExchange(
                    item["operation"],
                    dumps(item["request_json"]),
                    dumps(item["response_json"]),
                )
                for item in row["trace"]
            ),
            observation_schema_json=dumps(row["schema"])
            if row["schema_available"]
            else None,
            reset_observation_schema_json=dumps(row["reset_schema"])
            if row["reset_schema_available"]
            else None,
            telemetry_json=dumps(row["telemetry"])
            if row["telemetry_available"]
            else None,
            failure_phase=row["failure_phase"],
            failure_reason=row["failure_reason"],
            telemetry_error=row["telemetry_error"],
            telemetry_unavailable=row["telemetry_unavailable"],
        )
        samples.append(
            ReplayEvidence(
                row["scope"],
                sample,
                dumps(row["provider"]) if row["provider_available"] else None,
                row["cleanup_complete"],
            )
        )
    return replace(
        primary, replays=tuple(samples), replay_failure_reason=saved["failure_reason"]
    )


@pytest.mark.parametrize(
    "unsupported_call", [1, 2, 3, 4], ids=["session", "seed", "container", "judged"]
)
@pytest.mark.parametrize("limit", ["schema_numeric", "schema_depth", "ws_depth"])
def test_production_replay_prefix_survives_bundle_and_regrading(
    monkeypatch, tmp_path, unsupported_call, limit
):
    values = list(inputs(monkeypatch, judged=unsupported_call == 4))
    app, schema_raw, operations = protocol_app(unsupported_call, limit)
    monkeypatch.setattr(replay, "collect_runtime_evidence", collect_runtime_evidence)
    with served(app) as base_url:
        values[1].base_url = values[2].base_url = base_url
        primary = collect_runtime_evidence(
            base_url,
            values[4],
            episode_timeout_s=10,
            validation_token=values[3].env_vars["OPENENV_VALIDATION_TOKEN"],
        )
        assert primary.failure_reason is None
        values[5] = primary
        result = run(values)

    assert result.exchanges == primary.exchanges
    assert len(result.replays) == unsupported_call
    partial = result.replays[-1].evidence
    capability = "json_numeric_range" if limit == "schema_numeric" else "json_depth"
    phase = {
        1: "fresh session",
        2: "different-seed session",
        3: "fresh container replay",
        4: "judged replay sampling",
    }[unsupported_call]
    assert capability in partial.failure_reason
    assert capability in result.replay_failure_reason
    assert phase in result.replay_failure_reason
    if limit == "ws_depth":
        # Credential-aware collection retains the safe prefix, not an unscannable response.
        assert partial.failure_phase == "step"
        assert len(partial.exchanges) == 2
        assert partial.schema_response_json == schema_raw
        assert loads(partial.observation_schema_json)["x-exact"] == Decimal(
            "1.0000000000000001"
        )
        assert (
            loads(partial.reset_observation_schema_json)["x-integer"]
            == 9007199254740993
        )
        response = loads(partial.exchanges[0].response_json)
        assert response["data"]["observation"]["precise"] == Decimal(
            "1.0000000000000001"
        )
        assert response["data"]["observation"]["integer"] == 9007199254740993
    else:
        assert partial.failure_phase == "schema"
        assert not partial.exchanges
        assert partial.schema_response_json is None  # Credential scans did not finish.
    assert max(call for call, _ in operations) <= unsupported_call + 1
    if unsupported_call >= 3:
        container = result.replays[2]
        assert container.cleanup_complete is True
        assert json.loads(container.provider_json) == {
            "container_id": "fresh",
            "image_id": values[3].image_ref,
        }
        values[2].stop.assert_called_once()
        assert (
            values[0].start.call_args.args[0].env_vars["OPENENV_VALIDATION_TOKEN"]
            != values[3].env_vars["OPENENV_VALIDATION_TOKEN"]
        )
    else:
        values[0].start.assert_not_called()
    values[1].stop.assert_not_called()

    validation_report = report()
    validation_report.manifest.capabilities.llm_judged = unsupported_call == 4
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, result
    )
    direct = EpisodeDeterminismGrader().run(subject)
    assert direct.status is CheckStatus.SKIP
    assert RewardWellFormedGrader().run(subject).status is CheckStatus.PASS
    write_runtime_bundle(tmp_path, validation_report, evidence=result)
    rebuilt = rebuild_replays(tmp_path, primary)
    regraded = EpisodeDeterminismGrader().run(
        replace(subject, runtime_evidence=rebuilt)
    )
    assert regraded.status is direct.status
    assert regraded.evidence == direct.evidence
    assert regraded.measured == direct.measured
    saved = loads((tmp_path / "replays.json").read_text())
    row = saved["samples"][-1]
    assert row["failure_phase"] == partial.failure_phase
    assert row["failure_reason"] == partial.failure_reason
    assert row["redacted"] is False
    assert not row["omitted_trace_fields"] and not row["omitted_evidence_fields"]
    assert bool(rebuilt.replays[-1].evidence.exchanges) == (limit == "ws_depth")
    for candidate in (result, rebuilt):
        # A supported primary violation remains measurable despite incomplete siblings.
        rows = list(candidate.exchanges)
        rows[2] = replace(
            rows[2],
            response_json=rows[2].response_json.replace('"reward":0.5', '"reward":2'),
        )
        invalid_reward = replace(candidate, exchanges=tuple(rows))
        assert (
            RewardWellFormedGrader()
            .run(replace(subject, runtime_evidence=invalid_reward))
            .status
            is CheckStatus.FAIL
        )
        if unsupported_call > 1:
            # A completed session divergence also survives a later unsupported replay.
            sibling = candidate.replays[0]
            rows = list(sibling.evidence.exchanges)
            rows[0] = replace(
                rows[0],
                response_json=rows[0].response_json.replace(
                    '"counter":0', '"counter":7'
                ),
            )
            divergent = replace(
                sibling, evidence=replace(sibling.evidence, exchanges=tuple(rows))
            )
            invalid_replay = replace(
                candidate, replays=(divergent, *candidate.replays[1:])
            )
            assert (
                EpisodeDeterminismGrader()
                .run(replace(subject, runtime_evidence=invalid_replay))
                .status
                is CheckStatus.FAIL
            )
    for line in (tmp_path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
