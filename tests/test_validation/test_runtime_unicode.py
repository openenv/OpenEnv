"""Bounded wire evidence must survive the schema-worker transport."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import ObservationSchemaGrader
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.collector import MAX_MESSAGE_BYTES, MAX_TRACE_BYTES
from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange
from openenv.validation.runtime.schema_worker import MAX_INPUT_BYTES
from openenv.validation.types import CheckStatus


@pytest.mark.parametrize("payload_kind", ["unicode", "numeric"])
def test_valid_episode_within_wire_budgets_passes_schema_worker(tmp_path, payload_kind):
    if payload_kind == "unicode":
        response = json.dumps(
            {
                "type": "observation",
                "data": {
                    "observation": {"text": "é" * 300_000},
                    "reward": 1,
                    "done": False,
                },
            },
            ensure_ascii=False,
        )
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    else:
        # Compact exponent notation is legal JSON. Parsing then serializing this
        # episode expands it past the old worker cap despite fitting wire limits.
        response = (
            '{"type":"observation","data":{"observation":{"values":['
            + ",".join(["1e9"] * 125_000)
            + ']},"reward":1,"done":false}}'
        )
        schema = {"type": "object", "properties": {"values": {"type": "array"}}}
    exchanges = []
    for index in range(7):
        exchanges.append(
            WireExchange(
                operation="reset" if index == 0 else "step",
                request_json="{}",
                response_json=response,
            )
        )

    frame_sizes = [len(row.response_json.encode()) for row in exchanges]
    assert max(frame_sizes) < MAX_MESSAGE_BYTES
    assert sum(frame_sizes) < MAX_TRACE_BYTES
    subject = Subject(
        root=tmp_path,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        image_ref=None,
        running=None,
        outputs_dir=tmp_path,
        runtime_evidence=RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=json.dumps(schema),
        ),
    )

    result = ObservationSchemaGrader().run(subject)

    assert result.status is CheckStatus.PASS, result.evidence


def test_schema_worker_rejects_oversized_binary_input():
    worker = (
        Path(__file__).parents[2] / "src/openenv/validation/runtime/schema_worker.py"
    )
    payload = json.dumps({"schema_json": "{}", "observations": []}).encode()
    # Valid JSON plus whitespace demonstrates a byte-bound rejection rather than
    # a parse error or a crash after unbounded allocation.
    payload += b" " * (MAX_INPUT_BYTES + 1 - len(payload))

    result = subprocess.run(
        [sys.executable, "-I", str(worker)],
        input=payload,
        capture_output=True,
        timeout=5,
        env={},
        check=True,
    )

    assert json.loads(result.stdout) == ["schema worker input exceeds its size budget"]
