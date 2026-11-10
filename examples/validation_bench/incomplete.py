#!/usr/bin/env python3
"""Experiment 2: incomplete evidence, fail precedence and replay challenges.

Run with the existing venv, without installing anything:
  PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py --freeze-only
  PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py

All servers are bounded loopback processes. Provider measurements and Hub metadata
are explicit doubles: this does not launch Docker, HF Jobs or write to the Hub.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/test_validation"))

from openenv.validation import runner
from openenv.validation.graders import Subject
from openenv.validation.policy import load_policy
from openenv.validation.providers import ProviderError
from openenv.validation.providers.hf_sandbox import HFSandboxValidationProvider
from openenv.validation.report import ValidationReportV2
from openenv.validation.runtime import collector
from openenv.validation.runtime.artifacts import _redact
from openenv.validation.runtime.contracts import (
    HttpProbe,
    IsolationEvidence,
    LaunchSpec,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.runtime.json_values import dumps, loads
from openenv.validation.runtime.scheduler import execute_graders
from openenv.validation.types import Level, ProviderCapability
from support.runtime import FakeRunningSubject, FakeRuntimeProvider
from test_host_containment import contained
from test_resource_bounds import bounded

DEST = Path("/tmp/openenv-validation-bench-20261109")
FIXTURE = ROOT / "tests/fixtures/validation/runtime/served_probe"
SEEDS = (17, 271, 65537)
BASIC = ("reward_well_formed", "observation_schema", "state_contract")
SECRET = "exp2-synthetic-credential-canary"
ORACLE = "exp2-validator-owned-oracle-marker-20261109"


def expected(startup="pass", *, verdict="warn", replay="agreement", **checks):
    return {
        "verdict": verdict,
        "checks": {
            "runtime.startup": startup,
            **{"runtime." + k: v for k, v in checks.items()},
        },
        "replay": replay,
    }


# Frozen before any case execution. Expectations do not derive from observations.
MATRIX = {
    "completed_positive": expected(
        **{k: "pass" for k in BASIC},
        episode_isolation="pass",
        oracle_containment="pass",
        host_containment="pass",
        resource_bounds="pass",
    ),
    "interrupted_primary_prefix": expected(
        "error", verdict="fail", **{k: "skip" for k in BASIC}
    ),
    "interrupted_fresh_prefix": expected(
        "error", verdict="fail", episode_isolation="skip", **{k: "skip" for k in BASIC}
    ),
    "truncated_schema_http": expected(
        "fail", verdict="fail", **{k: "skip" for k in BASIC}
    ),
    "unsupported_schema_number": expected(
        "skip", replay="unsupported", **{k: "skip" for k in BASIC}
    ),
    "truncated_oracle_http": expected(
        oracle_containment="skip", episode_isolation="pass"
    ),
    "unsafe_http_omission": expected(oracle_containment="skip", replay="inconclusive"),
    "missing_measurements": expected(host_containment="skip", resource_bounds="skip"),
    "isolation_completed_prefix": expected(episode_isolation="skip"),
    "known_resource_fail_missing_storage": expected(
        verdict="fail", resource_bounds="fail", host_containment="skip"
    ),
    "known_reward_fail_missing_measurements": expected(
        verdict="fail",
        reward_well_formed="fail",
        resource_bounds="skip",
        host_containment="skip",
    ),
    "known_oracle_fail_unrelated_missing": expected(
        verdict="fail",
        oracle_containment="fail",
        resource_bounds="skip",
        replay="inconclusive",
    ),
    "precise_isolation_divergence": expected(verdict="fail", episode_isolation="fail"),
    "duplicate_wire_decisive_evidence": expected(
        verdict="fail", oracle_containment="fail"
    ),
    "duplicate_http_decisive_evidence": expected(
        verdict="fail", oracle_containment="fail"
    ),
    "decisive_http_redaction_challenge": expected(
        verdict="fail", oracle_containment="fail", replay="inconclusive"
    ),
    "hf_missing_runtime_metadata": expected("skip"),
    "hf_missing_image_binding": expected("skip"),
    "hf_mismatched_runtime_revision": expected("error", verdict="fail"),
    "hf_mismatched_source_revision": expected("error", verdict="fail"),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value, variant):
    return json.dumps(
        value,
        ensure_ascii=variant == 0,
        sort_keys=variant == 1,
        indent=2 if variant == 2 else None,
        separators=(",", ":") if variant == 0 else None,
    )


class IncompleteResponse(Response):
    """Send a real HTTP prefix, then let uvicorn close the incomplete response."""

    async def __call__(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(self.body) + 97).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": self.body, "more_body": True})


@contextmanager
def served(mode, variant, events):
    app = FastAPI()
    sessions = 0

    @app.get("/schema")
    def schema():
        if mode == "truncated_schema_http":
            return IncompleteResponse(b'{"observation":{"type":"object"')
        if mode == "unsupported_schema_number":
            return Response(
                '{"observation":{"minimum":1e9999999999999999999}}',
                media_type="application/json",
            )
        body = {
            "observation": {
                "type": "object",
                "required": ["counter", "reward", "done"],
                "properties": {
                    "counter": {"type": "number"},
                    "reward": {"type": "number"},
                    "done": {"type": "boolean"},
                },
            },
            "action": {"type": "object"},
            "state": {"type": "object"},
            "reset": {"type": "object"},
        }
        return Response(encode(body, variant), media_type="application/json")

    @app.get("/oracle/solve.sh")
    def oracle():
        if mode in {"unsafe_http_omission", "known_oracle_fail_unrelated_missing"}:
            return IncompleteResponse(("export API_KEY=" + SECRET + "\n").encode())
        if mode == "truncated_oracle_http":
            return IncompleteResponse(b"public truncated probe prefix")
        if mode == "duplicate_http_decisive_evidence":
            return Response(
                '{"value":' + json.dumps(ORACLE) + ',"value":"harmless"}',
                media_type="application/json",
            )
        if mode == "decisive_http_redaction_challenge":
            return Response(
                encode(
                    {"stdout": "export API_KEY=" + SECRET, "answer": ORACLE}, variant
                ),
                media_type="application/json",
            )
        return Response("denied", status_code=403)

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        nonlocal sessions
        sessions += 1
        session = sessions
        await connection.accept()
        steps = resets = 0
        episode_id = None
        try:
            while True:
                request = await connection.receive_json()
                operation = request["type"]
                if operation == "close":
                    await connection.close()
                    return
                if operation == "reset":
                    steps = 0
                    resets += 1
                    episode_id = request["data"]["episode_id"]
                elif operation == "step":
                    if (
                        mode == "isolation_completed_prefix"
                        and session == 2
                        and resets == 2
                    ):
                        await connection.close(code=1011)
                        return
                    steps += 1
                if operation == "state":
                    body = {
                        "type": "state",
                        "data": {"episode_id": episode_id, "step_count": steps},
                    }
                else:
                    observation = {"counter": steps}
                    if mode == "known_oracle_fail_unrelated_missing" and steps:
                        observation["public_solution"] = ORACLE
                    if mode == "duplicate_wire_decisive_evidence" and steps:
                        observation["value"] = "DUPLICATE_INSERT"
                    reward = (
                        -1
                        if mode == "known_reward_fail_missing_measurements" and steps
                        else (0.5 if steps else 0)
                    )
                    body = {
                        "type": "observation",
                        "data": {
                            "observation": observation,
                            "reward": reward,
                            "done": steps == 2,
                        },
                    }
                raw = encode(body, (variant + resets - 1) % 3)
                if mode == "duplicate_wire_decisive_evidence" and operation == "step":
                    raw = raw.replace(
                        '"DUPLICATE_INSERT"', json.dumps(ORACLE) + ',"value":"harmless"'
                    )
                if mode == "precise_isolation_divergence" and operation == "step":
                    # Exact decimal values differ, although binary floats round alike.
                    number = "1.0" if resets == 2 else "1.0000000000000001"
                    decoded = json.loads(raw)
                    decoded["data"]["observation"]["counter"] = "EXACT_NUMBER"
                    raw = encode(decoded, variant).replace('"EXACT_NUMBER"', number)
                events.append(
                    {
                        "session": session,
                        "reset": resets,
                        "operation": operation,
                        "request_sha256": sha(dumps(request).encode()),
                        "response_sha256": sha(raw.encode()),
                    }
                )
                await connection.send_text(raw)
        except WebSocketDisconnect:
            # The validation client has closed this fixture connection.
            pass

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        # Expected incomplete-response server errors are retained as fault events,
        # not printed with submitted body data.
        config = uvicorn.Config(
            app, log_level="critical", access_log=False, log_config=None
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if not server.started:
                raise RuntimeError("loopback server did not start")
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("loopback server did not stop")


class MeasuredSubject(FakeRunningSubject):
    """Synthetic provider observations; no real isolation/resource qualification."""

    def __init__(self, base_url, mode):
        super().__init__(base_url=base_url)
        self.mode = mode

    def inspect(self):
        return {
            "measurement_origin": "provider-double; no actual Docker/HF",
            "mode": self.mode,
        }

    def measure_containment(self):
        if self.mode in {
            "missing_measurements",
            "known_resource_fail_missing_storage",
            "known_reward_fail_missing_measurements",
            "known_oracle_fail_unrelated_missing",
        }:
            raise ProviderError("exp2 injected unavailable containment probe")
        return contained()

    def measure_resources(self):
        if self.mode in {
            "missing_measurements",
            "known_reward_fail_missing_measurements",
            "known_oracle_fail_unrelated_missing",
        }:
            raise ProviderError("exp2 injected unavailable resource probe")
        measured = bounded()
        if self.mode == "known_resource_fail_missing_storage":
            measured["cgroup"]["memory_max"] *= 2
            measured["containment"] = None
            measured["qualification"] = []
        return measured


def make_package(directory, seed, variant):
    package = directory / "subject"
    shutil.copytree(FIXTURE, package, ignore=shutil.ignore_patterns("__pycache__"))
    plan_path = package / "validation/runtime.json"
    plan = json.loads(plan_path.read_text())
    plan["reset"].update(seed=seed, episode_id=f"exp2-{seed}")
    plan_path.write_text(encode(plan, variant))
    manifest_path = package / "openenv.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["validation"]["resources"]["episode_timeout_s"] = 3.0
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=variant != 2))
    (package / "oracle/solve.sh").write_text(ORACLE + "\n")
    return package


def runtime_run(package, bundle, mode, variant):
    events, recordings, probes = [], [], []
    original_connect = collector.connect
    original_collect = collector.collect_runtime_evidence
    original_probes = runner._oracle_probes

    def collect_probes(*args, **kwargs):
        result = original_probes(*args, **kwargs)
        probes.extend(result)
        return result

    def collect(*args, **kwargs):
        call = len(recordings) + 1
        interrupt = (mode == "interrupted_primary_prefix" and call == 1) or (
            mode == "interrupted_fresh_prefix" and call == 3
        )

        class InterruptedConnection:
            def __init__(self, connection):
                self.connection = connection
                self.received = 0

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def recv(self, *a, **k):
                if self.received == 4:
                    raise KeyboardInterrupt
                self.received += 1
                return self.connection.recv(*a, **k)

        def connect(*a, **k):
            connection = original_connect(*a, **k)
            return InterruptedConnection(connection) if interrupt else connection

        with patch.object(collector, "connect", connect):
            try:
                evidence = original_collect(*args, **kwargs)
            except (
                collector.RuntimeCollectionInterrupted,
                collector.RuntimeCollectionUnsupported,
            ) as exc:
                recordings.append(exc.evidence)
                raise
            recordings.append(evidence)
            return evidence

    with served(mode, variant, events) as url:
        running = MeasuredSubject(url, mode)
        provider = FakeRuntimeProvider(
            name="exp2-local-protocol-provider-double-no-docker", subject=running
        )
        provider.capabilities = frozenset(
            {
                ProviderCapability.IMAGE_BUILD,
                ProviderCapability.EXEC,
                ProviderCapability.INSPECTED_ISOLATION,
                ProviderCapability.RESOURCE_ENFORCEMENT,
            }
        )
        with (
            patch.object(runner, "collect_runtime_evidence", collect),
            patch.object(runner, "_oracle_probes", collect_probes),
        ):
            report = runner.run_validation(
                package,
                max_level=Level.RUNTIME,
                provider=provider,
                artifacts_dir=bundle,
            )
    save(bundle.parent / "protocol-events.json", events)
    save(
        bundle.parent / "raw/collector-sessions.json",
        [asdict(recording) for recording in recordings],
    )
    # Unsafe bodies remain omitted. Exact byte hashes retain the raw provenance.
    save(
        bundle.parent / "raw/http-probes.json",
        [
            {
                "path": probe.path,
                "status": probe.status,
                "failure": probe.failure,
                "body_sha256": sha(probe.body),
                "body_bytes": len(probe.body),
                "safe_raw_body": probe.body.decode()
                if SECRET.encode() not in probe.body
                else "[unsafe body omitted]",
            }
            for probe in probes
        ],
    )
    session_names = ["collector", "isolation-sequential", "isolation-fresh"]
    prefix_integrity = []
    for session, recording in zip(session_names, recordings):
        saved = loads((bundle / f"{session}-evidence.json").read_text())
        trace = loads((bundle / f"{session}-trace.json").read_text())
        prefix_integrity.append(
            {
                "session": session,
                "count": len(trace),
                "count_matches": len(trace) == len(recording.exchanges),
                "failure_metadata_matches": saved["failure_phase"]
                == recording.failure_phase
                and saved["failure_reason"] == recording.failure_reason,
                "complete": saved["complete"],
            }
        )
    return (
        report,
        provider.capabilities,
        {
            "kind": "real loopback HTTP/WebSocket; production collector/runner/graders/artifact writer",
            "fault_origin": "collector recv KeyboardInterrupt wrapper after four real responses"
            if mode.startswith("interrupted_")
            else mode,
            "collection_calls": len(recordings),
            "exchange_counts": [len(e.exchanges) for e in recordings],
            "failure_phases": [e.failure_phase for e in recordings],
            "prefix_integrity": prefix_integrity,
            "http_body_hashes_preserved": all(
                row["body_sha256"] == sha(probe.body)
                and row["retained_bytes"] == len(probe.body)
                for row, probe in zip(
                    loads((bundle / "oracle-http.json").read_text()), probes
                )
            ),
            "cleanup_stopped": running.stopped,
            "build_calls": len(provider.builds),
            "start_calls": len(provider.launches),
            "jobs_created": 0,
        },
    )


def hf_run(package, bundle, mode):
    calls = []

    class Api:
        def __init__(self, token=None):
            pass

        def list_jobs_hardware(self):
            calls.append("list_jobs_hardware-double")
            return [SimpleNamespace(name="cpu-basic", accelerator=None)]

        def space_info(self, space_id):
            calls.append("space_info-double")
            raw = {"stage": "RUNNING", "sha": "a" * 40}
            if mode == "hf_missing_runtime_metadata":
                raw = {}
            if mode == "hf_mismatched_runtime_revision":
                raw["sha"] = "b" * 40
            return SimpleNamespace(
                sha=("b" if mode == "hf_mismatched_source_revision" else "a") * 40,
                runtime=SimpleNamespace(raw=raw),
            )

        def run_job(self, *args, **kwargs):
            return create_job()

    def create_job(*args, **kwargs):
        calls.append("JOB_CREATION_ATTEMPT")
        raise AssertionError("experiment forbids job creation")

    with (
        patch.object(huggingface_hub, "HfApi", Api),
        patch.object(
            huggingface_hub, "Sandbox", SimpleNamespace(create=create_job), create=True
        ),
    ):
        provider = HFSandboxValidationProvider("exp2-local-double/probe", "a" * 40)
        report = runner.run_validation(
            package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
        )
        # A made-up digest must not bypass the independent production start gate.
        try:
            provider.start(
                LaunchSpec(
                    image_ref="sha256:" + "a" * 64,
                    resources=report.manifest.resources,
                    network=report.manifest.network,
                    run_id="exp2-explicit-start-gate",
                )
            )
            direct_start = "UNEXPECTED_START"
        except ProviderError as exc:
            direct_start = type(exc).__name__ + ": " + str(exc)
    return (
        report,
        provider.capabilities,
        {
            "kind": "production HFSandboxValidationProvider with inert API/Sandbox doubles; no actual HF",
            "calls": calls,
            "jobs_created": calls.count("JOB_CREATION_ATTEMPT"),
            "direct_start_gate": direct_start,
        },
    )


def rebuild(bundle, session):
    path = bundle / f"{session}-evidence.json"
    if not path.exists():
        return None
    metadata = loads(path.read_text())
    if metadata is None:
        return None
    trace = loads((bundle / metadata["trace_file"]).read_text())
    return RuntimeEvidence(
        exchanges=tuple(
            WireExchange(
                row["operation"],
                dumps(row["request_json"]),
                dumps(row["response_json"]),
            )
            for row in trace
        ),
        observation_schema_json=dumps(metadata["observation_schema"])
        if metadata["schema_available"]
        else None,
        reset_observation_schema_json=dumps(metadata["reset_observation_schema"])
        if metadata["reset_schema_available"]
        else None,
        schema_response_json=dumps(metadata["schema_response"])
        if metadata["schema_response_available"]
        else None,
        failure_phase=metadata["failure_phase"],
        failure_reason=metadata["failure_reason"],
    )


def replay(bundle, package, report, capabilities):
    """Use only bundle evidence, plus the digest-pinned retained subject source.

    There is no production bundle reader. The adapter checks omission/redaction
    flags before claiming replay support; an unchecked replay is shown separately.
    """
    flags = []
    unsupported = False
    for path in bundle.glob("*-evidence.json"):
        metadata = loads(path.read_text())
        if not metadata:
            continue
        if any(
            metadata.get(k)
            for k in (
                "schema_parse_failed",
                "reset_schema_parse_failed",
                "schema_response_parse_failed",
            )
        ):
            unsupported = True
            flags.append(path.name + ": parse unsupported/failed")
        if metadata["redacted"]:
            flags.append(path.name + ": redacted/omitted")
    probe_rows = loads((bundle / "oracle-http.json").read_text())
    probes = []

    def body(row):
        if row.get("redacted") or row.get("body_omitted"):
            flags.append("oracle-http.json: redacted/omitted body")
        value = row["body"]
        return value.encode() if isinstance(value, str) else dumps(value).encode()

    for row in probe_rows:
        probes.append(
            HttpProbe(
                row["path"],
                row["status"],
                body(row),
                row["failure"],
                tuple(body(hop) for hop in row["redirect_bodies"]),
            )
        )
    recording = rebuild(bundle, "collector")
    sequential = rebuild(bundle, "isolation-sequential")
    fresh = rebuild(bundle, "isolation-fresh")
    inspection = loads((bundle / "run-manifest.json").read_text())["provider"]
    subject = Subject(
        root=package,
        manifest=report.manifest,
        image_ref=None,
        running=None,
        outputs_dir=bundle,
        runtime_evidence=recording,
        isolation_evidence=IsolationEvidence(sequential, fresh)
        if sequential is not None
        else None,
        oracle_http=tuple(probes),
        containment_evidence_json=dumps(inspection["containment"])
        if "containment" in inspection
        else None,
        resource_evidence_json=dumps(inspection["resources"])
        if "resources" in inspection
        else None,
    )
    registry = runner.default_grader_registry(load_policy("v2"))
    graders = [
        g
        for g in registry.select(report.manifest, Level.RUNTIME)
        if g.level == Level.RUNTIME
    ]
    prior = [r for r in report.results if r.check_id == "runtime.startup"]
    checked = execute_graders(
        graders, subject, provider_capabilities=capabilities, prior=prior
    )
    original = {r.check_id: r for r in report.results}
    comparisons = []
    for result in checked:
        before = original[result.check_id]
        comparisons.append(
            {
                "check_id": result.check_id,
                "original": before.status.value,
                "replayed": result.status.value,
                "status_agrees": before.status == result.status,
                "evidence_agrees": before.evidence == result.evidence,
            }
        )
    classification = (
        "unsupported" if unsupported else "inconclusive" if flags else "supported"
    )
    status_mismatches = [c for c in comparisons if not c["status_agrees"]]
    return {
        "classification": classification,
        "observed_outcome": ("mismatch" if status_mismatches else "agreement")
        if classification == "supported"
        else classification,
        "limits": sorted(set(flags)),
        "comparisons": comparisons,
        "supported_comparisons": len(comparisons) if not flags else 0,
        "status_mismatches": status_mismatches,
        "saved_known_fail_ids": [
            r.check_id
            for r in ValidationReportV2.model_validate_json(
                (bundle / "report.json").read_text()
            ).results
            if r.status.value in {"fail", "error"}
        ],
        "raw_numeric_tokens": "Decimal loader; no binary-float normalization",
        "source_digest_verified": runner.source_digest(package) == report.source_digest,
    }


def verify_bundle(bundle, report):
    checksums = []
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        checksums.append(
            {"file": filename, "match": sha((bundle / filename).read_bytes()) == digest}
        )
    text = "".join(path.read_text() for path in bundle.iterdir() if path.is_file())
    probes = loads((bundle / "oracle-http.json").read_text())
    return {
        "files": checksums,
        "all_checksums_match": all(c["match"] for c in checksums),
        "synthetic_credential_absent": SECRET not in text,
        # Report durations are binary floats, unlike exact subject JSON numbers.
        # Pydantic and stdlib can emit distinct shortest decimal spellings of the
        # same float; compare through the public report model.
        "saved_report_matches_expected_redaction": ValidationReportV2.model_validate_json(
            (bundle / "report.json").read_text()
        ).model_dump()
        == ValidationReportV2.model_validate(
            _redact(report.model_dump(mode="json"))
        ).model_dump(),
        "saved_decisive_statuses_preserved": {
            row["check_id"]: row["status"]
            for row in json.loads((bundle / "report.json").read_text())["results"]
        }
        == {row.check_id: row.status.value for row in report.results},
        "report_target_redacted": json.loads((bundle / "report.json").read_text())[
            "target"
        ]
        != report.target,
        "http_omissions": [
            {
                "path": p["path"],
                "status": p["status"],
                "failure": p["failure"],
                "body_omitted": p["body_omitted"],
                "redacted": p["redacted"],
                "body_sha256": p["body_sha256"],
                "retained_bytes": p["retained_bytes"],
            }
            for p in probes
        ],
    }


def run_case(base, name, seed, variant):
    directory = base / f"seed-{seed}" / name
    directory.mkdir(parents=True, exist_ok=False)
    package = make_package(directory, seed, variant)
    bundle = directory / "bundle"
    started = time.monotonic()
    report, capabilities, observations = (
        hf_run(package, bundle, name)
        if name.startswith("hf_")
        else runtime_run(package, bundle, name, variant)
    )
    actual = {
        "verdict": report.verdict.value,
        "checks": {r.check_id: r.status.value for r in report.results},
    }
    wanted = MATRIX[name]
    mismatches = []
    if actual["verdict"] != wanted["verdict"]:
        mismatches.append(
            {
                "field": "verdict",
                "expected": wanted["verdict"],
                "observed": actual["verdict"],
            }
        )
    for check_id, value in wanted["checks"].items():
        if actual["checks"].get(check_id) != value:
            mismatches.append(
                {
                    "field": check_id,
                    "expected": value,
                    "observed": actual["checks"].get(check_id),
                }
            )
    replayed = replay(bundle, package, report, capabilities)
    integrity = verify_bundle(bundle, report)
    # Frozen `agreement` asks for both support and equal grades. Capability is
    # labelled separately so a supported-but-mismatching replay is never PASS.
    expected_support = (
        "supported" if wanted["replay"] == "agreement" else wanted["replay"]
    )
    if replayed["classification"] != expected_support:
        mismatches.append(
            {
                "field": "replay",
                "expected": wanted["replay"],
                "observed": replayed["classification"],
            }
        )
    if replayed["classification"] == "supported" and replayed["status_mismatches"]:
        mismatches.append(
            {"field": "supported_replay", "observed": replayed["status_mismatches"]}
        )
    for field in (
        "all_checksums_match",
        "synthetic_credential_absent",
        "saved_report_matches_expected_redaction",
        "saved_decisive_statuses_preserved",
    ):
        if not integrity[field]:
            mismatches.append({"field": field, "expected": True, "observed": False})
    if observations["jobs_created"] or not replayed["source_digest_verified"]:
        mismatches.append(
            {"field": "zero_jobs_and_pinned_source", "observed": observations}
        )
    if observations.get("direct_start_gate") == "UNEXPECTED_START":
        mismatches.append(
            {
                "field": "direct_start_gate",
                "expected": "ProviderError",
                "observed": observations["direct_start_gate"],
            }
        )
    if not all(
        item["count_matches"] and item["failure_metadata_matches"]
        for item in observations.get("prefix_integrity", [])
    ) or not observations.get("http_body_hashes_preserved", True):
        mismatches.append(
            {"field": "raw_prefix_or_http_hashes", "observed": observations}
        )
    row = {
        "case": name,
        "seed": seed,
        "serialization_variant": variant,
        "expected": wanted,
        "observed": actual,
        "observations": observations,
        "replay": replayed,
        "integrity": integrity,
        "mismatches": mismatches,
        "elapsed_s": time.monotonic() - started,
        "bundle": str(bundle),
    }
    save(directory / "observation.json", row)
    print(
        json.dumps(
            {
                "case": name,
                "seed": seed,
                "verdict": actual["verdict"],
                "replay": replayed["classification"],
                "mismatches": len(mismatches),
            }
        ),
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    base = args.output / "exp2"
    base.mkdir(parents=True, exist_ok=True)
    frozen = {
        "experiment": "2",
        "head": "2781faaa59f00b7192145282950193d570d07659",
        "seeds": SEEDS,
        "serialization_variants": [
            "compact-unicode-escaped",
            "sorted-keys",
            "indented-unicode",
        ],
        "cases": MATRIX,
        "qualification": "local CPU protocol and synthetic provider observations only",
        "replay_rule": "redacted or omitted evidence is inconclusive; unsupported JSON is separate; saved FAIL must survive",
    }
    frozen_path = base / "expected-matrix.json"
    expected_bytes = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    if frozen_path.exists() and frozen_path.read_bytes() != expected_bytes:
        raise SystemExit("refusing to change frozen matrix")
    if args.freeze_only:
        frozen_path.write_bytes(expected_bytes)
        save(
            base / "freeze-provenance.json",
            {
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "sha256": sha(expected_bytes),
                "script_sha256": sha(Path(__file__).read_bytes()),
            },
        )
        print(
            f"Frozen {len(MATRIX)} cases x {len(SEEDS)} repeats: {sha(expected_bytes)}"
        )
        return 0
    if not frozen_path.exists():
        raise SystemExit(
            "freeze the expected matrix with --freeze-only before executing"
        )
    save(
        base / "execution-start.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "matrix_sha256": sha(expected_bytes),
            "script_sha256": sha(Path(__file__).read_bytes()),
        },
    )
    rows = []
    for variant, seed in enumerate(SEEDS):
        cases = list(MATRIX)
        random.Random(seed).shuffle(cases)
        for name in cases:
            rows.append(run_case(base, name, seed, variant))
    supported = [r for r in rows if r["replay"]["classification"] == "supported"]
    comparisons = [c for r in supported for c in r["replay"]["comparisons"]]
    challenges = [r for r in rows if r["replay"]["classification"] == "inconclusive"]
    summary = {
        "experiment": "incomplete evidence / false PASS / replay challenge",
        "matrix_sha256": sha(expected_bytes),
        "environment": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: importlib.metadata.version(p)
                for p in (
                    "huggingface_hub",
                    "httpx",
                    "websockets",
                    "uvicorn",
                    "pydantic",
                )
            },
        },
        "limits": [
            "Local CPU only; no actual Docker, HF Jobs, Hub writes or network calls outside loopback.",
            "Provider doubles inject synthetic resource/containment evidence; their PASS is grader coverage, not host qualification.",
            "Positive control still WARN because reserved/unmeasured checks remain SKIP.",
            "No production replay reader exists; experiment adapter honors redaction/omission/parse flags.",
            "Saved report retains known FAILs when raw HTTP evidence is deliberately omitted for credential safety.",
        ],
        "counts": {
            "cases_per_repeat": len(MATRIX),
            "repeats": len(SEEDS),
            "observations": len(rows),
            "expected_status_mismatches": sum(len(r["mismatches"]) for r in rows),
            "supported_replay_runs": len(supported),
            "supported_grader_comparisons": len(comparisons),
            "supported_grader_status_agreements": sum(
                c["status_agrees"] for c in comparisons
            ),
            "supported_grader_evidence_agreements": sum(
                c["evidence_agrees"] for c in comparisons
            ),
            "inconclusive_redacted_replay_runs": len(challenges),
            "unsupported_replay_runs": sum(
                r["replay"]["classification"] == "unsupported" for r in rows
            ),
            "all_bundles_checksum_valid": all(
                r["integrity"]["all_checksums_match"] for r in rows
            ),
            "jobs_created": sum(r["observations"]["jobs_created"] for r in rows),
        },
        "findings": [
            {"case": r["case"], "seed": r["seed"], "mismatches": r["mismatches"]}
            for r in rows
            if r["mismatches"]
        ],
        "unguarded_replay_challenges": [
            {
                "case": r["case"],
                "seed": r["seed"],
                "status_changes_if_omission_flags_ignored": r["replay"][
                    "status_mismatches"
                ],
                "saved_known_fail_ids": r["replay"]["saved_known_fail_ids"],
            }
            for r in challenges
        ],
        "observations": rows,
    }
    save(args.output / "exp2-report.json", summary)
    counts = summary["counts"]
    md = [
        "# Experiment 2: incomplete evidence and replay\n",
        f"Head: `{summary['environment']['head']}`. Matrix frozen before execution: `{sha(expected_bytes)}`.\n",
        f"{len(rows)} structured observations; {counts['expected_status_mismatches']} expected-status mismatches; {counts['jobs_created']} Jobs created.\n",
        f"Supported replay: {counts['supported_grader_status_agreements']}/{len(comparisons)} grader statuses agree across {len(supported)} runs. "
        f"Separate denominators: {len(challenges)} redacted/inconclusive and {counts['unsupported_replay_runs']} unsupported runs.\n",
        "| Case | Expected verdict | Observed verdicts (three repeats) | Target statuses | Expected replay | Observed replay |\n|---|---|---|---|---|---|",
    ]
    for name, want in MATRIX.items():
        matching = sorted(
            (r for r in rows if r["case"] == name), key=lambda r: r["seed"]
        )
        md.append(
            f"| {name} | {want['verdict']} | {', '.join(r['observed']['verdict'] for r in matching)} | "
            + "; ".join(
                k.removeprefix("runtime.") + "=" + v for k, v in want["checks"].items()
            )
            + f" | {want['replay']} | "
            + ", ".join(r["replay"]["observed_outcome"] for r in matching)
            + " |"
        )
    md.extend(
        [
            "\n## Interpretation\n",
            *["- " + item for item in summary["limits"]],
            "\n## Concrete artifact-fidelity finding\n\n"
            "Duplicate JSON keys on either the WebSocket or HTTP surface can contain a full oracle marker in an earlier value. "
            "The live oracle check and saved report retain FAIL. The artifact writer decodes away that value at "
            "`runtime/artifacts.py:_recording` or `_http_body`, without marking redaction/omission. "
            "The experiment's reconstructed oracle check becomes PASS. There is no production replay reader. "
            "These supported replays remain in the denominator and are labelled mismatch; this does not establish a live validation false PASS.\n",
            "\nThe decisive HTTP redaction challenge deliberately puts a full oracle marker next to a synthetic credential in an unsafe body. "
            "The production writer omits that body and retains its digest, byte count, omission flag and original report FAIL. "
            "Ignoring those flags can change a replayed oracle FAIL to PASS; these runs are inconclusive, excluded from the agreement denominator. "
            "This is a replay-consumer limitation, not a demonstrated production validation false PASS.\n",
            "\n## Reproduce\n\n```bash\nPYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py --freeze-only\n"
            "PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/incomplete.py\n```\n",
            "Use a fresh `--output` directory for another run. All raw production bundles, protocol event hashes, frozen matrix and per-case observations are under `exp2/`.\n",
        ]
    )
    (args.output / "exp2-report.md").write_text("\n".join(md))
    for suffix in ("json", "md"):
        shutil.copyfile(
            args.output / f"exp2-report.{suffix}", base / f"report.{suffix}"
        )
    # Flush the execution log before taking the final inventory.
    print(json.dumps(counts, sort_keys=True), flush=True)
    inventory = [
        f"{sha(path.read_bytes())}  {path.relative_to(base).as_posix()}\n"
        for path in sorted(base.rglob("*"))
        if path.is_file() and path != base / "SHA256SUMS"
    ]
    (base / "SHA256SUMS").write_text("".join(inventory))
    return int(bool(summary["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
