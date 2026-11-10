"""Exercise collection, gating and artifact binding without paid services."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.coding_audit.runner import (
    coding_budget,
    live_provider_id,
    normalize_coding_config,
    run_coding_validation,
)


@pytest.mark.parametrize("representation", ["dict", "list", "json-list"])
def test_live_hub_provider_mapping_shapes(representation):
    entry = SimpleNamespace(
        provider="deepinfra", provider_id="Qwen/Qwen3.8-27B", status="live"
    )
    mappings = (
        {"deepinfra": entry}
        if representation == "dict"
        else [vars(entry)]
        if representation == "json-list"
        else [entry]
    )
    assert (
        live_provider_id(
            SimpleNamespace(inference_provider_mapping=mappings), "deepinfra"
        )
        == "Qwen/Qwen3.8-27B"
    )


@pytest.mark.parametrize(
    "mappings",
    [
        None,
        [],
        [SimpleNamespace(provider="deepinfra", provider_id="model", status="staging")],
        [SimpleNamespace(provider="deepinfra", provider_id="model", status="live")] * 2,
    ],
)
def test_provider_mapping_must_be_unique_and_live(mappings):
    with pytest.raises(ValueError):
        live_provider_id(
            SimpleNamespace(inference_provider_mapping=mappings), "deepinfra"
        )


def configuration():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config["generation"]["candidate_count"] = 1
    config["validation"].update(reference_repeats=2, target_attempts=2)
    config["seeds"] = [1, 2]
    return config


class Backend:
    def __init__(self):
        self.number = 0
        self.target_calls = 0
        self.generated = "mkdir -p /app; echo 42 > /app/answer.txt"
        self.bad_verifier = False
        self.bad_raw = False
        self.target_error = False
        self.mutate = None
        self.transcript = []

    def execute_script(self, script, role):
        self.number += 1
        value = b"42\n" if "42" in script else b"41\n"
        files = {} if script == "true" else {"/app/answer.txt": value}
        digest = hashlib.sha256(
            json.dumps({k: v.hex() for k, v in files.items()}, sort_keys=True).encode()
        ).hexdigest()
        image = configuration()["execution"]["agent_image"]
        return {
            "files": files,
            "submission_digest": digest,
            "sandbox_id": f"solver-{self.number}",
            "image_ref": image,
            "cleanup_verified": True,
            "error": None,
            "transcript": list(self.transcript),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_seconds": 1,
        }

    def solve(self, model, seed, role):
        self.target_calls += 1
        result = self.execute_script("42" if seed == 2 else "41", role)
        if self.target_error:
            result["error"] = "provider failure"
        return result

    def generate_candidate(self, model, seed):
        if self.mutate:
            self.mutate()
        return {"script": self.generated, "error": None}

    def grade(self, submission):
        self.number += 1
        reward = int(
            submission["files"].get("/app/answer.txt") == b"42\n" or self.bad_verifier
        )
        return {
            "grade_sandbox_id": f"grader-{self.number}",
            "verifier_image_ref": configuration()["execution"]["verifier_image"],
            "reward": reward,
            "rawreward": b"0.999999999999999999999999"
            if self.bad_raw
            else str(reward).encode(),
            "cleanup_verified": True,
            "error": None,
            "transcript": [
                {
                    "exit_code": 0,
                    "output": f"Native reward: {reward}\n",
                    "output_truncated": False,
                }
            ],
        }

    def leakage_probes(self):
        return [
            {"name": name, "passed": True, "error": None}
            for name in (
                "private_paths_absent",
                "fresh_state_absent",
                "canary_not_in_transcript",
            )
        ]


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    for name in ("tests", "solution", "environment"):
        (root / name).mkdir(parents=True)
    (root / "task.toml").write_text('version = "1.0"\n')
    (root / "instruction.md").write_text("Write exactly 42 into /app/answer.txt.\n")
    (root / "tests/test.sh").write_text("#!/bin/bash\nprintf 1\n")
    (root / "solution/solve.sh").write_text("printf 42 > /app/answer.txt\n")
    (root / "environment/Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\n"
    )
    return root


def test_complete_collection_is_bound_and_keeps_model_difficulty(task, tmp_path):
    backend = Backend()
    output = tmp_path / "audit"
    report = run_coding_validation(task, configuration(), output, backend=backend)
    assert report["accepted"]
    assert not report["rfc008_complete"]
    assert not report["trainability_established"]
    assert backend.target_calls == 2
    assert report["checks"]["target_difficulty"]["solve_rate"] == 0.5
    evidence = json.loads((output / "evidence.json").read_text())
    assert {r["origin"] for r in evidence["references"]} == {"provided", "generated"}
    assert len(evidence["generation"]) == 1
    for row in evidence["trials"]:
        artifact = row["grader_transcript"]
        assert artifact["path"].endswith("/grader-transcript.json")
        data = (output / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert json.loads(data)[0]["output"] == f"Native reward: {row['reward']}\n"
    assert evidence["trials"][0]["reward"] == 0
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_bad_verifier_blocks_target_inference(task, tmp_path):
    backend = Backend()
    backend.bad_verifier = True
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert not report["accepted"]
    assert report["checks"]["verifier_rejection"]["status"] == "fail"
    assert backend.target_calls == 0


def test_failed_clean_candidate_excluded_without_overwriting_frozen_verifier(
    task, tmp_path
):
    backend = Backend()
    backend.generated = "echo wrong"
    before = (task / "tests/test.sh").read_bytes()
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert report["accepted"]
    assert len(report["checks"]["references"]["excluded"]) == 1
    assert (task / "tests/test.sh").read_bytes() == before


def test_raw_reward_precision_disagreement_cannot_pass(task, tmp_path):
    backend = Backend()
    backend.bad_raw = True
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert not report["accepted"]
    assert backend.target_calls == 0
    assert report["checks"]["reliability"]["status"] == "error"


def test_target_infra_failure_preserves_missing_not_zero(task, tmp_path):
    backend = Backend()
    backend.target_error = True
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    check = report["checks"]["target_difficulty"]
    assert not report["accepted"]
    assert check["completed"] == 0
    assert check["solve_rate"] is None


def test_changed_source_cannot_be_admitted(task, tmp_path):
    backend = Backend()
    backend.mutate = lambda: (task / "instruction.md").write_text("Changed task")
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert not report["accepted"]
    evidence = json.loads((tmp_path / "audit/evidence.json").read_text())
    assert evidence["collection_error"] == "ValueError"


def test_credentials_in_transcript_are_omitted_and_block_acceptance(
    task, tmp_path, monkeypatch
):
    token = "test-secret-not-a-real-token"
    monkeypatch.setenv("HF_SANDBOX_TOKEN", token)
    backend = Backend()
    backend.transcript = [{"content": token}]
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert not report["accepted"]
    for path in (tmp_path / "audit").rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes()


@pytest.mark.parametrize(
    "path,value", [("agent_image", "python:latest"), ("api_key", "secret")]
)
def test_mutable_images_and_embedded_credentials_rejected(path, value):
    config = configuration()
    config["execution"][path] = value
    with pytest.raises(ValueError):
        normalize_coding_config(config)


def test_budget_is_bounded_before_any_remote_execution(task, tmp_path):
    config = configuration()
    config["cloud"]["max_sandboxes"] = 1
    with pytest.raises(ValueError, match="budget"):
        run_coding_validation(task, config, tmp_path / "audit", backend=Backend())
    assert not (tmp_path / "audit").exists()
    assert (
        coding_budget(normalize_coding_config(configuration()))["inference_calls_max"]
        == 33
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_cli_dispatch_uses_example_runner_and_retains_evidence(
    task, tmp_path, monkeypatch, capsys, accepted
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration()))
    original = run_coding_validation
    backend = Backend()
    backend.bad_verifier = not accepted
    monkeypatch.setattr(
        "examples.coding_audit.runner.run_coding_validation",
        lambda task, config, output: original(task, config, output, backend=backend),
    )
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "examples.coding_audit",
            "--task",
            str(task),
            "--config",
            str(config),
            "--output",
            str(output),
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_module("examples.coding_audit", run_name="__main__")
    assert result.value.code == (0 if accepted else 1)
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"] is accepted
    assert json.loads(capsys.readouterr().out) == report
    assert (output / "evidence.json").is_file()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_cli_requires_persisted_evidence(task, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "examples.coding_audit",
            "--task",
            str(task),
            "--config",
            str(tmp_path / "missing.json"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_module("examples.coding_audit", run_name="__main__")
    assert result.value.code == 2
    assert "--output" in capsys.readouterr().err


def test_cli_input_failure_exits_two_without_starting_backend(
    task, tmp_path, monkeypatch, capsys
):
    from examples.coding_audit.__main__ import main

    monkeypatch.setattr(
        "examples.coding_audit.__main__.run_coding_validation",
        lambda *args: pytest.fail("input failure reached backend"),
    )
    assert (
        main(
            [
                "--task",
                str(task),
                "--config",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "audit"),
            ]
        )
        == 2
    )
    assert "input failure (FileNotFoundError)" in capsys.readouterr().err


def test_exact_json_threshold_cannot_round_into_a_pass(task, tmp_path):
    from decimal import Decimal

    from openenv.validation.runtime.json_values import loads

    config = loads(
        json.dumps(configuration()).replace(
            '"reward_max": 1,', '"reward_max": 1.00000000000000000001,'
        )
    )
    normalized = normalize_coding_config(config)
    assert normalized["validation"]["reward_max"] == Decimal("1.00000000000000000001")
    report = run_coding_validation(task, config, tmp_path / "audit", backend=Backend())
    assert not report["accepted"]
    saved = loads((tmp_path / "audit/config.json").read_text())
    assert saved["validation"]["reward_max"] == normalized["validation"]["reward_max"]


@pytest.mark.parametrize("value", ["", "missing"])
def test_execution_error_status_cannot_be_overwritten(task, tmp_path, value):
    class BrokenStatus(Backend):
        def execute_script(self, script, role):
            row = super().execute_script(script, role)
            if value == "missing":
                del row["error"]
            else:
                row["error"] = value
            return row

    backend = BrokenStatus()
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=backend
    )
    assert not report["accepted"]
    assert backend.target_calls == 0
    rows = json.loads((tmp_path / "audit/evidence.json").read_text())["trials"]
    assert all(row["error"] is not None for row in rows)


def test_verifier_identity_comes_from_verified_snapshot(task, tmp_path, monkeypatch):
    from examples.coding_audit import runner as coding_runner

    original = coding_runner.source_digest
    expected = original(task / "tests")

    def raced(path):
        if path == task / "tests":
            raise AssertionError(
                "mutable original verifier must not be hashed independently"
            )
        return original(path)

    monkeypatch.setattr(coding_runner, "source_digest", raced)
    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=Backend()
    )
    assert report["accepted"]
    assert report["verifier_digest"] == expected


def test_credential_metadata_is_failed_and_report_is_sanitized(
    task, tmp_path, monkeypatch
):
    secret = "dummy-execution-credential-for-regression"
    monkeypatch.setenv("HF_SANDBOX_TOKEN", secret)

    class DiagnosticBackend(Backend):
        def leakage_probes(self):
            return super().leakage_probes() + [
                {
                    "name": "diagnostic:" + secret,
                    "passed": True,
                    "error": None,
                    secret: "key",
                }
            ]

    report = run_coding_validation(
        task, configuration(), tmp_path / "audit", backend=DiagnosticBackend()
    )
    assert not report["accepted"]
    assert secret not in json.dumps(report)
    for path in (tmp_path / "audit").rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_credential_in_config_is_rejected_before_execution(task, tmp_path, monkeypatch):
    secret = "dummy-config-credential-for-regression"
    monkeypatch.setenv("HF_SANDBOX_TOKEN", secret)
    config = configuration()
    config["negative_cases"][0]["id"] = secret
    config["validation"]["required_negative_cases"][0] = secret
    with pytest.raises(ValueError, match="credentials"):
        run_coding_validation(task, config, tmp_path / "audit", backend=Backend())


def test_in_task_output_rejected_before_inference(task):
    backend = Backend()
    with pytest.raises(ValueError, match="outside"):
        run_coding_validation(
            task, configuration(), task / "audit.artifacts", backend=backend
        )
    assert backend.target_calls == 0
    assert not (task / "audit.artifacts").exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "environment/source/api.py",
        "instruction.md",
        "tests/test.sh",
        "solution/solve.sh",
    ],
)
def test_snapshot_rejects_hardlinked_artifacts_before_inference(
    task, tmp_path, monkeypatch, artifact
):
    import huggingface_hub
    from examples.coding_audit import hf

    m = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("test_hf.py"))))
    api, inference = m.API(), m.InferenceFactory()
    factory = m.Factory(api)
    api.model_info = lambda *args, **kwargs: SimpleNamespace(
        sha=m.MODEL["revision"],
        inference_provider_mapping={
            "deepinfra": {"provider_id": m.MODEL["model_id"], "status": "live"}
        },
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(hf, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(hf.Sandbox, "create", factory)
    monkeypatch.setattr(hf, "InferenceClient", inference)
    for name in ("HF_TOKEN", "HF_INFERENCE_TOKEN", "HF_SANDBOX_TOKEN"):
        monkeypatch.setenv(name, "synthetic-test-" + name)
    context_path = "environment/source/api.py"
    source = task / context_path
    source.parent.mkdir(parents=True)
    if artifact == context_path:
        source.hardlink_to(task / "solution/solve.sh")
    else:
        source.write_text("def public_answer(): return 42\n")
        (tmp_path / "linked-artifact").hardlink_to(task / artifact)
    assert (task / artifact).stat().st_nlink == 2
    config = configuration()
    config["generator"] = {**config["target"], "context_paths": [context_path]}
    config["execution"].update(
        agent_image=m.AGENT, verifier_image=m.VERIFIER, namespace="owner"
    )
    inference.responses = [m.response("true")]
    output = tmp_path / "audit"
    report = run_coding_validation(task, config, output)
    assert not inference.requests and not inference.kwargs
    assert not factory.created
    assert not report["accepted"]
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["collection_error"] == "HFBackendError"


@pytest.mark.parametrize("with_context", [False, True])
def test_real_backend_contract_reaches_difficulty_with_sdk_fakes(
    task, tmp_path, monkeypatch, with_context
):
    """Join the actual collector/backend/scorer, not just each seam in isolation."""
    from examples.coding_audit.hf import HFTaskBackend

    m = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("test_hf.py"))))
    api, inference = m.API(), m.InferenceFactory()
    factory = m.Factory(api)
    original = m.FakeSandbox.run

    def run(sandbox, argv, **kwargs):
        if sandbox.image == m.VERIFIER:
            sandbox.factory.reward = (
                b"1" if sandbox.files.data.get("/app/answer.txt") == b"42\n" else b"0"
            )
        else:
            command = " ".join(argv)
            sandbox.factory.submitted = (
                b"42\n" if "42" in command else b"41\n" if "41" in command else None
            )
        return original(sandbox, argv, **kwargs)

    monkeypatch.setattr(m.FakeSandbox, "run", run)
    config = configuration()
    context_path = "environment/source/pkg/api.py"
    context = b"def public_answer():\n    return 42\n"
    if with_context:
        source = task / context_path
        source.parent.mkdir(parents=True)
        source.write_bytes(context)
        config.setdefault("generator", dict(config["target"]))["context_paths"] = [
            context_path
        ]
    config["execution"].update(
        agent_image=m.AGENT, verifier_image=m.VERIFIER, namespace="owner"
    )
    inference.responses = [
        m.response('printf "42\\n" > /app/answer.txt'),
        m.response(None, calls=[m.terminal('printf "41\\n" > /app/answer.txt')]),
        m.response("done"),
        m.response(None, calls=[m.terminal('printf "42\\n" > /app/answer.txt')]),
        m.response("done"),
    ]
    backend = HFTaskBackend(
        task.resolve(),
        config["execution"],
        token="execution-only-test-token",
        inference_token="inference-only-test-token",
        api=api,
        sandbox_factory=factory,
        inference_factory=inference,
    )
    output = tmp_path / "audit"
    report = run_coding_validation(task, config, output, backend=backend)
    assert report["accepted"], report
    assert report["checks"]["target_difficulty"]["solve_rate"] == 0.5
    assert len(inference.requests) == 5
    evidence_path = output / "evidence.json"
    generated = json.loads(evidence_path.read_text())["generation"][0]
    if with_context:
        assert generated["candidate_context"] == [
            {
                "path": context_path,
                "runtime_path": "pkg/api.py",
                "size": len(context),
                "sha256": hashlib.sha256(context).hexdigest(),
            }
        ]
        assert context.decode() in inference.requests[0]["messages"][1]["content"]
    else:
        assert "candidate_context" not in generated
    assert all(
        request["messages"][1]["content"] == (task / "instruction.md").read_text()
        for request in inference.requests[1:]
    )
    checksums = dict(
        line.split("  ", 1)[::-1]
        for line in (output / "SHA256SUMS").read_text().splitlines()
    )
    assert (
        checksums["evidence.json"]
        == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    assert factory.created
    assert all(stage == "CANCELED" for stage in api.stages.values())


def test_grader_exception_keeps_sanitized_sha_bound_logs(task, tmp_path, monkeypatch):
    from examples.coding_audit.hf import HFTaskBackend

    m = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("test_hf.py"))))
    api, inference = m.API(), m.InferenceFactory()
    factory = m.Factory(api)
    original = m.FakeSandbox.run
    secret = "execution-only-test-token"

    def run(sandbox, argv, **kwargs):
        if sandbox.image == m.VERIFIER:
            kwargs["on_stdout"]("APT: partial output 雪 " + secret)
            raise RuntimeError("grader RPC failed: " + secret)
        return original(sandbox, argv, **kwargs)

    monkeypatch.setattr(m.FakeSandbox, "run", run)
    config = configuration()
    config["execution"].update(
        agent_image=m.AGENT, verifier_image=m.VERIFIER, namespace="owner"
    )
    inference.responses = [m.response('printf "42\\n" > /app/answer.txt')]
    backend = HFTaskBackend(
        task.resolve(),
        config["execution"],
        token=secret,
        inference_token="inference-only-test-token",
        api=api,
        sandbox_factory=factory,
        inference_factory=inference,
    )
    output = tmp_path / "audit"
    report = run_coding_validation(task, config, output, backend=backend)
    assert not report["accepted"]
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["target_not_run_reason"]
    for row in evidence["trials"]:
        assert row["error"] == "HF backend operation failed (RuntimeError)"
        assert row["reward"] is None and row["cleanup_verified"]
        artifact = row["grader_transcript"]
        data = (output / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        trace = json.loads(data)[0]
        assert trace["output"] == "APT: partial output 雪 [REDACTED]"
        assert trace["exit_code"] is None and not trace["output_truncated"]
    for path in output.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert all(stage == "CANCELED" for stage in api.stages.values())


@pytest.mark.parametrize("callbacks", ["single", "separate", "split_secret"])
@pytest.mark.parametrize("intersecting", [False, True])
def test_grader_overflow_discards_output_in_sha_bound_artifact(
    task, tmp_path, monkeypatch, callbacks, intersecting
):
    from examples.coding_audit.hf import HFTaskBackend

    m = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("test_hf.py"))))
    api, inference = m.API(), m.InferenceFactory()
    factory = m.Factory(api)
    original = m.FakeSandbox.run
    execution_secret = (
        "hf_SYNTHETIC_EXECUTION_TOKEN_REVIEW_ONLYh"
        if intersecting
        else "hf_SYNTHETIC_EXECUTION_TOKEN_FOR_REVIEW_ONLY"
    )
    inference_secret = "hf_SYNTHETIC_INFERENCE_TOKEN_REVIEW_ONLY"
    secret = inference_secret if intersecting else execution_secret
    cap, retained = 1048576, 28
    prefix = "雪" + "A" * (cap - len("雪".encode()) - retained)
    if intersecting:
        prefix = prefix[: -len(execution_secret[:-1])] + execution_secret[:-1]
        assert execution_secret in prefix + secret

    def run(sandbox, argv, **kwargs):
        if sandbox.image == m.VERIFIER:
            if callbacks == "single":
                kwargs["on_stdout"](prefix + secret)
            else:
                split = 7 if callbacks == "split_secret" else 0
                kwargs["on_stdout"](prefix + secret[:split])
                kwargs["on_stderr"](secret[split:])
            pytest.fail("grader output overflow was not rejected")
        return original(sandbox, argv, **kwargs)

    monkeypatch.setattr(m.FakeSandbox, "run", run)
    config = configuration()
    config["execution"].update(
        agent_image=m.AGENT,
        verifier_image=m.VERIFIER,
        namespace="owner",
        max_tool_output_bytes=cap,
    )
    inference.responses = [m.response('printf "42\\\\n" > /app/answer.txt')]
    backend = HFTaskBackend(
        task.resolve(),
        config["execution"],
        token=execution_secret,
        inference_token=inference_secret,
        api=api,
        sandbox_factory=factory,
        inference_factory=inference,
    )
    output = tmp_path / "audit"
    report = run_coding_validation(task, config, output, backend=backend)
    assert not report["accepted"]
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["target_not_run_reason"] and evidence["trials"]
    for row in evidence["trials"]:
        assert row["error"] == "terminal output exceeded byte limit"
        assert row["reward"] is None and row["cleanup_verified"]
        assert "raw_reward" not in row
        artifact = row["grader_transcript"]
        data = (output / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert json.loads(data) == [
            {
                "exit_code": None,
                "output": "",
                "output_truncated": True,
            }
        ]
        assert len(json.loads(data)[0]["output"].encode()) <= cap
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        data = (output / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert secret[:retained].encode() not in data
        assert execution_secret[:-1].encode() not in data
    assert api.stages and all(stage == "CANCELED" for stage in api.stages.values())


@pytest.mark.parametrize("value", [False, True])
def test_explicit_candidate_thinking_control_preserves_target(value):
    raw = configuration()
    raw["generator"] = {**raw["target"], "enable_thinking": value}
    normalized = normalize_coding_config(raw)
    assert normalized["generator"]["enable_thinking"] is value
    assert "enable_thinking" not in normalized["target"]


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_thinking_configuration_rejects_non_boolean(value):
    raw = configuration()
    raw["generator"] = {**raw["target"], "enable_thinking": value}
    with pytest.raises(ValueError, match="enable_thinking"):
        normalize_coding_config(raw)


@pytest.mark.parametrize("count", [0, 1, 32])
def test_generator_context_is_optional_explicit_and_preserves_target(count):
    raw = configuration()
    paths = [f"environment/source/pkg/api-{index}.py" for index in range(count)]
    raw["generator"] = {**raw["target"], "context_paths": paths}
    normalized = normalize_coding_config(raw)
    assert normalized["generator"]["context_paths"] == paths
    assert "context_paths" not in normalized["target"]
    assert "context_paths" not in normalize_coding_config(configuration())["generator"]


@pytest.mark.parametrize(
    "paths",
    [
        "environment/source/api.py",
        None,
        [1],
        ["environment/source/api.py"] * 2,
        [f"environment/source/{index}.py" for index in range(33)],
        ["environment/source"],
        ["environment/source/api.py/"],
        ["/environment/source/api.py"],
        ["environment/source/../tests/test.sh"],
        ["environment/source//api.py"],
        ["environment/source/./api.py"],
        ["environment/source/api\x00.py"],
        ["environment/source/a\nb.py"],
        ["environment/source/a\\b.py"],
        ["environment/source/a:b.py"],
        ["tests/test.sh"],
        ["solution/solve.sh"],
        ["validation/spec.json"],
        ["environment/source/tests/test.py"],
        ["environment/source/solution/solve.sh"],
        ["environment/source/validation/spec.json"],
        ["environment/source/.hidden/api.py"],
        ["environment/source/.env"],
        ["environment/source/secrets/token"],
        ["environment/source/reference/solve.py"],
        ["environment/source/spec/task.md"],
    ],
)
def test_generator_context_rejects_unsafe_paths(paths):
    raw = configuration()
    raw["generator"] = {**raw["target"], "context_paths": paths}
    with pytest.raises(ValueError, match="context_paths"):
        normalize_coding_config(raw)


def test_target_cannot_declare_context_paths():
    raw = configuration()
    raw["target"]["context_paths"] = []
    with pytest.raises(ValueError, match="target.context_paths"):
        normalize_coding_config(raw)
