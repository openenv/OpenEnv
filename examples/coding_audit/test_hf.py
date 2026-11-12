"""Offline SDK-shaped tests for the HF coding task execution boundary."""

import hashlib
import json
import os
import signal
import time
from pathlib import PurePosixPath
from types import SimpleNamespace

import httpx
import pytest
from examples.coding_audit.hf import (
    _deadline,
    HFBackendError,
    HFTaskBackend,
    submission_digest,
)
from huggingface_hub import hf_api, InferenceClient, Sandbox
from huggingface_hub.errors import SandboxError
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
)
from huggingface_hub.inference._providers.deepinfra import DeepInfraConversationalTask
from openenv.validation.runner import source_digest

AGENT = "python@sha256:" + "a" * 64
VERIFIER = "python@sha256:" + "b" * 64
MODEL = {
    "model_id": "Qwen/Qwen3.8-27B",
    "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "provider": "deepinfra",
    "provider_model_id": "Qwen/Qwen3.8-27B",
}


def test_real_sandbox_sdk_serializes_short_name_for_long_image(harness, monkeypatch):
    captured = {}
    image = (
        "registry.hf.space/example-organization-a-long-starter-image-name"
        "@sha256:" + "a" * 64
    )

    def capture(request):
        assert request.method == "POST" and request.url.host == "huggingface.co"
        captured.update(json.loads(request.content))
        return httpx.Response(400, json={"error": "offline request capture"})

    harness.backend.execution["agent_image"] = image
    harness.backend.sandbox_factory = Sandbox.create
    with httpx.Client(transport=httpx.MockTransport(capture)) as client:
        monkeypatch.setattr(hf_api, "get_session", lambda: client)
        record = harness.backend.execute_script(":", "nop")
    assert "BadRequestError" in record["error"]
    assert captured["dockerImage"] == image
    assert captured["labels"]["name"] == (
        "openenv-coding-" + captured["labels"]["openenv-coding-attempt"]
    )
    assert 0 < len(captured["labels"]["name"]) <= 100


def response(
    content="done", *, calls=None, model=None, fingerprint="fp-123", usage=None
):
    return ChatCompletionOutput.parse_obj_as_instance(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "object": "chat.completion",
            "model": model or MODEL["model_id"],
            "system_fingerprint": fingerprint,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": calls,
                    },
                }
            ],
            "usage": usage
            or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def terminal(command="printf answer > /app/answer.txt", identifier="call-1"):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": command})},
    }


def stream_event(*, delta=None, finish=None, choices=None, usage=None, **identity):
    return ChatCompletionStreamOutput.parse_obj_as_instance(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": MODEL["model_id"],
            "system_fingerprint": "fp-123",
            **identity,
            "choices": choices
            if choices is not None
            else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            "usage": usage,
        }
    )


def stream_response(reply):
    yield stream_event(
        delta=reply["choices"][0]["message"],
        finish=reply["choices"][0]["finish_reason"],
        model=reply["model"],
        system_fingerprint=reply["system_fingerprint"],
    )
    yield stream_event(
        choices=[],
        usage=reply["usage"],
        model=reply["model"],
        system_fingerprint=reply["system_fingerprint"],
    )


class Files:
    MAX_READ_BYTES = 512 * 1048576

    def __init__(self, initial=None):
        self.data = dict(initial or {})
        self.types = {}
        self.directories = {"/", "/tmp"}
        self.writes = []
        self.reads = []
        self.grow_on_read = False

    def stat(self, path):
        if path in self.directories:
            return SimpleNamespace(type="dir", size=0)
        if path not in self.data:
            raise SandboxError("missing", status_code=404)
        return SimpleNamespace(
            type=self.types.get(path, "file"), size=len(self.data[path])
        )

    def exists(self, path):
        try:
            self.stat(path)
            return True
        except SandboxError as exc:
            if exc.status_code == 404:
                return False
            raise

    def read(self, path):
        self.reads.append(path)
        data = self.data[path]
        if self.grow_on_read:
            data = data + b"x" * self.MAX_READ_BYTES
        if len(data) > self.MAX_READ_BYTES:
            raise SandboxError("too large")
        return data

    def write(self, path, data, mode=None):
        assert isinstance(data, (str, bytes))
        data = data.encode() if isinstance(data, str) else data
        self.data[path] = data
        self.writes.append((path, data, mode))

    def mkdir(self, path):
        self.directories.add(path)

    def delete(self, path):
        del self.data[path]


class FakeSandbox:
    def __init__(self, identity, image, api, factory):
        self.id = identity
        self.image = image
        self.files = Files(factory.initial)
        self.api = api
        self.factory = factory
        self.commands = []
        self.killed = False

    def run(self, argv, **kwargs):
        self.commands.append((argv, kwargs))
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is False
        assert kwargs["cwd"] in self.files.directories
        assert 0 < kwargs["timeout"] <= 120
        assert "env" not in kwargs
        if self.factory.output:
            kwargs["on_stdout"](self.factory.output)
        if self.image == VERIFIER:
            self.files.data["/logs/verifier/reward.txt"] = self.factory.reward
        elif self.factory.submitted is not None:
            self.files.data["/app/answer.txt"] = self.factory.submitted
        return SimpleNamespace(
            exit_code=self.factory.exit_code,
            stdout="",
            stderr="",
            timed_out=self.factory.timed_out,
        )

    def kill(self):
        self.killed = True
        if not self.api.ignore_cancel:
            self.api.stages[self.id] = "CANCELED"


class API:
    def __init__(self):
        self.images = {}
        self.stages = {}
        self.cancelled = []
        self.inspections = []
        self.ignore_cancel = False
        self.cancel_error = False
        self.inspect_error_after_cancel = False
        self.owner = "owner"
        self.image_override = None
        self.labels = {}

    def inspect_job(self, *, job_id, namespace=None, token=None):
        self.inspections.append((job_id, namespace, token))
        if self.inspect_error_after_cancel and job_id in self.cancelled:
            raise RuntimeError("inspection contains an execution-token-secret")
        return SimpleNamespace(
            id=job_id,
            docker_image=self.image_override or self.images[job_id],
            status=SimpleNamespace(stage=self.stages[job_id]),
            owner=SimpleNamespace(name=self.owner),
            labels=self.labels.get(job_id, {}),
        )

    def list_jobs(self, *, labels=None, timeout=None, namespace=None, token=None):
        return (
            self.inspect_job(job_id=job_id, namespace=namespace, token=token)
            for job_id in self.images
            if all(
                self.labels.get(job_id, {}).get(key) == value
                for key, value in (labels or {}).items()
            )
        )

    def cancel_job(self, *, job_id, namespace=None, token=None):
        self.cancelled.append(job_id)
        if self.cancel_error:
            raise RuntimeError("cancel failed")
        if not self.ignore_cancel:
            self.stages[job_id] = "CANCELED"


class Factory:
    def __init__(self, api):
        self.api = api
        self.created = []
        self.kwargs = []
        self.initial = {}
        self.output = ""
        self.submitted = b"answer"
        self.reward = b"1.0\n"
        self.exit_code = 0
        self.timed_out = False
        self.fail_after_create = False

    def __call__(self, **kwargs):
        identity = f"job-{len(self.created)}"
        self.kwargs.append(kwargs)
        self.api.images[identity] = kwargs["image"]
        self.api.stages[identity] = "RUNNING"
        self.api.labels[identity] = kwargs.get("labels", {})
        sandbox = FakeSandbox(identity, kwargs["image"], self.api, self)
        self.created.append(sandbox)
        if self.fail_after_create:
            raise RuntimeError("readiness failed after allocation")
        return sandbox


class InferenceFactory:
    def __init__(self):
        self.kwargs = []
        self.requests = []
        self.responses = []
        self.closed = 0

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return self

    def chat_completion(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        reply = self.responses.pop(0)
        if kwargs.get("stream") and isinstance(reply, ChatCompletionOutput):
            return stream_response(reply)
        return reply

    def close(self):
        self.closed += 1


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HF_INFERENCE_TOKEN", "HF_SANDBOX_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path.resolve()
    (root / "instruction.md").write_text("Write /app/answer.txt containing answer.")
    (root / "tests").mkdir()
    (root / "tests/test.sh").write_bytes(
        b"#!/bin/bash\nprintf 1 > /logs/verifier/reward.txt\n"
    )
    (root / "solution").mkdir()
    (root / "solution/solve.sh").write_text("PRIVATE_SOLUTION_SENTINEL")
    api = API()
    factory = Factory(api)
    inference = InferenceFactory()
    execution = {
        "agent_image": AGENT,
        "verifier_image": VERIFIER,
        "submission_paths": ["/app/answer.txt"],
        "namespace": "owner",
    }
    backend = HFTaskBackend(
        root,
        execution,
        token="execution-token-secret",
        inference_token="inference-token-secret",
        api=api,
        sandbox_factory=factory,
        inference_factory=inference,
        canary="PRIVATE_CANARY_SENTINEL",
    )
    return SimpleNamespace(
        root=root,
        backend=backend,
        api=api,
        factory=factory,
        inference=inference,
        execution=execution,
    )


def test_execution_grade_are_fresh_and_credentials_stay_in_controller(harness):
    h = harness
    submission = h.backend.execute_script(
        "printf answer > /app/answer.txt", "reference"
    )
    grade = h.backend.grade(submission)
    assert submission["error"] is None and submission["cleanup_verified"]
    assert grade["error"] is None and grade["cleanup_verified"]
    assert grade["reward"] == 1 and grade["rawreward"] == b"1.0\n"
    assert submission["sandbox_id"] != grade["grade_sandbox_id"]
    assert [item["image"] for item in h.factory.kwargs] == [AGENT, VERIFIER]
    for kwargs in h.factory.kwargs:
        assert kwargs["forward_hf_token"] is False
        assert kwargs["token"] == "execution-token-secret"
        assert (
            "env" not in kwargs and "secrets" not in kwargs and "volumes" not in kwargs
        )
    solver, verifier = h.factory.created
    assert not any(
        path.startswith(("/tests", "/solution")) for path in solver.files.data
    )
    assert (
        verifier.files.data["/tests/test.sh"] == (h.root / "tests/test.sh").read_bytes()
    )
    assert verifier.files.data["/app/answer.txt"] == b"answer"
    assert not any("solution" in path for path in verifier.files.data)
    assert b"PRIVATE_CANARY_SENTINEL" in verifier.files.data.values()
    assert submission["image_reference_kind"] == "control_plane_immutable_reference"
    assert len(submission["limitations"]) >= 4
    assert "token-secret" not in repr(submission) + repr(grade)
    assert h.api.cancelled == ["job-0", "job-1"]


def test_absent_submission_is_empty_and_can_legitimately_grade_zero(harness):
    h = harness
    h.factory.submitted = None
    submission = h.backend.execute_script(":", "nop")
    assert submission["files"] == {} and submission["error"] is None
    h.factory.reward = b"0"
    grade = h.backend.grade(submission)
    assert grade["error"] is None and grade["reward"] == 0


def test_native_verifier_reward_directory_exists_before_execution(harness, monkeypatch):
    h = harness
    submission = h.backend.execute_script(":", "nop")
    original_run = FakeSandbox.run

    def native_run(sandbox, argv, **kwargs):
        if sandbox.image == VERIFIER:
            assert "/logs/verifier" in sandbox.files.directories
        return original_run(sandbox, argv, **kwargs)

    monkeypatch.setattr(FakeSandbox, "run", native_run)
    grade = h.backend.grade(submission)
    assert grade["error"] is None
    assert grade["cleanup_verified"]


@pytest.mark.parametrize("phase", ["script", "grade"])
def test_nonzero_is_error_not_synthesized_reward(harness, phase):
    h = harness
    submission = h.backend.execute_script(":", "reference")
    h.factory.exit_code = 3
    record = (
        h.backend.grade(submission)
        if phase == "grade"
        else h.backend.execute_script("false", "target")
    )
    assert "nonzero" in record["error"]
    assert record.get("reward") is None and record["cleanup_verified"]


@pytest.mark.parametrize("failure", ["overflow", "timeout", "nonzero", "exception"])
def test_failed_grader_keeps_logs_only_without_overflow(harness, monkeypatch, failure):
    h = harness
    submission = h.backend.execute_script(":", "reference")
    h.backend.max_output = 17
    output = "setup\n雪💥雪💥" if failure == "overflow" else "setup\nfailed\n"

    def failed_run(sandbox, argv, **kwargs):
        kwargs["on_stdout"]("setup\n")
        kwargs["on_stderr"](output.removeprefix("setup\n"))
        if failure == "exception":
            raise RuntimeError("RPC failed after output")
        return SimpleNamespace(
            exit_code=3 if failure == "nonzero" else 0,
            timed_out=failure == "timeout",
        )

    monkeypatch.setattr(FakeSandbox, "run", failed_run)
    grade = h.backend.grade(submission)
    assert grade["error"] and grade["cleanup_verified"]
    assert grade["reward"] is None and grade["rawreward"] == b""
    trace = grade["transcript"][0]
    assert trace["output"] == ("" if failure == "overflow" else output)
    assert len(trace["output"].encode()) <= 17
    assert trace["output_truncated"] is (failure == "overflow")
    assert trace["exit_code"] == (
        3 if failure == "nonzero" else 0 if failure == "timeout" else None
    )


@pytest.mark.parametrize("cap", [17, 18])
def test_grader_output_stays_bounded_after_redaction(harness, cap):
    h = harness
    submission = h.backend.execute_script(":", "nop")
    backend = HFTaskBackend(
        h.root,
        dict(h.execution, max_tool_output_bytes=cap),
        token="secret",
        api=h.api,
        sandbox_factory=h.factory,
    )
    h.factory.output = "雪secret💥\n"
    h.factory.reward = b"0"
    grade = backend.grade(submission)
    assert grade["error"] is None and grade["reward"] == 0
    assert grade["transcript"] == [
        {
            "exit_code": 0,
            "output": "" if cap == 17 else "雪[REDACTED]💥\n",
            "output_truncated": cap == 17,
        }
    ]
    assert len(grade["transcript"][0]["output"].encode()) <= cap
    assert grade["cleanup_verified"]


@pytest.mark.parametrize("callbacks", ["single", "separate", "split_secret"])
@pytest.mark.parametrize(
    "credential", ["execution", "inference", "overlapping", "repeated", "intersecting"]
)
@pytest.mark.parametrize("retained", [1, 28])
def test_grader_overflow_discards_output(
    harness, monkeypatch, callbacks, credential, retained
):
    h = harness
    submission = h.backend.execute_script(":", "reference")
    credentials = {
        "execution": "hf_SYNTHETIC_EXECUTION_TOKEN_FOR_REVIEW_ONLY",
        "inference": "hf_SYNTHETIC_INFERENCE_TOKEN_FOR_REVIEW_ONLY",
    }
    if credential == "overlapping":
        credentials["execution"] = "SYNTHETIC_EXECUTION_TOKEN"
        credentials["inference"] = "hf_SYNTHETIC_EXECUTION_TOKEN_FOR_REVIEW_ONLY"
    elif credential == "repeated":
        credentials["execution"] = "hf_" + "X" * 22 + "hf_"
        credential = "execution"
    elif credential == "intersecting":
        credentials["execution"] = "hf_SYNTHETIC_EXECUTION_TOKEN_REVIEW_ONLYh"
        credentials["inference"] = "hf_SYNTHETIC_INFERENCE_TOKEN_REVIEW_ONLY"
    cap = 1048576
    backend = HFTaskBackend(
        h.root,
        dict(h.execution, max_tool_output_bytes=cap),
        token=credentials["execution"],
        inference_token=credentials["inference"],
        api=h.api,
        sandbox_factory=h.factory,
    )
    secret = credentials[
        "inference" if credential in ("overlapping", "intersecting") else credential
    ]
    prefix = "雪" + "A" * (cap - len("雪".encode()) - retained)
    if credential == "intersecting":
        complete = credentials["execution"]
        prefix = prefix[: -len(complete[:-1])] + complete[:-1]
        assert complete in prefix + secret

    def overflow(sandbox, argv, **kwargs):
        if callbacks == "single":
            kwargs["on_stdout"](prefix + secret + "\n")
        else:
            split = min(7, retained) if callbacks == "split_secret" else 0
            kwargs["on_stdout"](prefix + secret[:split])
            kwargs["on_stderr"](secret[split:] + "\n")
        pytest.fail("grader output overflow was not rejected")

    monkeypatch.setattr(FakeSandbox, "run", overflow)
    grade = backend.grade(submission)
    assert grade["error"] == "terminal output exceeded byte limit"
    assert grade["reward"] is None and grade["rawreward"] == b""
    assert grade["cleanup_verified"]
    trace = grade["transcript"][0]
    assert trace == {
        "exit_code": None,
        "output": "",
        "output_truncated": True,
    }
    assert len(trace["output"].encode()) <= cap
    assert secret[:retained] not in trace["output"]
    assert h.api.stages and all(stage == "CANCELED" for stage in h.api.stages.values())


def test_image_mismatch_cancels_before_any_workload_command(harness):
    h = harness
    h.api.image_override = VERIFIER
    record = h.backend.execute_script("dangerous", "reference")
    assert "image" in record["error"] and record["cleanup_verified"]
    assert not h.factory.created[0].commands
    assert h.api.cancelled == ["job-0"]


def test_rpc_ready_sandbox_waits_for_running_control_plane(harness, monkeypatch):
    h = harness
    inspect = h.api.inspect_job
    observed = []

    def lagged_inspect(**kwargs):
        job = inspect(**kwargs)
        if not h.api.cancelled:
            job.status.stage = "SCHEDULING" if len(observed) < 25 else "RUNNING"
            observed.append(job.status.stage)
        return job

    monkeypatch.setattr(h.api, "inspect_job", lagged_inspect)
    monkeypatch.setattr("examples.coding_audit.hf.time.sleep", lambda _: None)
    record = h.backend.execute_script(":", "reference")
    assert record["error"] is None
    assert observed == ["SCHEDULING"] * 25 + ["RUNNING"]
    assert record["cleanup_verified"]
    assert h.factory.created[0].commands


def test_scheduling_reconciliation_stops_at_caller_deadline(harness):
    h = harness
    inspect = h.api.inspect_job

    def never_running(**kwargs):
        job = inspect(**kwargs)
        if not h.api.cancelled:
            job.status.stage = "SCHEDULING"
        return job

    h.api.inspect_job = never_running
    h.backend.timeout = 0.02
    started = time.monotonic()
    record = h.backend.execute_script("dangerous", "reference")
    assert time.monotonic() - started < 0.5
    assert "deadline" in record["error"] and record["cleanup_verified"]
    assert not h.factory.created[0].commands


@pytest.mark.parametrize("stage", ["ERROR", "COMPLETED", "UNKNOWN"])
def test_nonrunning_control_plane_never_executes_and_is_cleaned_up(
    harness, monkeypatch, stage
):
    h = harness
    inspect = h.api.inspect_job
    attempts = []

    def never_running(**kwargs):
        job = inspect(**kwargs)
        if not h.api.cancelled:
            attempts.append(job.id)
            job.status.stage = stage
        return job

    monkeypatch.setattr(h.api, "inspect_job", never_running)
    monkeypatch.setattr("examples.coding_audit.hf.time.sleep", lambda _: None)
    record = h.backend.execute_script("dangerous", "reference")
    assert record["error"] is not None and record["cleanup_verified"]
    assert not h.factory.created[0].commands
    assert len(attempts) <= 20


def test_startup_reconciliation_rechecks_image_before_execution(harness, monkeypatch):
    h = harness
    inspect = h.api.inspect_job
    attempts = []

    def changed_image(**kwargs):
        job = inspect(**kwargs)
        if not h.api.cancelled:
            if attempts:
                job.docker_image = VERIFIER
            else:
                job.status.stage = "SCHEDULING"
            attempts.append(job.id)
        return job

    monkeypatch.setattr(h.api, "inspect_job", changed_image)
    monkeypatch.setattr("examples.coding_audit.hf.time.sleep", lambda _: None)
    record = h.backend.execute_script("dangerous", "reference")
    assert "image" in record["error"] and record["cleanup_verified"]
    assert not h.factory.created[0].commands
    assert len(attempts) == 2


@pytest.mark.parametrize("failure", ["nonterminal", "inspection"])
def test_cleanup_cannot_pass_from_cancel_return_alone(harness, failure):
    h = harness
    submission = h.backend.execute_script(":", "reference")
    h.api.ignore_cancel = failure == "nonterminal"
    h.api.inspect_error_after_cancel = failure == "inspection"
    grade = h.backend.grade(submission)
    assert grade["cleanup_verified"] is False
    assert grade["reward"] is None and "cleanup" in grade["error"]
    assert h.api.cancelled.count("job-1") == 3


def test_exact_kill_fallback_requires_independent_terminal_inspection(harness):
    h = harness
    h.api.cancel_error = True
    record = h.backend.execute_script(":", "reference")
    assert record["cleanup_verified"] and h.factory.created[0].killed
    assert h.api.inspections[-1][0] == record["sandbox_id"]


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "/app/../tests/x",
        "/app//x",
        "/app/./x",
        "/tests/x",
        "/solution/x",
        "/logs/verifier/reward.txt",
        "/",
        "//app/x",
    ],
)
def test_submission_path_preflight(harness, path):
    h = harness
    with pytest.raises(HFBackendError):
        HFTaskBackend(h.root, {**h.execution, "submission_paths": [path]})


@pytest.mark.parametrize("paths", [["/app/a", "/app/a/b"], ["/app/a", "/app/a"]])
def test_overlapping_paths_rejected(harness, paths):
    with pytest.raises(HFBackendError, match="overlap|repeat"):
        HFTaskBackend(harness.root, {**harness.execution, "submission_paths": paths})


@pytest.mark.parametrize(
    "mutation", ["extra_path", "tampered_digest", "not_bytes", "oversize"]
)
def test_grade_rejects_unsafe_transfer_before_launch(harness, mutation):
    h = harness
    files = {"/app/answer.txt": b"answer"}
    submission = {"files": files, "submission_digest": submission_digest(files)}
    if mutation == "extra_path":
        files["/tests/test.sh"] = b"fake reward"
    elif mutation == "tampered_digest":
        files["/app/answer.txt"] = b"changed"
    elif mutation == "not_bytes":
        files["/app/answer.txt"] = "answer"
    else:
        files["/app/answer.txt"] = b"x" * (h.backend.max_bytes + 1)
    grade = h.backend.grade(submission)
    assert grade["error"] and grade["reward"] is None
    assert not h.factory.created


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "symlink_parent"])
def test_local_verifier_artifacts_reject_nonregular_or_linked_paths(harness, kind):
    h = harness
    path = h.root / "tests/extra"
    if kind == "symlink":
        path.symlink_to(h.root / "instruction.md")
    elif kind == "hardlink":
        os.link(h.root / "instruction.md", path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(h.root / "solution", target_is_directory=True)
    grade = h.backend.grade({"files": {}, "submission_digest": submission_digest({})})
    assert grade["error"] and not h.factory.created


@pytest.mark.parametrize(
    "reward",
    [
        b"NaN",
        b"Infinity",
        b"1e9999",
        b'{"reward": true}',
        b'{"reward": "1"}',
        b'{"reward":1,"reward":0}',
        b"garbage",
    ],
)
def test_reward_requires_finite_trusted_numeric_bytes(harness, reward):
    h = harness
    h.factory.reward = reward
    grade = h.backend.grade({"files": {}, "submission_digest": submission_digest({})})
    assert grade["error"] and grade["reward"] is None and grade["cleanup_verified"]


def test_json_reward_collects_original_bytes(harness):
    h = harness
    h.factory.reward = b'{"reward":0.5,"detail":"ok"}\n'
    grade = h.backend.grade({"files": {}, "submission_digest": submission_digest({})})
    assert grade["reward"] == 0.5 and grade["rawreward"] == h.factory.reward


def test_provider_requests_base_id_with_distinct_token_and_records_source_limit(
    harness,
):
    h = harness
    h.inference.responses = [response("printf answer > /app/answer.txt")]
    generated = h.backend.generate_candidate(MODEL, 42)
    assert generated["error"] is None
    assert not h.factory.created
    request = h.inference.requests[0]
    assert request["model"] == MODEL["model_id"] and "@" not in request["model"]
    assert "seed" not in request and "tools" not in request
    assert h.inference.kwargs == [
        {"provider": "deepinfra", "api_key": "inference-token-secret", "timeout": 120}
    ]
    assert "PRIVATE_SOLUTION_SENTINEL" not in repr(request)
    assert "PRIVATE_CANARY_SENTINEL" not in repr(request)
    assert "test.sh" not in repr(request)
    metadata = generated["model_metadata"]
    assert metadata["source_revision"] == MODEL["revision"]
    assert metadata["served_weight_revision_pinned"] is False
    assert metadata["seed_supported"] is False
    assert metadata["responses"][0]["system_fingerprint"] == "fp-123"
    assert generated["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_seed_and_exact_provider_alias_are_explicit(harness):
    h = harness
    h.inference.responses = [response("true", model="provider/exact")]
    record = h.backend.generate_candidate(
        {**MODEL, "supports_seed": True, "provider_model_id": "provider/exact"}, 7
    )
    assert record["error"] is None
    assert h.inference.requests[0]["seed"] == 7
    h.inference.responses = [response("true", model="provider/unlisted")]
    record = h.backend.generate_candidate(MODEL, 7)
    assert "response model" in record["error"]
    assert record["model_metadata"]["responses"][0]["model"] == "provider/unlisted"


def test_top_p_is_forwarded_and_recorded(harness):
    h = harness
    h.inference.responses = [response("true")]
    record = h.backend.generate_candidate({**MODEL, "top_p": 0.1}, 7)
    assert record["error"] is None
    assert h.inference.requests[0]["top_p"] == 0.1
    assert record["model_metadata"]["requests"][0]["top_p"] == 0.1


def test_candidate_stream_assembles_text_and_final_usage_without_changing_recipe(
    harness,
):
    h = harness
    events = [
        stream_event(delta={"role": "assistant", "content": "printf "}),
        stream_event(delta={"reasoning_content": "Think 雪"}),
        stream_event(delta={"content": "answer > /app/answer.txt"}, finish="stop"),
        stream_event(
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    ]
    h.inference.responses = [(event for event in events)]
    record = h.backend.generate_candidate(
        {**MODEL, "max_tokens": 8192, "max_total_tokens": 32768, "top_p": 0.1}, 7
    )
    assert record["error"] is None
    assert record["script"] == "printf answer > /app/answer.txt"
    assert record["transcript"][-1]["reasoning_content"] == "Think 雪"
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 5}
    request = h.inference.requests[0]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["max_tokens"] == 8192 and request["top_p"] == 0.1
    assert request["model"] == MODEL["model_id"] and request["temperature"] == 0
    assert "tools" not in request and "seed" not in request
    assert record["model_metadata"]["requests"] == [
        {key: value for key, value in request.items() if key != "messages"}
    ]
    assert h.inference.closed == 1 and not h.factory.created


@pytest.mark.parametrize(
    "outcome", ["complete", "early_close", "provider_error", "bad_json"]
)
def test_real_sdk_stream_serializes_request_and_closes_http_response(
    harness, monkeypatch, outcome
):
    h = harness
    captured = {}
    closed = []
    events = list(stream_response(response("printf answer > /app/answer.txt")))

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            for event in events[:1] if outcome == "early_close" else events:
                yield ("data: " + json.dumps(event) + "\n\n").encode()
            if outcome == "provider_error":
                yield b'data: {"error": "provider failed"}\n\n'
            elif outcome == "bad_json":
                yield b"data: {broken\n\n"
            elif outcome == "complete":
                yield b"data: [DONE]\n\n"

        def close(self):
            closed.append(True)

    def capture(request):
        assert request.url.host == "api.deepinfra.com"
        captured.update(json.loads(request.content))
        return httpx.Response(200, stream=Body())

    monkeypatch.setattr(
        DeepInfraConversationalTask,
        "_prepare_mapping_info",
        lambda self, model: hf_api.InferenceProviderMapping(
            provider="deepinfra",
            hf_model_id=model,
            providerId=model,
            task="conversational",
            status="live",
        ),
    )
    h.backend.inference_factory = InferenceClient
    with httpx.Client(transport=httpx.MockTransport(capture)) as client:
        monkeypatch.setattr(
            "huggingface_hub.inference._client.get_session", lambda: client
        )
        record = h.backend.generate_candidate({**MODEL, "enable_thinking": False}, 0)
        assert closed == [True]
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["model"] == MODEL["model_id"]
    if outcome == "complete":
        assert record["error"] is None and record["script"].startswith("printf answer")
        assert record["usage"] == {"input_tokens": 10, "output_tokens": 5}
    else:
        assert record["error"] and record["script"] == ""


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("model_drift", "response model"),
        ("allowed_model_drift", "identity"),
        ("id_drift", "identity"),
        ("fingerprint_drift", "identity"),
        ("two_choices", "one assistant"),
        ("wrong_index", "one assistant"),
        ("boolean_index", "one assistant"),
        ("wrong_role", "assistant"),
        ("missing_role", "assistant"),
        ("nontext", "text"),
        ("nontext_list", "text"),
        ("tool_call", "tool"),
        ("no_stop", "incomplete"),
        ("length_finish", "incomplete"),
        ("no_usage", "usage"),
        ("missing_usage_field", "usage"),
        ("boolean_usage", "usage"),
        ("negative_usage", "usage"),
        ("inconsistent_usage", "usage"),
        ("early_usage", "usage"),
        ("extra_content", "stop"),
        ("duplicate_usage", "usage"),
        ("provider_error", "provider"),
        ("completion_budget", "token budget"),
        ("total_budget", "token budget"),
    ],
)
def test_candidate_stream_rejects_invalid_or_incomplete_sdk_events(
    harness, mutation, diagnostic
):
    h = harness
    events = [
        stream_event(delta={"role": "assistant", "content": "true"}),
        stream_event(finish="stop"),
        stream_event(
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    ]
    events = json.loads(json.dumps(events))
    if mutation == "model_drift":
        events[1]["model"] = "unlisted/model"
    elif mutation == "allowed_model_drift":
        events[1]["model"] = "configured/alias"
    elif mutation == "id_drift":
        events[1]["id"] = "chatcmpl-other"
    elif mutation == "fingerprint_drift":
        events[1]["system_fingerprint"] = "fp-other"
    elif mutation == "two_choices":
        events[0]["choices"] *= 2
    elif mutation in {"wrong_index", "boolean_index"}:
        events[0]["choices"][0]["index"] = True if mutation == "boolean_index" else 1
    elif mutation in {"wrong_role", "missing_role"}:
        events[0]["choices"][0]["delta"]["role"] = (
            None if mutation == "missing_role" else "user"
        )
    elif mutation == "nontext":
        events[0]["choices"][0]["delta"]["content"] = 7
    elif mutation == "nontext_list":
        events[0]["choices"][0]["delta"]["content"] = []
    elif mutation == "tool_call":
        events[0]["choices"][0]["delta"]["tool_calls"] = [terminal()]
    elif mutation == "no_stop":
        events = events[:1]
    elif mutation == "length_finish":
        events[1]["choices"][0]["finish_reason"] = "length"
    elif mutation == "no_usage":
        events = events[:2]
    elif mutation == "missing_usage_field":
        del events[-1]["usage"]["total_tokens"]
    elif mutation == "boolean_usage":
        events[-1]["usage"]["prompt_tokens"] = True
    elif mutation == "negative_usage":
        events[-1]["usage"].update(prompt_tokens=-1, total_tokens=4)
    elif mutation == "inconsistent_usage":
        events[-1]["usage"]["total_tokens"] = 16
    elif mutation == "early_usage":
        events[0]["usage"] = events[-1]["usage"]
    elif mutation == "extra_content":
        events.insert(2, stream_event(delta={"content": "false"}))
    elif mutation == "duplicate_usage":
        events.append(events[-1])
    elif mutation == "provider_error":
        events[-1]["error"] = "provider failed after partial output"
    elif mutation == "completion_budget":
        events[-1]["usage"].update(completion_tokens=2049, total_tokens=2059)
    elif mutation == "total_budget":
        events[-1]["usage"].update(prompt_tokens=16384, total_tokens=16389)
    closed = []

    def streaming():
        try:
            for event in events:
                yield ChatCompletionStreamOutput.parse_obj_as_instance(event)
        finally:
            closed.append(True)

    h.inference.responses = [streaming()]
    record = h.backend.generate_candidate(
        {**MODEL, "response_model_aliases": ["configured/alias"]}, 0
    )
    assert diagnostic in record["error"], record
    assert record["script"] == "" and not h.factory.created
    assert closed == [True] and h.inference.closed == 1
    assert record["model_metadata"]["requests"][0]["stream"] is True


@pytest.mark.parametrize("stage", ["before_first", "partial", "after_stop"])
def test_candidate_stream_provider_failures_preserve_request_and_partial_error(
    harness, stage
):
    h = harness
    closed = []

    def streaming():
        try:
            if stage != "before_first":
                yield stream_event(delta={"role": "assistant", "content": "true"})
            if stage == "after_stop":
                yield stream_event(finish="stop")
            raise RuntimeError("provider error with inference-token-secret")
        finally:
            closed.append(True)

    h.inference.responses = [streaming()]
    record = h.backend.generate_candidate(MODEL, 0)
    assert record["error"] == "HF backend operation failed (RuntimeError)"
    assert record["script"] == "" and not h.factory.created
    assert "inference-token-secret" not in repr(record)
    assert record["model_metadata"]["requests"][0]["stream"] is True
    assert closed == [True] and h.inference.closed == 1


@pytest.mark.parametrize("ending", ["provider_error", "extra_event", "over_budget"])
def test_failed_candidate_stream_retains_observed_final_usage(harness, ending):
    h = harness
    output_tokens = 2049 if ending == "over_budget" else 5

    def streaming():
        yield stream_event(delta={"role": "assistant", "content": "true"})
        yield stream_event(finish="stop")
        yield stream_event(
            choices=[],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": output_tokens,
                "total_tokens": 10 + output_tokens,
            },
        )
        if ending == "provider_error":
            raise RuntimeError("provider interrupted after usage")
        if ending == "extra_event":
            yield stream_event(choices=[])

    h.inference.responses = [streaming()]
    record = h.backend.generate_candidate(MODEL, 0)
    assert record["error"] and record["script"] == ""
    assert record["usage"] == {"input_tokens": 10, "output_tokens": output_tokens}
    assert record["input_tokens"] == 10 and record["output_tokens"] == output_tokens
    assert h.inference.closed == 1 and not h.factory.created


def test_failed_candidate_request_retains_metadata_and_closes_client(harness):
    h = harness

    def fail(**kwargs):
        raise RuntimeError("provider unavailable")

    h.inference.chat_completion = fail
    record = h.backend.generate_candidate({**MODEL, "top_p": 0.1}, 0)
    assert record["error"] == "HF backend operation failed (RuntimeError)"
    assert record["model_metadata"]["requests"][0]["top_p"] == 0.1
    assert record["model_metadata"]["requests"][0]["stream"] is True
    assert h.inference.closed == 1 and record["script"] == ""


@pytest.mark.parametrize("field", ["content", "reasoning", "reasoning_content"])
def test_candidate_stream_byte_bounds_stop_consuming_and_close(harness, field):
    h = harness
    h.backend.max_script = 16 if field == "content" else 65536
    (h.root / "instruction.md").write_text("write answer")
    continued = []

    def streaming():
        yield stream_event(delta={"role": "assistant", field: "雪" * 6})
        continued.append(True)
        pytest.fail("oversized stream was consumed after its byte limit")

    h.inference.responses = [streaming()]
    record = h.backend.generate_candidate(
        {**MODEL, "max_output_bytes": 512 if field == "content" else 16}, 0
    )
    assert "byte limit" in record["error"] and record["script"] == ""
    assert not continued and h.inference.closed == 1


def test_candidate_stream_keeps_serialized_output_byte_gate(harness):
    h = harness
    h.inference.responses = [response("雪" * 10)]
    record = h.backend.generate_candidate({**MODEL, "max_output_bytes": 64}, 0)
    assert "output exceeded byte limit" in record["error"]
    assert record["script"] == "" and h.inference.closed == 1


def test_candidate_stream_accepts_usage_on_final_stop(harness):
    h = harness
    h.inference.responses = [
        (
            event
            for event in [
                stream_event(
                    delta={"role": "assistant", "content": "true"},
                    finish="stop",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            ]
        )
    ]
    record = h.backend.generate_candidate(MODEL, 0)
    assert record["error"] is None and record["script"] == "true"
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_candidate_stream_is_interrupted_by_root_deadline_and_closes(harness):
    h = harness
    h.backend.timeout = 0.02
    closed = []

    def streaming():
        try:
            yield stream_event(delta={"role": "assistant", "content": "true"})
            time.sleep(1)
        finally:
            closed.append(True)

    h.inference.responses = [streaming()]
    started = time.monotonic()
    record = h.backend.generate_candidate(MODEL, 0)
    assert time.monotonic() - started < 0.5
    assert "deadline" in record["error"] and record["script"] == ""
    assert closed == [True] and h.inference.closed == 1


def test_unqualified_sdk_version_refused_before_launch(harness, monkeypatch):
    monkeypatch.setattr("examples.coding_audit.hf.version", lambda name: "0.0.0")
    with pytest.raises(HFBackendError, match="requires huggingface_hub"):
        HFTaskBackend(
            harness.backend.task_root,
            harness.backend.execution,
            sandbox_factory=harness.factory,
        )
    assert not harness.factory.created


@pytest.mark.parametrize(
    "model",
    [
        {},
        {**MODEL, "revision": "main"},
        {**MODEL, "model_id": MODEL["model_id"] + "@" + MODEL["revision"]},
        {**MODEL, "provider": "auto"},
        {**MODEL, "supports_tools": False},
    ],
)
def test_invalid_model_preflight_never_launches_or_calls_provider(harness, model):
    record = harness.backend.solve(model, 0)
    assert record["error"]
    assert not harness.factory.created and not harness.inference.requests


def test_target_tool_loop_is_bounded_and_uses_instruction_only(harness):
    h = harness
    h.inference.responses = [response(None, calls=[terminal()]), response("saved")]
    record = h.backend.solve(MODEL, 1)
    assert record["error"] is None and record["cleanup_verified"]
    assert record["files"] == {"/app/answer.txt": b"answer"}
    assert len(h.factory.created) == 1
    assert h.factory.created[0].commands[0][0] == [
        "/bin/bash",
        "-c",
        "printf answer > /app/answer.txt",
    ]
    assert h.inference.requests[0]["tools"][0]["function"]["name"] == "terminal"
    assert all("stream" not in request for request in h.inference.requests)
    assert h.inference.closed == 0
    assert "PRIVATE_SOLUTION_SENTINEL" not in repr(h.inference.requests)
    assert "PRIVATE_CANARY_SENTINEL" not in repr(h.inference.requests)
    assert record["input_tokens"] == 20 and record["output_tokens"] == 10


def test_candidate_context_binds_only_declared_public_bytes_to_prompt(
    harness, monkeypatch
):
    from examples.coding_audit import hf

    h = harness
    path = "environment/source/pkg/api.py"
    source = h.root / path
    source.parent.mkdir(parents=True)
    raw = 'def answer():\n    return "雪"\n'.encode()
    source.write_bytes(raw)
    (source.parent / "unused.py").write_text("UNDECLARED_PUBLIC_SENTINEL")
    (source.parent / ".hidden").mkdir()
    (source.parent / ".hidden/private.txt").write_text("HIDDEN_SENTINEL")
    reads = []
    original = hf._local_file

    def read(path, limit):
        reads.append(path.relative_to(h.root).as_posix())
        return original(path, limit)

    monkeypatch.setattr(hf, "_local_file", read)
    h.backend.workdir = "/workspace/runtime"
    h.inference.responses = [response("true")]
    record = h.backend.generate_candidate({**MODEL, "context_paths": [path]}, 0)
    assert record["error"] is None
    assert reads == ["instruction.md", path]
    prompt = h.inference.requests[0]["messages"][1]["content"]
    assert prompt.startswith((h.root / "instruction.md").read_text())
    assert path in prompt and "pkg/api.py" in prompt and "/workspace/runtime" in prompt
    assert raw.decode() in prompt
    assert (
        "UNDECLARED_PUBLIC_SENTINEL" not in prompt and "HIDDEN_SENTINEL" not in prompt
    )
    assert "PRIVATE_SOLUTION_SENTINEL" not in prompt
    assert record["candidate_context"] == [
        {
            "path": path,
            "runtime_path": "pkg/api.py",
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    ]
    assert "context_paths" not in h.inference.requests[0]


def test_default_candidate_prompt_and_target_protocol_are_unchanged(harness):
    h = harness
    h.inference.responses = [response("true"), response("done")]
    generated = h.backend.generate_candidate(MODEL, 0)
    solved = h.backend.solve(MODEL, 1)
    assert generated["error"] is None and solved["error"] is None
    assert "candidate_context" not in generated
    assert all(
        request["messages"][1]["content"] == (h.root / "instruction.md").read_text()
        for request in h.inference.requests
    )


@pytest.mark.parametrize(
    "kind", ["symlink", "symlink_parent", "hardlink", "directory", "fifo"]
)
def test_candidate_context_rejects_linked_or_nonregular_files_before_inference(
    harness, kind
):
    h = harness
    parent = h.root / "environment/source"
    parent.mkdir(parents=True)
    source = parent / "api.py"
    if kind == "symlink":
        source.symlink_to(h.root / "solution/solve.sh")
    elif kind == "symlink_parent":
        (h.root / "solution/api.py").write_text("PRIVATE_API_SENTINEL")
        (parent / "linked").symlink_to(h.root / "solution", target_is_directory=True)
        source = parent / "linked/api.py"
    elif kind == "hardlink":
        os.link(h.root / "solution/solve.sh", source)
    elif kind == "directory":
        source.mkdir()
    else:
        os.mkfifo(source)
    record = h.backend.generate_candidate(
        {**MODEL, "context_paths": [source.relative_to(h.root).as_posix()]}, 0
    )
    assert record["error"] and record["script"] == ""
    assert not h.inference.requests and not h.inference.kwargs
    assert not h.factory.created


@pytest.mark.parametrize(
    "contents",
    [
        [b"public\x00hidden"],
        [b"\xff"],
        [b"x" * 65537],
        [b"x" * 32768, b"y" * 32769],
    ],
)
def test_candidate_context_requires_bounded_utf8_without_nul(harness, contents):
    h = harness
    parent = h.root / "environment/source"
    parent.mkdir(parents=True)
    paths = []
    for index, raw in enumerate(contents):
        path = parent / f"public-{index}.txt"
        path.write_bytes(raw)
        paths.append(path.relative_to(h.root).as_posix())
    record = h.backend.generate_candidate({**MODEL, "context_paths": paths}, 0)
    assert record["error"] and record["script"] == ""
    assert not h.inference.requests and not h.inference.kwargs


@pytest.mark.parametrize(
    "size,budget,accepted", [(65536, 32768, True), (1024, 64, False)]
)
def test_candidate_context_uses_combined_byte_and_existing_message_budget(
    harness, size, budget, accepted
):
    h = harness
    path = h.root / "environment/source/api.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * size)
    h.inference.responses = [response("true")]
    record = h.backend.generate_candidate(
        {
            **MODEL,
            "context_paths": ["environment/source/api.py"],
            "max_total_tokens": budget,
        },
        0,
    )
    assert (record["error"] is None) is accepted
    assert bool(h.inference.requests) is accepted
    if not accepted:
        assert "context exceeds byte budget" in record["error"]


def test_target_context_is_rejected_before_sandbox_or_inference(harness):
    h = harness
    record = h.backend.solve(
        {**MODEL, "context_paths": ["environment/source/api.py"]}, 0
    )
    assert record["error"] and not h.factory.created
    assert not h.inference.requests and not h.inference.kwargs


@pytest.mark.parametrize(
    "path",
    [
        "solution/solve.sh",
        "environment/source/../../solution/solve.sh",
        "environment/source/tests/hidden.py",
        "environment/source/validation/spec.json",
        "environment/source/.hidden/secret.py",
        "environment/source/secrets/token",
        "environment/source/api\x00.py",
    ],
)
def test_invalid_candidate_context_is_rejected_without_reading_private_files(
    harness, monkeypatch, path
):
    from examples.coding_audit import hf

    h = harness
    reads = []
    original = hf._local_file

    def read(filename, limit):
        assert filename == h.root / "instruction.md"
        reads.append(filename)
        return original(filename, limit)

    monkeypatch.setattr(hf, "_local_file", read)
    record = h.backend.generate_candidate({**MODEL, "context_paths": [path]}, 0)
    assert record["error"] and record["script"] == ""
    assert reads == [h.root / "instruction.md"]
    assert not h.inference.requests and not h.inference.kwargs


def test_excess_parallel_tool_calls_execute_nothing_and_cleanup(harness):
    h = harness
    h.inference.responses = [
        response(None, calls=[terminal(identifier="a"), terminal(identifier="b")])
    ]
    record = h.backend.solve({**MODEL, "max_steps": 1}, 0)
    assert record["error"] is None and record["cleanup_verified"]
    assert record["truncated"] and record["termination_reason"] == "tool_call_limit"
    assert not h.factory.created[0].commands


def test_step_exhaustion_hands_actual_submission_to_grader(harness):
    h = harness
    h.inference.responses = [response(None, calls=[terminal()])]
    record = h.backend.solve({**MODEL, "max_steps": 1}, 0)
    assert record["error"] is None and record["files"] == {"/app/answer.txt": b"answer"}
    assert record["truncated"] and record["termination_reason"] == "step_limit"
    assert len(h.inference.requests) == 1 and record["cleanup_verified"]
    grade = h.backend.grade(record)
    assert grade["error"] is None and grade["reward"] == 1


def test_invalid_tool_name_is_not_executed(harness):
    h = harness
    call = terminal()
    call["function"]["name"] = "provider_request"
    h.inference.responses = [response(None, calls=[call])]
    record = h.backend.solve(MODEL, 0)
    assert "invalid terminal" in record["error"] and not h.factory.created[0].commands


def test_output_and_token_limits_are_fail_closed(harness):
    h = harness
    h.factory.output = "💥" * 4097
    record = h.backend.execute_script(":", "reference")
    assert "output exceeded" in record["error"] and record["cleanup_verified"]
    h.factory.output = ""
    h.inference.responses = [
        response(
            "true",
            usage={
                "prompt_tokens": 16000,
                "completion_tokens": 400,
                "total_tokens": 16400,
            },
        )
    ]
    record = h.backend.generate_candidate(MODEL, 0)
    assert "token budget" in record["error"]


def test_private_path_audit_blocks_before_execution(harness):
    h = harness
    h.factory.initial["/tests"] = b"private"
    record = h.backend.execute_script("true", "reference")
    assert "private path" in record["error"] and record["cleanup_verified"]
    assert not h.factory.created[0].commands


def test_leakage_probes_recheck_transcripts_without_extra_jobs(harness):
    h = harness
    h.factory.submitted = None
    probes = h.backend.leakage_probes()
    assert {row["name"] for row in probes} == {
        "private_paths_absent",
        "fresh_state_absent",
        "canary_not_in_transcript",
    }
    assert all(row["passed"] and row["error"] is None for row in probes)
    assert len(h.factory.created) == 2
    h.inference.responses = [response("# PRIVATE_CANARY_SENTINEL\ntrue")]
    h.backend.generate_candidate(MODEL, 0)
    probes = h.backend.leakage_probes()
    assert len(h.factory.created) == 2
    assert not next(row for row in probes if row["name"] == "canary_not_in_transcript")[
        "passed"
    ]


def test_fresh_state_marker_blocks_reused_filesystem(harness):
    h = harness
    first = h.backend.execute_script(":", "reference")
    h.factory.initial.update(h.factory.created[0].files.data)
    second = h.backend.execute_script(":", "target")
    assert first["error"] is None and "prior-state" in second["error"]
    assert not h.factory.created[1].commands
    assert not next(
        row for row in h.backend.leakage_probes() if row["name"] == "fresh_state_absent"
    )["passed"]


def test_state_marker_uses_configured_workdir(harness):
    h = harness
    h.execution["workdir"] = "/workspace/task"
    backend = HFTaskBackend(
        h.root,
        h.execution,
        token="execution-token-secret",
        inference_token="inference-token-secret",
        api=h.api,
        sandbox_factory=h.factory,
        inference_factory=h.inference,
    )
    assert PurePosixPath(backend.marker_path).parent == PurePosixPath("/workspace/task")
    assert PurePosixPath(backend.marker_path).name.startswith(".openenv-state-")
    first = backend.execute_script(":", "reference")
    h.factory.initial.update(h.factory.created[0].files.data)
    second = backend.execute_script(":", "target")
    assert first["error"] is None and "prior-state" in second["error"]


def test_submission_tree_digest_matches_canonical_algorithm(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/a").write_bytes(b"alpha")
    (tmp_path / "app/z").write_bytes(b"omega")
    assert submission_digest({"/app/z": b"omega", "/app/a": b"alpha"}) == source_digest(
        tmp_path
    )
    assert submission_digest({}) == hashlib.sha256(b"").hexdigest()


def test_deadline_interrupts_blocking_controller_and_restores_handler():
    handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(HFBackendError, match="deadline"):
        with _deadline(0.02):
            time.sleep(1)
    assert time.monotonic() - started < 0.5
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_strong_isolation_and_equal_credentials_rejected(harness):
    h = harness
    with pytest.raises(HFBackendError, match="strong"):
        HFTaskBackend(h.root, {**h.execution, "require_strong_isolation": True})
    with pytest.raises(HFBackendError, match="credentials must differ"):
        HFTaskBackend(h.root, h.execution, token="same", inference_token="same")


def test_runtime_tokens_are_redacted_and_credential_files_rejected(harness):
    h = harness
    h.factory.output = "server echoed execution-token-secret"
    record = h.backend.execute_script(":", "reference")
    assert "execution-token-secret" not in repr(record)
    assert "[REDACTED]" in repr(record["transcript"])
    h.factory.output = ""
    h.factory.submitted = b"inference-token-secret"
    record = h.backend.execute_script(":", "reference")
    assert "credential detected" in record["error"] and record["files"] == {}
    assert "inference-token-secret" not in repr(record)


def test_empty_submission_does_not_inherit_verifier_image_answer(harness):
    h = harness
    h.factory.initial["/app/answer.txt"] = b"image-provided-answer"
    grade = h.backend.grade({"files": {}, "submission_digest": submission_digest({})})
    assert grade["error"] is None
    assert "/app/answer.txt" not in h.factory.created[0].files.data


def test_verifier_private_image_paths_are_not_overlaid(harness):
    h = harness
    h.factory.initial["/tests"] = b"stale-tests"
    grade = h.backend.grade({"files": {}, "submission_digest": submission_digest({})})
    assert "preexisting private" in grade["error"] and grade["cleanup_verified"]
    assert not h.factory.created[0].commands


def test_remote_regular_file_and_read_growth_checks(harness):
    h = harness
    sandbox = h.factory(image=AGENT)
    sandbox.files.data["/app/answer.txt"] = b"answer"
    sandbox.files.types["/app/answer.txt"] = "symlink"
    with pytest.raises(HFBackendError, match="regular"):
        h.backend._collect(sandbox)
    sandbox.files.types.clear()
    sandbox.files.grow_on_read = True
    with pytest.raises(SandboxError):
        h.backend._collect(sandbox)
    assert sandbox.files.MAX_READ_BYTES == h.backend.max_bytes


def test_missing_execution_or_inference_credentials_never_fall_back_to_hf_token(
    harness, monkeypatch
):
    h = harness
    monkeypatch.setenv("HF_TOKEN", "persistence-token")
    backend = HFTaskBackend(
        h.root,
        h.execution,
        api=h.api,
        sandbox_factory=h.factory,
        inference_factory=h.inference,
    )
    assert "HF_SANDBOX_TOKEN" in backend.execute_script(":", "nop")["error"]
    assert "HF_INFERENCE_TOKEN" in backend.solve(MODEL, 0)["error"]
    assert not h.factory.created and not h.inference.requests


def test_timed_out_command_with_zero_exit_is_error(harness):
    harness.factory.timed_out = True
    record = harness.backend.execute_script(":", "reference")
    assert "timed out" in record["error"] and record["cleanup_verified"]


def test_cleanup_deadline_is_not_swallowed_by_retry_loop(harness):
    h = harness
    original = h.api.cancel_job

    def blocked_cancel(**kwargs):
        time.sleep(2)
        return original(**kwargs)

    h.api.cancel_job = blocked_cancel
    h.backend.cleanup_timeout = 0.02
    started = time.monotonic()
    record = h.backend.execute_script(":", "reference")
    assert time.monotonic() - started < 0.5
    assert record["error"] and not record["cleanup_verified"]


def test_canary_probe_checks_actual_unicode_message_text(harness):
    h = harness
    h.backend.canary = "private\n雪"
    h.backend._transcripts = [[{"content": "private\n雪"}]]
    assert not h.backend.leakage_probes()[-1]["passed"]


def test_failed_shell_command_is_observed_then_agent_repairs(harness, monkeypatch):
    h = harness
    original = FakeSandbox.run
    calls = []

    def run_with_failure(sandbox, argv, **kwargs):
        calls.append(argv)
        result = original(sandbox, argv, **kwargs)
        result.exit_code = 1 if len(calls) == 1 else 0
        return result

    monkeypatch.setattr(FakeSandbox, "run", run_with_failure)
    h.inference.responses = [
        response(None, calls=[terminal("grep missing absent", "a")]),
        response(None, calls=[terminal("printf answer > /app/answer.txt", "b")]),
        response("saved"),
    ]
    record = h.backend.solve(MODEL, 0)
    assert record["error"] is None and not record["truncated"]
    assert record["files"] == {"/app/answer.txt": b"answer"}
    observation = h.inference.requests[1]["messages"][-1]
    assert json.loads(observation["content"])["exit_code"] == 1
    assert h.backend.grade(record)["reward"] == 1


def test_execution_token_in_terminal_output_never_reaches_provider(harness):
    h = harness
    h.factory.output = "server echoed execution-token-secret"
    h.inference.responses = [response(None, calls=[terminal()]), response("done")]
    record = h.backend.solve(MODEL, 0)
    assert record["error"] is None
    assert "execution-token-secret" not in repr(h.inference.requests)
    assert "[REDACTED]" in repr(h.inference.requests[1])


def test_failed_startup_recovers_exact_labeled_job_and_independently_cleans(harness):
    h = harness
    h.factory.fail_after_create = True
    original = h.api.cancel_job
    cancelled = []

    def transient_cancel(**kwargs):
        cancelled.append(kwargs["job_id"])
        if len(cancelled) == 1:
            raise RuntimeError("transient controller failure")
        return original(**kwargs)

    h.api.cancel_job = transient_cancel
    record = h.backend.execute_script(":", "reference")
    assert record["error"] and record["cleanup_verified"]
    assert record["sandbox_id"] == "job-0" and cancelled == ["job-0", "job-0"]
    assert record["cleanup_job_ids"] == ["job-0"]
    assert h.api.stages["job-0"] == "CANCELED"


def test_transient_kill_exception_does_not_abort_cleanup_retries(harness, monkeypatch):
    h = harness
    h.api.cancel_error = True
    original = FakeSandbox.kill
    killed = []

    def transient_kill(sandbox):
        killed.append(sandbox.id)
        if len(killed) == 1:
            raise RuntimeError("transient kill failure")
        original(sandbox)

    monkeypatch.setattr(FakeSandbox, "kill", transient_kill)
    record = h.backend.execute_script(":", "reference")
    assert record["error"] is None and record["cleanup_verified"]
    assert killed == ["job-0", "job-0"]


def test_startup_recovery_retries_transient_listing_error(harness):
    h = harness
    h.factory.fail_after_create = True
    original = h.api.list_jobs
    calls = []

    def transient_listing(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("transient lookup failure")
        return original(**kwargs)

    h.api.list_jobs = transient_listing
    record = h.backend.execute_script(":", "reference")
    assert record["error"] and record["cleanup_verified"]
    assert len(calls) == 2 and h.api.cancelled == ["job-0"]


def test_rejected_length_stream_keeps_finish_and_terminal_usage(harness):
    h = harness
    h.inference.responses = [
        (
            ChatCompletionStreamOutput.parse_obj_as_instance(event)
            for event in (
                [
                    stream_event(delta={"role": "assistant", "content": "partial"}),
                    stream_event(finish="length"),
                    stream_event(
                        choices=[],
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 2048,
                            "total_tokens": 2058,
                        },
                    ),
                ]
            )
        )
    ]
    record = h.backend.generate_candidate(MODEL, 0)
    assert "incomplete" in record["error"] and record["script"] == ""
    assert record["model_metadata"]["responses"][0]["finish_reason"] == "length"
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 2048}
    assert h.inference.closed == 1 and not h.factory.created


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_thinking_flag_requires_an_explicit_boolean(harness, value):
    h = harness
    record = h.backend.generate_candidate({**MODEL, "enable_thinking": value}, 0)
    assert "enable_thinking must be a boolean" in record["error"]
    assert not h.inference.requests and record["script"] == ""


@pytest.mark.parametrize("language", ["bash", "sh", "shell"])
@pytest.mark.parametrize("closed", [False, True])
def test_fenced_candidate_keeps_raw_response_and_executes_only_bash(
    harness, language, closed
):
    h = harness
    body = "printf answer > /app/answer.txt\n"
    raw = "```" + language + "\n" + body + ("```\n" if closed else "")
    h.inference.responses = [response(raw)]
    record = h.backend.generate_candidate(MODEL, 0)
    assert record["error"] is None and record["script"] == body
    assert record["transcript"][-1]["content"] == raw
    metadata = record["script_format"]
    assert metadata["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert (
        metadata["script_sha256"]
        == hashlib.sha256(record["script"].encode()).hexdigest()
    )
    assert metadata["closing_fence_present"] is closed
    submission = h.backend.execute_script(record["script"], "generated")
    assert submission["error"] is None
    assert h.factory.created[0].commands[-1][0] == ["/bin/bash", "-c", record["script"]]


@pytest.mark.parametrize(
    "raw", ["```bash\n```", "```python\nprint('wrong language')\n```"]
)
def test_empty_or_non_bash_fence_does_not_create_a_candidate(harness, raw):
    h = harness
    h.inference.responses = [response(raw)]
    record = h.backend.generate_candidate(MODEL, 0)
    assert record["error"] and record["script"] == "" and not h.factory.created
