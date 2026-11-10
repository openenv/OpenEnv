"""Bounded HF Jobs coding trials for reviewed tasks, with controller-side inference.

Image receipts attest only the requested immutable control-plane reference. The
Sandbox RPC bearer reaches a workload-controlled server; this is not a hostile
code security boundary or a served-model weight revision guarantee.
"""

import hashlib
import json
import math
import os
import re
import signal
import stat
import threading
import time
import uuid
from contextlib import closing, contextmanager
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, InferenceClient, Sandbox
from huggingface_hub.errors import SandboxError

from .runner import _candidate_context_paths

_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
_PROMPT_FRAMING_ALLOWANCE = 8192
_LIMITATIONS = [
    "Immutable images are control-plane references, not measured runtime digests.",
    "Sandbox RPC exposes the execution bearer to the workload server; hostile-code credential isolation is not qualified.",
    "The verifier executes submitted code as the same user; verifier integrity against hostile submissions is not qualified.",
    "Inference Providers do not pin served weights to the recorded model source revision.",
    "Target prompt reservation uses a byte/framing assumption, not a serving-tokenizer or billing guarantee.",
]
_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class HFBackendError(ValueError):
    """A credential-free controller diagnostic."""


class _DeadlineExceeded(HFBackendError):
    pass


class _TargetBudgetStop(HFBackendError):
    def __init__(self, reason, *, response_received=False):
        super().__init__(reason)
        self.reason = reason
        self.response_received = response_received


@contextmanager
def _deadline(seconds):
    # SDK control-plane calls have no timeout argument. A controller must run on
    # the main Unix thread so a blocked synchronous SDK call can be interrupted.
    if threading.current_thread() is not threading.main_thread():
        raise HFBackendError("HF coding controller requires the main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise _DeadlineExceeded("controller deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL,
        min(seconds, previous_timer[0]) if previous_timer[0] else seconds,
        0.1,
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )


def _path(value):
    if not isinstance(value, str) or "\x00" in value:
        raise HFBackendError("paths must be absolute normalized file paths")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or value == "/"
        or value.startswith("//")
    ):
        raise HFBackendError("paths must be absolute normalized file paths")
    return value


def _under(path, parent):
    return path == parent or path.startswith(parent + "/")


def _candidate_script(raw):
    lines = raw.lstrip().splitlines(keepends=True)
    opening = lines[0].strip()
    if not opening.startswith("```"):
        return raw, None
    if opening not in {"```bash", "```sh", "```shell"}:
        raise HFBackendError("candidate code fence must declare Bash or sh")
    end = len(lines)
    while end > 1 and not lines[end - 1].strip():
        end -= 1
    closed = end > 1 and lines[end - 1].strip() == "```"
    script = "".join(lines[1 : end - 1] if closed else lines[1:])
    if not script.strip():
        raise HFBackendError("candidate response must contain a shell script")
    return script, {
        "wrapper": "markdown_bash",
        "closing_fence_present": closed,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
    }


def submission_digest(files):
    """Hash sorted relative paths, NUL, content, NUL as source_digest does."""
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.lstrip("/").encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _local_file(path, limit):
    # Walk descriptors so neither a final link nor a linked parent is followed.
    path = Path(path).absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        file_fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        with os.fdopen(file_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise HFBackendError("task artifacts must be regular unlinked files")
            if info.st_size > limit:
                raise HFBackendError("task artifact exceeds byte limit")
            content = stream.read(limit + 1)
            if len(content) > limit:
                raise HFBackendError("task artifact exceeds byte limit")
            return content, format(stat.S_IMODE(info.st_mode), "o")
    finally:
        os.close(fd)


def _plain(value):
    # HF generated output types derive from dict and preserve extra response fields.
    return json.loads(json.dumps(value))


def _retain_usage(receipt, usage):
    try:
        encoded = json.dumps(usage, allow_nan=False).encode()
        if len(encoded) > 65536:
            raise ValueError("oversized usage")
    except (TypeError, ValueError):
        receipt["usage_receipt_error"] = (
            "nonfinite, non-JSON or oversized usage details"
        )
        if isinstance(usage, dict):
            receipt["usage_counts"] = {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if type(usage.get(key)) is int and usage[key] >= 0
            }
    else:
        receipt["usage"] = usage


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise HFBackendError("duplicate JSON key")
        result[key] = value
    return result


class HFTaskBackend:
    """Execute each solve and grade in a fresh, dedicated HF Jobs Sandbox."""

    def __init__(
        self,
        task_root: Path,
        execution: dict,
        *,
        token=None,
        api=None,
        sandbox_factory=None,
        inference_factory=None,
        canary: str | None = None,
        inference_token=None,
    ):
        if version("huggingface_hub") != "1.33.0":
            raise HFBackendError(
                "this execution adapter requires huggingface_hub==1.33.0"
            )
        self.task_root = Path(task_root)
        self.execution = dict(execution)
        self.token = token if token is not None else os.environ.get("HF_SANDBOX_TOKEN")
        self.inference_token = (
            inference_token
            if inference_token is not None
            else os.environ.get("HF_INFERENCE_TOKEN")
        )
        credentials = [
            value
            for value in (self.token, self.inference_token, os.environ.get("HF_TOKEN"))
            if value
        ]
        if len(set(credentials)) != len(credentials):
            raise HFBackendError(
                "execution, inference and persistence credentials must differ"
            )
        if execution.get("strong_isolation") or execution.get(
            "require_strong_isolation"
        ):
            raise HFBackendError("strong verifier isolation is unsupported")
        for key in ("agent_image", "verifier_image"):
            if not isinstance(execution.get(key), str) or not _IMAGE.fullmatch(
                execution[key]
            ):
                raise HFBackendError(
                    f"{key} must be an immutable OCI name@sha256 reference"
                )
        self.workdir = _path(execution.get("workdir", "/app"))
        self.private_paths = list(
            dict.fromkeys(
                _path(item)
                for item in ["/tests", "/solution", *execution.get("private_paths", [])]
            )
        )
        self.reward_path = _path(
            execution.get("reward_path", "/logs/verifier/reward.txt")
        )
        paths = execution.get("submission_paths")
        if not isinstance(paths, list) or not paths:
            raise HFBackendError("submission_paths must be a nonempty file allowlist")
        self.submission_paths = [_path(item) for item in paths]
        for index, path in enumerate(self.submission_paths):
            if any(
                _under(path, protected) or _under(protected, path)
                for protected in [
                    "/tests",
                    "/solution",
                    self.reward_path,
                    *self.private_paths,
                ]
            ):
                raise HFBackendError("submission path overlaps private or reward paths")
            if any(
                _under(path, other) or _under(other, path)
                for other in self.submission_paths[:index]
            ):
                raise HFBackendError("submission paths overlap or repeat")
        self.timeout = self._bound(execution, "timeout_seconds", 120, 3600)
        self.max_bytes = self._bound(
            execution, "max_submission_bytes", 1048576, 16 * 1048576
        )
        self.max_script = self._bound(execution, "max_script_bytes", 65536, 1048576)
        self.max_output = self._bound(
            execution, "max_tool_output_bytes", 16384, 1048576
        )
        self.cleanup_attempts = self._bound(execution, "cleanup_attempts", 3, 10)
        self.cleanup_timeout = self._bound(execution, "cleanup_timeout_seconds", 10, 60)
        self.api = api if api is not None else HfApi(token=self.token)
        self.sandbox_factory = sandbox_factory or Sandbox.create
        self.inference_factory = inference_factory or InferenceClient
        self.canary = canary if canary is not None else uuid.uuid4().hex
        if not isinstance(self.canary, str) or not self.canary:
            raise HFBackendError("canary must be nonempty text")
        self.canary_digest = hashlib.sha256(self.canary.encode()).hexdigest()
        self.marker_path = str(
            PurePosixPath(self.workdir) / (".openenv-state-" + uuid.uuid4().hex)
        )
        self._audit = []
        self._transcripts = []
        self._owned_ids = set()
        self._credentials = tuple(credentials)

    def _redact(self, value):
        if isinstance(value, str):
            for secret in self._credentials:
                value = value.replace(secret, "[REDACTED]")
        elif isinstance(value, bytes):
            for secret in self._credentials:
                value = value.replace(secret.encode(), b"[REDACTED]")
        elif isinstance(value, dict):
            value = {
                self._redact(key): self._redact(item) for key, item in value.items()
            }
        elif isinstance(value, list):
            value = [self._redact(item) for item in value]
        return value

    def _has_canary(self, value):
        if isinstance(value, str):
            return self.canary in value
        if isinstance(value, dict):
            return any(self._has_canary(item) for item in value.values())
        if isinstance(value, list):
            return any(self._has_canary(item) for item in value)
        return False

    @staticmethod
    def _bound(config, key, default, maximum):
        value = config.get(key, default)
        if type(value) is not int or not 0 < value <= maximum:
            raise HFBackendError(f"{key} must be an integer from 1 to {maximum}")
        return value

    def _record(self):
        return {
            "files": {},
            "submission_digest": submission_digest({}),
            "sandbox_id": None,
            "image_ref": self.execution["agent_image"],
            "image_reference_kind": "control_plane_immutable_reference",
            "cleanup_verified": False,
            "error": None,
            "transcript": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "model_metadata": {},
            "limitations": list(_LIMITATIONS),
        }

    @staticmethod
    def _error(exc):
        return (
            str(exc)
            if isinstance(exc, HFBackendError)
            else f"HF backend operation failed ({type(exc).__name__})"
        )

    def _inspect(self, sandbox_id):
        job = self.api.inspect_job(
            job_id=sandbox_id,
            namespace=self.execution.get("namespace"),
            token=self.token,
        )
        if job.id != sandbox_id or (
            self.execution.get("namespace")
            and job.owner.name != self.execution["namespace"]
        ):
            raise HFBackendError("control-plane job identity mismatch")
        return job

    def _create(self, image, record, *, grader=False):
        if not self.token:
            raise HFBackendError("HF_SANDBOX_TOKEN execution credential is required")
        record["creation_label"] = uuid.uuid4().hex
        sandbox = self.sandbox_factory(
            image=image,
            flavor=self.execution.get("flavor", "cpu-basic"),
            idle_timeout=self.timeout,
            namespace=self.execution.get("namespace"),
            forward_hf_token=False,
            token=self.token,
            start_timeout=self.timeout,
            labels={
                "name": "openenv-coding-" + record["creation_label"],
                "openenv-coding-attempt": record["creation_label"],
            },
        )
        sandbox_id = sandbox.id
        # Register before inspection so image/identity failures still clean up.
        record["grade_sandbox_id" if grader else "sandbox_id"] = sandbox_id
        if sandbox_id in self._owned_ids:
            raise HFBackendError("sandbox identity was reused")
        self._owned_ids.add(sandbox_id)
        # SDK readiness checks the RPC server; the Jobs status can still lag.
        # Reconcile only SCHEDULING within the caller's existing deadline, and
        # recheck identity and the immutable image on every observation.
        for attempt in range(20):
            job = self._inspect(sandbox_id)
            if job.docker_image != image:
                raise HFBackendError("sandbox control-plane image mismatch")
            if job.status.stage == "RUNNING":
                break
            if job.status.stage != "SCHEDULING":
                raise HFBackendError("sandbox control-plane state is not runnable")
            if attempt < 19:
                time.sleep(0.5)
        else:
            raise HFBackendError("sandbox control-plane readiness did not converge")
        # The SDK stat/read pair otherwise permits up to 512 MiB per read.
        sandbox.files.MAX_READ_BYTES = self.max_bytes
        sandbox.files.mkdir(self.workdir)
        return sandbox

    def _recover_startup_jobs(self, record, pending):
        # create() can allocate a job before failing during readiness. The label
        # is unique to this attempt and is filtered again before any cancellation.
        jobs = self.api.list_jobs(
            labels={"openenv-coding-attempt": record["creation_label"]},
            namespace=self.execution.get("namespace"),
            token=self.token,
            timeout=self.cleanup_timeout,
        )
        for index, job in enumerate(jobs):
            if (
                index >= 4
                or (job.labels or {}).get("openenv-coding-attempt")
                != record["creation_label"]
            ):
                raise HFBackendError("startup cleanup ownership could not be verified")
            if (
                self.execution.get("namespace")
                and job.owner.name != self.execution["namespace"]
            ):
                raise HFBackendError("startup cleanup namespace mismatch")
            pending.add(job.id)
            self._owned_ids.add(job.id)
            record["cleanup_job_ids"] = sorted(pending)
            record[
                "grade_sandbox_id" if "grade_sandbox_id" in record else "sandbox_id"
            ] = sorted(pending)[0]

    def _cleanup(self, record, sandbox):
        sandbox_id = record.get("grade_sandbox_id") or record.get("sandbox_id")
        if sandbox_id is None and not record.get("creation_label"):
            return
        pending = {sandbox_id} if sandbox_id is not None else set()
        recovery_complete = sandbox_id is not None
        try:
            with _deadline(self.cleanup_timeout):
                for _ in range(self.cleanup_attempts):
                    if not recovery_complete:
                        try:
                            self._recover_startup_jobs(record, pending)
                            recovery_complete = bool(pending)
                        except _DeadlineExceeded:
                            raise
                        except Exception:
                            pass
                    if not pending:
                        continue
                    for owned_id in list(pending):
                        try:
                            self.api.cancel_job(
                                job_id=owned_id,
                                namespace=self.execution.get("namespace"),
                                token=self.token,
                            )
                        except _DeadlineExceeded:
                            raise
                        except Exception:
                            if sandbox is not None and sandbox.id == owned_id:
                                try:
                                    sandbox.kill()
                                except _DeadlineExceeded:
                                    raise
                                except Exception:
                                    pass
                        try:
                            if self._inspect(owned_id).status.stage in _TERMINAL:
                                pending.remove(owned_id)
                        except _DeadlineExceeded:
                            raise
                        except Exception:
                            continue
                    if not pending and recovery_complete:
                        record["cleanup_verified"] = True
                        break
        except Exception:
            pass
        if not record["cleanup_verified"]:
            record["error"] = record["error"] or "sandbox cleanup could not be verified"
            if "reward" in record:
                record["reward"] = None

    def _audit_solver(self, sandbox):
        evidence = {
            "sandbox_id": sandbox.id,
            "private_paths_absent": False,
            "fresh_state_absent": False,
        }
        self._audit.append(evidence)
        for path in self.private_paths:
            if sandbox.files.exists(path):
                raise HFBackendError("private path exists in solver sandbox")
        evidence["private_paths_absent"] = True
        if sandbox.files.exists(self.marker_path):
            raise HFBackendError("fresh sandbox contains prior-state marker")
        evidence["fresh_state_absent"] = True
        sandbox.files.write(self.marker_path, self.canary_digest.encode())

    def _run(self, sandbox, argv, end, *, cwd=None, trace=None):
        output = []
        size = 0

        def collect(chunk):
            nonlocal size
            raw = chunk.encode()
            if size + len(raw) > self.max_output:
                if trace is not None:
                    trace["output_truncated"] = True
                raise HFBackendError("terminal output exceeded byte limit")
            size += len(raw)
            output.append(chunk)

        try:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise HFBackendError("controller deadline exceeded")
            result = sandbox.run(
                argv,
                shell=False,
                cwd=cwd or self.workdir,
                timeout=remaining,
                check=False,
                capture_output=False,
                on_stdout=collect,
                on_stderr=collect,
            )
            if trace is not None:
                trace["exit_code"] = result.exit_code
            if result.timed_out:
                raise HFBackendError("sandbox command timed out")
            return result.exit_code, "".join(output)
        finally:
            if trace is not None:
                trace["output"] = "" if trace["output_truncated"] else "".join(output)

    def _collect(self, sandbox):
        files = {}
        remaining = self.max_bytes
        for path in self.submission_paths:
            try:
                entry = sandbox.files.stat(path)
            except SandboxError as exc:
                if exc.status_code == 404:
                    continue
                raise
            if (
                entry.type != "file"
                or type(entry.size) is not int
                or not 0 <= entry.size <= remaining
            ):
                raise HFBackendError("submission is not a bounded regular file")
            sandbox.files.MAX_READ_BYTES = remaining
            data = sandbox.files.read(path)
            if (
                not isinstance(data, bytes)
                or len(data) != entry.size
                or len(data) > remaining
            ):
                raise HFBackendError("submission changed size or exceeded byte limit")
            files[path] = data
            if any(secret.encode() in data for secret in self._credentials):
                raise HFBackendError("credential detected in submission artifact")
            remaining -= len(data)
        return files

    def execute_script(self, script: str, role: str) -> dict:
        """Execute a reference/candidate script without exposing controller credentials."""
        record = self._record()
        record["role"] = role
        sandbox = None
        started = time.monotonic()
        try:
            with _deadline(self.timeout):
                if (
                    not isinstance(script, str)
                    or len(script.encode()) > self.max_script
                ):
                    raise HFBackendError("script exceeds byte limit or is not text")
                sandbox = self._create(self.execution["agent_image"], record)
                self._audit_solver(sandbox)
                exit_code, output = self._run(
                    sandbox, ["/bin/bash", "-c", script], started + self.timeout
                )
                record["transcript"].append(
                    {
                        "role": role,
                        "script": script,
                        "output": output,
                        "exit_code": exit_code,
                    }
                )
                if exit_code != 0:
                    raise HFBackendError("task script exited nonzero")
                record["files"] = self._collect(sandbox)
                record["submission_digest"] = submission_digest(record["files"])
        except Exception as exc:
            record["error"] = self._error(exc)
        finally:
            self._cleanup(record, sandbox)
            record["latency_seconds"] = time.monotonic() - started
            self._transcripts.append(record["transcript"])
        return self._redact(record)

    def _model(self, model, seed, *, tools=False):
        model = dict(model)
        if tools and "context_paths" in model:
            raise HFBackendError("target.context_paths is unsupported")
        if (
            not isinstance(model.get("model_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["model_id"])
            or not isinstance(model.get("revision"), str)
            or not _REVISION.fullmatch(model["revision"])
            or not isinstance(model.get("provider"), str)
            or not model["provider"]
            or model["provider"] in {"auto", "cheapest", "fastest", "preferred"}
        ):
            raise HFBackendError(
                "model_id, pinned source revision and explicit provider are required"
            )
        if type(seed) is not int or seed < 0:
            raise HFBackendError("seed must be a nonnegative integer")
        if tools and model.get("supports_tools") is False:
            raise HFBackendError(
                "configured model does not support terminal function calling"
            )
        if "enable_thinking" in model and type(model["enable_thinking"]) is not bool:
            raise HFBackendError("enable_thinking must be a boolean")
        if type(model.get("supports_seed", False)) is not bool:
            raise HFBackendError("supports_seed must be an explicit boolean")
        aliases = model.get("response_model_aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(item, str) or not item for item in aliases
        ):
            raise HFBackendError(
                "response_model_aliases must be exact nonempty strings"
            )
        allowed = {model["model_id"], *aliases}
        if model.get("provider_model_id"):
            if not isinstance(model["provider_model_id"], str):
                raise HFBackendError("provider_model_id must be text")
            allowed.add(model["provider_model_id"])
        limits = {
            key: self._bound(model, key, default, maximum)
            for key, default, maximum in (
                ("max_steps", 8, 128),
                ("max_tokens", 2048, 32768),
                ("max_total_tokens", 16384, 1048576),
                ("max_output_bytes", 16384, 1048576),
            )
        }
        if not self.inference_token:
            raise HFBackendError("controller HF_INFERENCE_TOKEN is required")
        client = self.inference_factory(
            provider=model["provider"],
            api_key=self.inference_token,
            timeout=self.timeout,
        )
        return model, allowed, limits, client

    def _chat(self, client, model, allowed, limits, messages, record, seed, *, tools):
        used = record["input_tokens"] + record["output_tokens"]
        message_bytes = len(json.dumps(messages).encode())
        if message_bytes > limits["max_total_tokens"] * 4:
            if tools:
                raise _TargetBudgetStop("context_byte_limit")
            raise HFBackendError("chat context exceeds byte budget")
        remaining = limits["max_total_tokens"] - used
        if remaining <= 0:
            if tools:
                raise _TargetBudgetStop("total_token_limit")
            raise HFBackendError("model total token budget exhausted")
        request = {
            "messages": self._redact(messages),
            "model": model["model_id"],
            "max_tokens": min(limits["max_tokens"], remaining),
            "temperature": model.get("temperature", 0),
        }
        if "top_p" in model:
            value = model["top_p"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise HFBackendError("top_p must be a finite number in (0, 1]")
            request["top_p"] = value
        if "enable_thinking" in model:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": model["enable_thinking"]}
            }
        if model.get("supports_seed", False):
            request["seed"] = seed
        metadata = record["model_metadata"]
        preflight = None
        if tools:
            request.update(tools=[_TOOL], tool_choice="auto")
            encoded = json.dumps(request, allow_nan=False).encode()
            reserved_input = len(encoded) + _PROMPT_FRAMING_ALLOWANCE
            output_allowance = remaining - reserved_input
            preflight = {
                "method": "request_utf8_bytes_plus_framing_v1",
                "reservation_bytes": len(encoded),
                "request_bytes": len(encoded),
                "request_sha256": hashlib.sha256(encoded).hexdigest(),
                "framing_allowance": _PROMPT_FRAMING_ALLOWANCE,
                "input_token_reservation": reserved_input,
                "actual_tokens_before": used,
                "remaining_tokens": remaining,
                "output_token_cap": max(0, min(limits["max_tokens"], output_allowance)),
                "dispatched": False,
            }
            metadata.setdefault("preflights", []).append(preflight)
            if output_allowance <= 0:
                raise _TargetBudgetStop("input_reservation_limit")
            request["max_tokens"] = preflight["output_token_cap"]
            encoded = json.dumps(request, allow_nan=False).encode()
            preflight["request_bytes"] = len(encoded)
            preflight["request_sha256"] = hashlib.sha256(encoded).hexdigest()
        else:
            request.update(stream=True, stream_options={"include_usage": True})
        metadata.setdefault("requests", []).append(
            {key: value for key, value in request.items() if key != "messages"}
        )
        if tools:
            preflight["dispatched"] = True
            parsed = client.chat_completion(**request)
            response = _plain(parsed)
            # Access the typed nested value: outer SDK dicts can omit extras.
            response["usage"] = _plain(parsed.usage)
            choices = response.get("choices")
            receipt = {
                "model": response.get("model"),
                "system_fingerprint": response.get("system_fingerprint"),
                "id": response.get("id"),
                "seed_sent": request.get("seed"),
                "finish_reason": choices[0].get("finish_reason")
                if isinstance(choices, list)
                and len(choices) == 1
                and isinstance(choices[0], dict)
                else None,
                "response_validated": False,
                "discarded": True,
                "tool_calls_processed": 0,
            }
            metadata.setdefault("responses", []).append(receipt)
            _retain_usage(receipt, response["usage"])
        else:
            response = self._candidate_response(
                client, request, allowed, limits, record
            )
            receipt = metadata["responses"][-1]
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise HFBackendError("provider response omitted token usage")
        for source, target in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
        ):
            value = usage.get(source)
            if type(value) is not int or value < 0:
                raise HFBackendError("provider returned invalid token usage")
            if tools:
                record[target] += value
                record["usage"][target] += value
        if (
            type(usage.get("total_tokens")) is not int
            or usage["total_tokens"] < 0
            or usage["total_tokens"]
            != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            raise HFBackendError("provider returned inconsistent token usage")
        if receipt.get("usage_receipt_error"):
            raise HFBackendError("provider returned invalid usage details")
        violations = []
        if usage["completion_tokens"] > request["max_tokens"]:
            violations.append("response_output_cap")
        if (
            record["input_tokens"] + record["output_tokens"]
            > limits["max_total_tokens"]
        ):
            violations.append("cumulative_total_cap")
        if tools and usage["prompt_tokens"] > preflight["input_token_reservation"]:
            violations.append("input_reservation_contract")
        if violations:
            receipt["budget_violations"] = violations
            raise HFBackendError(
                "provider exceeded token budget: " + ", ".join(violations)
            )
        if response.get("model") not in allowed:
            raise HFBackendError(
                "provider response model does not match configured exact identities"
            )
        choices = response.get("choices", [])
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0].get("message"), dict)
        ):
            raise HFBackendError("provider must return exactly one assistant message")
        message = choices[0]["message"]
        if message.get("role") != "assistant":
            raise HFBackendError("provider must return an assistant message")
        record["transcript"].append(message)
        if len(json.dumps(message).encode()) > limits["max_output_bytes"]:
            raise HFBackendError("model output exceeded byte limit")
        finish = choices[0].get("finish_reason")
        if finish == "length" and tools:
            receipt["response_validated"] = True
            raise _TargetBudgetStop("response_output_limit", response_received=True)
        if finish not in {"stop", "tool_calls"}:
            raise HFBackendError("model response was incomplete")
        if finish == "tool_calls" and (
            not isinstance(message.get("tool_calls"), list) or not message["tool_calls"]
        ):
            raise HFBackendError("invalid terminal function calls")
        receipt["response_validated"] = True
        return message

    def _candidate_response(self, client, request, allowed, limits, record):
        # The pinned SDK yields parsed deltas but owns HTTP contexts on the client.
        response = {}
        parts = {"content": []}
        size = script_size = 0
        stopped = False
        finish_reason = None
        usage = None
        try:
            with closing(client.chat_completion(**request)) as stream:
                for parsed in stream:
                    event = _plain(parsed)
                    if not isinstance(event, dict) or event.get("error") is not None:
                        raise HFBackendError("provider returned a stream error")
                    # Outer SDK dicts omit extra fields on nested dataclasses.
                    if isinstance(event.get("choices"), list):
                        for index, choice in enumerate(parsed.choices):
                            event["choices"][index]["delta"] = _plain(choice.delta)
                    if not response:
                        response = {
                            key: event.get(key)
                            for key in ("model", "id", "system_fingerprint")
                        }
                        record["model_metadata"]["responses"].append(
                            {**response, "seed_sent": request.get("seed")}
                        )
                    if event.get("model") not in allowed:
                        raise HFBackendError(
                            "provider response model does not match configured exact identities"
                        )
                    if (
                        not isinstance(event.get("id"), str)
                        or not event["id"]
                        or any(
                            event.get(key) != value for key, value in response.items()
                        )
                    ):
                        raise HFBackendError(
                            "provider stream response identity changed"
                        )
                    if usage is not None:
                        raise HFBackendError("provider returned data after final usage")
                    choices = event.get("choices")
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise HFBackendError(
                            "provider must stream exactly one assistant choice"
                        )
                    if choices:
                        if stopped:
                            raise HFBackendError(
                                "provider returned a choice after stop"
                            )
                        choice = choices[0]
                        if (
                            not isinstance(choice, dict)
                            or type(choice.get("index")) is not int
                            or choice["index"] != 0
                            or not isinstance(choice.get("delta"), dict)
                        ):
                            raise HFBackendError(
                                "provider must stream exactly one assistant choice at index 0"
                            )
                        delta = choice["delta"]
                        if delta.get("role") not in {None, "assistant"} or (
                            not parts["content"] and delta.get("role") != "assistant"
                        ):
                            raise HFBackendError(
                                "provider must stream an assistant message"
                            )
                        for key, value in delta.items():
                            if key == "role" or value is None:
                                continue
                            if key == "tool_calls" and value == []:
                                continue
                            if key not in {"content", "reasoning", "reasoning_content"}:
                                raise HFBackendError(
                                    "candidate stream contains tool calls or unsupported deltas"
                                )
                            if not isinstance(value, str):
                                raise HFBackendError(
                                    "provider stream delta must be text"
                                )
                            length = len(value.encode())
                            size += length
                            if key == "content":
                                script_size += length
                            if (
                                size > limits["max_output_bytes"]
                                or script_size > self.max_script
                            ):
                                raise HFBackendError("model output exceeded byte limit")
                            if value:
                                parts.setdefault(key, []).append(value)
                        # Empty assistant role events still establish the message.
                        if not parts["content"]:
                            parts["content"].append("")
                        finish = choice.get("finish_reason")
                        if finish is not None:
                            finish_reason = finish
                            record["model_metadata"]["responses"][-1][
                                "finish_reason"
                            ] = finish
                            if finish not in {"stop", "length"}:
                                raise HFBackendError("model response was incomplete")
                            stopped = True
                    elif not stopped:
                        raise HFBackendError(
                            "model response was incomplete before usage"
                        )
                    event_usage = event.get("usage")
                    if event_usage is not None:
                        event_usage = _plain(parsed.usage)
                        _retain_usage(
                            record["model_metadata"]["responses"][-1], event_usage
                        )
                        if not stopped:
                            raise HFBackendError("provider returned usage before stop")
                        if not isinstance(event_usage, dict) or any(
                            type(event_usage.get(key)) is not int
                            or event_usage[key] < 0
                            for key in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            )
                        ):
                            raise HFBackendError(
                                "provider returned invalid token usage"
                            )
                        if event_usage["total_tokens"] != (
                            event_usage["prompt_tokens"]
                            + event_usage["completion_tokens"]
                        ):
                            raise HFBackendError(
                                "provider returned inconsistent token usage"
                            )
                        usage = event_usage
                        # Usage already observed remains evidence if the stream
                        # fails later. The caller accounts only non-streamed calls.
                        for source, target in (
                            ("prompt_tokens", "input_tokens"),
                            ("completion_tokens", "output_tokens"),
                        ):
                            record[target] += usage[source]
                            record["usage"][target] += usage[source]
                        if (
                            usage["completion_tokens"] > request["max_tokens"]
                            or record["input_tokens"] + record["output_tokens"]
                            > limits["max_total_tokens"]
                        ):
                            raise HFBackendError("provider exceeded token budget")
                    elif not choices:
                        raise HFBackendError("provider response omitted token usage")
                if not stopped:
                    raise HFBackendError("model response was incomplete")
                if usage is None:
                    raise HFBackendError("provider response omitted token usage")
                if finish_reason != "stop":
                    raise HFBackendError("model response was incomplete")
            response.update(
                usage=usage,
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            **{key: "".join(value) for key, value in parts.items()},
                        },
                    }
                ],
            )
            return response
        except Exception:
            if parts["content"]:
                record["transcript"].append(
                    {
                        "role": "assistant",
                        **{key: "".join(value) for key, value in parts.items()},
                    }
                )
            raise
        finally:
            client.close()

    def _model_metadata(self, model, seed):
        return {
            "requested_model": model["model_id"],
            "source_revision": model["revision"],
            "provider": model["provider"],
            "provider_model_id": model.get("provider_model_id"),
            "response_model_aliases": model.get("response_model_aliases", []),
            "seed": seed,
            "seed_supported": model.get("supports_seed", False),
            "served_weight_revision_pinned": False,
            "responses": [],
        }

    def _instruction(self):
        raw, _ = _local_file(self.task_root / "instruction.md", self.max_script)
        return raw.decode("utf-8")

    def _candidate_instruction(self, model, record):
        instruction = self._instruction()
        if "context_paths" not in model:
            return instruction
        paths = _candidate_context_paths(model["context_paths"])
        record["candidate_context"] = []
        parts = [instruction]
        remaining = 65536
        for path in paths:
            raw, _ = _local_file(self.task_root / path, remaining)
            remaining -= len(raw)
            if b"\0" in raw:
                raise HFBackendError("candidate context contains NUL")
            text = raw.decode("utf-8")
            runtime_path = (
                PurePosixPath(path).relative_to("environment/source").as_posix()
            )
            record["candidate_context"].append(
                {
                    "path": path,
                    "runtime_path": runtime_path,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            parts.append(
                f"Public source path: {path}\n"
                f"Runtime filename relative to {self.workdir}: {runtime_path}\n"
                f"---\n{text}\n---"
            )
        return "\n\n".join(parts)

    def generate_candidate(self, model: dict, seed: int) -> dict:
        """Generate a shell script from the instruction and explicit public context."""
        record = self._record()
        record["script"] = ""
        started = time.monotonic()
        try:
            with _deadline(self.timeout):
                instruction = self._candidate_instruction(model, record)
                model, allowed, limits, client = self._model(model, seed)
                record["model_metadata"] = self._model_metadata(model, seed)
                messages = [
                    {
                        "role": "system",
                        "content": "Return only a bash script that attempts the task. Do not use Markdown fences.",
                    },
                    {"role": "user", "content": instruction},
                ]
                record["transcript"] = messages
                message = self._chat(
                    client, model, allowed, limits, messages, record, seed, tools=False
                )
                script = message.get("content")
                if (
                    not isinstance(script, str)
                    or not script.strip()
                    or message.get("tool_calls")
                ):
                    raise HFBackendError(
                        "candidate response must contain a shell script"
                    )
                if len(script.encode()) > self.max_script:
                    raise HFBackendError("candidate script exceeds byte limit")
                script, formatting = _candidate_script(script)
                if formatting is not None:
                    record["script_format"] = formatting
                record["script"] = script
        except Exception as exc:
            record["error"] = self._error(exc)
        finally:
            record["latency_seconds"] = time.monotonic() - started
            self._transcripts.append(record["transcript"])
        return self._redact(record)

    def solve(self, model: dict, seed: int, role="target") -> dict:
        """Run a bounded terminal-tool agent with inference kept in the controller."""
        record = self._record()
        record["role"] = role
        record.update(truncated=False, termination_reason="completed")
        sandbox = None
        started = time.monotonic()
        try:
            with _deadline(self.timeout):
                model, allowed, limits, client = self._model(model, seed, tools=True)
                record["model_metadata"] = self._model_metadata(model, seed)
                messages = [
                    {
                        "role": "system",
                        "content": "Complete the task using the terminal function. Commands run in the task workspace. Finish with a short message after saving your solution files.",
                    },
                    {"role": "user", "content": self._instruction()},
                ]
                record["transcript"] = messages
                sandbox = self._create(self.execution["agent_image"], record)
                self._audit_solver(sandbox)
                calls_used = validated_responses = 0
                record["model_metadata"]["completed_response_turns"] = 0
                for _ in range(limits["max_steps"]):
                    try:
                        message = self._chat(
                            client,
                            model,
                            allowed,
                            limits,
                            messages,
                            record,
                            seed,
                            tools=True,
                        )
                    except _TargetBudgetStop as stop:
                        if not validated_responses and not stop.response_received:
                            raise HFBackendError(
                                "initial target prompt cannot fit token budget"
                            ) from None
                        record.update(truncated=True, termination_reason=stop.reason)
                        break
                    calls = message.get("tool_calls") or []
                    if not calls:
                        record["model_metadata"]["responses"][-1]["discarded"] = False
                        validated_responses += 1
                        record["model_metadata"]["completed_response_turns"] = (
                            validated_responses
                        )
                        break
                    if not isinstance(calls, list):
                        raise HFBackendError("invalid terminal function calls")
                    if calls_used + len(calls) > limits["max_steps"]:
                        record.update(
                            truncated=True, termination_reason="tool_call_limit"
                        )
                        break
                    ids = set()
                    # Reject ambiguous identities before executing any call in a batch.
                    for call in calls:
                        if (
                            not isinstance(call, dict)
                            or call.get("type") != "function"
                            or not isinstance(call.get("id"), str)
                            or not call["id"]
                            or call["id"] in ids
                            or not isinstance(call.get("function"), dict)
                            or not isinstance(call["function"].get("name"), str)
                            or not isinstance(call["function"].get("arguments"), str)
                        ):
                            raise HFBackendError("invalid terminal function call")
                        ids.add(call["id"])
                    for call in calls:
                        calls_used += 1
                        problem = None
                        if call["function"]["name"] != "terminal":
                            problem = "Only the terminal function is available; use terminal(command: string)."
                        else:
                            try:
                                arguments = json.loads(
                                    call["function"]["arguments"],
                                    object_pairs_hook=_unique_pairs,
                                )
                                if (
                                    not isinstance(arguments, dict)
                                    or set(arguments) != {"command"}
                                    or not isinstance(arguments["command"], str)
                                    or len(arguments["command"].encode())
                                    > self.max_script
                                ):
                                    problem = "terminal requires exactly one bounded string argument named command."
                            except (ValueError, TypeError):
                                problem = "terminal arguments must be a unique JSON object with one command string."
                        if problem:
                            content = json.dumps({"error": problem})
                            if len(content.encode()) > self.max_output:
                                raise HFBackendError(
                                    "tool error feedback exceeds byte limit"
                                )
                            record["model_metadata"]["tool_error_count"] = (
                                record["model_metadata"].get("tool_error_count", 0) + 1
                            )
                        else:
                            exit_code, output = self._run(
                                sandbox,
                                ["/bin/bash", "-c", arguments["command"]],
                                started + self.timeout,
                            )
                            content = json.dumps(
                                {"exit_code": exit_code, "output": output}
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": content,
                            }
                        )
                        receipt = record["model_metadata"]["responses"][-1]
                        receipt["tool_calls_processed"] += 1
                        receipt["discarded"] = False
                    validated_responses += 1
                    record["model_metadata"]["completed_response_turns"] = (
                        validated_responses
                    )
                else:
                    record.update(truncated=True, termination_reason="step_limit")
                record["files"] = self._collect(sandbox)
                record["submission_digest"] = submission_digest(record["files"])
        except Exception as exc:
            record["error"] = self._error(exc)
        finally:
            self._cleanup(record, sandbox)
            record["latency_seconds"] = time.monotonic() - started
            self._transcripts.append(record["transcript"])
        return self._redact(record)

    def _test_files(self):
        root = self.task_root / "tests"
        if root.is_symlink() or not root.is_dir():
            raise HFBackendError("task tests must be a regular directory")
        files = []
        remaining = 16 * 1048576
        for index, path in enumerate(root.rglob("*")):
            if index >= 4096:
                raise HFBackendError("too many verifier filesystem entries")
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            relative = path.relative_to(root).as_posix()
            raw, mode = _local_file(path, remaining)
            remaining -= len(raw)
            files.append(("/tests/" + relative, raw, mode))
        return sorted(files)

    def grade(self, submission: dict) -> dict:
        """Grade exact allowlisted submission bytes in a separate verifier Sandbox."""
        record = {
            "grade_sandbox_id": None,
            "verifier_image_ref": self.execution["verifier_image"],
            "image_reference_kind": "control_plane_immutable_reference",
            "reward": None,
            "rawreward": b"",
            "transcript": [],
            "cleanup_verified": False,
            "error": None,
            "limitations": list(_LIMITATIONS),
        }
        sandbox = None
        started = time.monotonic()
        try:
            with _deadline(self.timeout):
                files = submission["files"]
                if (
                    submission.get("error")
                    or not isinstance(files, dict)
                    or any(
                        path not in self.submission_paths or not isinstance(raw, bytes)
                        for path, raw in files.items()
                    )
                    or sum(map(len, files.values())) > self.max_bytes
                ):
                    raise HFBackendError("invalid submission transfer")
                if submission.get("submission_digest") != submission_digest(files):
                    raise HFBackendError("submission digest mismatch")
                files = dict(files)
                tests = self._test_files()
                command = self.execution.get(
                    "verifier_command", ["/bin/bash", "/tests/test.sh"]
                )
                if (
                    not isinstance(command, list)
                    or not command
                    or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
                ):
                    raise HFBackendError("verifier_command must be an argv list")
                sandbox = self._create(
                    self.execution["verifier_image"], record, grader=True
                )
                if sandbox.files.exists("/tests") or sandbox.files.exists("/solution"):
                    raise HFBackendError(
                        "verifier image contains preexisting private paths"
                    )
                for path in self.submission_paths:
                    if sandbox.files.exists(path):
                        if sandbox.files.stat(path).type != "file":
                            raise HFBackendError(
                                "verifier submission destination is not a regular file"
                            )
                        sandbox.files.delete(path)
                for path, raw, mode in tests:
                    sandbox.files.write(path, raw, mode=mode)
                sandbox.files.write(
                    "/tests/.openenv-canary-" + uuid.uuid4().hex, self.canary.encode()
                )
                for path, raw in files.items():
                    sandbox.files.write(path, raw)
                # Harbor normally mounts this directory. Fresh Sandbox graders
                # need it before a native verifier can write its reward file.
                sandbox.files.mkdir(str(PurePosixPath(self.reward_path).parent))
                # A stale image-provided reward must not become new evidence.
                if sandbox.files.exists(self.reward_path):
                    sandbox.files.delete(self.reward_path)
                trace = {"exit_code": None, "output": "", "output_truncated": False}
                record["transcript"].append(trace)
                exit_code, _ = self._run(
                    sandbox, command, started + self.timeout, trace=trace
                )
                if exit_code != 0:
                    raise HFBackendError("verifier command exited nonzero")
                entry = sandbox.files.stat(self.reward_path)
                if (
                    entry.type != "file"
                    or type(entry.size) is not int
                    or not 0 < entry.size <= min(self.max_bytes, 65536)
                ):
                    raise HFBackendError("reward is not a bounded regular file")
                sandbox.files.MAX_READ_BYTES = min(self.max_bytes, 65536)
                raw = sandbox.files.read(self.reward_path)
                if len(raw) != entry.size:
                    raise HFBackendError("reward changed during collection")
                if any(secret.encode() in raw for secret in self._credentials):
                    raise HFBackendError("credential detected in reward artifact")
                record["rawreward"] = raw
                text = raw.decode("utf-8").strip()
                if text.startswith("{"):
                    value = json.loads(
                        text, parse_float=Decimal, object_pairs_hook=_unique_pairs
                    )[self.execution.get("reward_key", "reward")]
                else:
                    value = Decimal(text)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (Decimal, int))
                    or not Decimal(value).is_finite()
                    or not math.isfinite(float(value))
                ):
                    raise HFBackendError("reward must be finite numeric data")
                record["reward"] = float(value)
        except Exception as exc:
            record["error"] = self._error(exc)
            record["reward"] = None
        finally:
            self._cleanup(record, sandbox)
            record["latency_seconds"] = time.monotonic() - started
            self._transcripts.append(record.get("transcript", []))
        record = self._redact(record)
        for trace in record["transcript"]:
            if len(trace["output"].encode()) > self.max_output:
                trace["output"] = ""
                trace["output_truncated"] = True
        return record

    def leakage_probes(self) -> list:
        """Audit private paths, two fresh solver instances, and accumulated transcripts."""
        probes = []
        while len(self._audit) < 2:
            record = self.execute_script(":", "leakage_probe")
            if record["error"]:
                probes.append(
                    {
                        "name": "fresh_state_absent",
                        "passed": False,
                        "error": record["error"],
                        "evidence": list(self._audit),
                    }
                )
                break
        for name in ("private_paths_absent", "fresh_state_absent"):
            if any(probe["name"] == name for probe in probes):
                continue
            probes.append(
                {
                    "name": name,
                    "error": None,
                    "passed": len(self._audit) >= 2
                    and all(item[name] for item in self._audit),
                    "evidence": list(self._audit),
                }
            )
        probes.append(
            {
                "name": "canary_not_in_transcript",
                "error": None,
                "passed": not self._has_canary(self._transcripts),
                "evidence": {
                    "canary_digest": self.canary_digest,
                    "transcript_count": len(self._transcripts),
                },
            }
        )
        return probes
