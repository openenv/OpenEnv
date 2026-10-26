"""Bounded Docker-local execution for validation, independent of core providers.

Public and no-network modes are enforced; allowlists are refused. No-network subjects
are reached through a trusted helper sharing only their network namespace. Runtime
hardening is an execution baseline; it does not certify containment.
"""

import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..manifest import ExecutionDeclaration
from ..runtime.contracts import LaunchSpec, NetworkEvidence
from ..types import ProviderCapability
from . import ExecResult, ProviderError, StartupError, UnsupportedCapability
from ._netns import (
    ExecRelayBridge,
    HARDENING,
    HELPER_IMAGE,
    PROBE_SCRIPT,
    SINK_SCRIPT,
    WAIT_READY_SCRIPT,
)


_LABEL = "org.openenv.validation.run"
_MAX_OUTPUT = 65536
_EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    ".ssh",
    ".aws",
    ".azure",
    ".config",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "outputs",
    ".validation",
}
_SECRET_NAME = re.compile(
    r"^(?:\.env(?:\..*)?|\.netrc|.*_secrets?\..*|"
    r"id_(?:rsa|ed25519).*|id_ecdsa(?:\..*)?|"
    r"credentials(?:\.json)?|secrets?|secrets\.(?:json|toml|ya?ml))$|"
    r"\.(?:pem|key|p12|pfx)$",
    re.I | re.S,
)
_TOKEN = re.compile(
    r"(?:hf_[A-Za-z0-9]{8,}|(?:sk|ghp|github_pat)[-_][A-Za-z0-9_-]{8,}|"
    r"(?i:bearer)\s+\S+|(?i:password|token|api[_-]?key|secret)\s*[=:]\s*\S+)"
)


def _safe_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return (
        _TOKEN.sub("[REDACTED]", value)
        .encode("utf-8")[-_MAX_OUTPUT:]
        .decode("utf-8", "ignore")
    )


def _command(
    argv: list[str], timeout_s: float, max_bytes: int = _MAX_OUTPUT
) -> tuple[int, str, str]:
    """Drain both pipes while retaining bounded tails; kill the client process group."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ProviderError("Operation deadline elapsed")
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise ProviderError("Could not execute the Docker client") from exc
    buffers = [bytearray(), bytearray()]

    def drain(pipe, output):
        try:
            while chunk := pipe.read(8192):
                output.extend(chunk)
                if len(output) > max_bytes:
                    del output[:-max_bytes]
        finally:
            pipe.close()

    threads = [
        threading.Thread(target=drain, args=(pipe, output), daemon=True)
        for pipe, output in zip((process.stdout, process.stderr), buffers)
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group can exit between wait() and killpg().
            pass
        process.wait()
        raise ProviderError("Docker operation exceeded its deadline") from None
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group can exit before this cleanup signal.
            pass
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    return (
        process.returncode,
        buffers[0].decode("utf-8", "replace"),
        buffers[1].decode("utf-8", "replace"),
    )


def _contained(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ProviderError("Build paths must remain inside the package") from None
    if path.is_symlink():
        raise ProviderError("Build paths may not be symbolic links")
    return path


def _snapshot(root: Path, destination: Path, max_bytes: int, deadline: float) -> None:
    """Copy source bytes, excluding links, caches, known credential names and outputs."""
    total = 0
    files = 0

    def copy_tree(directory_fd: int, target_dir: Path, depth: int = 0):
        nonlocal total, files
        if depth > 64:
            raise ProviderError("Build snapshot exceeds its directory depth budget")
        # Descriptor-relative opens also reject a link swapped in during copying.
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                if name in _EXCLUDED or _SECRET_NAME.search(name):
                    continue
                if time.monotonic() >= deadline:
                    raise ProviderError("Build snapshot exceeded its deadline")
                if entry.is_symlink():
                    raise ProviderError("Build snapshots do not accept symbolic links")
                files += 1
                if files > 100000:
                    raise ProviderError("Build snapshot exceeds its file-count budget")
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    info = os.fstat(child_fd)
                    target = target_dir / name
                    if stat.S_ISDIR(info.st_mode):
                        target.mkdir()
                        copy_tree(child_fd, target, depth + 1)
                    elif stat.S_ISREG(info.st_mode):
                        with target.open("wb") as output:
                            while chunk := os.read(child_fd, 1024 * 1024):
                                total += len(chunk)
                                if total > max_bytes:
                                    raise ProviderError(
                                        "Build snapshot exceeds its size budget"
                                    )
                                if time.monotonic() >= deadline:
                                    raise ProviderError(
                                        "Build snapshot exceeded its deadline"
                                    )
                                output.write(chunk)
                        target.chmod(info.st_mode & 0o777)
                    else:
                        raise ProviderError("Build snapshots accept regular files only")
                finally:
                    os.close(child_fd)

    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            copy_tree(root_fd, destination)
        finally:
            os.close(root_fd)
    except OSError:
        raise ProviderError(
            "Build snapshot could not safely read package files"
        ) from None


def _ensure_image(image: str, timeout_s: float = 300) -> None:
    """Pull a pinned validator image once, outside any subject deadline."""
    code, _, _ = _command(["docker", "image", "inspect", image], 30)
    if code == 0:
        return
    code, _, stderr = _command(["docker", "pull", image], timeout_s)
    if code:
        raise ProviderError(
            "Could not pull the network helper image: " + _safe_text(stderr[-1024:])
        )


def _container_ip(inspect_output: str) -> str:
    """The container's bridge address; Podman reports it per network only."""
    settings = json.loads(inspect_output)[0]["NetworkSettings"]
    address = settings.get("IPAddress") or next(
        (
            network.get("IPAddress")
            for network in (settings.get("Networks") or {}).values()
            if network.get("IPAddress")
        ),
        "",
    )
    if not re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", address or ""):
        raise ProviderError("Network sink has no IPv4 bridge address")
    return address


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DockerValidationProvider:
    """Build isolated source snapshots and launch owned, resource-bounded subjects."""

    name = "docker-local"
    capabilities = frozenset(
        {
            ProviderCapability.IMAGE_BUILD,
            ProviderCapability.EXEC,
            ProviderCapability.NETWORK_POLICY,
        }
    )
    supported_network_modes = frozenset({"public", "no-network"})

    def __init__(
        self,
        *,
        build_timeout_s: float = 600,
        max_context_bytes: int = 2**30,
        helper_image: str = HELPER_IMAGE,
    ):
        if not math.isfinite(build_timeout_s) or build_timeout_s <= 0:
            raise ValueError("build_timeout_s must be positive and finite")
        if max_context_bytes <= 0:
            raise ValueError("max_context_bytes must be positive")
        if "@sha256:" not in helper_image:
            raise ValueError("helper_image must be pinned by digest")
        self.build_timeout_s = build_timeout_s
        self.max_context_bytes = max_context_bytes
        self.helper_image = helper_image

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        deadline = time.monotonic() + self.build_timeout_s
        try:
            root = root.resolve(strict=True)
        except OSError:
            raise ProviderError("Package directory could not be read") from None
        _contained(root, execution.dockerfile)
        _contained(root, execution.context)
        with tempfile.TemporaryDirectory(
            prefix="openenv-validation-build-"
        ) as directory:
            staging = Path(directory) / "source"
            staging.mkdir()
            _snapshot(root, staging, self.max_context_bytes, deadline)
            dockerfile = _contained(staging, execution.dockerfile)
            context = _contained(staging, execution.context)
            if not dockerfile.is_file() or not context.is_dir():
                raise ProviderError(
                    "Build Dockerfile and context must exist in the snapshot"
                )
            iidfile = Path(directory) / "image-id"
            code, _, stderr = _command(
                [
                    "docker",
                    "build",
                    "--iidfile",
                    str(iidfile),
                    "--file",
                    str(dockerfile),
                    str(context),
                ],
                deadline - time.monotonic(),
            )
            if code != 0:
                raise StartupError("Image build failed: " + _safe_text(stderr[-4096:]))
            image = iidfile.read_text().strip() if iidfile.exists() else ""
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
                raise ProviderError("Docker did not return an immutable image ID")
            return image

    def start(self, spec: LaunchSpec) -> "DockerRunningSubject":
        if spec.network.mode not in self.supported_network_modes:
            raise UnsupportedCapability(
                f"docker-local does not enforce {spec.network.mode}"
            )
        if spec.resources.gpus or spec.resources.gpu_types:
            raise UnsupportedCapability("docker-local does not provide GPU isolation")
        if not math.isfinite(spec.resources.cpu):
            raise UnsupportedCapability("CPU budget must be finite")
        isolated = spec.network.mode == "no-network"
        if isolated:
            # The helper is the only control path; fetch it outside the startup deadline.
            _ensure_image(self.helper_image)
        name = f"openenv-validation-{spec.run_id}-{uuid.uuid4().hex[:12]}"
        subject = DockerRunningSubject(name, spec, helper_image=self.helper_image)
        # tmpfs and /dev/shm share the declared aggregate writable-byte allowance.
        budget = spec.resources.disk_mb * 1024 * 1024
        shm_bytes = min(64 * 1024 * 1024, budget // 2)
        argv = [
            "docker",
            "create",
            "--name",
            name,
            "--label",
            f"{_LABEL}={spec.run_id}",
            # Forced removal must not wait for a graceful stop. Docker kills at
            # once; Podman honours this per-container timeout before SIGKILL.
            "--stop-timeout",
            "0",
            "--init",
            "--user",
            "65532:65532",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            f"{spec.resources.memory_mb}m",
            "--memory-swap",
            f"{spec.resources.memory_mb}m",
            "--cpus",
            str(spec.resources.cpu),
            *(
                ["--network", "none"]
                if isolated
                else ["--network", "bridge", "--publish", "127.0.0.1::8000"]
            ),
            "--shm-size",
            str(shm_bytes),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={budget - shm_bytes},mode=1777",
            "--log-driver",
            "json-file",
            "--log-opt",
            "max-size=1m",
            "--log-opt",
            "max-file=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
        ]
        for key, value in sorted(spec.env_vars.items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.append(spec.image_ref)
        deadline = time.monotonic() + spec.startup_timeout_s
        try:
            subject._run(argv, deadline - time.monotonic())
            details = subject._raw_inspect(deadline - time.monotonic())
            mounts = details.get("Mounts", [])
            if any(mount.get("Type") in {"bind", "volume"} for mount in mounts):
                raise StartupError("Image declares unsupported persistent mounts")
            subject._run(["docker", "start", name], deadline - time.monotonic())
            if isolated:
                subject._start_helper(deadline - time.monotonic())
                code, _, _ = subject._helper_exec(
                    WAIT_READY_SCRIPT,
                    [str(max(deadline - time.monotonic() - 1, 0.1))],
                    deadline - time.monotonic(),
                )
                if code != 0:
                    raise StartupError(
                        "Subject did not return HTTP 200 from /health before its deadline"
                    )
                subject._bridge = ExecRelayBridge(subject._helper)
                subject.base_url = f"http://127.0.0.1:{subject._bridge.port}"
                return subject
            details = subject._raw_inspect(deadline - time.monotonic())
            ports = details.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp")
            if not ports or ports[0].get("HostIp") != "127.0.0.1":
                raise StartupError("Docker did not bind the API to host loopback")
            port = int(ports[0]["HostPort"])
            if not 0 < port < 65536:
                raise StartupError("Docker returned an invalid API port")
            subject.base_url = f"http://127.0.0.1:{port}"
            # Never send local validation requests through a host HTTP proxy.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _NoRedirects()
            )
            while time.monotonic() < deadline:
                try:
                    with opener.open(
                        subject.base_url + "/health",
                        timeout=min(1.0, deadline - time.monotonic()),
                    ) as response:
                        if response.status == 200:
                            return subject
                except (OSError, urllib.error.URLError):
                    # Connection failures are expected while the server starts.
                    pass
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            raise StartupError(
                "Subject did not return HTTP 200 from /health before its deadline"
            )
        except BaseException as exc:
            try:
                subject.stop()
            except ProviderError as cleanup:
                raise StartupError(
                    f"Startup failed; cleanup also failed: {cleanup}"
                ) from None
            if isinstance(exc, (KeyboardInterrupt, SystemExit, StartupError)):
                raise
            raise StartupError(_safe_text(str(exc), subject._secrets)) from None


class DockerRunningSubject:
    """An owned container; only this container is removed during cleanup."""

    def __init__(self, name: str, spec: LaunchSpec, helper_image: str = HELPER_IMAGE):
        self.name = name
        self.run_id = spec.run_id
        self.base_url = ""
        self._stopped = False
        self._secrets = tuple(spec.env_vars.values())
        self._mode = spec.network.mode
        self._helper_image = helper_image
        self._helper: str | None = None
        self._bridge: ExecRelayBridge | None = None
        # Run-owned helper, sink and control containers, removed before the subject.
        self._auxiliary: list[str] = []

    def _owned_create(self, name: str, argv: list[str], timeout_s: float) -> None:
        self._auxiliary.append(name)
        self._run(
            ["docker", "create", "--name", name, "--label", f"{_LABEL}={self.run_id}"]
            + argv,
            timeout_s,
        )
        self._run(["docker", "start", name], timeout_s)

    def _start_helper(self, timeout_s: float) -> None:
        """Join only the subject's network namespace from the validator's own image."""
        if self._helper is not None:
            return
        helper = f"{self.name}-netns"
        self._owned_create(
            helper,
            [
                "--network",
                f"container:{self.name}",
                *HARDENING,
                self._helper_image,
                "sleep",
                "infinity",
            ],
            timeout_s,
        )
        self._helper = helper

    def _helper_exec(self, script: str, args: list[str], timeout_s: float):
        return _command(
            ["docker", "exec", self._helper, "python3", "-c", script, *args], timeout_s
        )

    def measure_network(self, timeout_s: float = 120) -> dict:
        """
        Measure reachability of a validator-owned sink from the subject's namespace.

        The same probes run from a control container with ordinary networking, so a
        probe kind counts as evidence only when the sink is demonstrably reachable.

        Args:
            timeout_s (`float`, *optional*, defaults to `120`):
                Budget for creating the sink and running both probe sets.

        Returns:
            `dict`: a validated [`~openenv.validation.runtime.contracts.NetworkEvidence`].
        """
        _ensure_image(self._helper_image)
        deadline = time.monotonic() + timeout_s

        def remaining():
            return deadline - time.monotonic()

        self._start_helper(remaining())
        sink = f"{self.name}-sink"
        self._owned_create(
            sink,
            [
                "--network",
                "bridge",
                *HARDENING,
                self._helper_image,
                "python3",
                "-c",
                SINK_SCRIPT,
            ],
            remaining(),
        )
        try:
            sink_ip = _container_ip(self._run(["docker", "inspect", sink], remaining()))
            control = f"{self.name}-control"
            self._auxiliary.append(control)
            code, out, _ = _command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    control,
                    "--label",
                    f"{_LABEL}={self.run_id}",
                    "--network",
                    "bridge",
                    *HARDENING,
                    self._helper_image,
                    "python3",
                    "-c",
                    PROBE_SCRIPT,
                    sink_ip,
                    "15",
                ],
                remaining(),
            )
            if code:
                raise ProviderError("Control network probe failed")
            control_probes = json.loads(out)["probes"]
            code, out, _ = self._helper_exec(PROBE_SCRIPT, [sink_ip], remaining())
            if code:
                raise ProviderError("Subject namespace probe failed")
            measured = json.loads(out)
            mode = (
                self._raw_inspect(remaining()).get("HostConfig", {}).get("NetworkMode")
            )
            evidence = NetworkEvidence.model_validate(
                {
                    "requested_mode": self._mode,
                    "subject_network_mode": str(mode),
                    "namespace": measured["namespace"],
                    "probes": [
                        {
                            "kind": kind,
                            "control": control_probes[kind],
                            "subject": measured["probes"][kind],
                        }
                        for kind in ("tcp", "udp", "icmp")
                    ],
                }
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("Network probes returned invalid evidence") from exc
        finally:
            self._remove_owned(sink)
        return evidence.model_dump(mode="json")

    def _run(self, argv: list[str], timeout_s: float, max_bytes: int = _MAX_OUTPUT):
        code, stdout, stderr = _command(argv, timeout_s, max_bytes)
        if code:
            raise ProviderError(
                "Docker operation failed: " + _safe_text(stderr[-4096:], self._secrets)
            )
        return stdout

    def _raw_inspect(self, timeout_s: float = 10) -> dict:
        output = self._run(["docker", "inspect", self.name], timeout_s)
        try:
            details = json.loads(output)[0]
            if details["Config"].get("Labels", {}).get(_LABEL) != self.run_id:
                raise ProviderError("Container ownership label does not match")
            return details
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            raise ProviderError("Docker returned invalid inspection evidence") from None

    def inspect(self) -> dict:
        """Return selected effective settings without image env, command or host paths."""
        details = self._raw_inspect()
        host = details.get("HostConfig", {})
        return {
            "container_id": details.get("Id"),
            "image_id": details.get("Image"),
            "running": details.get("State", {}).get("Running", False),
            "oom_killed": details.get("State", {}).get("OOMKilled", False),
            "user": details.get("Config", {}).get("User"),
            "limits": {
                key: host.get(key)
                for key in (
                    "Memory",
                    "MemorySwap",
                    "NanoCpus",
                    "PidsLimit",
                    "ReadonlyRootfs",
                    "CapDrop",
                    "SecurityOpt",
                    "Init",
                    "Tmpfs",
                    "ShmSize",
                    "NetworkMode",
                )
            },
            "mounts": [
                {key: mount.get(key) for key in ("Type", "Destination", "RW")}
                for mount in details.get("Mounts", [])
            ],
        }

    def logs(self, max_bytes: int = _MAX_OUTPUT) -> str:
        if not 0 < max_bytes <= _MAX_OUTPUT:
            raise ValueError("max_bytes must be between 1 and 65536")
        code, stdout, stderr = _command(
            ["docker", "logs", "--tail", "1000", self.name], 10, max_bytes
        )
        if code:
            raise ProviderError("Could not read subject logs")
        return (
            _safe_text(stdout + stderr, self._secrets)
            .encode("utf-8")[-max_bytes:]
            .decode("utf-8", "ignore")
        )

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        if (
            not argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or not argv[0]
        ):
            raise ValueError(
                "exec requires a nonempty argument vector without null bytes"
            )
        if self._stopped:
            raise ProviderError("Subject has stopped")
        started = time.monotonic()
        try:
            code, stdout, stderr = _command(
                ["docker", "exec", "--", self.name, *argv], timeout_s
            )
        except ProviderError as exc:
            # Killing a docker exec client alone does not kill in-container children.
            try:
                self.stop()
            except ProviderError as cleanup:
                raise cleanup from exc
            raise
        return ExecResult(
            code,
            _safe_text(stdout, self._secrets),
            _safe_text(stderr, self._secrets),
            time.monotonic() - started,
        )

    def _owner(self, name: str | None = None) -> str | None:
        """Return the owner label of this subject's container, or `None` if absent.

        A listing reports a missing container as an empty result on every engine,
        whereas inspect errors are engine-specific text. Only the name and owner
        label are requested, so arbitrary image ENV/labels cannot exceed the
        bounded output and block cleanup.
        """
        name = name or self.name
        code, output, _ = _command(
            [
                "docker",
                "ps",
                "--all",
                "--no-trunc",
                "--filter",
                f"name={name}",
                "--format",
                f'{{{{.Names}}}}\t{{{{.Label "{_LABEL}"}}}}',
            ],
            10,
        )
        if code:
            raise ProviderError("Docker could not list containers")
        # The name filter matches substrings; only an exact name is this container.
        for line in output.splitlines():
            listed, _, owner = line.partition("\t")
            if listed == name:
                return owner
        return None

    def _remove_owned(self, name: str) -> None:
        """Remove a run-owned auxiliary container, verified by listing."""
        owner = self._owner(name)
        if owner is None:
            return
        if owner != self.run_id:
            raise ProviderError("Refusing cleanup of a container with another owner")
        _command(["docker", "rm", "--force", "--volumes", name], 10)
        if self._owner(name) is not None:
            raise ProviderError("Container removal could not be independently verified")

    def stop(self) -> None:
        if self._stopped:
            return
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        # Helpers share the subject's namespace; remove them before the subject.
        auxiliary_cleanup_failed = False
        for name in reversed(self._auxiliary):
            try:
                self._remove_owned(name)
            except ProviderError:
                # One failed helper must not leave the other containers running.
                auxiliary_cleanup_failed = True
        try:
            owner = self._owner()
        except ProviderError:
            raise ProviderError(
                "Could not verify container ownership for cleanup"
            ) from None
        if owner is not None:
            if owner != self.run_id:
                raise ProviderError(
                    "Refusing cleanup of a container with another owner"
                )
            code, _, stderr = _command(
                ["docker", "rm", "--force", "--volumes", self.name], 10
            )
            try:
                remaining = self._owner()
            except ProviderError:
                raise ProviderError(
                    "Container removal could not be independently verified"
                ) from None
            if remaining is not None:
                if code:
                    raise ProviderError(
                        "Could not remove validation container: "
                        + _safe_text(stderr[-4096:], self._secrets)
                    )
                raise ProviderError(
                    "Container removal could not be independently verified"
                )
        if auxiliary_cleanup_failed:
            raise ProviderError("Could not verify network helper cleanup")
        self._stopped = True
