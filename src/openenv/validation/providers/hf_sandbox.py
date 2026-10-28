"""Run a Hugging Face Space as a validation subject in a dedicated HF Sandbox.

A Space revision pins the downloaded source, not the image selected by HF Jobs.
This integration has no supported source-to-image digest resolver, so runtime
startup reports the missing immutable_image capability before creating a job.
The sandbox subject below retains bounded serving, exec and cleanup support for
provider qualification once that binding is available.
"""

import json
import re
import shlex
import time
from pathlib import Path

import httpx

from ...core.containers.runtime.hf_sandbox_provider import _LocalAuthProxy
from ..manifest import ExecutionDeclaration
from ..runtime.contracts import LaunchSpec
from ..types import ProviderCapability
from . import ExecResult, ProviderError, StartupError, UnsupportedCapability

_TERMINAL_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
_MAX_OUTPUT = 65536


def _server_command(dockerfile: str) -> list[str]:
    """The effective `CMD` of a Dockerfile, which a Sandbox job does not run itself."""
    if re.search(r"(?im)^#\s*escape\s*=\s*`\s*$", dockerfile):
        raise UnsupportedCapability("hf-sandbox requires backslash Dockerfile escapes")
    stages = {}
    stage = {}
    stage_number = 0
    lines = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    # Docker removes escaped newlines before interpreting an instruction.
    lines = lines.replace("\\\n", "")
    if "<<" in lines:
        raise UnsupportedCapability(
            "missing provider capability: dockerfile_heredoc; "
            "hf-sandbox does not interpret heredocs or literal << markers"
        )
    for line in lines.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        instruction, rest = parts[0].upper(), parts[1]
        if instruction == "FROM":
            base = rest.split()
            if base[0].startswith("--platform="):
                base = base[1:]
            stage = stages.get(base[0].lower(), {}).copy()
            stages[str(stage_number)] = stage
            stage_number += 1
            if len(base) == 3 and base[1].upper() == "AS":
                stages[base[2].lower()] = stage
        elif instruction in {"CMD", "ENTRYPOINT", "SHELL"}:
            stage[instruction] = rest
    if stage.get("ENTRYPOINT") not in (None, "[]"):
        raise UnsupportedCapability("hf-sandbox does not run Dockerfile ENTRYPOINT")
    rest = stage.get("CMD")
    if rest is None:
        raise StartupError("The Space Dockerfile declares no CMD to start the server")
    command = ["/bin/sh", "-c", rest]
    if rest.startswith("["):
        try:
            command = json.loads(rest)
        except ValueError:
            pass  # Not valid exec form, so Docker runs it as shell form.
    if command[:2] == ["/bin/sh", "-c"] and "SHELL" in stage:
        raise UnsupportedCapability("hf-sandbox does not run Dockerfile SHELL")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) for part in command)
    ):
        raise StartupError("The Space Dockerfile declares no CMD to start the server")
    return command


def _app_port(readme: str) -> int:
    """`app_port` from the Space card front matter; the Hub default is 7860."""
    front = readme.split("---", 2)[1] if readme.startswith("---") else ""
    match = re.search(r"^app_port:\s*(\d+)\s*$", front, re.M)
    return int(match.group(1)) if match else 7860


class HFSandboxValidationProvider:
    """
    Inspect a Hub Space for validation on HF Jobs hardware, including GPU flavors.

    Args:
        space_id (`str`):
            The Space to run, e.g. `"openenv/echo_env"`.
        revision (`str`):
            The full commit hash of the validated Space files; resolve a branch or
            tag first. Runtime remains unsupported until an immutable image can
            be bound to this source commit.
        flavor (`str`, *optional*, defaults to `"cpu-basic"`):
            HF Jobs hardware flavor, as listed by `hf jobs hardware`.
        token (`str`, *optional*):
            Hugging Face token; defaults to the locally saved one.
    """

    name = "hf-sandbox"
    supported_network_modes = frozenset({"public"})

    def __init__(
        self,
        space_id: str,
        revision: str,
        *,
        flavor: str = "cpu-basic",
        token: str | None = None,
    ):
        from huggingface_hub import HfApi

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"revision must be a full commit hash, not {revision!r}")
        self.space_id = space_id
        self.revision = revision
        self.flavor = flavor
        self._api = HfApi(token=token)
        hardware = {h.name: h for h in self._api.list_jobs_hardware()}
        if flavor not in hardware:
            raise ValueError(f"unknown HF Jobs flavor {flavor!r}")
        accelerator = hardware[flavor].accelerator
        self.gpus = int(accelerator.quantity) if accelerator else 0
        self.capabilities = frozenset(
            {
                ProviderCapability.IMAGE_BUILD,
                ProviderCapability.EXEC,
                ProviderCapability.REMOTE,
                *([ProviderCapability.GPU] if self.gpus else []),
            }
        )
        self._command: list[str] = []
        self._port = 7860

    def _check_space(self) -> None:
        """Compare source and reported runtime metadata without claiming image provenance."""
        try:
            info = self._api.space_info(self.space_id)
        except Exception as exc:
            raise ProviderError(f"Could not read Space {self.space_id}") from exc
        if info.sha != self.revision:
            raise ProviderError(
                f"Space {self.space_id} is at revision {info.sha}, not the "
                f"validated revision {self.revision}"
            )
        # Raw runtime metadata is not provenance for a separately launched Job.
        runtime = info.runtime.raw if info.runtime else {}
        stage, runtime_revision = runtime.get("stage"), runtime.get("sha")
        if stage not in (None, "RUNNING") or runtime_revision not in (
            None,
            self.revision,
        ):
            raise ProviderError(
                f"Space {self.space_id} reports stage {stage}, runtime revision "
                f"{runtime_revision}; expected RUNNING at source revision {self.revision}; "
                "job image identity remains unverified"
            )
        if stage is None or runtime_revision is None:
            raise UnsupportedCapability(
                "missing provider capability: space_runtime_metadata; "
                "Space runtime stage or revision is unavailable; "
                "job image identity remains unverified; no job launched"
            )

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        """Read the server command and port from the validated Space files."""
        try:
            dockerfile = (root / execution.dockerfile).read_text()
        except OSError:
            raise StartupError(
                f"The Space has no Dockerfile at {execution.dockerfile}"
            ) from None
        self._command = _server_command(dockerfile)
        readme = root / "README.md"
        self._port = _app_port(readme.read_text() if readme.is_file() else "")
        self._check_space()
        raise UnsupportedCapability(
            "missing provider capability: immutable_image; "
            f"Space source revision {self.revision} matches the Hub runtime revision, "
            "but no verified source-to-image digest binding is available; no job launched"
        )

    def start(self, spec: LaunchSpec) -> "HFSandboxSubject":
        if spec.network.mode not in self.supported_network_modes:
            raise UnsupportedCapability(
                f"hf-sandbox does not enforce {spec.network.mode}"
            )
        if spec.resources.gpus > self.gpus:
            raise UnsupportedCapability(
                f"flavor {self.flavor} has {self.gpus} GPUs, "
                f"{spec.resources.gpus} declared"
            )
        self._check_space()
        raise UnsupportedCapability(
            "missing provider capability: immutable_image; "
            f"no verified image digest for Space source revision {self.revision}; "
            "no job launched"
        )


class HFSandboxSubject:
    """A dedicated sandbox job serving one Space; stopping cancels the job."""

    def __init__(self, sandbox, provider: HFSandboxValidationProvider):
        self.base_url = ""
        self._sandbox = sandbox
        self._provider = provider
        self._proxy: _LocalAuthProxy | None = None
        self._stopped = False

    def _serve(self, command: list[str], port: int, timeout_s: float) -> None:
        self._sandbox.run(
            shlex.join(command)
            + f" 2>&1 | {{ head -c {_MAX_OUTPUT} > /tmp/openenv-server.log; "
            "cat > /dev/null; }",
            shell=True,
            background=True,
        )
        target = self._sandbox.proxy_url_for(port, "/")
        headers = self._sandbox.proxy_headers
        deadline = time.monotonic() + timeout_s
        # Until the subject binds its port, the sandbox's own server answers the
        # proxy, including a healthy /health; wait for the subject's listener first.
        listening = f"grep -qi ':{port:04X} 0\\+:0000 0A' /proc/net/tcp /proc/net/tcp6"
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            while time.monotonic() < deadline:
                try:
                    check = self._sandbox.run(
                        listening, shell=True, check=False, timeout=5
                    )
                    if check.exit_code == 0:
                        response = client.get(
                            target + "health", headers=headers, timeout=5
                        )
                        if response.status_code == 200:
                            break
                except httpx.HTTPError:
                    pass  # The server is still starting.
                time.sleep(1)
            else:
                raise StartupError(
                    "Subject did not return HTTP 200 from /health before its deadline"
                )
        self._proxy = _LocalAuthProxy(target_url=target, headers=headers)
        self.base_url = self._proxy.start()

    def inspect(self) -> dict:
        return {
            "job_id": self._sandbox.id,
            "image": f"hf.co/spaces/{self._provider.space_id}",
            # No source-bound image digest has been verified.
            "image_verified": False,
            "space_revision": self._provider.revision,
            "built_revision": None,
            "flavor": self._provider.flavor,
        }

    def logs(self, max_bytes: int = _MAX_OUTPUT) -> str:
        result = self._sandbox.run(
            ["tail", "-c", str(max_bytes), "/tmp/openenv-server.log"], check=False
        )
        return result.stdout[-max_bytes:]

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        if self._stopped:
            raise ProviderError("Subject has stopped")
        started = time.monotonic()
        result = self._sandbox.run(argv, shell=False, timeout=timeout_s, check=False)
        return ExecResult(
            result.exit_code,
            result.stdout[-_MAX_OUTPUT:],
            result.stderr[-_MAX_OUTPUT:],
            time.monotonic() - started,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        self._sandbox.kill()
        # kill() logs rather than raises on failure, so confirm with the Jobs API.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            stage = self._provider._api.inspect_job(
                job_id=self._sandbox.id
            ).status.stage
            if stage in _TERMINAL_STAGES:
                self._stopped = True
                return
            time.sleep(2)
        raise ProviderError(f"HF Sandbox job {self._sandbox.id} did not stop")
