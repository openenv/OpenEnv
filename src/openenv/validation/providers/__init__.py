"""Validation provider protocol.

Validation needs more than start/stop: network-policy enforcement and in-sandbox exec
are grader dependencies. Rather than widen the core provider ABCs, validation defines
its own protocol and adapts the core providers (Docker-local, HF Sandbox) behind it.

GPU is a provider capability, not an unsupported package category: packages declaring
GPU resources validate on providers that offer GPUs; elsewhere the affected checks
SKIP with the capability named.

Universal invariants inherited from core: internal port 8000, readiness =
`GET /health` returning 200.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..manifest import ExecutionDeclaration
from ..runtime.contracts import LaunchSpec
from ..types import ProviderCapability


class ProviderError(RuntimeError):
    """A bounded, sanitized provider operation failed."""


class StartupError(ProviderError):
    """A subject did not become healthy within its startup deadline."""


class UnsupportedCapability(ProviderError):
    """The provider cannot enforce the requested execution contract."""


@dataclass(frozen=True)
class ExecResult:
    """Result of executing a command inside the running sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


@runtime_checkable
class RunningSubject(Protocol):
    """A started validation subject: reachable, execable, stoppable."""

    base_url: str

    def inspect(self) -> dict: ...

    def logs(self, max_bytes: int = 65536) -> str: ...

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult: ...

    def stop(self) -> None: ...


@runtime_checkable
class ValidationProvider(Protocol):
    """
    Starts validation subjects in a sandbox with declared capabilities.

    A grader whose `requires_provider` names a capability the provider lacks is
    SKIPped with the capability named. The runner checks the manifest's declared
    network mode and GPU requirements before building. The provider must enforce
    the complete launch specification or refuse it. Enforcing `no-network` or
    `allowlist` modes requires the `NETWORK_POLICY` capability.
    """

    name: str
    capabilities: frozenset[ProviderCapability]
    supported_network_modes: frozenset[str]

    def build(self, root: Path, execution: ExecutionDeclaration) -> str: ...

    def start(self, spec: LaunchSpec) -> RunningSubject: ...
