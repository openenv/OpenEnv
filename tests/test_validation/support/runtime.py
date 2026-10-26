"""Small reusable fakes; real protocol and Docker tests establish execution evidence."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from openenv.validation.manifest import ExecutionDeclaration
from openenv.validation.providers import ExecResult
from openenv.validation.runtime.contracts import (
    LaunchSpec,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.types import ProviderCapability


def exchange(operation, request, response):
    """Build immutable evidence without hiding malformed wire values via coercion."""
    return WireExchange(operation, json.dumps(request), json.dumps(response))


def evidence(*exchanges, observation_schema=None):
    """Build one collector result from exact test exchanges and schema data."""
    schema = None if observation_schema is None else json.dumps(observation_schema)
    return RuntimeEvidence(tuple(exchanges), observation_schema_json=schema)


@dataclass
class FakeRunningSubject:
    """Inert subject recording teardown and exec, with no network or Docker calls."""

    base_url: str = "http://127.0.0.1:1"
    stopped: bool = False
    commands: list[list[str]] = field(default_factory=list)

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        self.commands.append(list(argv))
        return ExecResult(0, "", "", 0.0)

    def inspect(self) -> dict:
        return {"test_only": True}

    def logs(self, max_bytes: int = 65_536) -> str:
        return ""

    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeRuntimeProvider:
    """Validation-only launcher fake that records explicit launch requests."""

    name: str = "fake-runtime"
    capabilities: frozenset = frozenset(
        {ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC}
    )
    supported_network_modes: frozenset[str] = frozenset({"public"})
    subject: FakeRunningSubject = field(default_factory=FakeRunningSubject)
    launches: list[LaunchSpec] = field(default_factory=list)
    builds: list[Path] = field(default_factory=list)

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        self.builds.append(root)
        return "sha256:" + "a" * 64

    def start(self, spec: LaunchSpec) -> FakeRunningSubject:
        self.launches.append(spec)
        return self.subject
