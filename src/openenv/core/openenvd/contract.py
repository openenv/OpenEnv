# SPDX-License-Identifier: BSD-3-Clause

"""The `openenvd:` contract: zones, per-zone policies and phases.

A manifest declares three zones. Each zone sets a policy ceiling (network,
resources, OCI runtime) and each container narrows it. The kernel enforces the
narrowing on its own (nested cgroup limits, per-container network namespaces);
`validate_manifest` reports violations early with a readable error.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestError(ValueError):
    """Raised when an `openenvd:` block violates the contract."""


class ZoneKind(str, Enum):
    AGENT = "agent"
    SERVICES = "services"
    OBSERVERS = "observers"


class Phase(str, Enum):
    """The canonical phases of a unit. Exactly one holds at any time."""

    PROVISIONING = "provisioning"
    READY = "ready"
    RUNNING = "running"
    FROZEN = "frozen"
    SEALED = "sealed"
    GRADING = "grading"
    CLOSED = "closed"
    FAILED = "failed"


AGENT_PHASES: tuple[Phase, ...] = (Phase.READY, Phase.RUNNING)
"""Agent-zone containers live only here. Manifests can't change it."""

ASSET_PHASES: frozenset[Phase] = frozenset({Phase.FROZEN, Phase.GRADING})
"""The only phases in which a container may read privileged assets."""

_EGRESS_RANK = {"none": 0, "relays-only": 1, "allowlist": 2}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Resources(_Strict):
    """cgroup v2 limits. `None` means no limit at this level.

    Attributes:
        memory_mb (`int`, *optional*):
            `memory.max`, in MiB.
        pids (`int`, *optional*):
            `pids.max`.
        cpu (`float`, *optional*):
            CPUs, written to `cpu.max` as quota over a 100ms period.
    """

    memory_mb: int | None = Field(default=None, gt=0)
    pids: int | None = Field(default=None, gt=0)
    cpu: float | None = Field(default=None, gt=0)

    def within(self, ceiling: "Resources") -> list[str]:
        problems = []
        for name in ("memory_mb", "pids", "cpu"):
            mine, cap = getattr(self, name), getattr(ceiling, name)
            if cap is not None and mine is not None and mine > cap:
                problems.append(f"{name} {mine} exceeds the zone ceiling {cap}")
        return problems

    def narrowed(self, child: "Resources") -> "Resources":
        """The effective limits of a child under this ceiling."""
        merged = {}
        for name in ("memory_mb", "pids", "cpu"):
            mine, cap = getattr(child, name), getattr(self, name)
            values = [v for v in (mine, cap) if v is not None]
            merged[name] = min(values) if values else None
        return Resources(**merged)


class NetworkPolicy(_Strict):
    """Where a zone or container may connect.

    Attributes:
        egress (`str`, *optional*, defaults to `"none"`):
            `none`, `relays-only` (only the recording relays openenvd mounts in)
            or `allowlist` (the relays plus the named endpoints in `allow`).
        allow (`list[str]`, *optional*):
            Named endpoints, resolved by the control plane.
        model (`str`, *optional*):
            Model provider reachable through the model proxy.
    """

    egress: Literal["none", "relays-only", "allowlist"] = "none"
    allow: list[str] = Field(default_factory=list)
    model: str | None = None

    def within(self, ceiling: "NetworkPolicy") -> list[str]:
        problems = []
        if _EGRESS_RANK[self.egress] > _EGRESS_RANK[ceiling.egress]:
            problems.append(
                f"egress {self.egress!r} is wider than the zone's {ceiling.egress!r}"
            )
        extra = sorted(set(self.allow) - set(ceiling.allow))
        if extra:
            problems.append(f"allow {extra} is not in the zone allowlist")
        return problems


class RuntimePolicy(_Strict):
    """Which OCI runtime creates the zone's containers.

    Attributes:
        oci (`str`, *optional*, defaults to `"crun"`):
            `crun`, `runc` or `runsc`.
        require (`bool`, *optional*, defaults to `False`):
            Refuse to start if this runtime is missing, instead of using `crun`.
    """

    oci: Literal["crun", "runc", "runsc"] = "crun"
    require: bool = False


class Expose(_Strict):
    host: str
    port: int = Field(gt=0, lt=65536)


class ContainerSpec(_Strict):
    """One container in a zone. Every field narrows the zone's policy.

    Attributes:
        argv (`list[str]`, *optional*):
            Command to run. The env container defaults to the env server.
        rootfs (`str`, *optional*, defaults to `"/"`):
            Directory used as the container's read-only root.
        env (`dict[str, str]`, *optional*):
            Extra environment variables.
        resources (`Resources`, *optional*):
            Leaf cgroup limits.
        network (`NetworkPolicy`, *optional*):
            Narrows the zone's network policy.
        phases (`list[Phase]`, *optional*):
            Observers only: the phases this container lives in.
        reads (`list[str]`, *optional*):
            Observers only: `assets.<name>`, `workspace`, `cgroups`,
            `services.<name>.state` or `<observer>.output`.
        expose (`Expose`, *optional*):
            Services only: the hostname and port the agent zone may call.
        state (`str`, *optional*):
            Services only: the directory holding the service's state.
        output (`str`, *optional*):
            Observers only: the file a downstream observer may read.
        after (`str`, *optional*):
            Observers only: start after this observer's cgroup is empty.
        port (`int`, *optional*):
            The port the process listens on inside its own network namespace.
    """

    argv: list[str] | None = None
    rootfs: str = "/"
    env: dict[str, str] = Field(default_factory=dict)
    resources: Resources = Field(default_factory=Resources)
    network: NetworkPolicy | None = None
    phases: list[Phase] | None = None
    reads: list[str] = Field(default_factory=list)
    expose: Expose | None = None
    state: str | None = None
    output: str | None = None
    after: str | None = None
    port: int | None = Field(default=None, gt=0, lt=65536)


class ZoneSpec(_Strict):
    """A zone's ceiling plus its containers.

    Attributes:
        runtime (`RuntimePolicy`, *optional*):
            OCI runtime for every container in the zone.
        network (`NetworkPolicy`, *optional*):
            Ceiling for the containers' network.
        resources (`Resources`, *optional*):
            Limits on the zone's cgroup, shared by all its containers.
        containers (`dict[str, ContainerSpec]`, *optional*):
            The zone's containers, by name.
        during_grading (`str`, *optional*, defaults to `"frozen"`):
            Services only: `frozen` or `serve-observers`.
    """

    runtime: RuntimePolicy = Field(default_factory=RuntimePolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    resources: Resources = Field(default_factory=Resources)
    containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    during_grading: Literal["frozen", "serve-observers"] = "frozen"

    def effective_resources(self, name: str) -> Resources:
        return self.resources.narrowed(self.containers[name].resources)


def _default_agent_zone() -> ZoneSpec:
    return ZoneSpec(
        network=NetworkPolicy(egress="relays-only"),
        containers={"env": ContainerSpec()},
    )


class Zones(_Strict):
    agent: ZoneSpec = Field(default_factory=_default_agent_zone)
    services: ZoneSpec = Field(default_factory=ZoneSpec)
    observers: ZoneSpec = Field(default_factory=ZoneSpec)

    @model_validator(mode="before")
    @classmethod
    def _zone_defaults(cls, data: Any) -> Any:
        """Fill each zone's unset policies with that zone type's defaults."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        given = data.get("agent")
        if isinstance(given, dict):
            merged = dict(given)
            network = merged.get("network", {})
            if isinstance(network, dict):
                merged["network"] = {
                    **_default_agent_zone().network.model_dump(),
                    **network,
                }
            merged.setdefault("containers", {"env": {}})
            data["agent"] = merged
        return data

    def get(self, kind: ZoneKind) -> ZoneSpec:
        return getattr(self, kind.value)


class Manifest(_Strict):
    """The `openenvd:` block of `openenv.yaml`.

    Attributes:
        enabled (`bool`, *optional*, defaults to `True`):
            Whether openenvd runs this environment.
        app (`str`, *optional*):
            `module:attr` of the environment's ASGI app. Defaults to the
            top-level `app` of `openenv.yaml`.
        port (`int`, *optional*, defaults to `8000`):
            Port the unit listens on. Defaults to the top-level `port`.
        zones (`Zones`, *optional*):
            The agent, services and observers zones.
        privileged_assets (`dict[str, str]`, *optional*):
            Asset name to path, relative to the asset root.
        workspace_seed (`str`, *optional*):
            Directory used as the read-only lower layer of `/workspace`.
        kill_deadline_s (`float`, *optional*, defaults to `10.0`):
            How long `agent/` may take to empty before it counts as misbehavior.
    """

    enabled: bool = True
    app: str | None = None
    port: int = Field(default=8000, gt=0, lt=65536)
    zones: Zones = Field(default_factory=Zones)
    privileged_assets: dict[str, str] = Field(default_factory=dict)
    workspace_seed: str | None = None
    kill_deadline_s: float = Field(default=10.0, gt=0)

    def container_phases(self, zone: ZoneKind, name: str) -> tuple[Phase, ...]:
        """The phases in which a container lives."""
        if zone is ZoneKind.AGENT:
            return AGENT_PHASES
        if zone is ZoneKind.SERVICES:
            return (
                Phase.READY,
                Phase.RUNNING,
                Phase.FROZEN,
                Phase.SEALED,
                Phase.GRADING,
            )
        return tuple(self.zones.observers.containers[name].phases or ())


def _is_under(path: str, root: str) -> bool:
    p, r = Path(path), Path(root)
    return p == r or r in p.parents


def _read_target(read: str) -> tuple[str, str | None]:
    head, _, rest = read.partition(".")
    return head, rest or None


def validate_manifest(manifest: Manifest) -> list[str]:
    """Return every contract violation in `manifest`, or an empty list.

    Args:
        manifest (`Manifest`):
            The parsed `openenvd:` block.

    Returns:
        `list[str]`: Human-readable problems, each naming the zone and container.
    """
    problems: list[str] = []
    zones = manifest.zones

    for kind in ZoneKind:
        zone = zones.get(kind)
        for name, spec in zone.containers.items():
            where = f"zones.{kind.value}.containers.{name}"
            if spec.network is not None:
                problems += [f"{where}: {p}" for p in spec.network.within(zone.network)]
            problems += [f"{where}: {p}" for p in spec.resources.within(zone.resources)]

    agent = zones.agent
    if "env" not in agent.containers:
        problems.append("zones.agent: an 'env' container is required")
    if agent.network.egress != "relays-only" or agent.network.allow:
        problems.append(
            "zones.agent: egress is fixed to relays-only so every call is recorded"
        )
    for name, spec in agent.containers.items():
        where = f"zones.agent.containers.{name}"
        if spec.phases is not None:
            problems.append(f"{where}: agent-zone phases are fixed to [ready, running]")
        if spec.reads:
            problems.append(f"{where}: only observers may declare reads")
        if spec.expose or spec.state or spec.output or spec.after:
            problems.append(f"{where}: expose/state/output/after are not agent fields")

    services = zones.services
    for name, spec in services.containers.items():
        where = f"zones.services.containers.{name}"
        if spec.reads:
            problems.append(f"{where}: only observers may declare reads")
        if spec.phases is not None:
            problems.append(f"{where}: service phases follow the unit")
        if spec.output or spec.after:
            problems.append(f"{where}: output/after are observer fields")
        if spec.expose is not None and spec.port is None:
            problems.append(f"{where}: expose needs the service's listening port")

    observers = zones.observers
    if observers.containers and observers.during_grading != "frozen":
        problems.append("zones.observers: during_grading only applies to services")
    for name, spec in observers.containers.items():
        where = f"zones.observers.containers.{name}"
        if not spec.phases:
            problems.append(f"{where}: observers must list at least one phase")
        if spec.expose or spec.state:
            problems.append(f"{where}: expose/state are service fields")
        phases = set(spec.phases or ())
        for read in spec.reads:
            head, rest = _read_target(read)
            if head == "assets":
                if rest not in manifest.privileged_assets:
                    problems.append(f"{where}: unknown asset {rest!r}")
                if phases - ASSET_PHASES:
                    problems.append(
                        f"{where}: reads assets, so its phases must be within "
                        "[frozen, grading]"
                    )
            elif head == "services":
                svc, _, field = (rest or "").partition(".")
                if svc not in services.containers or field != "state":
                    problems.append(f"{where}: unknown read {read!r}")
                elif services.containers[svc].state is None:
                    problems.append(f"{where}: service {svc!r} declares no state")
            elif head in ("workspace", "cgroups"):
                if rest is not None:
                    problems.append(f"{where}: unknown read {read!r}")
            elif head in observers.containers and rest == "output":
                if observers.containers[head].output is None:
                    problems.append(f"{where}: observer {head!r} declares no output")
            else:
                problems.append(f"{where}: unknown read {read!r}")
        if spec.after is not None:
            if spec.after not in observers.containers or spec.after == name:
                problems.append(f"{where}: after must name another observer")
        if (
            any(r.startswith("assets.") for r in spec.reads)
            and "workspace" in spec.reads
        ):
            problems.append(
                f"{where}: reads both assets and workspace; split it into a runner "
                "(workspace, no assets) and a verdict that reads the runner's output"
            )

    hosts = [
        s.expose.host for s in services.containers.values() if s.expose is not None
    ]
    if len(hosts) != len(set(hosts)):
        problems.append("zones.services: expose hosts must be unique")
    return problems


def parse_manifest(data: dict[str, Any] | None) -> Manifest:
    """Parse and validate an `openenvd:` mapping.

    Args:
        data (`dict` or `None`):
            The value of the `openenvd:` key.

    Returns:
        [`Manifest`]: The validated manifest.

    Raises:
        `ManifestError`: If the block is malformed or breaks a zone rule.
    """
    try:
        manifest = Manifest.model_validate(data or {})
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError("; ".join(problems))
    return manifest


def load_manifest(path: str | Path) -> Manifest:
    """Load the `openenvd:` block from an `openenv.yaml` file.

    Args:
        path (`str` or `Path`):
            Path to `openenv.yaml`.

    Returns:
        [`Manifest`]: The validated manifest. A file without the block yields
        defaults, with `app` and `port` taken from the top level.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    block = dict(raw.get("openenvd") or {})
    if raw.get("app") and not block.get("app"):
        block["app"] = raw["app"]
    if raw.get("port") and not block.get("port"):
        block["port"] = raw["port"]
    return parse_manifest(block)
