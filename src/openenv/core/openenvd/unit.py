# SPDX-License-Identifier: BSD-3-Clause

"""One environment unit: zones of containers, the relays between them, and the phases.

openenvd runs in the unit's parent namespaces. For each episode it builds:

- `zones/agent/env`: the unmodified env server, listening on `env.sock`;
- `zones/agent/harness` (optional): sees only `MCP_URL` and `MODEL_BASE_URL`,
  each forwarded to a relay in the control plane;
- `zones/services/<name>`: hidden services, reachable from the env container
  only through a service relay;
- `zones/observers/<name>`: read-only sidecars that run in the phases they list.

Every container's only way out is its own socket directory, mounted at
`/run/openenvd`. Graders start only after the agent zone is empty.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from .cgroups import CgroupTree
from .contract import AGENT_PHASES, ContainerSpec, Manifest, Phase, ZoneKind
from .custody import copy_validated, CustodyError
from .oci import ContainerPlan, MountPlan, write_bundle
from .phases import Actor, PhaseError, PhaseMachine, Transition
from .relays import (
    env_relay_app,
    model_proxy_app,
    serve_asgi_on_unix,
    service_relay_app,
    stop_server,
)
from .runtime import (
    bind_mount,
    ContainerRuntime,
    mount_overlay,
    RuntimeFailure,
    unmount,
)
from .shim import shim_spec

logger = logging.getLogger(__name__)

SOCKET_DIR = "/run/openenvd"
WORKSPACE = "/workspace"
HARNESS_PORTS = {"mcp": 8000, "model": 8001}
PLACEHOLDER_KEY = "openenvd-placeholder"
_BASE_UID = 200_000
_UID_RANGE = 65_536
_PRINCIPAL_UID = 1000


class AgentMisbehavior(RuntimeError):
    """The agent's own subtree caused the failure. The episode scores 0."""


class InfrastructureError(RuntimeError):
    """A fault outside the agent's subtree. Never a score, never the agent's fault."""


@dataclass
class ModelRoute:
    """Where the model proxy sends a harness's calls.

    Attributes:
        base_url (`str`):
            Upstream base URL, for example the trainer's sampler.
        key (`str`):
            The real credential. It stays in the control plane.
        provider (`str`, *optional*, defaults to `"anthropic"`):
            `anthropic` or `openai`; decides the auth header.
    """

    base_url: str
    key: str
    provider: str = "anthropic"


@dataclass
class UnitPaths:
    """Where openenvd keeps its state. None of it is mounted into a zone.

    Attributes:
        state (`Path`):
            Episode overlays, private asset copy, runtime logs.
        run (`Path`):
            Bundles, OCI runtime state and the per-container socket directories.
        assets (`Path`):
            The environment's privileged assets, as shipped.
        seed (`Path`, *optional*):
            Read-only lower layer for `/workspace`.
    """

    state: Path
    run: Path
    assets: Path
    seed: Path | None = None

    @property
    def hidden(self) -> list[str]:
        return [str(self.state.parent), str(self.assets)]


class Launcher(Protocol):
    async def start(self, plan: ContainerPlan, bundle_dir: Path) -> None: ...
    async def stop(self, name: str) -> None: ...
    async def wait(self, name: str, timeout: float) -> bool: ...


@dataclass
class OciLauncher:
    """Starts each container through an OCI runtime."""

    runtime: ContainerRuntime
    unit_id: str
    hidden: list[str] = field(default_factory=list)

    def _cid(self, name: str) -> str:
        return f"openenv-{self.unit_id}-{name}"

    async def start(self, plan: ContainerPlan, bundle_dir: Path) -> None:
        config = write_bundle(bundle_dir, plan)
        spec = json.loads(config.read_text())
        masked = spec.setdefault("linux", {}).setdefault("maskedPaths", [])
        masked += [p for p in self.hidden if p not in masked]
        config.write_text(json.dumps(spec, indent=2))
        await self.runtime.delete(self._cid(plan.name))
        await self.runtime.run(self._cid(plan.name), bundle_dir)

    async def stop(self, name: str) -> None:
        await self.runtime.kill(self._cid(name))
        await self.runtime.delete(self._cid(name))

    async def wait(self, name: str, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.runtime.wait(self._cid(name)), timeout)
            return True
        except asyncio.TimeoutError:
            return False


def _ensure_mountpoints(mounts: list[MountPlan], root: Path = Path("/")) -> None:
    """Create missing mount destinations under `root`.

    A container whose rootfs is a read-only bind of `/` can't create its own
    mountpoints, and an image's directories under a tmpfs such as `/run` are
    hidden once the machine boots, so openenvd makes them before the start.
    """
    for m in mounts:
        dest = root / m.destination.lstrip("/")
        if dest.exists():
            continue
        if Path(m.source).is_dir():
            dest.mkdir(mode=0o755, parents=True)
        else:
            dest.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            dest.touch()


@dataclass
class _Container:
    zone: ZoneKind
    name: str
    spec: ContainerSpec
    index: int

    @property
    def cgroup(self) -> str:
        return f"zones/{self.zone.value}/{self.name}"

    @property
    def key(self) -> str:
        return f"{self.zone.value}-{self.name}"


@dataclass
class Verdict:
    """What grading produced for one episode."""

    status: str
    observers: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


class Unit:
    """Builds and drives one environment unit.

    Args:
        manifest ([`Manifest`]):
            The validated `openenvd:` block.
        paths (`UnitPaths`):
            Private directories of the control plane.
        launcher (`Launcher`):
            Creates containers.
        cgroups ([`CgroupTree`]):
            The unit's cgroup root.
        model (`ModelRoute`, *optional*):
            Upstream for the harness's model calls.
        python (`str`, *optional*, defaults to `"/usr/local/bin/python3"`):
            Interpreter inside the rootfs, used for the shim and the env server.
    """

    def __init__(
        self,
        manifest: Manifest,
        paths: UnitPaths,
        launcher: Launcher,
        cgroups: CgroupTree,
        *,
        model: ModelRoute | None = None,
        python: str = "/usr/local/bin/python3",
    ) -> None:
        self.manifest = manifest
        self.paths = paths
        self.launcher = launcher
        self.cgroups = cgroups
        self.model = model
        self.python = python
        self.machine = PhaseMachine(on_transition=self._on_transition)
        self._finishing: asyncio.Task | None = None
        self.episode_id: str | None = None
        self.verdict: Verdict | None = None
        self._servers: list[Any] = []
        self._mounted: list[Path] = []
        self._running: set[str] = set()
        self._lock = asyncio.Lock()
        self._containers = self._enumerate()
        self._root_binds: set[Path] = set()
        # Traverse-only: containers' runtimes resolve mount sources through these
        # paths from inside their user namespaces. Every zone masks them.
        for d in (paths.state, paths.run):
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o711)

    # ------------------------------------------------------------------ layout

    def _enumerate(self) -> dict[str, _Container]:
        out: dict[str, _Container] = {}
        index = 0
        for kind in ZoneKind:
            for name, spec in self.manifest.zones.get(kind).containers.items():
                c = _Container(kind, name, spec, index)
                out[c.key] = c
                index += 1
        return out

    @property
    def phase(self) -> Phase:
        return self.machine.phase

    def _zone_containers(self, kind: ZoneKind) -> list[_Container]:
        return [c for c in self._containers.values() if c.zone is kind]

    def _episode_dir(self) -> Path:
        assert self.episode_id is not None
        return self.paths.state / "episodes" / self.episode_id

    def _sock_dir(self, c: _Container) -> Path:
        return self.paths.run / "sock" / c.key

    def _uid(self, c: _Container) -> int:
        return _BASE_UID + c.index * _UID_RANGE

    def _on_transition(self, t: Transition) -> None:
        logger.info(
            "episode %s: %s -> %s (%s, %s)",
            self.episode_id,
            t.from_phase.value,
            t.to_phase.value,
            t.actor.value,
            t.reason,
        )

    # ------------------------------------------------------------------ plans

    def _mounts_for(self, c: _Container) -> list[MountPlan]:
        sock = self._sock_dir(c)
        mounts = [
            MountPlan(str(sock), SOCKET_DIR, options=["rbind", "nosuid", "nodev"])
        ]
        ep = self._episode_dir()
        if c.zone is ZoneKind.AGENT and c.name == "env":
            mounts.append(
                MountPlan(
                    str(ep / "workspace" / "merged"), WORKSPACE, options=["rbind"]
                )
            )
            hosts = sock / "hosts"
            mounts.append(MountPlan(str(hosts), "/etc/hosts", options=["rbind", "ro"]))
        elif c.zone is ZoneKind.SERVICES and c.spec.state:
            mounts.append(
                MountPlan(str(ep / "services" / c.name / "merged"), c.spec.state)
            )
        elif c.zone is ZoneKind.OBSERVERS:
            ro = ["rbind", "ro", "nosymfollow", "nodev", "nosuid", "noexec"]
            out = ep / "observers" / c.name / "out"
            mounts.append(
                MountPlan(str(out), "/out", options=["rbind", "nosuid", "nodev"])
            )
            view = self._view_dir(c)
            kinds = {r.partition(".")[0] for r in c.spec.reads}
            if "workspace" in kinds:
                src = ep / "workspace" / "merged"
                mounts.append(MountPlan(str(src), WORKSPACE, options=ro))
            if "assets" in kinds:
                mounts.append(MountPlan(str(view / "assets"), "/assets", options=ro))
            if "services" in kinds:
                mounts.append(
                    MountPlan(str(view / "services"), "/services", options=ro)
                )
            if any(r.endswith(".output") for r in c.spec.reads):
                src = ep / "observers" / c.name / "inputs"
                mounts.append(MountPlan(str(src), "/inputs", options=ro))
        return mounts

    def _argv(self, c: _Container) -> list[str]:
        if c.spec.argv:
            return list(c.spec.argv)
        if c.zone is ZoneKind.AGENT and c.name == "env":
            if not self.manifest.app:
                raise InfrastructureError("manifest has no app for the env container")
            return [
                self.python,
                "-m",
                "uvicorn",
                self.manifest.app,
                "--uds",
                f"{SOCKET_DIR}/env.sock",
                "--no-access-log",
            ]
        raise InfrastructureError(f"{c.key} has no argv")

    def _env(self, c: _Container) -> dict[str, str]:
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if c.zone is ZoneKind.AGENT and c.name == "harness":
            base = "http://127.0.0.1"
            env.update(
                MCP_URL=f"{base}:{HARNESS_PORTS['mcp']}/mcp",
                MODEL_BASE_URL=f"{base}:{HARNESS_PORTS['model']}",
                ANTHROPIC_BASE_URL=f"{base}:{HARNESS_PORTS['model']}",
                ANTHROPIC_API_KEY=PLACEHOLDER_KEY,
                OPENAI_BASE_URL=f"{base}:{HARNESS_PORTS['model']}/v1",
                OPENAI_API_KEY=PLACEHOLDER_KEY,
            )
        if c.zone is ZoneKind.OBSERVERS:
            env["OPENENVD_PHASE"] = self.phase.value
        if c.zone is ZoneKind.AGENT and c.name == "env":
            for svc, address in self._service_addresses():
                key = svc.name.upper().replace("-", "_")
                env[f"OPENENVD_SERVICE_{key}"] = (
                    f"http://{svc.spec.expose.host}:{svc.spec.expose.port}"
                )
        for name, path in (
            ("SOCKETS", SOCKET_DIR),
            ("WORKSPACE", WORKSPACE),
            ("OUT", "/out"),
            ("INPUTS", "/inputs"),
            ("ASSETS", "/assets"),
            ("SERVICES", "/services"),
        ):
            env[f"OPENENVD_{name}"] = path
        env.update(c.spec.env)
        return env

    def _forwards(self, c: _Container) -> list[dict]:
        if c.zone is ZoneKind.AGENT and c.name == "harness":
            return [
                {
                    "kind": "tcp_to_unix",
                    "host": "127.0.0.1",
                    "port": port,
                    "path": f"{SOCKET_DIR}/{name}.sock",
                }
                for name, port in HARNESS_PORTS.items()
            ]
        if c.zone is ZoneKind.AGENT and c.name == "env":
            return [
                {
                    "kind": "tcp_to_unix",
                    "host": address,
                    "port": svc.spec.expose.port,
                    "path": f"{SOCKET_DIR}/svc-{svc.name}.sock",
                }
                for svc, address in self._service_addresses()
            ]
        if c.zone is ZoneKind.SERVICES and c.spec.port:
            return [
                {
                    "kind": "unix_to_tcp",
                    "path": f"{SOCKET_DIR}/svc.sock",
                    "host": "127.0.0.1",
                    "port": c.spec.port,
                }
            ]
        if c.zone is ZoneKind.OBSERVERS and c.spec.network and c.spec.network.model:
            return [
                {
                    "kind": "tcp_to_unix",
                    "host": "127.0.0.1",
                    "port": HARNESS_PORTS["model"],
                    "path": f"{SOCKET_DIR}/model.sock",
                }
            ]
        return []

    def _service_addresses(self) -> list[tuple[_Container, str]]:
        exposed = [c for c in self._zone_containers(ZoneKind.SERVICES) if c.spec.expose]
        return [(c, f"127.0.0.{10 + i}") for i, c in enumerate(exposed)]

    def _rootfs(self, rootfs: str) -> str:
        """`/` can't be pivoted onto, so containers get a recursive bind of it."""
        if rootfs != "/":
            return rootfs
        target = self.paths.run / "rootfs"
        if target not in self._root_binds:
            bind_mount(Path("/"), target, read_only=True, nodev=False, nosuid=False)
            self._root_binds.add(target)
        return str(target)

    def _plan(self, c: _Container) -> tuple[ContainerPlan, dict]:
        zone = self.manifest.zones.get(c.zone)
        resources = zone.resources.narrowed(c.spec.resources)
        sock = self._sock_dir(c)
        plan = ContainerPlan(
            name=c.key,
            zone=c.zone,
            argv=self._argv(c),
            env=self._env(c),
            rootfs=self._rootfs(c.spec.rootfs),
            uid=self._uid(c),
            gid=self._uid(c),
            hostname=c.name,
            cgroups_path="/" + c.cgroup,
            mounts=self._mounts_for(c),
            resources=resources,
            shim_spec_path=f"{SOCKET_DIR}/shim.json",
            python=self.python,
            userns=True,
        )
        workdir = WORKSPACE if c.zone is ZoneKind.AGENT and c.name == "env" else "/tmp"
        spec = shim_spec(
            argv=plan.argv,
            env=plan.env,
            forwards=self._forwards(c),
            workdir=workdir,
        )
        (sock / "shim.json").write_text(json.dumps(spec))
        return plan, spec

    # ------------------------------------------------------------------ files

    def _prepare_dirs(self) -> None:
        ep = self._episode_dir()
        ep.mkdir(parents=True, exist_ok=True)
        for d in (ep.parent, ep):
            d.chmod(0o711)
        assets = self.paths.state / "assets"
        if not assets.exists():
            assets.mkdir()
            for name, rel in self.manifest.privileged_assets.items():
                src = self.paths.assets / rel
                dst = assets / name
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    dst.mkdir()
                    shutil.copyfile(src, dst / Path(rel).name)
            for root, dirs, files in os.walk(assets):
                os.chmod(root, 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o644)
        for c in self._containers.values():
            sock = self._sock_dir(c)
            if sock.exists():
                shutil.rmtree(sock)
            sock.mkdir(parents=True)
            for d in (sock.parent, sock):
                d.chmod(0o755)
            self._chown(sock, c)
        seed = self.paths.seed
        lower = seed if seed and seed.exists() else ep / "empty"
        lower.mkdir(exist_ok=True)
        self._overlay(lower, ep / "workspace", self._containers["agent-env"])
        for c in self._zone_containers(ZoneKind.SERVICES):
            if c.spec.state:
                svc_lower = ep / "services" / c.name / "seed"
                svc_lower.mkdir(parents=True, exist_ok=True)
                self._overlay(svc_lower, ep / "services" / c.name, c)
        for c in self._zone_containers(ZoneKind.OBSERVERS):
            out = ep / "observers" / c.name / "out"
            out.mkdir(parents=True, exist_ok=True)
            (ep / "observers" / c.name / "inputs").mkdir(exist_ok=True)
            self._chown(out, c)
        env_sock = self._sock_dir(self._containers["agent-env"])
        lines = ["127.0.0.1 localhost", "::1 localhost"]
        lines += [
            f"{addr} {svc.spec.expose.host}" for svc, addr in self._service_addresses()
        ]
        (env_sock / "hosts").write_text("\n".join(lines) + "\n")

    def _chown(self, path: Path, c: _Container) -> None:
        uid = self._uid(c) + _PRINCIPAL_UID
        try:
            os.chown(path, uid, uid)
        except PermissionError:
            pass

    def _overlay(self, lower: Path, root: Path, owner: _Container) -> None:
        upper, work, merged = root / "upper", root / "work", root / "merged"
        try:
            mount_overlay(lower, upper, work, merged)
            self._mounted.append(merged)
        except RuntimeFailure:
            merged.mkdir(parents=True, exist_ok=True)
            if lower.exists():
                shutil.copytree(lower, merged, dirs_exist_ok=True)
        self._chown(merged, owner)
        self._chown(upper, owner)

    # ------------------------------------------------------------------ relays

    async def _serve(self, app: Any, path: Path) -> None:
        self._servers.append(await serve_asgi_on_unix(app, str(path)))

    async def _start_relays(self) -> None:
        env = self._containers["agent-env"]
        env_sock = str(self._sock_dir(env) / "env.sock")
        for svc, _ in self._service_addresses():
            upstream = str(self._sock_dir(svc) / "svc.sock")
            app = service_relay_app(upstream)
            await self._serve(app, self._sock_dir(env) / f"svc-{svc.name}.sock")
        harness = self._containers.get("agent-harness")
        if harness is not None:
            hdir = self._sock_dir(harness)
            relay = env_relay_app(env_sock, on_activity=self.note_activity)
            await self._serve(relay, hdir / "mcp.sock")
            if self.model is not None:
                proxy = model_proxy_app(
                    self.model.base_url,
                    self.model.key,
                    provider=self.model.provider,
                    placeholder_key=PLACEHOLDER_KEY,
                )
                await self._serve(proxy, hdir / "model.sock")
        for c in self._zone_containers(ZoneKind.OBSERVERS):
            if c.spec.network and c.spec.network.model and self.model is not None:
                proxy = model_proxy_app(
                    self.model.base_url,
                    self.model.key,
                    provider=self.model.provider,
                    placeholder_key=PLACEHOLDER_KEY,
                )
                await self._serve(proxy, self._sock_dir(c) / "model.sock")

    async def _stop_relays(self) -> None:
        servers, self._servers = self._servers, []
        for server in servers:
            try:
                await stop_server(server)
            except Exception:
                logger.exception("relay did not stop cleanly")

    # ------------------------------------------------------------------ containers

    async def _start(self, c: _Container) -> None:
        plan, _ = self._plan(c)
        if c.spec.rootfs == "/":
            _ensure_mountpoints(plan.mounts)
        self.cgroups.create(c.cgroup)
        self.cgroups.set_limits(c.cgroup, plan.resources)
        bundle = self.paths.run / "bundles" / c.key
        bundle.mkdir(parents=True, exist_ok=True)
        for d in (bundle.parent, bundle):
            d.chmod(0o711)
        await self.launcher.start(plan, bundle)
        self._running.add(c.key)

    async def _stop(self, c: _Container) -> None:
        if c.key in self._running:
            self._running.discard(c.key)
            try:
                await self.launcher.stop(c.key)
            except Exception:
                logger.exception("could not stop %s", c.key)

    async def _wait_socket(self, path: Path, timeout: float, *, http: bool) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                if not http:
                    return True
                try:
                    transport = httpx.AsyncHTTPTransport(uds=str(path))
                    async with httpx.AsyncClient(transport=transport) as client:
                        r = await client.get("http://env/health", timeout=1.0)
                    if r.status_code < 500:
                        return True
                except httpx.HTTPError:
                    pass
            await asyncio.sleep(0.05)
        return False

    def _zone_resources(self) -> None:
        for kind in ZoneKind:
            rel = f"zones/{kind.value}"
            self.cgroups.create(rel)
            self.cgroups.set_limits(rel, self.manifest.zones.get(kind).resources)

    def _observers_for(self, phase: Phase) -> list[_Container]:
        return [
            c
            for c in self._zone_containers(ZoneKind.OBSERVERS)
            if phase in self.manifest.container_phases(ZoneKind.OBSERVERS, c.name)
        ]

    # ------------------------------------------------------------------ lifecycle

    def info(self) -> dict[str, Any]:
        """What the orchestrator may know: the phase and the episode's verdict."""
        return {
            "isolation": "enabled",
            "phase": self.phase.value,
            "episode_id": self.episode_id,
            "verdict": self.verdict.status if self.verdict else None,
        }

    def verdict_info(self) -> dict[str, Any]:
        """The last episode's grading results, for the orchestrator."""
        if self.verdict is None:
            return {"episode_id": self.episode_id, "status": None}
        return {
            "episode_id": self.episode_id,
            "status": self.verdict.status,
            "reason": self.verdict.reason,
            "observers": self.verdict.observers,
        }

    async def reset(self) -> dict[str, Any]:
        """Tear down the current episode, if any, and provision a fresh one."""
        async with self._lock:
            if self.phase not in (Phase.CLOSED, Phase.FAILED):
                await self._teardown()
                if self.phase is not Phase.FAILED:
                    self.machine.phase = Phase.CLOSED
            self.machine.transition(Phase.PROVISIONING, Actor.ORCHESTRATOR, "reset")
            self.episode_id = uuid.uuid4().hex[:12]
            self.verdict = None
            try:
                await self._provision()
            except Exception as exc:
                await self._fail(f"provisioning failed: {exc}")
                raise InfrastructureError("episode_unavailable") from exc
            self.machine.transition(Phase.READY, Actor.SYSTEM, "healthy")
            self._start_observers(Phase.READY)
            return self.info()

    async def _provision(self) -> None:
        self._prepare_dirs()
        self._zone_resources()
        for c in self._zone_containers(ZoneKind.SERVICES):
            await self._start(c)
        for c in self._zone_containers(ZoneKind.SERVICES):
            if c.spec.port and not await self._wait_socket(
                self._sock_dir(c) / "svc.sock", 30, http=False
            ):
                raise InfrastructureError(f"service {c.name} did not start")
        await self._start_relays()
        env = self._containers["agent-env"]
        await self._start(env)
        if not await self._wait_socket(self._sock_dir(env) / "env.sock", 60, http=True):
            raise InfrastructureError("env server did not become healthy")
        harness = self._containers.get("agent-harness")
        if harness is not None:
            await self._start(harness)

    def _start_observers(self, phase: Phase) -> None:
        for c in self._observers_for(phase):
            if c.key not in self._running:
                asyncio.create_task(self._run_background_observer(c))

    async def _run_background_observer(self, c: _Container) -> None:
        try:
            self._stage_inputs(c)
            await self._start(c)
        except Exception:
            logger.exception("observer %s failed to start", c.name)

    def note_activity(self) -> None:
        """The orchestrator or agent sent a step: the episode is running."""
        if self.phase is Phase.READY:
            self.machine.transition(Phase.RUNNING, Actor.SYSTEM, "first step")

    async def inspect(self) -> dict[str, Any]:
        """Freeze the agent zone and services, and run observers that list `frozen`."""
        async with self._lock:
            self.machine.transition(Phase.FROZEN, Actor.ORCHESTRATOR, "inspect")
            self.cgroups.freeze("zones/agent")
            self.cgroups.freeze("zones/services")
            await self.cgroups.wait_frozen("zones/agent", 5)
            results = await self._run_observers(self._observers_for(Phase.FROZEN))
            return {"phase": self.phase.value, "observers": results}

    async def resume(self) -> dict[str, Any]:
        async with self._lock:
            self.machine.transition(Phase.RUNNING, Actor.ORCHESTRATOR, "resume")
            self.cgroups.thaw("zones/services")
            self.cgroups.thaw("zones/agent")
            return self.info()

    async def on_episode_done(self, reason: str = "done") -> None:
        """The environment reported `done`: stop the agent and grade in the background.

        Returns at once, so the relay can deliver the final observation first.
        """
        if self._finishing is None or self._finishing.done():
            self._finishing = asyncio.create_task(self._finish(reason))

    async def end_episode(self) -> dict[str, Any]:
        """Orchestrator: end the episode now, stop the agent zone and grade."""
        if self.phase not in (Phase.READY, Phase.RUNNING, Phase.FROZEN):
            raise PhaseError(f"no episode to end in {self.phase.value}")
        self._finishing = asyncio.create_task(self._finish("ended_by_orchestrator"))
        await self._finishing
        return self.info()

    async def wait_finished(self) -> None:
        if self._finishing is not None:
            await self._finishing

    async def _finish(self, reason: str) -> None:
        await asyncio.sleep(0.2)
        async with self._lock:
            if self.phase not in (Phase.READY, Phase.RUNNING, Phase.FROZEN):
                return
            actor = (
                Actor.ENVIRONMENT if self.phase is not Phase.FROZEN else Actor.SYSTEM
            )
            try:
                await self._stop_agent(reason, actor)
            except AgentMisbehavior as exc:
                self.verdict = Verdict("agent_misbehavior", reason=str(exc))
                await self._close()
                return
            await self._grade()

    async def _stop_agent(self, reason: str, actor: Actor) -> None:
        """Kill the agent zone and wait until the kernel reports it empty.

        Moves the unit to `sealed`: the agent's outputs are fixed from here on.
        """
        if self.manifest.zones.services.during_grading == "frozen":
            self.cgroups.freeze("zones/services")
        self.cgroups.thaw("zones/agent")
        atomic = self.cgroups.kill("zones/agent")
        for c in self._zone_containers(ZoneKind.AGENT):
            await self._stop(c)
        emptied = await self.cgroups.wait_empty(
            "zones/agent", self.manifest.kill_deadline_s
        )
        logger.info("agent zone stopped (atomic=%s, emptied=%s)", atomic, emptied)
        if not emptied:
            raise AgentMisbehavior("agent zone did not empty before the deadline")
        self.machine.transition(Phase.SEALED, actor, reason)

    async def _grade(self) -> None:
        self.machine.transition(Phase.GRADING, Actor.SYSTEM, "populated 0")
        try:
            results = await self._run_observers(self._observers_for(Phase.GRADING))
        except AgentMisbehavior as exc:
            self.verdict = Verdict("agent_misbehavior", reason=str(exc))
        except InfrastructureError as exc:
            self.verdict = Verdict("infrastructure_error", reason=str(exc))
        except Exception as exc:
            logger.exception("grading failed")
            self.verdict = Verdict("infrastructure_error", reason=type(exc).__name__)
        else:
            self.verdict = Verdict("graded", observers=results)
        await self._close()

    def _order(self, observers: list[_Container]) -> list[_Container]:
        names = {c.name: c for c in observers}
        ordered: list[_Container] = []
        seen: set[str] = set()

        def visit(c: _Container) -> None:
            if c.name in seen:
                return
            seen.add(c.name)
            if c.spec.after and c.spec.after in names:
                visit(names[c.spec.after])
            ordered.append(c)

        for c in observers:
            visit(c)
        return ordered

    async def _run_observers(self, observers: list[_Container]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        ep = self._episode_dir()
        for c in self._order([o for o in observers if o.key not in self._running]):
            self._stage_inputs(c)
            await self._start(c)
            finished = await self.launcher.wait(c.key, timeout=300)
            stats = self.cgroups.stats(c.cgroup)
            await self._stop(c)
            await self.cgroups.wait_empty(c.cgroup, 5)
            if stats.get("memory_events", {}).get("oom_kill"):
                raise InfrastructureError(f"observer {c.name} ran out of memory")
            if not finished:
                raise InfrastructureError(f"observer {c.name} timed out")
            verdict_file = ep / "observers" / c.name / "out" / "verdict.json"
            result: Any = None
            if verdict_file.is_file() and not verdict_file.is_symlink():
                try:
                    result = json.loads(verdict_file.read_text()[:1_000_000])
                except ValueError:
                    result = {"error": "verdict.json is not JSON"}
            results[c.name] = result
        return results

    def _view(self, src: Path, target: Path) -> None:
        """Expose `src` read-only, without following links, at `target`."""
        if target in self._mounted:
            return
        bind_mount(src, target, read_only=True, nosymfollow=True, noexec=True)
        self._mounted.append(target)

    def _view_dir(self, c: _Container) -> Path:
        return self.paths.run / "views" / c.key

    def _stage_inputs(self, c: _Container) -> None:
        """Build the observer's read-only views, then copy upstream outputs in."""
        ep = self._episode_dir()
        view = self._view_dir(c)
        for kind in ("assets", "services"):
            (view / kind).mkdir(parents=True, exist_ok=True)
        for d in (view.parent, view, view / "assets", view / "services"):
            d.chmod(0o755)
        for read in c.spec.reads:
            head, _, rest = read.partition(".")
            if head == "assets":
                self._view(self.paths.state / "assets" / rest, view / "assets" / rest)
            elif head == "services":
                svc = rest.split(".")[0]
                src = ep / "services" / svc / "merged"
                self._view(src, view / "services" / svc)
        for read in c.spec.reads:
            head, _, rest = read.partition(".")
            if rest == "output":
                src = ep / "observers" / head / "out"
                dst = ep / "observers" / c.name / "inputs" / head
                if dst.exists():
                    shutil.rmtree(dst)
                try:
                    copy_validated(src, dst, max_bytes=256 << 20, max_files=10_000)
                except CustodyError as exc:
                    raise AgentMisbehavior(f"{head} output: {exc}") from exc
                for root, dirs, files in os.walk(dst):
                    for name in [root, *(os.path.join(root, f) for f in files)]:
                        self._chown(Path(name), c)

    async def close(self) -> dict[str, Any]:
        """Orchestrator `close`: end the episode without grading."""
        async with self._lock:
            if self.phase in (Phase.READY, Phase.RUNNING, Phase.FROZEN):
                try:
                    await self._stop_agent("closed", Actor.ORCHESTRATOR)
                except AgentMisbehavior as exc:
                    self.verdict = Verdict("agent_misbehavior", reason=str(exc))
            await self._close()
            return self.info()

    async def _close(self) -> None:
        await self._teardown()
        if self.phase in (Phase.GRADING, Phase.SEALED):
            self.machine.transition(Phase.CLOSED, Actor.SYSTEM, "verdict")
        elif self.phase is Phase.FAILED:
            self.machine.transition(Phase.CLOSED, Actor.SYSTEM, "cleanup")

    async def _teardown(self) -> None:
        for kind in ZoneKind:
            rel = f"zones/{kind.value}"
            try:
                self.cgroups.thaw(rel)
                self.cgroups.kill(rel)
            except Exception:
                pass
        for c in list(self._containers.values()):
            await self._stop(c)
        for kind in ZoneKind:
            await self.cgroups.wait_empty(f"zones/{kind.value}", 5)
        await self._stop_relays()
        for merged in reversed(self._mounted):
            unmount(merged)
        self._mounted.clear()
        try:
            self.cgroups.remove("zones")
        except Exception:
            logger.exception("could not remove the zone cgroups")

    async def _fail(self, reason: str) -> None:
        logger.error("unit failed: %s", reason)
        try:
            self.machine.transition(Phase.FAILED, Actor.SYSTEM, reason)
        except PhaseError:
            pass
        self.verdict = Verdict("infrastructure_error", reason=reason)
        await self._teardown()

    async def shutdown(self) -> None:
        async with self._lock:
            await self._teardown()
            for target in self._root_binds:
                unmount(target)
            self._root_binds.clear()


__all__ = [
    "AGENT_PHASES",
    "AgentMisbehavior",
    "InfrastructureError",
    "ModelRoute",
    "OciLauncher",
    "Unit",
    "UnitPaths",
    "Verdict",
]
