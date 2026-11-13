# SPDX-License-Identifier: BSD-3-Clause

"""openenvd as PID 1 of an environment unit.

Every image starts openenvd. When it can isolate (it runs privileged: inside a
microVM, or a container started with `--privileged`), it builds the zones and
serves the unit's one listener. When it can't, it logs why and runs the
environment's server directly, exactly as the image always has.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path

from .cgroups import CgroupTree
from .contract import load_manifest, Manifest
from .preflight import preflight
from .relays import stop_server
from .runtime import OciRuntime
from .surfaces import serve_surfaces, surfaces_app
from .unit import ModelRoute, OciLauncher, Unit, UnitPaths

logger = logging.getLogger("openenvd")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openenvd", description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("openenv.yaml"))
    p.add_argument("--assets", type=Path, help="asset root (default: manifest dir)")
    p.add_argument("--seed", type=Path, help="read-only seed for /workspace")
    p.add_argument("--state", type=Path, default=Path("/var/lib/openenvd"))
    p.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--port", type=int, help="listen port (default: the manifest's `port`)"
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--tokens",
        type=Path,
        help="JSON file with the orchestrator token; read once, then deleted",
    )
    p.add_argument("--model-url", help="upstream for harness model calls")
    p.add_argument(
        "--model-key-file", type=Path, help="file holding the real model key; deleted"
    )
    p.add_argument("--model-provider", default="anthropic")
    p.add_argument(
        "--require-isolation",
        action="store_true",
        default=os.environ.get("OPENENVD_REQUIRE_ISOLATION") == "1",
        help="refuse to start instead of running without isolation",
    )
    p.add_argument("--check", action="store_true", help="run the enforcement check")
    p.add_argument(
        "--preflight", action="store_true", help="say whether isolation would be on"
    )
    return p


def _read_once(path: Path) -> str:
    text = path.read_text()
    try:
        path.unlink()
    except OSError:
        pass
    return text


def _orchestrator_token(path: Path | None) -> str:
    if path is None:
        token = os.environ.pop("OPENENVD_ORCHESTRATOR_TOKEN", "")
    else:
        token = json.loads(_read_once(path)).get("orchestrator", "")
    if not token:
        token = secrets.token_urlsafe(32)
        logger.warning("no orchestrator token given; generated one for this run")
        print(f"OPENENVD_ORCHESTRATOR_TOKEN={token}", flush=True)
    return token


def build_unit(args: argparse.Namespace, manifest: Manifest, runtime: str) -> Unit:
    """Adopt the cgroup root and assemble the unit."""
    state = args.state.resolve()
    paths = UnitPaths(
        state=state / "state",
        run=state / "run",
        assets=(args.assets or args.manifest.parent).resolve(),
        seed=args.seed
        or (Path(manifest.workspace_seed) if manifest.workspace_seed else None),
    )
    for d in (paths.state, paths.run):
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o711)
    asset_sources = {
        str(
            (paths.assets / rel).parent
            if (paths.assets / rel).is_file()
            else paths.assets / rel
        )
        for rel in manifest.privileged_assets.values()
    }
    hidden = sorted({str(state), "/run/secrets", *asset_sources})
    cgroups = CgroupTree(args.cgroup_root)
    cgroups.adopt_self("control")
    oci = OciRuntime(runtime, paths.run / "oci", paths.state / "logs")
    launcher = OciLauncher(oci, secrets.token_hex(4), hidden=hidden)
    model = None
    if args.model_url:
        key = _read_once(args.model_key_file).strip() if args.model_key_file else ""
        model = ModelRoute(args.model_url, key, args.model_provider)
    return Unit(manifest, paths, launcher, cgroups, model=model, python=args.python)


def run_unisolated(args: argparse.Namespace, manifest: Manifest) -> int:
    """Replace openenvd with the environment's own server, as the image always ran."""
    if not manifest.app:
        print("openenvd: openenv.yaml names no app to serve", file=sys.stderr)
        return 2
    port = str(args.port or manifest.port)
    argv = [args.python, "-m", "uvicorn", manifest.app, "--host", args.host]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(args.python, [*argv, "--port", port])
    return 0  # not reached


async def serve(args: argparse.Namespace, manifest: Manifest, runtime: str) -> int:
    unit = build_unit(args, manifest, runtime)
    if args.check:
        from .check import run_check

        result = await run_check(unit)
        print(json.dumps(result.to_json(), indent=2))
        await unit.shutdown()
        for failure in result.failures:
            print(f"openenvd check: {failure}", file=sys.stderr)
        return 1 if result.failures else 0
    port = args.port or manifest.port
    orchestrator = _orchestrator_token(args.tokens)
    env_sock = str(unit._sock_dir(unit._containers["agent-env"]) / "env.sock")
    app = surfaces_app(unit, env_sock, orchestrator_token=orchestrator)
    server = await serve_surfaces(app, args.host, port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    logger.info("isolation enabled; listening on %s:%s", args.host, port)
    print("openenvd ready", flush=True)
    await stop.wait()
    await stop_server(server)
    await unit.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("OPENENVD_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    result = preflight(args.cgroup_root, manifest.zones.agent.runtime.oci)
    if args.preflight:
        print(
            json.dumps(
                {"isolation": result.enabled, "reasons": result.reasons}, indent=2
            )
        )
        return 0
    reasons = list(result.reasons)
    if not manifest.enabled:
        reasons = ["openenvd.enabled is false in openenv.yaml"]
    if reasons:
        if args.require_isolation or args.check:
            for reason in reasons:
                print(f"openenvd: cannot isolate: {reason}", file=sys.stderr)
            return 3
        logger.warning("isolation disabled: %s", "; ".join(reasons))
        logger.warning("serving the environment directly, without openenvd")
        return run_unisolated(args, manifest)
    return asyncio.run(serve(args, manifest, result.runtime))
