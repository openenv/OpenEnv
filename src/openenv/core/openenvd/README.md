# openenvd

openenvd is the control plane of an environment unit (RFC 009, [#1053](https://github.com/huggingface/OpenEnv/issues/1053)).
It runs inside the unit as PID 1, in the parent namespaces, and builds three
zones of containers in child namespaces:

| Zone | Containers | Can reach |
|------|------------|-----------|
| `agent` | `env` (the unmodified env server), optional `harness` | Only its own socket directory: relays for MCP, the model and hidden services |
| `services` | hidden simulators, for example `airbnb-sim` | Nothing. The `env` container calls them through a relay |
| `observers` | grader, judge, metrics | Read-only views of the workspace, assets and service state |

The orchestrator talks to one listener (`:8100`): `/ws` (Gym, token), `/mcp`
(agent, unchanged), and `/info`, `/verdict`, `/reset_unit`, `/inspect`,
`/resume` and `/end_episode` (token).

## Modules

| Module | Role |
|--------|------|
| `contract.py` | The `openenvd:` manifest block: zones, per-zone policy ceilings, phases |
| `phases.py` | Canonical phases and who may cause each transition |
| `unit.py` | Builds an episode: overlays, socket dirs, relays, containers, stopping the agent, grading |
| `preflight.py` | Can openenvd isolate here? Root, namespaces, a writable cgroup v2 root and an OCI runtime |
| `cgroups.py` | Zone subtrees: limits, freeze, atomic kill, `populated 0` |
| `oci.py`, `seccomp.py`, `runtime.py` | OCI bundles and the `crun` driver |
| `shim.py`, `forwarder.py` | Container PID 1: reaps children, runs the loopback forwarders |
| `relays.py`, `surfaces.py` | Relays (placeholder model keys, hidden-service proxying) and the external listener |
| `custody.py` | Validated copies of agent output for graders |
| `check.py` | `openenv check enforcement`: probe containers try what each zone must not do |

## Isolation on or off

openenvd is in every image. At startup it runs the preflight:

- **Privileged** (a microVM, or `docker run --privileged --cgroupns=private`):
  isolation is enabled. openenvd builds the zones and serves the unit's
  listener on the manifest's `port`.
- **Anything else** (a plain container, a Space, macOS): openenvd logs
  `isolation disabled` with the reasons and runs the environment's server
  directly, exactly as the image always has.

`--require-isolation` (or `OPENENVD_REQUIRE_ISOLATION=1`) turns the fallback
into a refusal, for production launchers that must never run without it.
`openenv check preflight` says which way it would go here.

## Running

```bash
python -m openenv.core.openenvd --manifest openenv.yaml --tokens tokens.json
```

Container paths are also exposed as `OPENENVD_WORKSPACE`, `OPENENVD_OUT`,
`OPENENVD_INPUTS`, `OPENENVD_ASSETS`, `OPENENVD_SERVICES` and
`OPENENVD_SOCKETS`. The env container gets `OPENENVD_SERVICE_<NAME>` for each
hidden service.

Containers run on a read-only bind of the image's `/`, so they can read
everything in the image except the control-plane state, secrets and declared
assets, which are hidden. openenvd creates any missing mountpoints itself.
Only the mounts it makes are writable: `/tmp`, `/workspace` (env), `/out`
(observers) and each service's `state` path.

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd
# Live, in Docker: a privileged run (isolated) and a plain run (fallback)
tests/core/test_openenvd/integration/run.sh
```
