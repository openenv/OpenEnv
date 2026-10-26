"""
Stress test simulating large-scale agentic RL with AWM environments.

Simulates one RL step: 2000 environments reset in parallel, each runs a
multi-turn episode (random tool calls with LLM-like latency), then closes.

Phases per session:
  1. connect + reset           — env startup
  2. list_tools                — tool discovery
  3. N turns of tool calls     — simulate multi-turn agent interaction
     (random tool, empty args, random "thinking" delay between turns)
  4. done + close              — episode end

Usage:
    # Terminal 1: Start server
    PYTHONPATH=src:envs uv run uvicorn \
        envs.agent_world_model_env.server.app:app \
        --host 0.0.0.0 --port 8899

    # Terminal 2: Run RL simulation (default 2000 envs)
    PYTHONPATH=src:envs uv run python \
        envs/agent_world_model_env/example_stress_test.py

    # Custom scale
    PYTHONPATH=src:envs uv run python \
        envs/agent_world_model_env/example_stress_test.py \
        --scale 500 --concurrency 100 --min-turns 1 --max-turns 3
"""

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx
import psutil
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rl_stress")

BASE_URL = os.environ.get("AWM_BASE_URL", "http://localhost:8899")
CLIENT_TIMEOUT: float = 600.0

# Scenarios to cycle through
SCENARIOS = [
    "e_commerce_33",
    "inventory_management_7",
    "document_management_5",
    "billing_payments_3",
    "hris_employee_management_1",
]

# RL simulation defaults
MIN_TURNS = 3
MAX_TURNS = 20
# Simulate LLM rollout time: uniform [min, max] seconds
LLM_THINK_MIN = 1.0
LLM_THINK_MAX = 20.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SessionResult:
    session_id: int
    scenario: str
    task_idx: int
    num_turns: int  # planned turns
    turns_completed: int = 0
    connect_s: float = 0.0
    reset_s: float = 0.0
    list_tools_s: float = 0.0
    tool_call_latencies: list[float] = field(default_factory=list)
    done_s: float = 0.0
    total_s: float = 0.0
    success: bool = False
    error: str | None = None
    num_tools: int = 0
    tools_discovered: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System resource monitor
# ---------------------------------------------------------------------------
class ResourceMonitor:
    """Periodically samples CPU and memory in a background task."""

    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._samples: list[dict] = []
        self._task: asyncio.Task | None = None
        self._process = psutil.Process(os.getpid())
        self._server_pid: int | None = None

    def start(self, server_pid: int | None = None):
        self._server_pid = server_pid
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            sample = {
                "time": time.monotonic(),
                "system_cpu_pct": psutil.cpu_percent(interval=0),
                "system_mem_pct": psutil.virtual_memory().percent,
                "system_mem_used_gb": round(
                    psutil.virtual_memory().used / (1024**3), 2
                ),
                "client_mem_mb": round(self._process.memory_info().rss / (1024**2), 1),
            }
            if self._server_pid:
                try:
                    server_proc = psutil.Process(self._server_pid)
                    children = server_proc.children(recursive=True)
                    server_mem = server_proc.memory_info().rss
                    for child in children:
                        try:
                            server_mem += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    sample["server_tree_mem_mb"] = round(server_mem / (1024**2), 1)
                    sample["server_children"] = len(children)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._samples.append(sample)
            await asyncio.sleep(self._interval)

    def summary(self) -> dict:
        if not self._samples:
            return {}
        cpu_vals = [s["system_cpu_pct"] for s in self._samples]
        mem_vals = [s["system_mem_used_gb"] for s in self._samples]
        client_mem = [s["client_mem_mb"] for s in self._samples]
        result = {
            "samples": len(self._samples),
            "cpu_pct": {
                "mean": round(statistics.mean(cpu_vals), 1),
                "max": round(max(cpu_vals), 1),
            },
            "system_mem_gb": {
                "min": round(min(mem_vals), 2),
                "max": round(max(mem_vals), 2),
            },
            "client_mem_mb": {
                "min": round(min(client_mem), 1),
                "max": round(max(client_mem), 1),
            },
        }
        server_mem = [
            s["server_tree_mem_mb"] for s in self._samples if "server_tree_mem_mb" in s
        ]
        if server_mem:
            result["server_tree_mem_mb"] = {
                "min": round(min(server_mem), 1),
                "max": round(max(server_mem), 1),
            }
        server_children = [
            s["server_children"] for s in self._samples if "server_children" in s
        ]
        if server_children:
            result["server_subprocess_peak"] = max(server_children)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def latency_stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "count": len(s),
        "min": round(min(s), 3),
        "p50": round(s[len(s) // 2], 3),
        "p90": round(s[int(len(s) * 0.9)], 3),
        "p99": round(s[int(len(s) * 0.99)], 3),
        "max": round(max(s), 3),
        "mean": round(statistics.mean(s), 3),
    }


async def check_server(url: str) -> int | None:
    """Check server is up, return server PID if available."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/docs", timeout=10)
        resp.raise_for_status()
    # Try to find server PID by matching the port in the URL
    from urllib.parse import urlparse

    port = str(urlparse(url).port or "8899")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "uvicorn" in cmdline and port in cmdline:
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


async def fetch_server_stats(url: str) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/stats", timeout=5)
            return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single session: full RL episode
# ---------------------------------------------------------------------------
@dataclass
class ProgressCounters:
    """Shared counters updated by each session for live progress reporting."""

    done: int = 0
    ok: int = 0
    fail: int = 0
    resets_done: int = 0
    turns_done: int = 0


async def run_rl_episode(
    session_id: int,
    scenario: str,
    task_idx: int,
    num_turns: int,
    reset_semaphore: asyncio.Semaphore,
    interact_semaphore: asyncio.Semaphore,
    counters: ProgressCounters,
) -> SessionResult:
    """Simulate a full RL episode: reset -> list_tools -> N tool calls -> done."""
    r = SessionResult(
        session_id=session_id,
        scenario=scenario,
        task_idx=task_idx,
        num_turns=num_turns,
    )
    session_start = time.monotonic()
    phase = "init"
    env = AWMEnv(
        base_url=BASE_URL, message_timeout_s=CLIENT_TIMEOUT, connect_timeout_s=60.0
    )

    try:
        # -- Phase 1: connect + reset (rate-limited to avoid thundering herd) --
        async with reset_semaphore:
            phase = "connect"
            t0 = time.monotonic()
            await env.connect()
            r.connect_s = time.monotonic() - t0

            phase = "reset"
            t0 = time.monotonic()
            result = await env.reset(scenario=scenario, task_idx=task_idx)
            r.reset_s = time.monotonic() - t0

            if result.observation.reward_type not in ("reset_ok", "reset_warning"):
                r.error = f"reset failed: {result.observation.error}"
                counters.done += 1
                counters.fail += 1
                return r

            r.num_tools = result.observation.num_tools or 0
            counters.resets_done += 1

        # -- Phase 2: list_tools --
        phase = "list_tools"
        t0 = time.monotonic()
        result = await env.step(ListToolsAction())
        r.list_tools_s = time.monotonic() - t0

        # Collect tool names for random calling
        obs = result.observation
        if hasattr(obs, "tools") and obs.tools:
            r.tools_discovered = [
                t.get("name", t.get("tool_name", ""))
                for t in obs.tools
                if isinstance(t, dict)
            ]
        if not r.tools_discovered:
            r.tools_discovered = ["unknown_tool"]

        # -- Phase 3: multi-turn tool calling (simulate agent interaction) --
        async with interact_semaphore:
            for turn in range(num_turns):
                phase = f"turn_{turn}"

                # Simulate LLM thinking time (async sleep = non-blocking)
                think_time = random.uniform(LLM_THINK_MIN, LLM_THINK_MAX)
                await asyncio.sleep(think_time)

                # Pick a random tool and call with empty args (will fail, that's fine)
                tool_name = random.choice(r.tools_discovered)
                t0 = time.monotonic()
                try:
                    result = await env.step(
                        CallToolAction(tool_name=tool_name, arguments={})
                    )
                except Exception:
                    # Tool call failure is expected (no args), just measure latency
                    pass
                r.tool_call_latencies.append(time.monotonic() - t0)
                r.turns_completed += 1
                counters.turns_done += 1

        # -- Phase 4: done + close --
        phase = "done"
        t0 = time.monotonic()
        await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )
        r.done_s = time.monotonic() - t0

        r.success = True
        counters.ok += 1

    except Exception as e:
        r.error = f"[{phase}] {type(e).__name__}: {str(e)[:200]}"
        counters.fail += 1
    finally:
        r.total_s = time.monotonic() - session_start
        counters.done += 1
        try:
            await env.close()
        except Exception:
            pass

    return r


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------
async def progress_reporter(
    counters: ProgressCounters,
    total: int,
    total_turns: int,
    monitor: ResourceMonitor,
    interval: float = 10.0,
):
    """Periodically log progress while the test runs."""
    start = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - start
        in_flight = total - counters.done

        stats = await fetch_server_stats(BASE_URL)
        server_sessions = stats.get("total_sessions", "?") if stats else "?"

        # Current resource snapshot
        samples = monitor._samples
        last = samples[-1] if samples else {}
        cpu = last.get("system_cpu_pct", "?")
        mem = last.get("system_mem_used_gb", "?")
        server_mem = last.get("server_tree_mem_mb", "?")
        children = last.get("server_children", "?")

        log.info(
            f"[{elapsed:.0f}s] episodes={counters.done}/{total} "
            f"ok={counters.ok} fail={counters.fail} "
            f"resets={counters.resets_done} "
            f"turns={counters.turns_done}/{total_turns} "
            f"in_flight={in_flight} | "
            f"server={server_sessions} subprocs={children} | "
            f"cpu={cpu}% mem={mem}GB server={server_mem}MB"
        )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
async def run_rl_step(
    scale: int,
    concurrency: int,
    min_turns: int,
    max_turns: int,
) -> tuple[list[SessionResult], dict]:
    """Run one RL step: launch `scale` episodes in parallel."""

    log.info("=" * 78)
    log.info(
        f"RL STEP SIMULATION: {scale} envs, concurrency={concurrency}, "
        f"turns={min_turns}-{max_turns}, timeout={CLIENT_TIMEOUT}s"
    )
    log.info("=" * 78)

    # Discover server PID for resource monitoring
    server_pid = await check_server(BASE_URL)
    log.info(f"Server reachable (pid={server_pid})")

    monitor = ResourceMonitor(interval=2.0)
    monitor.start(server_pid)

    # Two semaphores:
    # - reset_semaphore: limits concurrent resets (heavy: subprocess spawn)
    # - interact_semaphore: limits concurrent multi-turn interaction
    reset_semaphore = asyncio.Semaphore(concurrency)
    interact_semaphore = asyncio.Semaphore(scale)  # no limit on interaction

    # Pre-assign turns per session
    turn_counts = [random.randint(min_turns, max_turns) for _ in range(scale)]
    total_planned_turns = sum(turn_counts)

    counters = ProgressCounters()

    # Launch progress reporter
    progress_task = asyncio.create_task(
        progress_reporter(counters, scale, total_planned_turns, monitor)
    )

    wall_start = time.monotonic()

    tasks = []
    for i in range(scale):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        task_idx = i % 10
        tasks.append(
            run_rl_episode(
                session_id=i,
                scenario=scenario,
                task_idx=task_idx,
                num_turns=turn_counts[i],
                reset_semaphore=reset_semaphore,
                interact_semaphore=interact_semaphore,
                counters=counters,
            )
        )

    completed = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - wall_start

    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    await monitor.stop()

    resource_summary = monitor.summary()

    # --------------- Report ---------------
    ok = [r for r in completed if r.success]
    failed = [r for r in completed if not r.success]
    total_turns = sum(r.turns_completed for r in completed)
    total_planned = sum(r.num_turns for r in completed)

    log.info("")
    log.info(f"{'=' * 78}")
    log.info(
        f"RESULTS: {len(ok)}/{scale} succeeded, {len(failed)} failed, wall={wall_s:.1f}s"
    )
    log.info(f"Total turns: {total_turns}/{total_planned} completed")
    log.info(f"{'=' * 78}")

    # Latency distributions
    for label, values in [
        ("connect", [r.connect_s for r in ok]),
        ("reset", [r.reset_s for r in ok]),
        ("list_tools", [r.list_tools_s for r in ok]),
        ("tool_call", [lat for r in ok for lat in r.tool_call_latencies]),
        ("done", [r.done_s for r in ok]),
        ("episode_total", [r.total_s for r in ok]),
    ]:
        stats = latency_stats(values)
        if stats:
            log.info(f"  {label:>14s}: {json.dumps(stats)}")

    # Resource summary
    log.info("")
    log.info(f"  {'RESOURCES':>14s}: {json.dumps(resource_summary)}")

    # Turn distribution
    if ok:
        turn_dist = [r.num_turns for r in ok]
        log.info(
            f"  {'turns/episode':>14s}: min={min(turn_dist)} max={max(turn_dist)} "
            f"mean={statistics.mean(turn_dist):.1f}"
        )

    # Failures
    if failed:
        log.warning("")
        log.warning(f"  {len(failed)} failures:")
        for r in failed[:20]:
            log.warning(
                f"    session {r.session_id} ({r.scenario}/{r.task_idx}, "
                f"turns={r.turns_completed}/{r.num_turns}): {r.error}"
            )
        if len(failed) > 20:
            log.warning(f"    ... and {len(failed) - 20} more")

    return list(completed), resource_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="AWM stress test — simulates large-scale agentic RL"
    )
    p.add_argument(
        "--scale",
        type=int,
        default=2000,
        help="Number of parallel environments per RL step (default: 2000)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="Max concurrent resets (default: 256)",
    )
    p.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Min tool-call turns per episode (default: 3)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max tool-call turns per episode (default: 20)",
    )
    p.add_argument(
        "--think-min",
        type=float,
        default=1.0,
        help="Min LLM rollout time per turn in seconds (default: 1.0)",
    )
    p.add_argument(
        "--think-max",
        type=float,
        default=20.0,
        help="Max LLM rollout time per turn in seconds (default: 20.0)",
    )
    p.add_argument(
        "--url",
        default="http://localhost:8899",
        help="Server base URL (default: http://localhost:8899)",
    )
    p.add_argument(
        "--client-timeout",
        type=float,
        default=600.0,
        help="Client message timeout in seconds (default: 600)",
    )
    return p.parse_args()


async def main():
    args = parse_args()
    global BASE_URL, CLIENT_TIMEOUT, MIN_TURNS, MAX_TURNS, LLM_THINK_MIN, LLM_THINK_MAX
    BASE_URL = args.url
    CLIENT_TIMEOUT = args.client_timeout
    MIN_TURNS = args.min_turns
    MAX_TURNS = args.max_turns
    LLM_THINK_MIN = args.think_min
    LLM_THINK_MAX = args.think_max

    log.info(f"AWM RL Stress Test — server: {BASE_URL}")
    try:
        await check_server(BASE_URL)
    except Exception as e:
        log.error(f"Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    results, resources = await run_rl_step(
        args.scale, args.concurrency, args.min_turns, args.max_turns
    )

    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok

    log.info("")
    log.info("=" * 78)
    log.info("FINAL SUMMARY")
    log.info("=" * 78)
    log.info(
        f"  scale={args.scale}  concurrency={args.concurrency}  ok={ok}  fail={fail}"
    )
    log.info(
        f"  turns_range=[{args.min_turns},{args.max_turns}]  "
        f"total_turns={sum(r.turns_completed for r in results)}"
    )
    if resources:
        log.info(f"  resources={json.dumps(resources)}")

    if fail > 0:
        log.error("SOME EPISODES FAILED")
        sys.exit(1)
    else:
        log.info("ALL EPISODES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
