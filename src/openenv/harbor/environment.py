"""The OpenEnv environment: Task API for discovery, one MCP tool for execution.

`run_rollout` is a single long tool call rather than `reset`/`step`, because OpenEnv's HTTP handlers
construct and close an environment per request while a Harbor rollout is one stateful 60-600s run
that the harness drives. There is no meaningful `step()` to expose while opencode drives itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation

from .models import HarborState

# A rollout is minutes, not seconds. OpenEnv's MCP tools default to 30s and there is no config knob,
# so the only way to raise it is to shadow `step`/`step_async` and inject a default.
_ROLLOUT_TIMEOUT_S = 1800.0


def _trials_dir() -> Path:
    """Where per-rollout artifacts are written.

    Overridable because the default lives on `/tmp`, which is node-local, shared between users, and
    cleared: one sweep of 12 harnesses x 15 tasks left 13,187 trial directories totalling 35 GB
    there, and a 16,000-rollout run projects to ~42 GB. Traces are the durable product of an eval --
    the reward is one float, the trace is the evidence -- so they belong on a filesystem the user
    chose.

    Set `OPENENV_HARBOR_TRIALS_DIR` to relocate. The default is unchanged so existing deployments
    behave exactly as before.
    """
    return Path(
        os.environ.get("OPENENV_HARBOR_TRIALS_DIR") or "/tmp/openenv-harbor-trials"
    )


class HarborEnvironment(MCPEnvironment):
    """Per-session environment exposing `run_rollout` and `capabilities` over MCP."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Server-wide config, set once by `serving.build_app`. Class-level because OpenEnv builds a
    # throwaway instance per `/metadata` and `/schema` request, and `__init__` must stay cheap and
    # credential-free or the docs page cannot load.
    _datasets: list[str] = []
    _llm_url: str = ""
    _model: str = ""
    # The validated LLM report from startup. Rebuilding it per request would mean re-probing the
    # endpoint on every `capabilities()` call, so the startup verdict (including `ok`) is carried.
    _llm: dict[str, Any] = {}

    @classmethod
    def configure(
        cls,
        *,
        datasets: list[str],
        llm_url: str = "",
        model: str = "",
        llm: dict[str, Any] | None = None,
    ) -> None:
        cls._datasets = list(datasets)
        cls._llm_url = llm_url
        cls._model = model
        cls._llm = dict(llm or {})

    def __init__(self) -> None:
        from fastmcp import FastMCP

        from .capabilities import capabilities as _capabilities
        from .tasks import HarborTaskProvider

        self._provider = HarborTaskProvider(self._datasets)
        self._state = HarborState(episode_id=str(uuid4()), llm_url=self._llm_url)

        mcp = FastMCP("harbor_env")

        @mcp.tool
        async def run_rollout(
            split: str = "",
            task_index: int = 0,
            harness: str = "opencode",
            sandbox: str = "e2b",
            reward_key: str = "",
            keep_sandbox: bool = False,
            force_build: bool = False,
            llm_url: str = "",
            model: str = "",
            api_key: str = "",
            auth_header: str = "",
            agent_timeout_sec: float = 0.0,
            agent_step_limit: int = 0,
            sampling: dict[str, Any] | None = None,
            provider: str = "openai",
            purpose: str = "auto",
            eval_sampling: dict[str, Any] | None = None,
        ) -> str:
            """Run one Harbor rollout and return a JSON `HarborRolloutResult`.

            `harness`, `sandbox` AND the engine are all per-call, so consecutive rollouts can use
            different agents, different backends and different engines against the same server.

            `agent_timeout_sec` overrides the agent's execution timeout. `agent_step_limit` caps
            captured model calls, including auxiliary calls, and sets a native step limit where
            supported. Zero leaves calls uncapped. `sampling={"temperature": ...}` pins an explicit
            full-vocabulary training policy; use the same temperature as the trainer.

            Naming `llm_url` probes that engine (once per engine, then cached) and decides this
            rollout's tier from what it can actually return: token ids and processed logprobs mean
            `train`, anything less means `eval`. That is why a trainer and an eval run can share one
            server — the dataset tree and the sandbox templates are the expensive things to host, and
            the engine is the cheap, changing part.
            """
            return await self._run_rollout(
                split,
                task_index,
                harness,
                sandbox,
                reward_key,
                keep_sandbox,
                force_build,
                llm_url,
                model,
                api_key,
                auth_header,
                agent_timeout_sec,
                agent_step_limit,
                sampling,
                provider,
                purpose,
                eval_sampling,
            )

        @mcp.tool
        def capabilities() -> str:
            """Harnesses, sandboxes, datasets and LLM status for this server."""
            caps = _capabilities(
                datasets=self._datasets,
                llm=self._llm or {"url": self._llm_url, "model": self._model},
            )
            return json.dumps(caps.to_dict())

        @mcp.tool
        def list_tasks(split: str = "", start: int = 0, stop: int = 20) -> str:
            """A window of tasks in a split, for browsing without pulling all of them."""
            return json.dumps(self._provider.get_task_range(split, start, stop))

        super().__init__(mcp)

    # --- Task API (OpenEnv discovers these by duck typing) ----------------
    def list_splits(self) -> list[dict[str, Any]]:
        return self._provider.list_splits()

    def num_tasks(self, split: str) -> int:
        return self._provider.num_tasks(split)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self._provider.list_tasks(split)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self._provider.get_task(split, index)

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        return self._provider.get_task_range(split, start, stop)

    # --- Environment ------------------------------------------------------
    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **_: Any
    ) -> Observation:
        """New episode. Boots nothing — a sandbox is created per `run_rollout`, not per reset."""
        service = self._service()
        self._state = HarborState(
            episode_id=episode_id or str(uuid4()),
            llm_url=self._llm_url,
            intercept_url=getattr(service, "public_url", "") if service else "",
        )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": "Call run_rollout(split=..., task_index=..., harness=..., sandbox=...)",
                "datasets": self._datasets,
            },
        )

    def _step_impl(
        self, action: Any, timeout_s: float | None = None, **_: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action {type(action).__name__}; "
                "use CallToolAction(name='run_rollout', ...)"
            },
        )

    def step(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return super().step(action, timeout_s=timeout_s or _ROLLOUT_TIMEOUT_S, **kwargs)

    async def step_async(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return await super().step_async(
            action, timeout_s=timeout_s or _ROLLOUT_TIMEOUT_S, **kwargs
        )

    @property
    def state(self) -> HarborState:
        return self._state

    # --- internals --------------------------------------------------------
    @staticmethod
    def _service() -> Any:
        from .serving import HarborService

        return HarborService.current()

    async def _run_rollout(
        self,
        split: str,
        task_index: int,
        harness: str,
        sandbox: str,
        reward_key: str,
        keep_sandbox: bool,
        force_build: bool,
        llm_url: str = "",
        model: str = "",
        api_key: str = "",
        auth_header: str = "",
        agent_timeout_sec: float = 0.0,
        agent_step_limit: int = 0,
        sampling: dict[str, Any] | None = None,
        provider: str = "openai",
        purpose: str = "auto",
        eval_sampling: dict[str, Any] | None = None,
    ) -> str:
        from .models import HarborRolloutResult
        from .rollout import run_rollout as _run

        service = self._service()
        if service is None:
            return HarborRolloutResult(
                ok=False,
                error="server not initialised: no capture proxy is running",
                harness=harness,
                sandbox=sandbox,
            ).model_dump_json()

        split = split or (self._datasets[0] if self._datasets else "")
        try:
            task_dir = self._provider.task_dir(split, int(task_index))
        except Exception as exc:  # noqa: BLE001
            return HarborRolloutResult(
                ok=False, error=str(exc)[:400], harness=harness, sandbox=sandbox
            ).model_dump_json()

        # This tool is `async` on purpose. FastMCP dispatches a SYNC tool body through
        # `anyio.to_thread.run_sync` with no limiter (`fastmcp/utilities/async_utils.py`), which
        # uses anyio's default `CapacityLimiter(40)` — and the body holds that worker thread for the
        # whole rollout. That capped the server at 40 concurrent rollouts no matter what
        # `MAX_CONCURRENT_ENVS` said, and invisibly: `rollout.py` starts its clock inside the body,
        # after admission, so queue time never appeared in `wall_s`. Measured 40 concurrent for a
        # sync tool vs 400 for an async one at the same request count.
        #
        # Staying on the loop also keeps Harbor's teardown correct: `Trial._finalize` SHIELDS
        # `agent_environment.stop(...)`, and the old `asyncio.run`-per-rollout closed the loop as
        # soon as the coroutine returned, cancelling that shielded task and leaking the sandbox.
        async def _resolve_and_run():
            """Settle which engine serves this rollout, then run it.

            The probe is async and cached per engine, so it has to happen inside the coroutine rather
            than at call time. Without a named engine this falls back to whatever the server booted
            with, which is what every existing caller gets.
            """
            from openenv.core.harness.capture.sessions import Upstream

            pool = service.capture.app.state.upstreams
            if llm_url:
                upstream = Upstream(
                    llm_url=llm_url,
                    model=model,
                    api_key=api_key or None,
                    auth_header=auth_header or "Authorization",
                    provider=provider,
                )
                client, level = await pool.resolve(upstream)
                served = client.served_model or model
            else:
                upstream = None
                client, level = pool.default
                # Read from the live service rather than class config: it is the same value the proxy
                # was built with, so the result cannot claim a level the proxy is not running at.
                level = getattr(service, "capture_level", "tokens")
                served = service.model
            return await _run(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                registry=service.capture.registry,
                intercept_url=service.public_url,
                model=served,
                trials_dir=_trials_dir(),
                dataset=split,
                reward_key=reward_key,
                keep_sandbox=keep_sandbox,
                force_build=force_build,
                capture_level=level,
                purpose=purpose,
                eval_sampling=eval_sampling,
                upstream=upstream,
                inference=client,
                # 0 defers to the task file. This bounds agent execution only;
                # Harbor's environment build/healthcheck timeouts govern setup.
                # It is not a wall-clock deadline for the whole RPC.
                agent_timeout_sec=agent_timeout_sec or None,
                agent_step_limit=agent_step_limit or None,
                sampling=sampling,
            )

        result = await _resolve_and_run()

        self._state.rollouts_completed += 1
        self._state.last_reward = result.reward
        self._state.last_task_id = result.task_id
        self._state.last_trial_name = result.trial_name
        return result.model_dump_json()
