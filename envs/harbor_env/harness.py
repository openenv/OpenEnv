# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loop-owning sessions backed by a deployed `harbor_env` server.

    factory = HarborSessionFactory(server_url="http://harbor:8000", split="...", llm_url="http://vllm:8000")
    session = factory.create(prompt)      # prompt carries the task's instruction
    session.wait_for_completion()         # the server runs the agent; this blocks
    trace = session.fetch_proxy_trace()   # per-turn records for a trainer
    reward = session.verify(transcript)   # the task's own verifier, forwarded

This is the shape TRL's `HarnessRolloutWorker` consumes in loop-owning mode, so the training script
is the stock one and nothing has to be added to TRL. The work happens on a server that owns the
sandbox, the agent and the capture proxy. The trainer needs no sandbox credentials, no harness
installed, and no capture plumbing of its own.

WHY THE ENGINE IS AN ARGUMENT HERE. `run_rollout` takes `llm_url` per call, so a trainer points the
server at the vLLM it is currently syncing weights into. The server probes that engine and the tier
follows from what it can return: token ids plus processed logprobs give a trainable rollout, anything
less gives an eval one. That is what lets a training run and an eval run share one server.

The trace carries the engine's prompt ids, sampled ids, behavior logprobs and full per-token loss
mask. No prompt is reconstructed locally. For training, pass `sampling={"temperature": ...}` with
the trainer's temperature: the session pins the full-vocabulary policy across all harnesses.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.harness.capture.sessions import evaluation_sampling
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.harness.training import TrainingTrace
from openenv.harbor.client import HarborEnv
from openenv.harbor.contract import to_trace_entries, to_training_trace
from openenv.harbor.models import HarborRolloutResult

logger = logging.getLogger(__name__)


def instruction_id(instruction: str) -> str:
    """Stable id for a task, from its instruction text.

    TRL's loop-owning path forwards only `prompt` to the factory — extra dataset columns never arrive,
    and `seed` is the group counter rather than a dataset index. So the task has to be recoverable from
    the prompt itself. Hashing the instruction keeps the prompt a real prompt (readable in logged
    completions, and what the agent is actually asked to do) instead of smuggling an index through it.
    """
    return hashlib.sha1(instruction.strip().encode()).hexdigest()


def _decoded_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call `arguments` as a mapping rather than a JSON string, for chat-template rendering.

    The wire format keeps `arguments` as a string, which is what the OpenAI schema specifies. XML-style
    templates — Qwen3.5's among them — iterate it, and iterating a string raises
    `Can only get item pairs from a mapping`, so a render that should be a measurement becomes a crash.

    TRL does the same thing before rendering (`_decode_tool_call_arguments`), so skipping it here would
    also mean measuring something other than what TRL actually feeds the template.
    """
    import json as _json

    out: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            out.append(message)
            continue
        decoded = []
        for call in calls:
            function = call.get("function") or {}
            arguments = function.get("arguments", call.get("arguments"))
            if not isinstance(arguments, str):
                decoded.append(call)
                continue
            try:
                parsed = _json.loads(arguments or "{}")
            except ValueError:
                # Not JSON. Pass it through: a template that cannot render it should fail loudly
                # rather than have this function invent a shape.
                decoded.append(call)
                continue
            new_call = dict(call)
            if call.get("function"):
                new_call["function"] = {**function, "arguments": parsed}
            else:
                new_call["arguments"] = parsed
            decoded.append(new_call)
        out.append({**message, "tool_calls": decoded})
    return out


def measure_prompt_skew(
    result: HarborRolloutResult, tokenizer, **template_kwargs
) -> dict[str, Any]:
    """How exactly a local re-render reproduces the engine's own prompt, per turn.

    The number nobody has by default. TRL re-renders prompts because `TraceEntry` carries no prompt
    ids; this says what that costs for a given model and harness instead of assuming it is free or
    assuming it is fatal. `exact_match_frac == 1.0` means the loop-owning path is lossless here.

    Returns:
        `dict` with keys:
            - `turns` (`int`): turns compared.
            - `exact_match_frac` (`float`): fraction whose re-render matched token for token.
            - `worst_common_prefix` (`int`): shortest agreeing prefix seen, in tokens.
            - `length_deltas` (`list[int]`): re-rendered length minus the engine's, per turn.
    """
    compared = 0
    exact = 0
    worst_prefix = -1
    deltas: list[int] = []
    for turn in result.turns or []:
        if turn.discarded or not turn.prompt_token_ids or not turn.request_messages:
            continue
        rendered = tokenizer.apply_chat_template(
            _decoded_arguments(turn.request_messages),
            tools=turn.request_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        engine = list(turn.prompt_token_ids)
        compared += 1
        deltas.append(len(rendered) - len(engine))
        if rendered == engine:
            exact += 1
        prefix = 0
        for a, b in zip(rendered, engine):
            if a != b:
                break
            prefix += 1
        worst_prefix = prefix if worst_prefix < 0 else min(worst_prefix, prefix)
    return {
        "turns": compared,
        "exact_match_frac": (exact / compared) if compared else 0.0,
        "worst_common_prefix": max(worst_prefix, 0),
        "length_deltas": deltas,
    }


class HarborSession(ResourceSession):
    """One Harbor rollout, run by the server, read back here.

    Created idle rather than already-running: `wait_for_completion` is what starts it. The server call
    is a single blocking request that boots the sandbox, runs the agent to completion and grades the
    workspace, so there is nothing useful to do between `create` and `wait`.
    """

    def __init__(
        self,
        *,
        env: HarborEnv,
        split: str,
        task_index: int,
        instruction: str,
        harness: str,
        sandbox: str,
        llm_url: str,
        model: str,
        reward_key: str = "",
        api_key: str = "",
        auth_header: str = "",
        agent_timeout_sec: float = 0.0,
        agent_step_limit: int = 0,
        sampling: dict[str, Any] | None = None,
        provider: str = "openai",
        purpose: str = "auto",
        eval_sampling: dict[str, Any] | None = None,
        owns_env: bool = False,
    ) -> None:
        self._env = env
        # Whether closing this session should close the client too. True when the factory handed this
        # session a client of its own, which it does for every rollout — see `create`.
        self._owns_env = owns_env
        self._split = split
        self._task_index = task_index
        self._instruction = instruction
        self._harness = harness
        self._sandbox = sandbox
        self._llm_url = llm_url
        self._model = model
        self._reward_key = reward_key
        self._api_key = api_key
        self._auth_header = auth_header
        self._agent_timeout_sec = agent_timeout_sec
        self._agent_step_limit = agent_step_limit
        self._sampling = training_sampling(sampling) if sampling is not None else None
        self._provider = provider
        self._purpose = purpose
        self._eval_sampling = eval_sampling
        self.result: HarborRolloutResult | None = None

    # --- ResourceSession -----------------------------------------------------

    def initial_messages(self) -> list[Message]:
        return [{"role": "user", "content": self._instruction}]

    def list_tools(self) -> list[Tool]:
        # The agent owns its own tool loop inside the sandbox; none are exposed to the harness.
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=(
                "HarborSession does not expose external tool calls; the agent owns its own loop "
                "inside the server's sandbox."
            )
        )

    def verify(
        self, transcript: list[Message], final_state: Any | None = None
    ) -> VerifyResult:
        """The task's own verifier score, as the server reported it.

        Never recomputed here, and never defaulted to 0: a rollout that did not run has no grade, and
        scoring that as zero teaches a policy that a crashed rollout is as good as a wrong answer.
        """
        if self.result is None:
            return VerifyResult(env_reward=None, done=True)
        return VerifyResult(env_reward=self.result.reward, done=True)

    def close(self) -> None:
        # The server owns the sandbox and tears it down with the trial, so there is nothing to release
        # there. The CLIENT is ours, though: one websocket per session, and leaving it open holds an
        # env session on the server until `max_concurrent_envs` is exhausted.
        if self._owns_env and self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001 - a client that will not close must not fail a rollout
                logger.warning("closing this session's client failed", exc_info=True)
            self._env = None

    # --- loop-owning extensions ---------------------------------------------

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Run the rollout and block until the server is done. Returns a process-style exit code.

        `0` means the server reported a usable rollout. Non-zero means it did not, and the reason is on
        `self.result.error` — returned rather than raised, because a rollout failing is an outcome the
        trainer has to score around, while an exception here would take down the rollout loop and with
        it every training rank waiting on the next batch.

        `timeout_s` overrides agent execution time, as `agent_timeout_sec` does.
        Harbor's sandbox setup and teardown have separate timeouts; this is not
        an end-to-end RPC deadline.
        """
        try:
            self.result = self._env.run_rollout(
                split=self._split,
                task_index=self._task_index,
                harness=self._harness,
                sandbox=self._sandbox,
                reward_key=self._reward_key,
                llm_url=self._llm_url,
                model=self._model,
                api_key=self._api_key,
                auth_header=self._auth_header,
                provider=self._provider,
                purpose=self._purpose,
                eval_sampling=self._eval_sampling,
                # `is not None`, not `or`: 0 is a documented value meaning "defer to the task
                # file", and `or` silently replaces it with the factory default.
                agent_step_limit=self._agent_step_limit,
                **({"sampling": self._sampling} if self._sampling is not None else {}),
                agent_timeout_sec=(
                    timeout_s if timeout_s is not None else self._agent_timeout_sec
                ),
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.warning(
                "harbor rollout call failed for %s[%d]: %s: %s",
                self._split,
                self._task_index,
                type(exc).__name__,
                exc,
            )
            self.result = None
            return 1
        if not self.result.ok:
            logger.warning(
                "harbor rollout %s[%d] not ok: %s",
                self._split,
                self._task_index,
                self.result.error,
            )
        else:
            self._warn_if_oversized()
        return 0 if self.result.ok else 1

    # A packed row much beyond this has, in practice, meant an OOM in the loss step rather than a
    # slow one. Deliberately a log line and not a rejection: what to do about a huge rollout is the
    # trainer's decision, and dropping it here would silently shrink a GRPO group.
    OVERSIZED_TRAINABLE_TOKENS = 250_000

    def _warn_if_oversized(self) -> None:
        """Name the cause of an OOM that otherwise surfaces in the wrong place.

        AsyncGRPO packs every turn of a rollout into ONE training row, and each turn re-sends the
        whole conversation, so packed length grows with the SQUARE of the turn count. When that row is
        too large the failure appears much later as `CUBLAS_STATUS_ALLOC_FAILED` inside the chunked LM
        head's backward — an OOM wearing a cuBLAS mask, in a stack frame that mentions neither the
        rollout nor its turn count. This line is what connects the two.
        """
        result = self.result
        if result is None:
            return
        tokens = getattr(result, "n_trainable_tokens", 0) or 0
        if tokens > self.OVERSIZED_TRAINABLE_TOKENS:
            logger.warning(
                "harbor rollout %s[%d] packs %d trainable tokens across %d turns; a row this large "
                "has meant an OOM in the loss step. Cap the agent with agent_step_limit.",
                self._split,
                self._task_index,
                tokens,
                getattr(result, "n_turns", -1),
            )

    def fetch_training_trace(self) -> TrainingTrace:
        """Return producer-validated tokens and masks after the rollout finishes."""
        if self.result is None:
            raise RuntimeError("rollout produced no capture")
        return to_training_trace(self.result)

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Per-turn captured records, in TRL's `TraceEntry` shape.

        Empty for an eval rollout, and that is the point: an eval endpoint yields a reward and a
        readable trace but no token fields, so there is nothing to train on and this says so by being
        empty rather than by handing back rows of zeros.
        """
        if self.result is None:
            return []
        if self.result.rollout_type != "train":
            logger.warning(
                "rollout is %s (capture_level=%s), so it carries no trainable turns",
                self.result.rollout_type,
                self.result.capture_level,
            )
            return []
        return to_trace_entries(self.result)


class HarborSessionFactory(ResourceSessionFactory[HarborSession]):
    """Turns a prompt into a Harbor rollout on a deployed server.

    Args:
        server_url (`str`):
            Root of a running `harbor_env` server, e.g. `http://localhost:8000`.
        split (`str`, *optional*):
            Which dataset the server should draw tasks from. Defaults to the server's first.
        llm_url (`str`, *optional*):
            Engine for these rollouts — for training, the same vLLM the trainer syncs weights into,
            which is what makes the rollouts on-policy. Omit to use the server's default engine.
        harness (`str`, *optional*, defaults to `"opencode"`):
            Which agent runs in the sandbox.
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Harbor sandbox backend.
        agent_step_limit (`int`, *optional*, defaults to `0`):
            Model-call budget at the proxy, including auxiliary calls. Also sets a native step
            limit when supported. `0` leaves model calls uncapped.
        agent_timeout_sec (`float`, *optional*, defaults to `600.0`):
            Override the agent execution timeout. Sandbox setup has separate timeouts.
        sampling (`dict`, *optional*):
            Explicit full-vocabulary training policy with the trainer's temperature. Validated
            before opening clients and forwarded on every session, regardless of harness.
        provider (`str`, *optional*, defaults to `"openai"`):
            Wire protocol of `llm_url`, e.g. `"anthropic"` for a native Anthropic endpoint.
        purpose (`str`, *optional*, defaults to `"auto"`):
            `"train"` fails each rollout up front when the engine cannot return token ids, instead
            of running it as an eval rollout with nothing to train on. `"eval"` never exports a
            training contract. `"auto"` takes whatever the engine's probed tier gives.
        eval_sampling (`dict`, *optional*):
            Sampling for eval rollouts (`temperature`, `top_p`, `top_k`). Requires `purpose="eval"`.

    Examples:

    ```python
    factory = HarborSessionFactory(
        server_url="http://localhost:8000",
        split="AdithyaSK/data_agent_rl_environment_train",
        llm_url="http://localhost:8001",
        model="Qwen/Qwen3.5-2B",
    )
    ```
    """

    def __init__(
        self,
        server_url: str,
        *,
        split: str = "",
        llm_url: str = "",
        model: str = "",
        harness: str = "opencode",
        sandbox: str = "e2b",
        reward_key: str = "",
        api_key: str = "",
        auth_header: str = "",
        agent_timeout_sec: float = 600.0,
        agent_step_limit: int = 0,
        sampling: dict[str, Any] | None = None,
        provider: str = "openai",
        purpose: str = "auto",
        eval_sampling: dict[str, Any] | None = None,
        num_tasks: int | None = None,
        indices: list[int] | None = None,
        max_message_size_mb: float = 4096.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.harness = harness
        self.sandbox = sandbox
        self.llm_url = llm_url
        self.model = model
        self.reward_key = reward_key
        self.api_key = api_key
        self.auth_header = auth_header
        self.agent_timeout_sec = agent_timeout_sec
        self.agent_step_limit = agent_step_limit
        self.sampling = training_sampling(sampling) if sampling is not None else None
        # Checked here for the same reason as `sampling`: the server rejects these per rollout, so
        # a bad value would otherwise fail every rollout of the run one at a time.
        if purpose not in {"auto", "eval", "train"}:
            raise ValueError("purpose must be auto, eval, or train")
        if purpose == "eval" and sampling is not None:
            raise ValueError("eval purpose cannot apply a training sampling override")
        if evaluation_sampling(eval_sampling) and purpose != "eval":
            raise ValueError("eval_sampling requires explicit eval purpose")
        self.provider = provider
        self.purpose = purpose
        self.eval_sampling = eval_sampling
        self._num_tasks = num_tasks
        # Specific tasks, rather than the first N of the split. Which tasks a group trains on decides
        # whether it can learn anything at all: a task every generation solves and one none solves both
        # give reward_std 0. The band that splits has to be chosen per model — the suite's own
        # difficulty numbers were measured with a different harness and point the wrong way here.
        # `is not None`, not truthiness: an EMPTY list means a caller's selection matched nothing,
        # and falling back to "all tasks" there would silently train on the whole split instead of
        # saying so. Only an omitted argument means "no selection".
        if indices is not None and not list(indices):
            raise ValueError(
                "indices is empty: no tasks would be selected. Pass None to use the whole split."
            )
        self._indices = list(indices) if indices is not None else None
        # A rollout result is quadratic in turns (each turn carries its whole prompt), so the 100 MB
        # default is reachable: a 262-turn rollout exceeded it and closed the connection.
        self._max_message_size_mb = max_message_size_mb
        self._env: HarborEnv | None = None
        self._split = split
        self._by_instruction: dict[str, int] = {}
        self._tasks: list[dict[str, Any]] = []

    def __getstate__(self) -> dict[str, Any]:
        """Drop the live client when this factory is pickled.

        TRL spawns its rollout loop in a separate process and pickles the factory into it. Building
        the dataset in the parent calls `prompt_rows()` -> `tasks()` -> `_client()`, which binds an
        httpx client and a websocket — so without this the pickle either fails outright or the child
        inherits a connection owned by another process and every rollout dies on it.

        The task list and the instruction map are kept: they are plain data, they cost a round trip to
        rebuild, and the child needs the same mapping the parent's dataset was built from.
        """
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    # Built lazily so it is created in whichever process actually uses it, and dropped on pickling by
    # `__getstate__` above.
    def _client(self) -> HarborEnv:
        if self._env is None:
            self._env = HarborEnv(base_url=self.server_url)
            if not self._split:
                splits = self._env.splits()
                if not splits:
                    raise RuntimeError(f"{self.server_url} serves no splits")
                self._split = splits[0]["name"]
        return self._env

    def new_client(self) -> HarborEnv:
        """A fresh client for one rollout. Overridable so a caller can substitute a transport.

        Separate from `_client()`, which is the factory's own long-lived connection for task metadata.
        """
        return HarborEnv(
            base_url=self.server_url, max_message_size_mb=self._max_message_size_mb
        )

    def tasks(self) -> list[dict[str, Any]]:
        """The tasks this factory can run, fetched once from the server."""
        if not self._tasks:
            env = self._client()
            total = env.num_tasks(self._split)
            if self._indices is not None:
                bad = [i for i in self._indices if not 0 <= i < total]
                if bad:
                    raise IndexError(
                        f"index(es) {bad[:5]} out of range for {self._split!r} ({total} tasks)"
                    )
                # Normalised the same way as the range branch below: `get_task` returns a
                # `HarborTaskRef` model while `get_task_range` returns dicts, and everything after
                # this point does `task.get(...)`.
                self._tasks = [
                    _as_dict(env.get_task(self._split, i)) for i in self._indices
                ]
            else:
                stop = min(total, self._num_tasks) if self._num_tasks else total
                self._tasks = [
                    _as_dict(t) for t in env.get_task_range(self._split, 0, stop)
                ]
            self._by_instruction = {}
            collisions: dict[str, int] = {}
            for i, task in enumerate(self._tasks):
                key = instruction_id(task.get("instruction") or "")
                index = int(task.get("index", i))
                if key in self._by_instruction:
                    collisions[key] = collisions.get(key, 1) + 1
                    continue
                self._by_instruction[key] = index
            if collisions:
                # Last-write-wins here would be invisible and wrong: two tasks with identical
                # instructions produce two dataset rows that both resolve to one index, so a group
                # trains on a task it was not given while every log line looks normal. Keeping the
                # first and saying how many were shadowed at least makes it findable.
                logger.warning(
                    "%d task(s) share an instruction with an earlier task and are unreachable "
                    "through prompt lookup; the first occurrence wins",
                    sum(collisions.values()) - len(collisions),
                )
        return self._tasks

    def prompt_rows(self) -> list[dict[str, Any]]:
        """Dataset rows for a trainer: the instruction as the prompt, plus columns worth logging.

        All `num_generations` of a group share a row, so they all get the same task and the group
        baseline is well formed without any seed plumbing.
        """
        return [
            {
                "prompt": [{"role": "user", "content": t.get("instruction") or ""}],
                "task_name": t.get("task_name") or "",
                "task_index": int(t.get("index", i)),
            }
            for i, t in enumerate(self.tasks())
        ]

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> HarborSession:
        instruction = _instruction_of(task)
        self.tasks()  # ensures the instruction -> index map exists
        index = self._by_instruction.get(instruction_id(instruction))
        if index is None:
            raise KeyError(
                "this prompt does not match any task on the server. Build the dataset from "
                "`prompt_rows()` so the instruction the trainer sends is the one the server has."
            )
        # A CLIENT PER SESSION, never the factory's shared one. The MCP transport sends and then
        # receives on one socket with no request-id correlation, so two rollouts sharing a client raise
        # `ConcurrencyError: cannot call recv while another coroutine is already running recv`. With
        # `num_generations` rollouts in flight that is every rollout of every step, instantly, and each
        # one comes back unscorable — a run that looks like it is working and trains on nothing.
        #
        # The factory keeps its own client for task metadata, which is fetched once and sequentially.
        return HarborSession(
            env=self.new_client(),
            owns_env=True,
            split=self._split,
            task_index=index,
            instruction=instruction,
            harness=self.harness,
            sandbox=self.sandbox,
            llm_url=self.llm_url,
            model=self.model,
            reward_key=self.reward_key,
            api_key=self.api_key,
            auth_header=self.auth_header,
            agent_timeout_sec=self.agent_timeout_sec,
            agent_step_limit=self.agent_step_limit,
            sampling=self.sampling,
            provider=self.provider,
            purpose=self.purpose,
            eval_sampling=self.eval_sampling,
        )


def _as_dict(task: Any) -> dict[str, Any]:
    """A task record as a plain dict, whichever shape the client handed back.

    `HarborEnv.get_task` returns a `HarborTaskRef` model and `get_task_range` returns dicts, so code
    downstream that calls `task.get(...)` breaks on one and not the other depending on which path
    selected the tasks.
    """
    if hasattr(task, "model_dump"):
        return task.model_dump()
    return dict(task)


def _instruction_of(task: Any) -> str:
    """The instruction text out of whatever the worker passed as `task`."""
    if isinstance(task, list) and task:
        last = task[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    if isinstance(task, dict):
        return str(task.get("instruction") or task.get("content") or "")
    return str(task or "")
