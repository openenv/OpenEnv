# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate a policy on an environment's tasks with the pieces of the collect pipeline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import openai
from openenv.core.evals.base import EvalHarness
from openenv.core.harness.collect import _rollout_final_state
from openenv.core.harness.rollout import (
    _resolve_env_reward,
    _tool_result_reward,
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    ModelStep,
    ResourceSessionFactory,
    VerifyResult,
)
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

# Failures the policy's output cannot cause: a lost connection to the environment,
# and the model endpoint's connection, timeout, authentication, routing, rate-limit
# and server errors. Any other exception raised during a rollout is the policy's.
_INFRA_ERRORS = (
    ConnectionError,
    ConnectionClosed,
    openai.APIConnectionError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_DEFAULTS: dict[str, Any] = {
    "limit": None,
    "seeds": [0],
    "samples_per_task": 1,
    "temperature": 0.0,
    "success_threshold": 1.0,
    "max_turns": 9,
    "max_tokens": 200,
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _resolve_parameters(eval_parameters: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the seven evaluation settings with their defaults, after checking them.

    Args:
        eval_parameters (`Mapping[str, Any]`):
            The caller's parameters. Keys other than the seven are ignored.

    Returns:
        `dict[str, Any]`: `limit`, `seeds`, `samples_per_task`, `temperature`,
        `success_threshold`, `max_turns` and `max_tokens`.

    Raises:
        `ValueError`: If a value has the wrong type or range; the message names the key.
    """
    p = {key: eval_parameters.get(key, default) for key, default in _DEFAULTS.items()}
    if p["limit"] is not None and not (_is_int(p["limit"]) and p["limit"] >= 1):
        raise ValueError(f"limit must be None or an int >= 1, got {p['limit']!r}")
    seeds = p["seeds"]
    if (
        not isinstance(seeds, (list, tuple))
        or not seeds
        or not all(_is_int(seed) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            f"seeds must be a non-empty list of distinct ints, got {seeds!r}"
        )
    for key in ("samples_per_task", "max_turns", "max_tokens"):
        if not (_is_int(p[key]) and p[key] >= 1):
            raise ValueError(f"{key} must be an int >= 1, got {p[key]!r}")
    if not (_is_real(p["temperature"]) and p["temperature"] >= 0):
        raise ValueError(f"temperature must be a number >= 0, got {p['temperature']!r}")
    if p["samples_per_task"] > 1 and p["temperature"] == 0:
        raise ValueError("temperature must be > 0 when samples_per_task > 1")
    if not _is_real(p["success_threshold"]):
        raise ValueError(
            f"success_threshold must be a finite number, got {p['success_threshold']!r}"
        )
    return {
        **p,
        "seeds": list(seeds),
        "temperature": float(p["temperature"]),
        "success_threshold": float(p["success_threshold"]),
    }


@dataclass
class _Episode:
    """One evaluated episode, as the harness classified it."""

    task: Mapping[str, Any]
    seed: int
    sample: int
    outcome: str
    reward: float | None = None
    rollout: HarnessRolloutResult | None = None
    verify: VerifyResult | None = None
    finished: bool = False
    error: str | None = None
    turns: int | None = field(init=False, default=None)
    tokens: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.rollout is not None:
            self.turns = self.rollout.metrics.get("turns")
            self.tokens = len(self.rollout.completion_ids) or None


def _mean(values: list[Any]) -> float | None:
    """Mean of `values`, or None when there is none or any of them is None."""
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def _success_rate(episodes: list[_Episode], threshold: float) -> float | None:
    kept = [e for e in episodes if e.outcome in ("scored", "policy_error")]
    if not kept:
        return None
    return sum(_solved(e, threshold) for e in kept) / len(kept)


def _solved(episode: _Episode, threshold: float) -> bool:
    return episode.outcome == "scored" and episode.reward >= threshold


class EnvEvalHarness(EvalHarness):
    """
    Evaluate a policy on a list of an environment's tasks, one episode at a time.

    Each episode goes through the pieces `openenv collect` uses: the session
    factory starts it with its task and seed, the harness adapter runs it with the
    model step, and `session.verify` closes it. The harness walks the tasks, each
    task with every seed, and each seed `samples_per_task` times, in the order a
    `SequentialSampler` gives, then aggregates the rewards the environment
    returned. It never computes, rounds or fills in a reward.

    Each episode gets one outcome:

    - `scored`: the environment ended the episode with a reward;
    - `unscored`: the environment ended it with a reward of None, a failure of its own;
    - `policy_error`: the harness ended it because of the policy, whatever reward an
      earlier step returned: a reply without a tool call, the turn limit, or an
      exception the policy's output can cause or whose cause is unclear (a tool call
      that cannot be parsed or names no tool, a request past the context length);
    - `infra_error`: a failure the policy cannot cause: the session could not be
      created, the connection to the environment was lost, the model endpoint could
      not be reached or answered with an authentication, rate-limit or server error,
      or `verify` forwarded a reward that differs from the environment's.

    An exception counts as `infra_error` only when it is a builtin `ConnectionError`
    (such as `SessionLostError`), a `websockets` `ConnectionClosed`, or one of the
    `openai` client's connection, timeout, authentication, permission, not-found,
    rate-limit and server errors. Anything else counts as `policy_error`: an
    environment step that times out, an error frame from the server, and the same
    failures raised by another model client, such as an Anthropic rate limit.

    `unscored` and `infra_error` episodes are left out of `success_rate`,
    `mean_reward` and `pass@k`, and reported apart. A `policy_error` counts as not
    solved, so a policy cannot raise its score by failing to answer.

    `run` reads seven keys of `eval_parameters` and ignores the others, such as
    `model` and `llm_endpoint`, which stay in the result's config as a record:

    | Key | Default | Meaning |
    |-----|---------|---------|
    | `limit` | `None` | Evaluate only the first `limit` tasks. |
    | `seeds` | `[0]` | Seeds crossed with each task. |
    | `samples_per_task` | `1` | Episodes per task and seed, for `pass@k`. |
    | `temperature` | `0.0` | Passed to the model step; must be > 0 when `samples_per_task > 1`. |
    | `success_threshold` | `1.0` | Reward at or above which an episode is solved. |
    | `max_turns` | `9` | The harness's turn limit. |
    | `max_tokens` | `200` | Passed to the model step. |

    Args:
        session_factory ([`~openenv.core.harness.ResourceSessionFactory`]):
            Creates one session per episode from a task spec and a seed.
        harness_adapter ([`~openenv.core.harness.HarnessAdapter`]):
            Runs each episode with `run_white_box`.
        tasks (`Iterable[Mapping[str, Any]]`):
            The task specs to evaluate, in order. Each needs a unique string `id`;
            `difficulty` and `group`, when present, give the breakdowns.
        model_step ([`~openenv.core.harness.ModelStep`]):
            The policy, as `CollectRunner.run` takes it, for example
            `build_model_step(OpenAIClient(...))` for an OpenAI-compatible endpoint.
            It receives `temperature` and `max_tokens` in its `sampling` argument;
            `mean_completion_tokens` needs it to return `completion_ids`.

    Raises:
        `ValueError`: If `tasks` is empty, or a task is not a mapping with a
            string `id`, or two tasks share an `id`.

    Examples:

    ```python
    harness = EnvEvalHarness(
        session_factory,
        MCPHarnessAdapter(),
        list(client.iter_tasks("test")),
        model_step=build_model_step(OpenAIClient(POLICY_URL, None, model="policy")),
    )
    result = harness.run_from_config(config)
    result.scores["success_rate"], result.scores["error_rate"]
    ```
    """

    def __init__(
        self,
        session_factory: ResourceSessionFactory,
        harness_adapter: HarnessAdapter,
        tasks: Iterable[Mapping[str, Any]],
        *,
        model_step: ModelStep,
    ) -> None:
        tasks = list(tasks)
        if not tasks:
            raise ValueError("tasks is empty")
        seen: set[str] = set()
        for position, task in enumerate(tasks):
            if not isinstance(task, Mapping) or not isinstance(task.get("id"), str):
                raise ValueError(
                    f"task at position {position} is not a mapping with a string 'id'"
                )
            if task["id"] in seen:
                raise ValueError(f"task id {task['id']!r} appears more than once")
            seen.add(task["id"])
        self._session_factory = session_factory
        self._harness_adapter = harness_adapter
        self._tasks = tasks
        self._model_step = model_step

    def run(
        self,
        harness_version: str,
        library_versions: dict[str, str],
        dataset: str,
        eval_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate the policy and return its scores.

        Exceptions from closing a session are not classified and propagate, as
        does the error of a final reward that is not a number.

        Args:
            harness_version (`str`):
                Not used here; `run_from_config` keeps it in the result's config.
            library_versions (`dict[str, str]`):
                Not used here; `run_from_config` keeps it in the result's config.
            dataset (`str`):
                Not used here; `run_from_config` keeps it in the result's config.
            eval_parameters (`dict[str, Any]`):
                The settings of the class's table; other keys are ignored.

        Returns:
            `dict[str, Any]`, JSON-safe, with keys:
                - `num_tasks`, `num_episodes` (`int`): tasks after `limit`, and episodes.
                - `success_rate` (`float` or `None`): solved over scored and
                  `policy_error` episodes; `success_rate_stderr`, its binomial
                  standard error, which treats episodes as independent.
                - `mean_reward` (`float` or `None`): over scored episodes only, so
                  read it with `error_rate`.
                - `mean_turns`, `mean_completion_tokens` (`float` or `None`): over
                  the episodes whose rollout finished; None if one lacks the count.
                - `pass@k`, `pass@k_num_tasks` for each `k` up to
                  `samples_per_task`: the unbiased estimator, averaged over the
                  tasks with at least `k` scored or `policy_error` episodes, and
                  how many tasks that is.
                - `error_rate`, `unscored_rate`, `infra_error_rate` (`float`): shares
                  of all episodes.
                - `by_difficulty`, `by_group` (`dict[str, float | None]`): success
                  rate per value of that task key.
                - `per_task` (`dict[str, str]`): `solved` when every scored or
                  `policy_error` episode of the task is solved, `left_out` when it
                  has none, else `not_solved`.

        Raises:
            `ValueError`: If a setting is invalid; nothing has been run.
        """
        params = _resolve_parameters(eval_parameters)
        tasks = self._tasks[: params["limit"]]

        limits = HarnessRunLimits(
            max_turns=params["max_turns"],
            sampling={
                "temperature": params["temperature"],
                "max_tokens": params["max_tokens"],
            },
        )
        episodes: list[_Episode] = []
        for task in tasks:
            for seed in params["seeds"]:
                for sample in range(params["samples_per_task"]):
                    episode_id = f"ep-{len(episodes):06d}"
                    episode = self._run_episode(task, seed, sample, episode_id, limits)
                    if episode.error is not None:
                        logger.warning(
                            "%s %s: %s", episode_id, episode.outcome, episode.error
                        )
                    episodes.append(episode)

        scores = self._scores(tasks, episodes, params)
        return scores

    def _run_episode(
        self,
        task: Mapping[str, Any],
        seed: int,
        sample: int,
        episode_id: str,
        limits: HarnessRunLimits,
    ) -> _Episode:
        """Run one episode and classify it by who ended it (see the class docstring)."""
        try:
            session = self._session_factory.create(
                task, seed=seed, episode_id=episode_id
            )
        except Exception as exc:
            return _Episode(task, seed, sample, "infra_error", error=_describe(exc))
        try:
            rollout = None
            try:
                rollout = self._harness_adapter.run_white_box(
                    model_step=self._model_step, session=session, limits=limits
                )
                verify = session.verify(
                    transcript=rollout.messages,
                    final_state=_rollout_final_state(rollout),
                )
            except Exception as exc:
                outcome = (
                    "infra_error" if isinstance(exc, _INFRA_ERRORS) else "policy_error"
                )
                return _Episode(
                    task, seed, sample, outcome, rollout=rollout, error=_describe(exc)
                )
        finally:
            session.close()

        finished = {"rollout": rollout, "verify": verify, "finished": True}
        trace = rollout.tool_trace
        if not trace or not trace[-1].result.done:
            return _Episode(task, seed, sample, "policy_error", **finished)
        if _tool_result_reward(trace[-1].result) is None:
            reward = None if verify.env_reward is None else float(verify.env_reward)
        else:
            try:
                reward = _resolve_env_reward(rollout, verify)
            except ValueError as exc:
                return _Episode(
                    task, seed, sample, "infra_error", error=_describe(exc), **finished
                )
        outcome = "unscored" if reward is None else "scored"
        return _Episode(task, seed, sample, outcome, reward=reward, **finished)

    @staticmethod
    def _scores(
        tasks: list[Mapping[str, Any]],
        episodes: list[_Episode],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        threshold = params["success_threshold"]
        by_task: dict[str, list[_Episode]] = {task["id"]: [] for task in tasks}
        for episode in episodes:
            if episode.outcome in ("scored", "policy_error"):
                by_task[episode.task["id"]].append(episode)

        success_rate = _success_rate(episodes, threshold)
        num_kept = sum(len(kept) for kept in by_task.values())
        finished = [e for e in episodes if e.finished]
        scores: dict[str, Any] = {
            "num_tasks": len(tasks),
            "num_episodes": len(episodes),
            "success_rate": success_rate,
            "success_rate_stderr": None
            if success_rate is None
            else math.sqrt(success_rate * (1 - success_rate) / num_kept),
            "mean_reward": _mean([e.reward for e in episodes if e.outcome == "scored"]),
            "mean_turns": _mean([e.turns for e in finished]),
            "mean_completion_tokens": _mean([e.tokens for e in finished]),
        }
        counts = [
            (len(kept), sum(_solved(e, threshold) for e in kept))
            for kept in by_task.values()
        ]
        for k in range(1, params["samples_per_task"] + 1):
            estimates = [
                1 - math.comb(n - solved, k) / math.comb(n, k)
                for n, solved in counts
                if n >= k
            ]
            scores[f"pass@{k}"] = _mean(estimates)
            scores[f"pass@{k}_num_tasks"] = len(estimates)
        for outcome, key in (
            ("policy_error", "error_rate"),
            ("unscored", "unscored_rate"),
            ("infra_error", "infra_error_rate"),
        ):
            scores[key] = sum(e.outcome == outcome for e in episodes) / len(episodes)
        for key, name in (("difficulty", "by_difficulty"), ("group", "by_group")):
            buckets: dict[str, list[_Episode]] = {}
            for task in tasks:
                if key in task:
                    buckets.setdefault(str(task[key]), [])
            for episode in episodes:
                if key in episode.task:
                    buckets[str(episode.task[key])].append(episode)
            scores[name] = {
                value: _success_rate(members, threshold)
                for value, members in buckets.items()
            }
        per_task = {}
        for task_id, kept in by_task.items():
            if not kept:
                per_task[task_id] = "left_out"
            elif all(_solved(e, threshold) for e in kept):
                per_task[task_id] = "solved"
            else:
                per_task[task_id] = "not_solved"
        scores["per_task"] = per_task
        return scores


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "EnvEvalHarness",
]
