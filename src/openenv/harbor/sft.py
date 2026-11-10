# SPDX-License-Identifier: BSD-3-Clause

"""Turn Harbor rollouts into supervised fine-tuning rows.

An evaluation rollout against a hosted model carries no token ids, so it can't feed RL training, but its trace is
supervised data as it stands. The agent's last model call holds the whole conversation as the harness sent it
(system prompt, task, every earlier assistant turn with its tool calls, every tool output), and that call plus its
reply is one SFT row. This is the text-level export: it works on `eval` and `train` rollouts alike, and never goes
through the token-level training contract of `openenv.harbor.contract`.

Rows are in the conversational format with a `tools` column, which TRL's `SFTTrainer` and chat templates read
directly. Each row keeps its reward. Filtering is the caller's choice (`min_reward`), since a recorder shouldn't
decide a training policy.

Examples:

```python
from openenv.harbor.sft import export_sft

rows, stats = export_sft(results, min_reward=1.0)
```
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .models import HarborRolloutResult


@dataclass
class SFTExportStats:
    """What `export_sft` did with each rollout."""

    rollouts: int = 0
    rows: int = 0
    below_min_reward: int = 0
    ungraded: int = 0
    no_agent_turn: int = 0
    ended_on_tool_call: list[str] = field(default_factory=list)


def rollout_reward(result: HarborRolloutResult, reward_key: str = "") -> float | None:
    """
    The reward a rollout was graded with.

    `result.reward` when it's set. Otherwise the verifier reported several named rewards and none was picked as the
    training signal: `reward_key` names it, and a single named reward is used as it is.

    Args:
        result ([`HarborRolloutResult`]):
            One rollout.
        reward_key (`str`, *optional*, defaults to `""`):
            Key of `result.rewards` to fall back to, or a comma-separated preference order.

    Returns:
        `float` or `None`: the reward, or `None` when the rollout was not graded.
    """
    if result.reward is not None:
        return result.reward
    if reward_key:
        # A literal key first, then a comma-separated preference order, as `openenv harbor rollout --reward-key`.
        for key in [reward_key] + [k.strip() for k in reward_key.split(",")]:
            if key in result.rewards:
                return result.rewards[key]
    if len(result.rewards) == 1:
        return next(iter(result.rewards.values()))
    return None


def _normalize_tool_calls(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style tool calls with `arguments` as a dict.

    Captured replies carry flat `{name, arguments}` and request messages nest them under `function`, and providers
    send `arguments` as a JSON string. Chat templates such as Qwen3.5's iterate over the arguments, so they're parsed.
    """
    out = []
    for call in calls:
        fn = call.get("function", call)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {
                    "_raw": args
                }  # keep what the model wrote rather than drop the call
        normalized = {
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args or {}},
        }
        if call.get("id"):
            normalized["id"] = call["id"]
        out.append(normalized)
    return out


def _last_agent_turn(result: HarborRolloutResult):
    turns = [
        t
        for t in result.turns
        if t.role == "agent" and not t.discarded and t.request_messages
    ]
    return turns[-1] if turns else None


def ended_on_tool_call(result: HarborRolloutResult) -> bool:
    """
    Whether the agent stopped right after a tool call, with no final reply.

    It usually means a model call failed and the harness gave up, so the task is worth running again rather than
    counting as a wrong answer.
    """
    turn = _last_agent_turn(result)
    return turn is not None and turn.finish_reason == "tool_calls"


def to_sft_row(
    result: HarborRolloutResult, reward_key: str = ""
) -> dict[str, Any] | None:
    """
    One conversational SFT row from a rollout, or `None` if it has no agent turn.

    The row is the agent's last request to the model plus the reply to it. Auxiliary calls (session titles,
    summaries) and discarded retries are not part of it.

    Args:
        result ([`HarborRolloutResult`]):
            One rollout.
        reward_key (`str`, *optional*, defaults to `""`):
            Passed to [`rollout_reward`].

    Returns:
        `dict` with keys:
            - `messages` (`list[dict]`): the conversation, ending with the agent's last reply.
            - `tools` (`list[dict]`): the tools the harness offered.
            - `reward` (`float` or `None`): the rollout's reward.
            - `rewards` (`dict[str, float]`): every reward the verifier reported.
            - `task_name`, `dataset`, `harness`, `rollout_type` (`str`): where the row comes from.
            - `n_turns` (`int`): model calls in the rollout.
    """
    turn = _last_agent_turn(result)
    if turn is None:
        return None
    reply: dict[str, Any] = {"role": "assistant", "content": turn.text or ""}
    if turn.tool_calls:
        reply["tool_calls"] = turn.tool_calls
    messages = [dict(m) for m in turn.request_messages] + [reply]
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    return {
        "messages": messages,
        "tools": list(turn.request_tools or []),
        "reward": rollout_reward(result, reward_key),
        "rewards": dict(result.rewards),
        "task_name": result.task_name,
        "dataset": result.dataset,
        "harness": result.harness,
        "rollout_type": result.rollout_type,
        "n_turns": result.n_turns,
    }


def export_sft(
    results: Iterable[HarborRolloutResult | dict[str, Any]],
    min_reward: float | None = None,
    reward_key: str = "",
) -> tuple[list[dict[str, Any]], SFTExportStats]:
    """
    SFT rows from Harbor rollouts.

    Args:
        results (`Iterable` of [`HarborRolloutResult`] or `dict`):
            Rollouts, as returned by `run_batch` or read back from `openenv harbor rollout --out`.
        min_reward (`float`, *optional*):
            Keep only rollouts graded at least this. Ungraded rollouts are dropped too. Unset, every rollout with an
            agent turn becomes a row, with its reward on it.
        reward_key (`str`, *optional*, defaults to `""`):
            Passed to [`rollout_reward`].

    Returns:
        `tuple` of the rows and an [`SFTExportStats`].

    Examples:

    ```python
    import json
    from openenv.harbor.sft import export_sft

    rows, stats = export_sft(json.load(open("result.json")), min_reward=1.0)
    ```
    """
    rows, stats = [], SFTExportStats()
    for item in results:
        result = (
            item
            if isinstance(item, HarborRolloutResult)
            else HarborRolloutResult.model_validate(item)
        )
        stats.rollouts += 1
        if ended_on_tool_call(result):
            stats.ended_on_tool_call.append(result.task_name)
        row = to_sft_row(result, reward_key)
        if row is None:
            stats.no_agent_turn += 1
            continue
        if min_reward is not None:
            if row["reward"] is None:
                stats.ungraded += 1
                continue
            if row["reward"] < min_reward:
                stats.below_min_reward += 1
                continue
        rows.append(row)
    stats.rows = len(rows)
    return rows, stats
