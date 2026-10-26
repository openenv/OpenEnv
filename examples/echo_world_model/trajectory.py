"""Rollout trajectory with **per-token role masks** — the schema piece OpenEnv is missing.

OpenEnv's `EpisodeRecord` is message-level (`messages` + `tool_trace`). To switch
on ECHO's env-token loss, a trainer must know, *per token*, which tokens were the
agent's **actions** vs the environment's **observations** (and, finer, real
`env_output` vs harness `warning` boilerplate). That is exactly the set of masks
ECHO's reference code carries (`completion_masks` / `completion_observation_masks`
/ `completion_warning_masks`).

This module makes that concrete: a `Trajectory` is a list of role-tagged
`Segment`s, and `tokenize_trajectory()` returns aligned per-token boolean masks
ready for `echo_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# token roles
CONTEXT = "context"  # system prompt / task — given, not a loss target
ACTION = "action"  # agent/assistant tokens — the GRPO (policy-gradient) target
ENV_OUTPUT = "env_output"  # real tool/world output — the ECHO world-model target
WARNING = "warning"  # harness boilerplate — excluded from env loss by default


@dataclass
class Segment:
    role: str
    text: str


@dataclass
class Trajectory:
    """One rollout, segmented by role, plus its terminal reward."""

    segments: list[Segment]
    reward: float
    task_prompt: str = ""


def tokenize_trajectory(
    tokenizer,
    traj: Trajectory,
    *,
    world_loss_target: str = "env_only",
) -> dict[str, torch.Tensor]:
    """Render a trajectory to token ids + aligned per-token role masks.

    Tokenizes each segment independently (``add_special_tokens=False``) and
    concatenates, so every token's role is known exactly.

    Args:
        world_loss_target: ``"env_only"`` (default) puts only real `env_output`
            tokens under the world-model loss — excluding harness ``warning``
            boilerplate the model could memorize; ``"all"`` includes warnings.

    Returns dict of 1-D tensors: ``input_ids``, ``action_mask``, ``obs_mask``,
    ``warning_mask`` (the last three boolean, aligned to ``input_ids``).
    """
    import torch

    ids: list[int] = []
    roles: list[str] = []
    for seg in traj.segments:
        seg_ids = tokenizer(seg.text, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        roles.extend([seg.role] * len(seg_ids))

    input_ids = torch.tensor(ids, dtype=torch.long)
    role_arr = roles
    action_mask = torch.tensor([r == ACTION for r in role_arr], dtype=torch.bool)
    warning_mask = torch.tensor([r == WARNING for r in role_arr], dtype=torch.bool)
    if world_loss_target == "all":
        obs_mask = torch.tensor(
            [r in (ENV_OUTPUT, WARNING) for r in role_arr], dtype=torch.bool
        )
    else:  # env_only
        obs_mask = torch.tensor([r == ENV_OUTPUT for r in role_arr], dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "action_mask": action_mask,
        "obs_mask": obs_mask,
        "warning_mask": warning_mask,
    }
