"""ECHO core — role-masked trajectories + the env-token world-model loss.

A compact, self-contained distillation of **RFC 010** (ECHO for OpenEnv),
kept inside this example so it runs standalone against OpenEnv ``main``. One idea:

    L_ECHO = L_GRPO(action tokens) + λ · CE(observation tokens)

The policy already computes logits for the environment's observation tokens in the
*same forward pass*, so making it **predict** those tokens is ~free, "the world is
a free loss function." See ``rfcs/010-echo-env-token-world-model.md``,
``microsoft/echo-rl`` on SkyRL, and arXiv:2605.24517.

The only thing OpenEnv is missing to switch this on is a **per-token role mask** in
the rollout. This module makes that concrete; ``awm.py`` produces one from a real
upstream environment (``envs/agent_world_model_env``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ── per-token roles ──────────────────────────────────────────────────────────
CONTEXT = "context"  # system prompt / task — given, never a loss target
ACTION = "action"  # agent tokens — the GRPO (policy-gradient) target
ENV_OUTPUT = "env_output"  # real tool/world output — the ECHO world-model target
WARNING = "warning"  # harness boilerplate — excluded from the env loss by default

ROLES = (CONTEXT, ACTION, ENV_OUTPUT, WARNING)


@dataclass
class Segment:
    """A contiguous run of text with a single role."""

    role: str
    text: str


@dataclass
class Trajectory:
    """One rollout, segmented by role, plus its terminal reward."""

    segments: list[Segment]
    reward: float = 0.0
    task_prompt: str = ""
    meta: dict = field(default_factory=dict)


def tokenize_trajectory(
    tokenizer,
    traj: Trajectory,
    *,
    world_loss_target: str = "env_only",
) -> dict[str, torch.Tensor]:
    """Render a trajectory to token ids + aligned per-token role masks.

    Each segment is tokenized independently (``add_special_tokens=False``) and
    concatenated, so every token's role is known exactly.

    Note: tokenizing per-segment yields *exact* role masks but can differ from
    tokenizing the full string at segment boundaries for sub-word (BPE) tokenizers.
    For role accounting and this reference that's fine; a trainer that needs
    byte-exact parity should tokenize once and assign roles by character offsets.

    Args:
        world_loss_target: ``"env_only"`` (default) puts only real ``env_output``
            tokens under the world-model loss — excluding the harness ``warning``
            boilerplate a model could trivially memorize; ``"all"`` includes them.

    Returns 1-D tensors ``input_ids``, ``action_mask``, ``obs_mask``,
    ``warning_mask`` (the last three boolean, aligned to ``input_ids``).
    """
    ids: list[int] = []
    roles: list[str] = []
    for seg in traj.segments:
        seg_ids = tokenizer(seg.text, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        roles.extend([seg.role] * len(seg_ids))

    input_ids = torch.tensor(ids, dtype=torch.long)
    action_mask = torch.tensor([r == ACTION for r in roles], dtype=torch.bool)
    warning_mask = torch.tensor([r == WARNING for r in roles], dtype=torch.bool)
    if world_loss_target == "all":
        obs_mask = torch.tensor(
            [r in (ENV_OUTPUT, WARNING) for r in roles], dtype=torch.bool
        )
    else:  # env_only
        obs_mask = torch.tensor([r == ENV_OUTPUT for r in roles], dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "action_mask": action_mask,
        "obs_mask": obs_mask,
        "warning_mask": warning_mask,
    }


def echo_loss(
    logits: torch.Tensor,  # [B, T, V]
    input_ids: torch.Tensor,  # [B, T]
    action_mask: torch.Tensor,  # [B, T] bool — agent/action tokens (GRPO target)
    obs_mask: torch.Tensor,  # [B, T] bool — env observation tokens (ECHO target)
    advantages: torch.Tensor,  # [B] per-sequence advantage (e.g. GRPO group-relative)
    *,
    world_model_coeff: float = 0.05,  # λ. 0.0 == vanilla GRPO. Keep small.
    use_rl: bool = True,  # False == verifier-free (env-token loss only)
) -> tuple[torch.Tensor, dict[str, float]]:
    """The hybrid ECHO loss over a batch of rollouts.

    Next-token convention: position ``t`` predicts token ``t+1``; a token's role
    mask is taken from the *target* token (``t+1``).

    Two reference behaviours the tests pin:
      * ``world_model_coeff == 0`` ⇒ vanilla GRPO (action-token policy gradient only).
      * ``use_rl is False``        ⇒ verifier-free world modeling (env-token CE only) —
        the "bootstrap before you have a grader" mode.
    """
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    targets = input_ids[:, 1:]  # [B, T-1]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    a = action_mask[:, 1:].float()
    o = obs_mask[:, 1:].float()

    # GRPO: REINFORCE on the agent's action tokens, weighted by the (per-sequence)
    # advantage. Needs reward variance to carry signal — exactly why ECHO's dense
    # env term helps when reward is sparse or collapses.
    if use_rl:
        adv = advantages.unsqueeze(-1).to(token_logp.dtype)  # [B, 1]
        l_grpo = -(token_logp * a * adv).sum() / a.sum().clamp(min=1.0)
    else:
        l_grpo = token_logp.new_zeros(())

    # ECHO: cross-entropy (== -logp) on the environment's observation tokens,
    # length-normalized. Dense, always present, ~free.
    l_env = -(token_logp * o).sum() / o.sum().clamp(min=1.0)

    loss = l_grpo + world_model_coeff * l_env
    return loss, {
        "loss": float(loss.detach()),
        "l_grpo": float(l_grpo.detach()),
        "l_env": float(l_env.detach()),
    }
