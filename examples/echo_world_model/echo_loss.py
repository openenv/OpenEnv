"""The ECHO objective — ``L_ECHO = L_GRPO(action tokens) + λ · L_env(observation tokens)``.

The whole idea in one function: the model already computes logits for the
environment's observation tokens in the same forward pass, so adding a
cross-entropy loss that makes it *predict those observations* costs nothing
extra — "the world is a free loss function."

Two reference behaviors this function makes exact (and the tests pin):
- ``world_model_coeff = 0`` ⇒ vanilla GRPO (action-token policy gradient only).
- ``use_rl = False``        ⇒ verifier-free world modeling (env-token CE only) —
  the "bootstrap before you have a grader" mode.

The env term is a length-normalized cross-entropy on the observation tokens —
i.e. SFT on the tool-response tokens, which (per Prime Intellect) is just RL with
a constant positive advantage, so it reuses the exact same forward/backward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def echo_loss(
    logits: torch.Tensor,  # [B, T, V]
    input_ids: torch.Tensor,  # [B, T]
    action_mask: torch.Tensor,  # [B, T] bool — agent/action tokens (GRPO target)
    obs_mask: torch.Tensor,  # [B, T] bool — env observation tokens (world-model target)
    advantages: torch.Tensor,  # [B] per-sequence advantage (e.g. GRPO group-relative)
    *,
    world_model_coeff: float = 0.05,  # λ. 0.0 == vanilla GRPO. Keep small (collapse risk).
    use_rl: bool = True,  # False == verifier-free (env-token loss only)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the hybrid ECHO loss over a batch of rollouts.

    Next-token convention: position ``t`` predicts token ``t+1``; a token's role
    mask is taken from the *target* token (``t+1``).
    """
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    targets = input_ids[:, 1:]  # [B, T-1]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    a = action_mask[:, 1:].float()  # action target tokens
    o = obs_mask[:, 1:].float()  # observation target tokens

    # GRPO: REINFORCE on the agent's action tokens, weighted by the (per-sequence)
    # advantage. Needs reward variance to carry signal — which is exactly why
    # ECHO's dense env term helps when reward is sparse / collapses.
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
