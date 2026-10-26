"""Tests for the ECHO-on-AWM adapter + loss. Run: ``pytest -q`` (from this dir)."""

from __future__ import annotations

import json
import pathlib

import torch
from awm import awm_episode_to_trajectory
from echo import ACTION, CONTEXT, echo_loss, ENV_OUTPUT, tokenize_trajectory, WARNING

HERE = pathlib.Path(__file__).parent
EPISODE = json.loads((HERE / "fixtures" / "awm_ecommerce_episode.json").read_text())


class _CharTok:
    def __init__(self, corpus: str) -> None:
        chars = sorted(set(corpus))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self.stoi[c] for c in text]}


def _tok_traj(world_loss_target: str = "env_only"):
    traj = awm_episode_to_trajectory(EPISODE)
    tok = _CharTok("".join(s.text for s in traj.segments))
    return traj, tokenize_trajectory(tok, traj, world_loss_target=world_loss_target)


# ── adapter / role mask ──────────────────────────────────────────────────────
def test_roles_present_and_ordered():
    traj = awm_episode_to_trajectory(EPISODE)
    roles = [s.role for s in traj.segments]
    assert roles[0] == CONTEXT  # context first
    assert ACTION in roles and ENV_OUTPUT in roles  # real tool-use produces both


def test_warning_role_separated_from_env_output():
    """A `warning` field becomes its own WARNING segment, never folded into env_output."""
    episode = {
        "scenario": "synthetic",
        "task": "demo warning handling",
        "tools": ["t"],
        "steps": [
            {
                "action": {"tool_name": "t", "arguments": {}},
                "observation": {
                    "tool_name": "t",
                    "tool_result": {"ok": True},
                    "warning": "prices may change at checkout",
                },
            }
        ],
        "reward": 0.0,
    }
    traj = awm_episode_to_trajectory(episode)
    roles = [s.role for s in traj.segments]
    assert WARNING in roles and ENV_OUTPUT in roles
    warn_seg = next(s for s in traj.segments if s.role == WARNING)
    env_seg = next(s for s in traj.segments if s.role == ENV_OUTPUT)
    assert "may change at checkout" in warn_seg.text
    assert "may change at checkout" not in env_seg.text


def test_action_precedes_its_env_output():
    """The model must be conditioned on its action before predicting the world."""
    traj = awm_episode_to_trajectory(EPISODE)
    first_action = next(i for i, s in enumerate(traj.segments) if s.role == ACTION)
    first_env = next(i for i, s in enumerate(traj.segments) if s.role == ENV_OUTPUT)
    assert first_action < first_env


def test_masks_partition_tokens():
    _, t = _tok_traj()
    a, o, w = t["action_mask"], t["obs_mask"], t["warning_mask"]
    # disjoint
    assert not bool((a & o).any())
    assert not bool((a & w).any())
    assert not bool((o & w).any())
    # together they never exceed the sequence length
    assert int((a | o | w).sum()) <= a.numel()


def test_free_signal_dominates():
    """ECHO's whole pitch: observation tokens outnumber action tokens."""
    _, t = _tok_traj()
    n_action = int(t["action_mask"].sum())
    n_obs = int(t["obs_mask"].sum())
    assert n_action > 0 and n_obs > 0
    assert n_obs > n_action


def test_warning_excluded_by_default_included_on_all():
    _, env_only = _tok_traj("env_only")
    _, all_obs = _tok_traj("all")
    # env_only excludes warning tokens; "all" folds them into the obs mask
    assert int(all_obs["obs_mask"].sum()) == int(env_only["obs_mask"].sum()) + int(
        env_only["warning_mask"].sum()
    )


# ── loss invariants ──────────────────────────────────────────────────────────
def _logits_for(t):
    input_ids = t["input_ids"].unsqueeze(0)
    torch.manual_seed(0)
    vocab = int(input_ids.max()) + 2
    logits = torch.randn(1, input_ids.shape[1], vocab)
    masks = {k: v.unsqueeze(0) for k, v in t.items() if k != "input_ids"}
    return input_ids, logits, masks


def test_lambda_zero_is_vanilla_grpo():
    _, t = _tok_traj()
    input_ids, logits, m = _logits_for(t)
    adv = torch.tensor([1.0])
    _, parts = echo_loss(
        logits, input_ids, m["action_mask"], m["obs_mask"], adv, world_model_coeff=0.0
    )
    assert abs(parts["loss"] - parts["l_grpo"]) < 1e-6  # env term not added


def test_verifier_free_is_pure_env_ce():
    _, t = _tok_traj()
    input_ids, logits, m = _logits_for(t)
    adv = torch.tensor([1.0])
    _, parts = echo_loss(
        logits,
        input_ids,
        m["action_mask"],
        m["obs_mask"],
        adv,
        world_model_coeff=1.0,
        use_rl=False,
    )
    assert parts["l_grpo"] == 0.0
    assert abs(parts["loss"] - parts["l_env"]) < 1e-6  # pure CE at coeff=1.0
    assert parts["l_env"] > 0.0


def test_lambda_scales_env_term():
    """λ multiplies the env-CE term (verifier-free isolates it for an exact check)."""
    _, t = _tok_traj()
    input_ids, logits, m = _logits_for(t)
    adv = torch.tensor([1.0])
    _, parts = echo_loss(
        logits,
        input_ids,
        m["action_mask"],
        m["obs_mask"],
        adv,
        world_model_coeff=0.05,
        use_rl=False,
    )
    assert abs(parts["loss"] - 0.05 * parts["l_env"]) < 1e-6


def test_loss_is_finite_and_differentiable():
    _, t = _tok_traj()
    input_ids = t["input_ids"].unsqueeze(0)
    torch.manual_seed(0)
    vocab = int(input_ids.max()) + 2
    logits = torch.randn(1, input_ids.shape[1], vocab, requires_grad=True)
    m = {k: v.unsqueeze(0) for k, v in t.items() if k != "input_ids"}
    loss, _ = echo_loss(
        logits,
        input_ids,
        m["action_mask"],
        m["obs_mask"],
        torch.tensor([1.0]),
        world_model_coeff=0.05,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
