"""Train a small model to predict the terminal's responses — the ECHO env-token
world-model loss, with the reward term OFF (verifier-free).

This is the CPU-runnable heart of the demo. No grader, no teacher: the model just
acts in the env and learns to predict what the environment returns. We measure on
**held-out** tasks (different questions, same tool dynamics) so this is
generalization, not memorization — directly reproducing ECHO's verifier-free
result in miniature ("the world is a free loss function").

For the full hybrid `L_GRPO(actions) + λ·L_env(obs)` and the ~2.3×/pass@1 results,
see `echo_loss.py` (+ its tests) and `backends/` (SkyRL / Tinker / Foundry Fine-Tuning).

Run:  python train_echo.py --steps 120
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from echo_loss import echo_loss
from mini_terminal_env import TEST_TASKS, TRAIN_TASKS
from rollout import oracle_rollout
from trajectory import ENV_OUTPUT, tokenize_trajectory


def build_batch(tokenizer, trajs, world_loss_target="env_only"):
    toks = [
        tokenize_trajectory(tokenizer, t, world_loss_target=world_loss_target)
        for t in trajs
    ]
    maxlen = max(t["input_ids"].numel() for t in toks)
    pad = tokenizer.eos_token_id
    n = len(toks)
    input_ids = torch.full((n, maxlen), pad, dtype=torch.long)
    action_mask = torch.zeros((n, maxlen), dtype=torch.bool)
    obs_mask = torch.zeros((n, maxlen), dtype=torch.bool)
    attn = torch.zeros((n, maxlen), dtype=torch.long)
    for i, t in enumerate(toks):
        L = t["input_ids"].numel()
        input_ids[i, :L] = t["input_ids"]
        action_mask[i, :L] = t["action_mask"]
        obs_mask[i, :L] = t["obs_mask"]
        attn[i, :L] = 1
    return {
        "input_ids": input_ids,
        "action_mask": action_mask,
        "obs_mask": obs_mask,
        "attention_mask": attn,
    }


@torch.no_grad()
def eval_env_ce(model, batch):
    """Mean cross-entropy (nats/token) the model assigns to held-out env-output tokens."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    _, stats = echo_loss(
        out.logits,
        batch["input_ids"],
        batch["action_mask"],
        batch["obs_mask"],
        advantages=torch.zeros(batch["input_ids"].shape[0]),
        world_model_coeff=1.0,
        use_rl=False,
    )
    return stats["l_env"]


@torch.no_grad()
def predict_observation(model, tokenizer, task):
    """Teacher-forced world-model readout: at each position of the first env
    observation, what token does the model predict? Returns (cmd, actual,
    predicted, token_accuracy)."""
    traj = oracle_rollout(task)
    idx = next(i for i, s in enumerate(traj.segments) if s.role == ENV_OUTPUT)
    prefix = "".join(s.text for s in traj.segments[:idx])
    actual = traj.segments[idx].text
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    obs_ids = tokenizer(actual, add_special_tokens=False)["input_ids"]
    full = torch.tensor([prefix_ids + obs_ids])
    logits = model(full).logits[0]
    start = len(prefix_ids) - 1
    pred_ids = logits[start : start + len(obs_ids)].argmax(-1).tolist()
    acc = sum(p == t for p, t in zip(pred_ids, obs_ids)) / max(len(obs_ids), 1)
    predicted = tokenizer.decode(pred_ids, skip_special_tokens=True)
    cmd = prefix.split("$")[-1].split("[stdout]")[0].strip()
    return cmd, actual.strip(), predicted.strip(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="echo_run.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading {args.model} (CPU) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.train()

    train_batch = build_batch(tokenizer, [oracle_rollout(t) for t in TRAIN_TASKS])
    test_batch = build_batch(tokenizer, [oracle_rollout(t) for t in TEST_TASKS])

    before = eval_env_ce(model, test_batch)
    cmd0, actual0, pred0, acc0 = predict_observation(model, tokenizer, TEST_TASKS[0])
    print(f"\nHeld-out env-token CE before training: {before:.3f} nats/token\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    hist_train, hist_test, xs = [], [], []
    best_ce, best_state, best_step = before, None, 0
    import copy

    for step in range(1, args.steps + 1):
        out = model(
            input_ids=train_batch["input_ids"],
            attention_mask=train_batch["attention_mask"],
        )
        loss, stats = echo_loss(
            out.logits,
            train_batch["input_ids"],
            train_batch["action_mask"],
            train_batch["obs_mask"],
            advantages=torch.zeros(len(TRAIN_TASKS)),
            world_model_coeff=1.0,
            use_rl=False,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 1:
            model.eval()
            test_ce = eval_env_ce(model, test_batch)
            model.train()
            hist_train.append(stats["l_env"])
            hist_test.append(test_ce)
            xs.append(step)
            flag = ""
            if test_ce < best_ce:
                best_ce, best_step = test_ce, step
                best_state = copy.deepcopy(model.state_dict())
                flag = "  *best"
            print(
                f"step {step:4d}  train env-CE {stats['l_env']:.3f}  |  held-out env-CE {test_ce:.3f}{flag}"
            )

    # restore the best (generalizing) checkpoint for the final report
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    after = best_ce
    cmd1, actual1, pred1, acc1 = predict_observation(model, tokenizer, TEST_TASKS[0])

    print("\n" + "=" * 64)
    print("RESULT — the model learned to predict the world, verifier-free")
    print("=" * 64)
    print(
        f"held-out env-token CE:  {before:.3f}  ->  {after:.3f} nats/token "
        f"({100 * (before - after) / before:+.0f}%, best @ step {best_step})\n"
    )
    print(f"teacher-forced prediction of `{cmd1}` output on a HELD-OUT task:")
    print(f"  actual    : {actual1!r}")
    print(f"  before    : {pred0!r}   (token acc {acc0:.0%})")
    print(f"  after ECHO: {pred1!r}   (token acc {acc1:.0%})")

    plt.figure(figsize=(7, 4.2))
    plt.plot(xs, hist_train, label="train env-token CE", lw=2)
    plt.plot(xs, hist_test, label="held-out env-token CE", lw=2)
    plt.xlabel("step")
    plt.ylabel("cross-entropy (nats/token)")
    plt.title("ECHO verifier-free: the model learns to predict the environment")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"\nsaved curve -> {args.out}")


if __name__ == "__main__":
    main()
