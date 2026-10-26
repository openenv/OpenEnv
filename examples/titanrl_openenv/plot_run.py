# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot a TitanRL <-> OpenEnv run from the artifacts it already writes.

The recipes in ``config_registry.py`` enable TensorBoard alongside W&B, so every
run leaves an ``events.out.tfevents.*`` file and a ``rollout_samples.jsonl``
under its dump folder. This turns those into the figures in ``README.md``,
without needing a W&B account or network access::

    python examples/titanrl_openenv/plot_run.py \\
        /path/to/torchtitan/outputs/rl/openenv_chess \\
        --out examples/titanrl_openenv/assets

Four panels, chosen to show the things that actually went wrong while getting
this recipe to train:

``reward``
    Mean episode reward per step, train and validation, with the train +/- std
    band. The band is the interesting part: GRPO can only learn from *within
    group* reward spread, and a run whose std sits at zero is a run whose
    batches are all untrainable.
``components``
    The two reward terms separately -- the environment's own sparse reward
    (``OpenEnvReward``) and the shaped position score
    (``OpenEnvShapingReward``) -- so it is visible which one is carrying the
    signal.
``trainability``
    Fraction of groups with zero reward std, straight from TitanRL's own
    ``group_zero_std_frac``. This is the number that decides whether the run
    survives; everything in the "sparse environment" section below is in
    service of keeping it near zero.
``rollouts``
    Turns per rollout and truncation rate, i.e. how much of a game the model
    actually gets to play inside its token budget.

Requires ``matplotlib`` and ``tensorboard``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Optional


def _load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Read every scalar series out of the run's TensorBoard event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    return {
        tag: [(s.step, s.value) for s in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def _series(
    scalars: dict[str, list[tuple[int, float]]], *tags: str
) -> tuple[list[int], list[float]]:
    """First of ``tags`` that exists, as ``(steps, values)``.

    Several candidates per series because the metric namespace has moved
    before (``rollout_reward/_mean`` was once ``reward/_mean``); a renamed tag
    should degrade to an empty panel, not a stack trace.
    """
    for tag in tags:
        points = scalars.get(tag)
        if points:
            return [p[0] for p in points], [p[1] for p in points]
    return [], []


def _plot_reward(ax, scalars) -> None:
    steps, mean = _series(scalars, "rollout_reward/_mean", "reward/_mean")
    _, std = _series(scalars, "rollout_reward/_std", "reward/_std")
    if steps:
        ax.plot(steps, mean, label="train", color="#1f77b4")
        if len(std) == len(mean):
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            # The spread, not the level, is what makes a batch trainable.
            ax.fill_between(
                steps, lo, hi, alpha=0.2, color="#1f77b4", label="train ±1 std"
            )
    v_steps, v_mean = _series(scalars, "validation_reward/_mean")
    if v_steps:
        ax.plot(v_steps, v_mean, "o--", label="validation", color="#d62728")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("Episode reward")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.legend(fontsize="small")


# Rubric reward-fn class name -> label. TitanRL keys the per-component metrics
# by class name, which is why the shaping reward is its own class.
_COMPONENTS = {
    "OpenEnvReward": ("env reward (chess outcome)", "#2ca02c"),
    "OpenEnvShapingReward": ("shaping (position eval)", "#ff7f0e"),
}


def _plot_components(ax, scalars) -> None:
    plotted = False
    for name, (label, color) in _COMPONENTS.items():
        steps, values = _series(
            scalars,
            f"rollout_reward/component/{name}/mean",
            f"reward/component/{name}/mean",
        )
        if steps:
            ax.plot(steps, values, label=label, color=color)
            plotted = True
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("Reward components")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean component value")
    if plotted:
        ax.legend(fontsize="small")


def _plot_rollouts(ax, scalars) -> None:
    steps, turns = _series(scalars, "rollout/num_turns/mean", "num_turns/mean")
    if steps:
        ax.plot(steps, turns, label="turns per rollout", color="#9467bd")
        ax.legend(loc="upper left", fontsize="small")
    ax.set_title("Rollout shape")
    ax.set_xlabel("training step")
    ax.set_ylabel("turns")

    t_steps, trunc = _series(
        scalars, "rollout/truncation_rate/mean", "truncation_rate/mean"
    )
    if t_steps:
        twin = ax.twinx()
        twin.plot(t_steps, trunc, "--", label="truncation rate", color="#8c564b")
        twin.set_ylabel("truncation rate")
        twin.set_ylim(-0.05, 1.05)
        twin.legend(loc="upper right", fontsize="small")


def _plot_trainability(ax, scalars) -> None:
    """Fraction of GRPO groups whose rollouts all scored the same.

    The metric this example exists to keep at zero. A group with zero reward
    std produces zero advantage for every one of its rollouts, so it trains
    nothing; at 1.0 across ten consecutive batches TitanRL aborts the run.
    """
    steps, frac = _series(
        scalars,
        "rollout_reward/group_zero_std_frac/mean",
        "reward/group_zero_std_frac/mean",
    )
    if steps:
        ax.plot(steps, frac, color="#d62728", label="groups with zero reward std")
        ax.fill_between(steps, 0.0, frac, alpha=0.15, color="#d62728")
        ax.legend(fontsize="small")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("Untrainable groups (lower is better)")
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of groups")


def plot_scalars(run_dir: str, out_path: str) -> Optional[str]:
    """Write the three-panel training figure. Returns the path, or None."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scalars = _load_scalars(run_dir)
    if not scalars:
        print(f"no TensorBoard scalars under {run_dir}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    _plot_reward(axes[0][0], scalars)
    _plot_components(axes[0][1], scalars)
    _plot_trainability(axes[1][0], scalars)
    _plot_rollouts(axes[1][1], scalars)
    fig.suptitle(f"TitanRL on OpenEnv chess — {os.path.basename(run_dir)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def summarize_rollouts(run_dir: str) -> dict[str, object]:
    """Summarize ``rollout_samples.jsonl``: statuses, rewards, advantage spread.

    Two things about this file are easy to misread:

    * It is **not** a census of the rollouts. TitanRL's recorder applies a
      ``KeepExtremeRewardsFilter``, which keeps only the highest- and
      lowest-reward rollout of each group — two lines per group, whatever
      ``num_samples_per_prompt`` is. So ``nonzero_advantage`` is an *upper
      bound* on how much of the batch carries gradient: if even the two
      extremes tie, the whole group is flat (that is the signature of the
      untrainable-batch abort), but the converse does not follow.
    * Validation rollouts are recorded too, each as its own single-rollout
      group with no advantage. Left in, they look like half the run producing
      no gradient, so they are counted separately here.
    """
    path = os.path.join(run_dir, "rollout_samples.jsonl")
    if not os.path.exists(path):
        return {}

    statuses: collections.Counter = collections.Counter()
    turn_counts: collections.Counter = collections.Counter()
    rewards: list[float] = []
    advantages: list[float] = []
    groups: set = set()
    num_validation = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_validation"):
                num_validation += 1
                continue
            statuses[row.get("status")] += 1
            turn_counts[len(row.get("turns") or [])] += 1
            groups.add(row.get("group_id"))
            rewards.append(float(row.get("reward") or 0.0))
            advantages.append(float(row.get("advantage") or 0.0))

    if not rewards:
        return {"validation_rollouts": num_validation}

    return {
        "recorded_train_rollouts": len(rewards),
        "validation_rollouts": num_validation,
        "groups": len(groups),
        "statuses": dict(statuses.most_common()),
        "turns": dict(sorted(turn_counts.items())),
        "distinct_rewards": len(set(round(r, 6) for r in rewards)),
        # Of the recorded extremes, how many carry a gradient. Zero here means
        # every group was flat and the run is about to abort.
        "nonzero_advantage": sum(1 for a in advantages if abs(a) > 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run dump folder (--dump-folder).")
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the figure (default: the run folder).",
    )
    parser.add_argument(
        "--name",
        default="training_curves.png",
        help="Figure filename.",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.run_dir):
        # TitanRL writes the dump folder relative to where training ran --
        # usually the torchtitan root, not this repo.
        parser.error(f"no such run folder: {args.run_dir}")

    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    written = plot_scalars(args.run_dir, os.path.join(out_dir, args.name))
    if written:
        print(f"wrote {written}")

    summary = summarize_rollouts(args.run_dir)
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
