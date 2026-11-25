# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot actual training metrics and a matched held-out evaluation comparison."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def wilson_interval(correct, total):
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def read_records(path):
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Evaluation must contain nonempty, unique question IDs")
    if any(row.get("guided_demonstration") for row in records):
        raise ValueError("Guided demonstrations cannot enter the evaluation comparison")
    return records


def main():
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base, trained = read_records(args.base), read_records(args.trained)
    if len(base) != len(trained):
        raise ValueError("Evaluation sizes differ")
    for before, after in zip(base, trained, strict=True):
        if any(before[key] != after[key] for key in ["id", "question", "answer"]):
            raise ValueError("Evaluation questions or order differ")
        if before["conversation"][:2] != after["conversation"][:2]:
            raise ValueError("Initial evaluation prompts differ")
    state = json.loads(args.trainer_state.read_text())
    history = [row for row in state["log_history"] if "reward" in row]
    if not history:
        raise ValueError("Trainer state has no recorded reward curve")
    total = len(base)
    summary = {
        "num_questions": total,
        "training_steps": state["global_step"],
        "models": {},
    }
    for name, records in [("Base", base), ("GRPO", trained)]:
        correct = sum(bool(row["exact_match"]) for row in records)
        summary["models"][name] = {
            "correct": correct,
            "exact_match": correct / total,
            "exact_match_wilson_95_interval": wilson_interval(correct, total),
            "mean_reward": float(np.mean([row["reward"] for row in records])),
            "mean_recursive_child_runs": float(
                np.mean([row["delegations"] for row in records])
            ),
            "emitted_tool_calls": sum(row["emitted_tool_calls"] for row in records),
            "parsed_tool_calls": sum(row["parsed_tool_calls"] for row in records),
        }
    changes = np.array(
        [
            int(after["exact_match"]) - int(before["exact_match"])
            for before, after in zip(base, trained, strict=True)
        ]
    )
    rng = np.random.default_rng(2026)
    bootstraps = rng.choice(changes, size=(10000, total), replace=True).mean(axis=1)
    summary["paired_accuracy_change"] = float(changes.mean())
    summary["paired_bootstrap_95_interval"] = np.quantile(
        bootstraps, [0.025, 0.975]
    ).tolist()
    if np.unique(changes).size == 1:
        summary["paired_bootstrap_95_interval"] = None
        summary["paired_bootstrap_note"] = (
            "All sampled pairs have the same change, so the bootstrap is degenerate. "
            "This does not establish zero uncertainty on other questions."
        )
    summary["uncertainty_method"] = (
        "Per-model Wilson 95% intervals; paired percentile bootstrap, 10,000 resamples, seed 2026. One training run."
    )
    summary["source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [
            ("trainer_state", args.trainer_state),
            ("base", args.base),
            ("trained", args.trained),
        ]
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "evaluation.csv").open("w", newline="") as stream:
        columns = [
            "question_id",
            "base_correct",
            "grpo_correct",
            "base_reward",
            "grpo_reward",
            "base_recursive_child_runs",
            "grpo_recursive_child_runs",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for before, after in zip(base, trained, strict=True):
            writer.writerow(
                {
                    "question_id": before["id"],
                    "base_correct": int(before["exact_match"]),
                    "grpo_correct": int(after["exact_match"]),
                    "base_reward": before["reward"],
                    "grpo_reward": after["reward"],
                    "base_recursive_child_runs": before["delegations"],
                    "grpo_recursive_child_runs": after["delegations"],
                }
            )
    with (output / "training.csv").open("w", newline="") as stream:
        columns = [
            "step",
            "reward",
            "loss",
            "grad_norm",
            "learning_rate",
            "step_time",
            "completions/clipped_ratio",
            "frac_reward_zero_std",
            "entropy",
            "reward_std",
            "tools/call_frequency",
            "tools/failure_frequency",
        ]
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "openenv-hotpotqa",
        }
    )
    figure = plt.figure(figsize=(8, 6.5), layout="constrained")
    grid = figure.add_gridspec(2, 2)
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    axes[0].plot(
        [row["step"] for row in history],
        [row["reward"] for row in history],
        color="#2369a8",
        linewidth=1.5,
        marker=".",
    )
    axes[0].set(
        title="Training reward",
        xlabel="Optimizer step",
        ylabel="Mean episode reward",
        ylim=(-0.1, 1.0),
    )
    labels = list(summary["models"])
    models = list(summary["models"].values())
    accuracies = np.array([model["exact_match"] for model in models]) * 100
    intervals = (
        np.array([model["exact_match_wilson_95_interval"] for model in models]) * 100
    )
    axes[1].bar(labels, accuracies, color=["#9ba8b5", "#2369a8"], width=0.55)
    axes[1].errorbar(
        labels,
        accuracies,
        yerr=[accuracies - intervals[:, 0], intervals[:, 1] - accuracies],
        fmt="none",
        color="#273444",
        capsize=5,
    )
    axes[1].set(
        title=f"Held-out exact match (n={total})\n95% Wilson intervals",
        ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    for index, model in enumerate(models):
        axes[1].text(index, 96, f"{model['correct']}/{total}", ha="center", va="top")
    children = [model["mean_recursive_child_runs"] for model in models]
    axes[2].bar(labels, children, color=["#9ba8b5", "#2369a8"], width=0.55)
    axes[2].set(
        title="Recursive child runs",
        ylabel="Mean per held-out question",
        ylim=(0, max(1, max(children) * 1.3)),
    )
    for index, value in enumerate(children):
        axes[2].text(
            index, value + 0.03 * axes[2].get_ylim()[1], f"{value:.2f}", ha="center"
        )
    figure.savefig(output / "results.png", dpi=180)
    figure.savefig(output / "results.svg", metadata={"Date": None})
    svg = output / "results.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
