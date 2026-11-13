"""Compare causal baselines on every eligible episode start; not a frontier-agent trial."""

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.policies import reservations
from server.engine import EPISODE_DAYS, Replay


def evaluate():
    result = {}
    for split in ("train", "test"):
        result[split] = {}
        for policy in (
            "zero",
            "previous_day",
            "fixed_quantile",
            "rolling_quantile",
            "perfect_information",
        ):
            env = Replay(split)
            costs = []
            for start in env.starts:
                env.reset(0)
                env.start = start  # Offline orchestration only; not an agent tool.
                view = env.observe()
                frozen = reservations(view, "fixed_quantile")
                for _ in range(EPISODE_DAYS):
                    if policy == "perfect_information":
                        # Offline lower bound reads unrevealed demand; NEVER a causal policy.
                        action = env.trace[env.start + env.turn]
                    else:
                        action = reservations(view, policy, frozen)
                    view = env.reserve(action)
                costs.append(env.total_cost)
            result[split][policy] = {
                "episodes": len(costs),
                "mean_episode_modeled_cost": mean(costs),
                "min_episode_modeled_cost": min(costs),
                "max_episode_modeled_cost": max(costs),
            }
    return result


if __name__ == "__main__":
    report = evaluate()
    text = json.dumps(report, indent=2) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    print(text)
