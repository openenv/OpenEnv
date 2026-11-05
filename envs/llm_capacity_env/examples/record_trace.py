"""Record and independently replay full local MCP transitions of causal baselines.

These are deterministic heuristic-policy traces, not LLM reasoning traces, and not
an implementation of automatic server-side session telemetry / RFC 008 grading.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openenv.core.env_server.mcp_types import CallToolAction

from examples.policies import reservations
from server.capacity_environment import CapacityEnvironment


def unwrap(observation):
    assert observation.error is None, observation.error
    result = observation.result
    if hasattr(result, "data"):
        return result.data
    return result["data"]


def record(out, policy, seed=23):
    path = out / f"{policy}-seed{seed}.jsonl"
    env = CapacityEnvironment(split="test")
    records = []
    try:
        initial = env.reset(seed=seed, episode_id=f"trace-{policy}-{seed}")
        view = unwrap(env.step(CallToolAction(tool_name="observe_demand_history", arguments={})))
        assert view == initial.metadata
        records.append(
            {
                "type": "reset",
                "policy_kind": "deterministic_heuristic",
                "policy": policy,
                "seed": seed,
                "split": "test",
                "observation": view,
            }
        )
        for turn in range(14):
            action = {
                "tool_name": "reserve_capacity",
                "arguments": {"reservations": reservations(view, policy)},
            }
            observation = env.step(CallToolAction(**action))
            after = unwrap(observation)
            records.append(
                {
                    "type": "transition",
                    "step": turn + 1,
                    "observation": view,
                    "action": action,
                    "reward": observation.reward,
                    "terminal": observation.done,
                    "next_observation": after,
                }
            )
            assert observation.done == (turn == 13)
            assert observation.reward == after["previous"]["reward"]
            view = after
        path.write_text(
            "".join(json.dumps(x, allow_nan=False, separators=(",", ":")) + "\n" for x in records)
        )
    finally:
        env.close()
    return path


def replay(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    header = records[0]
    env = CapacityEnvironment(split=header["split"])
    total = 0.0
    try:
        initial = env.reset(seed=header["seed"])
        assert initial.metadata == header["observation"]
        for row in records[1:]:
            before = unwrap(
                env.step(CallToolAction(tool_name="observe_demand_history", arguments={}))
            )
            assert before == row["observation"]
            result = env.step(CallToolAction(**row["action"]))
            assert unwrap(result) == row["next_observation"]
            assert result.reward == row["reward"] and result.done == row["terminal"]
            total += result.reward
        assert len(records) == 15 and result.done
        cost = row["next_observation"]["cumulative_modeled_cost"]
        assert abs(total + cost / 1000) < 1e-10
        return {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "policy": header["policy"],
            "steps": 14,
            "replay_exact": True,
            "episode_modeled_cost": cost,
            "episode_reward": total,
        }
    finally:
        env.close()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/traces")
    out.mkdir(parents=True, exist_ok=True)
    results = [replay(record(out, p)) for p in ("zero", "rolling_quantile")]
    report = {
        "passed": True,
        "scope": "Local MCP baseline traces and exact replay; no LLM run or automatic subject-telemetry certification",
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "traces": results,
    }
    (out / "TRACE_RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
