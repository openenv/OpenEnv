"""Batch bridge: the GPU worker receives public prompts and raw environment rewards."""

import json
import math
import sys

from openenv.core.generic_client import GenericEnvClient


def main():
    payload = json.load(sys.stdin)
    operation = payload["operation"]
    split = payload["split"]
    indices = payload["indices"]
    if split not in {"train", "test"} or len(indices) > 100:
        raise ValueError("unsupported split or batch size")
    if any(
        type(i) is not int or i < 0 or i >= (1000 if split == "train" else 50)
        for i in indices
    ):
        raise ValueError("only uploaded four-card 24 tasks are supported")
    if operation not in {"tasks", "score"}:
        raise ValueError("unsupported operation")
    expressions = payload.get("expressions", [])
    if operation == "score" and (
        len(expressions) != len(indices)
        or any(not isinstance(s, str) or len(s) > 2048 for s in expressions)
    ):
        raise ValueError("invalid expressions")
    output = {"tasks": [], "rewards": [], "task_ids": [], "valid": []}
    with GenericEnvClient(base_url="http://127.0.0.1:8012").sync() as env:
        for row, index in enumerate(indices):
            result = env.reset(
                split=split,
                index=index,
                seed=20261104,
                episode_id=f"{split}-{index}",
                reward_mode="shaped",
            )
            obs = result.observation
            if result.done or obs["level"] != "24" or obs["solution"] is not None:
                raise RuntimeError("reset contract failed")
            if operation == "tasks":
                output["tasks"].append(
                    {
                        "index": index,
                        "task_id": obs["task_id"],
                        "prompt": [
                            {"role": "system", "content": obs["system_prompt"]},
                            {"role": "user", "content": obs["prompt"]},
                        ],
                    }
                )
            else:
                result = env.step({"expression": expressions[row]})
                if (
                    not result.done
                    or result.reward is None
                    or not math.isfinite(result.reward)
                ):
                    raise RuntimeError("missing finite terminal reward")
                output["rewards"].append(float(result.reward))
                output["task_ids"].append(obs["task_id"])
                output["valid"].append(bool(result.observation["result"]["valid"]))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
