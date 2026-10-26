# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tinker==0.22.7",
# ]
# ///

"""Train the verifier-free ECHO env-token objective with Tinker."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from mini_terminal_env import TEST_TASKS, TRAIN_TASKS  # noqa: E402
from rollout import oracle_rollout  # noqa: E402
from trajectory import ENV_OUTPUT, Trajectory  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 16
LEARNING_RATE = 2e-4


def build_echo_data(
    tinker_types: Any,
    tokenizer: Any,
    trajectories: Sequence[Trajectory],
) -> list[Any]:
    """Tokenize rollouts and put loss weight only on shifted env-output tokens."""
    encoded: list[tuple[list[int], list[bool]]] = []
    env_token_count = 0
    for trajectory in trajectories:
        token_ids: list[int] = []
        env_mask: list[bool] = []
        for segment in trajectory.segments:
            segment_ids = tokenizer.encode(segment.text, add_special_tokens=False)
            token_ids.extend(segment_ids)
            env_mask.extend([segment.role == ENV_OUTPUT] * len(segment_ids))
        if len(token_ids) < 2:
            raise ValueError("ECHO trajectories must contain at least two tokens")
        shifted_env_mask = env_mask[1:]
        env_token_count += sum(shifted_env_mask)
        encoded.append((token_ids, shifted_env_mask))

    if env_token_count == 0:
        raise ValueError("ECHO data must contain at least one env_output target token")

    return [
        tinker_types.Datum(
            model_input=tinker_types.ModelInput.from_ints(tokens=token_ids[:-1]),
            loss_fn_inputs={
                "target_tokens": token_ids[1:],
                "weights": [float(is_env) / env_token_count for is_env in env_mask],
            },
        )
        for token_ids, env_mask in encoded
    ]


def evaluate_env_ce(training_client: Any, data: Sequence[Any]) -> float:
    """Return mean cross-entropy over the globally normalized env tokens."""
    result = training_client.forward(list(data), "cross_entropy").result()
    weighted_logprob = 0.0
    for output, datum in zip(result.loss_fn_outputs, data, strict=True):
        weighted_logprob += sum(
            logprob * weight
            for logprob, weight in zip(
                output["logprobs"].tolist(),
                datum.loss_fn_inputs["weights"].tolist(),
                strict=True,
            )
        )
    return -weighted_logprob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required")

    import tinker

    print(f"Creating Tinker LoRA trainer for {args.model}")
    training_client = tinker.ServiceClient().create_lora_training_client(
        base_model=args.model,
        rank=LORA_RANK,
    )
    tokenizer = training_client.get_tokenizer()
    train_data = build_echo_data(
        tinker.types, tokenizer, [oracle_rollout(task) for task in TRAIN_TASKS]
    )
    heldout_data = build_echo_data(
        tinker.types, tokenizer, [oracle_rollout(task) for task in TEST_TASKS]
    )

    before = evaluate_env_ce(training_client, heldout_data)
    for _ in range(args.steps):
        backward = training_client.forward_backward(train_data, "cross_entropy")
        optimizer = training_client.optim_step(
            tinker.types.AdamParams(learning_rate=LEARNING_RATE)
        )
        backward.result()
        optimizer.result()
    after = evaluate_env_ce(training_client, heldout_data)

    print(f"held-out env-token CE: {before:.3f} -> {after:.3f} nats/token")


if __name__ == "__main__":
    main()
