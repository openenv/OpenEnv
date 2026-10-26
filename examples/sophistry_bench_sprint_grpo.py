# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Plain `openenv`: there is no `core` extra. The published extras are
#     # daytona, aca, modal and inspect, and uv only warns before installing
#     # the base package anyway, so the old spelling polluted every job log.
#     "openenv",
#     "trl",
#     "datasets",
#     "torch",
#     "transformers",
# ]
# ///

"""Train a policy on `sophistry_bench_sprint_env` with TRL's GRPOTrainer.

Single-step env, so this is a plain prompt -> completion -> reward GRPO setup:
no `environment_factory`/tool-calling needed. Uses `GenericEnvClient` so the
script only depends on `openenv` from PyPI, which also makes it runnable
as a standalone `uv` script, including via Hugging Face Jobs:

    hf jobs uv run examples/sophistry_bench_sprint_grpo.py --flavor a10g-small \
        --secrets HF_TOKEN -- --push-to-hub --out your-username/sophistry-grpo

Connects via a manually-built `UVProvider` + `GenericEnvClient` rather than
`from_env()` + `.sync()`: this env only allows one concurrent session
(`SUPPORTS_CONCURRENT_SESSIONS = False`), and `from_env()` + `.sync()` can
leave behind an orphaned first connection that occupies that single slot (see
https://github.com/huggingface/OpenEnv/pull/854). Needs the `project_path`
git-clone fix from that PR; until it's released, override the `openenv`
dependency above with a git ref of it.

Run locally:
    python examples/sophistry_bench_sprint_grpo.py --n-episodes 64 --steps 50
"""

from __future__ import annotations

import argparse

from datasets import Dataset
from openenv import GenericEnvClient
from openenv.core.containers.runtime.uv_provider import UVProvider
from trl import GRPOConfig, GRPOTrainer

SPACE_REPO_ID = "openenv-community/sophistry_bench_sprint_env"


def _completion_text(completion) -> str:
    """TRL passes either a list of chat messages or a raw string, depending
    on whether the model/dataset use chat templating."""
    if isinstance(completion, list):
        if not completion or not isinstance(completion[-1], dict):
            raise ValueError(f"Unexpected completion shape from TRL: {completion!r}")
        return completion[-1]["content"]
    if isinstance(completion, str):
        return completion
    raise ValueError(f"Unexpected completion type from TRL: {type(completion)!r}")


def build_dataset(client, n_episodes: int) -> Dataset:
    """Walk `reset(seed=i)` to get a fixed, replayable set of advocacy tasks.
    Each row carries the `seed` needed to re-derive the same task later, in
    the reward function."""
    rows = []
    for i in range(n_episodes):
        obs = client.reset(seed=i).observation
        rows.append({"prompt": [{"role": "user", "content": obs["prompt"]}], "seed": i})
    return Dataset.from_list(rows)


def make_reward_func(client):
    """Re-running `reset(seed=...)` before each `step(...)` recreates the
    exact task the completion was sampled for -- the server is
    single-session, so this runs sequentially against one client."""

    def reward_func(completions, seed, **kwargs) -> list[float]:
        assert len(completions) == len(seed), (
            f"completions/seed length mismatch: {len(completions)} vs {len(seed)}"
        )
        rewards = []
        for completion, s in zip(completions, seed):
            client.reset(seed=s)
            result = client.step({"text": _completion_text(completion)})
            rewards.append(result.reward)
        return rewards

    return reward_func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-episodes", type=int, default=64, help="Dataset size.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Total rollouts sampled per step (must be divisible by --num-generations).",
    )
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--out", default="sophistry-grpo-Qwen2.5-0.5B")
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    if args.per_device_batch_size % args.num_generations != 0:
        ap.error("--per-device-batch-size must be divisible by --num-generations")

    provider = UVProvider(
        project_path=f"git+https://huggingface.co/spaces/{SPACE_REPO_ID}",
        app="sophistry_bench_sprint_env.server.app:app",
        context_timeout_s=180.0,  # cold clone + dependency install can be slow
    )
    base_url = provider.start()
    provider.wait_for_ready()

    with GenericEnvClient(base_url=base_url, provider=provider).sync() as client:
        dataset = build_dataset(client, args.n_episodes)

        config = GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_batch_size,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            bf16=True,  # halves the [batch, len, vocab] logits tensor at fp32
            gradient_checkpointing=True,
            logging_steps=1,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.out if args.push_to_hub else None,
        )

        trainer = GRPOTrainer(
            model=args.model,
            reward_funcs=make_reward_func(client),
            train_dataset=dataset,
            args=config,
        )
        trainer.train()
        trainer.save_model(args.out)
        print(f"Saved fine-tuned model to {args.out}")

        if args.push_to_hub:
            trainer.push_to_hub()
            print(f"Pushed fine-tuned model to https://huggingface.co/{args.out}")


if __name__ == "__main__":
    main()
