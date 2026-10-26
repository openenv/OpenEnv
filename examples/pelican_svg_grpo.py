# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openenv",
#     "trl",
#     "trackio",
#     "datasets",
#     "torch",
#     "transformers",
# ]
# ///

"""Train a small policy to draw with TRL's GRPOTrainer on `pelican_svg_env`.

Single-step env, so this is a plain prompt -> completion -> reward GRPO setup.

**The vision judge is off during training, on purpose.** Every judged sample
costs two multimodal API calls, which at a few hundred rollouts a step is both
slow and expensive, and a judge that varies run to run makes the reward curve
unreadable. With the judge disabled the environment scores on the deterministic
structural layer alone and the whole reward is reproducible arithmetic.

That choice sets up the experiment worth running. The structural layer can be
satisfied by two circles, a bar and a blob: it is a proxy, and the interesting question
is whether a model optimising it learns to *draw* or learns to *game it*. So the
script scores a held-out sample with the judge before and after training. If
structure goes up and the judged score does not follow, the policy found the
proxy rather than the task, and that is a result rather than a failure.

Run on Hugging Face Jobs with a GPU:

    hf jobs uv run examples/pelican_svg_grpo.py --flavor a10g-small \\
        --secrets HF_TOKEN -- --steps 60 --push-to-hub \\
        --out your-username/pelican-svg-grpo

Run locally:

    python examples/pelican_svg_grpo.py --steps 5 --n-episodes 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

from datasets import Dataset
from openenv import GenericEnvClient
from openenv.core.containers.runtime.uv_provider import UVProvider
from trl import GRPOConfig, GRPOTrainer

SPACE_REPO_ID = "sergiopaniego/pelican-svg-env"
APP = "pelican_svg_env.server.app:app"

# Simon Willison's original prompt. The rest of the catalogue is variations of
# ours, so training and reporting both stay on this one.
TASK_ID = "pelican_bicycle"


def _completion_text(completion) -> str:
    """TRL hands over either a list of chat messages or a raw string."""
    if isinstance(completion, list):
        if not completion or not isinstance(completion[-1], dict):
            raise ValueError(f"Unexpected completion shape from TRL: {completion!r}")
        return completion[-1]["content"]
    if isinstance(completion, str):
        return completion
    raise ValueError(f"Unexpected completion type from TRL: {type(completion)!r}")


def start_env(disable_judge: bool) -> tuple[str, UVProvider]:
    """Boot the environment from its Space repo and return its base URL."""
    env_vars = {"PELICAN_SVG_DISABLE_JUDGE": "1"} if disable_judge else {}
    provider = UVProvider(
        project_path=f"git+https://huggingface.co/spaces/{SPACE_REPO_ID}",
        app=APP,
        env_vars=env_vars,
        context_timeout_s=180.0,  # cold clone plus dependency install is slow
    )
    base_url = provider.start()
    provider.wait_for_ready()
    return base_url, provider


def build_dataset(
    prompt: str, n_episodes: int, model: str, enable_thinking: bool
) -> Dataset:
    """One row per episode, all on the same pinned task.

    The prompt is identical every row, which is what we want: the variation
    being learned from is in the sampled completions, not in the question.

    The chat template is applied here rather than left to the trainer so that
    `enable_thinking` can be set. Qwen3 is a hybrid reasoning model with thinking
    **on by default**: left alone, a 0.6B spends its whole completion budget
    inside `<think>` and never emits an SVG, so every rollout scores zero and
    GRPO has no variance to learn from. Pre-formatting to a plain string prompt
    also sidesteps differences in how TRL versions forward template kwargs.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return Dataset.from_list([{"prompt": text} for _ in range(n_episodes)])


def fetch_prompt(base_url: str) -> str:
    """Read the task prompt over a short-lived connection.

    Deliberately not reusing a long-lived client. An idle WebSocket held across
    the probe, which loads a model and generates two dozen completions on the
    GPU, gets closed by the server underneath it ("ConnectionClosedOK") and takes
    the whole run down before step one.
    """
    with GenericEnvClient(base_url=base_url).sync() as client:
        return client.reset(task_id=TASK_ID).observation["prompt"]


def make_reward_func(client):
    """Score each completion through the environment, never in the trainer."""

    def reward_func(completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            client.reset(task_id=TASK_ID)
            result = client.step({"response": _completion_text(completion)})
            rewards.append(float(result.reward or 0.0))
        return rewards

    return reward_func


def judged_probe(base_url: str, texts, save_to: pathlib.Path | None = None) -> dict:
    """Re-score finished completions with the judge switched on.

    Run before and after training. Structure rising while this stays flat is the
    signature of a policy that found the proxy rather than the task.

    The drawings themselves are kept, not just their scores. A before-and-after
    pair of actual pictures is the only way a reader can judge whether "the
    structural score rose" means anything, and a run that discards them leaves
    the claim resting on the author's word.
    """
    from openenv import GenericEnvClient as Client

    samples, gate_failures = [], 0
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
    with Client(base_url=base_url).sync() as client:
        for index, text in enumerate(texts):
            client.reset(task_id=TASK_ID)
            result = client.step({"response": text})
            observation = result.observation
            passed = bool(observation.get("gate_passed", False))
            gate_failures += not passed
            record = {
                "index": index,
                "reward": float(result.reward or 0.0),
                "structure": observation.get("structure_score", 0.0),
                "semantic": observation.get("semantic_score", 0.0),
                "gate_passed": passed,
                "violations": observation.get("violations", []),
                "caption": (
                    (observation.get("breakdown") or {}).get("judge") or {}
                ).get("caption", ""),
            }
            if save_to is not None:
                svg = _extract_svg(text)
                name = f"{index:03d}.svg" if svg else f"{index:03d}.txt"
                (save_to / name).write_text(svg or text)
                record["file"] = name
            samples.append(record)

    def mean(key):
        values = [s[key] for s in samples]
        return round(statistics.mean(values), 4) if values else 0.0

    summary = {
        "n": len(samples),
        "reward": mean("reward"),
        "structure": mean("structure"),
        "semantic": mean("semantic"),
        "gate_failures": gate_failures,
    }
    if save_to is not None:
        (save_to / "samples.json").write_text(
            json.dumps({"summary": summary, "samples": samples}, indent=2)
        )
    return summary


def _extract_svg(text: str) -> str:
    """Pull the last complete SVG document out of a completion, if there is one."""
    matches = re.findall(r"<svg\b.*?</svg\s*>", text or "", re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else ""


def sample_completions(
    model_path: str,
    prompt: str,
    count: int,
    max_new_tokens: int,
    enable_thinking: bool = False,
):
    """Greedy-ish samples from a checkpoint, for the before and after probe.

    Applies the chat template with the same `enable_thinking` the training
    dataset used. Probing in a different template mode than the one the policy
    was trained under would measure the wrong thing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text] * count, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )


def report_proxy_gap(before: dict, after: dict) -> None:
    """Say plainly whether the policy learned the task or just the proxy.

    Compares how much each layer gained rather than testing either against a
    fixed epsilon. A judge that recognises nothing scores near zero either way,
    so what identifies the behaviour is the disparity between the gains and not
    their absolute size.
    """
    structure_gain = after["structure"] - before["structure"]
    semantic_gain = after["semantic"] - before["semantic"]
    print(
        f"structure {before['structure']:.3f} -> {after['structure']:.3f} "
        f"({structure_gain:+.3f}), semantic {before['semantic']:.3f} -> "
        f"{after['semantic']:.3f} ({semantic_gain:+.3f}), gate failures "
        f"{before['gate_failures']} -> {after['gate_failures']}"
    )
    if structure_gain <= 0.05:
        print("The structural score did not move. Nothing was learned to speak of.")
        return
    # A judge that recognises nothing scores near zero, so compare the gains.
    ratio = structure_gain / max(semantic_gain, 1e-6)
    if after["semantic"] < 0.1 and ratio > 3.0:
        print(
            f"WARNING: structure gained {ratio:.0f}x more than the judged score, "
            f"which ended at {after['semantic']:.3f}. The policy learned to "
            "satisfy the deterministic layer without drawing anything a judge "
            "recognises. Read the structural gain as reward hacking, not skill."
        )
    else:
        print(
            "Both layers moved together, so the structural gain is backed by "
            "drawings a judge recognises."
        )


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Qwen3-0.6B is the smallest recent generation. There is no "
        "Qwen3.5 or Qwen3.6 at this size; the smallest of those is Qwen3.5-2B.",
    )
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Let a hybrid reasoning model think before answering. Off by "
        "default: at this size the thinking eats the completion budget and no "
        "SVG comes out, so every rollout scores zero.",
    )
    ap.add_argument("--n-episodes", type=int, default=256, help="Dataset size.")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Sequences per forward pass. This is the memory knob, not a quality "
        "knob: the logits tensor is [batch, length, vocab] and accelerate upcasts "
        "it to fp32 regardless of bf16, so 8 x 2560 x 151936 asks for 12.4 GB and "
        "dies on a 24 GB A10G. Keep it small and raise "
        "--gradient-accumulation-steps instead.",
    )
    ap.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Trades steps for memory. The GRPO group size is "
        "per-device-batch-size x this, so the group stays large while each "
        "forward pass stays small.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Completions per prompt. Must divide "
        "per-device-batch-size x gradient-accumulation-steps.",
    )
    ap.add_argument(
        "--max-completion-length",
        type=int,
        default=2048,
        help="An SVG needs room. Too small and every rollout is truncated_svg, "
        "so the reward is zero everywhere and GRPO has no gradient to work with.",
    )
    ap.add_argument(
        "--probe-samples",
        type=int,
        default=24,
        help="Completions to re-score with the judge before and after training. "
        "Zero skips the probe. Do not go small: at n=4 a run whose training "
        "reward rose 16-fold produced identical before and after probes, which "
        "is not evidence of anything either way.",
    )
    ap.add_argument("--out", default="pelican-svg-grpo-Qwen3-0.6B")
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the checkpoint. Worth doing: a Job's filesystem goes away "
        "with the Job, so a run without this leaves nothing behind but stdout.",
    )
    ap.add_argument(
        "--trackio-space",
        default=None,
        help="Space id to stream the run to, so the curves are public and "
        "anyone can check them instead of taking a screenshot on trust. "
        "Defaults to the value of --out.",
    )
    ap.add_argument(
        "--judge-in-reward",
        action="store_true",
        help="Include the vision judge in the training reward. Off by default "
        "because it costs two multimodal calls per rollout and makes the reward "
        "non-deterministic. Turning it on is the only way to find out whether a "
        "policy games a reward the judge is part of, which the deterministic-only "
        "runs cannot answer.",
    )
    ap.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable trackio. Only useful for a throwaway smoke run.",
    )
    args = ap.parse_args()

    generation_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    if generation_batch % args.num_generations != 0:
        ap.error(
            "--per-device-batch-size x --gradient-accumulation-steps "
            f"({generation_batch}) must be divisible by --num-generations "
            f"({args.num_generations})"
        )

    probe_dir = pathlib.Path(args.out) / "probe"
    base_url, provider = start_env(disable_judge=not args.judge_in_reward)
    judge_url: str | None = None
    judge_provider = None
    try:
        # Phases are kept apart on purpose. Sampling completions takes minutes of
        # GPU time with no environment traffic, and a connection left open across
        # it is closed by the server, so each phase opens its own.
        prompt = fetch_prompt(base_url)
        dataset = build_dataset(
            prompt, args.n_episodes, args.model, args.enable_thinking
        )

        before = None
        if args.probe_samples:
            if args.judge_in_reward:
                # The training environment already has the judge on, so booting a
                # second one would clone the repo and reinstall for no reason.
                judge_url = base_url
            else:
                judge_url, judge_provider = start_env(disable_judge=False)
            texts = sample_completions(
                args.model,
                prompt,
                args.probe_samples,
                args.max_completion_length,
                enable_thinking=args.enable_thinking,
            )
            before = judged_probe(judge_url, texts, save_to=probe_dir / "before")
            print(f"judged probe before training: {before}")

        # Deliberately not `provider=provider`: that makes the client's
        # __exit__ shut the server down, and with --judge-in-reward the
        # after-probe scores against that same server. The provider is owned by
        # the `finally` below, which runs after every probe.
        with GenericEnvClient(base_url=base_url).sync() as client:
            config = GRPOConfig(
                output_dir=args.out,
                max_steps=args.steps,
                learning_rate=args.lr,
                per_device_train_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_generations=args.num_generations,
                max_completion_length=args.max_completion_length,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                report_to=[] if args.no_tracking else "trackio",
                **(
                    {}
                    if args.no_tracking
                    else {"trackio_space_id": args.trackio_space or args.out}
                ),
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

            rewards = [
                entry["reward"]
                for entry in trainer.state.log_history
                if "reward" in entry
            ]
            if rewards:
                head = statistics.mean(rewards[: max(1, len(rewards) // 5)])
                tail = statistics.mean(rewards[-max(1, len(rewards) // 5) :])
                print(
                    f"reward first fifth {head:.4f} -> last fifth {tail:.4f} "
                    f"(delta {tail - head:+.4f}) over {len(rewards)} logged steps"
                )

        after = None
        if args.probe_samples and judge_url:
            texts = sample_completions(
                args.out,
                prompt,
                args.probe_samples,
                args.max_completion_length,
                enable_thinking=args.enable_thinking,
            )
            after = judged_probe(judge_url, texts, save_to=probe_dir / "after")
            print(f"judged probe after training: {after}")
            if before:
                print(
                    "structure "
                    f"{before['structure']:.3f} -> {after['structure']:.3f}, "
                    "semantic "
                    f"{before['semantic']:.3f} -> {after['semantic']:.3f}"
                )
                if args.probe_samples < 16:
                    print(
                        f"NOTE: the probe is n={args.probe_samples}, too small "
                        "to separate a real gain from noise. Treat the "
                        "comparison as indicative only."
                    )
                report_proxy_gap(before, after)

        summary = {
            "model": args.model,
            "steps": args.steps,
            "reward_curve": rewards,
            "judged_before": before,
            "judged_after": after,
        }
        print("SUMMARY " + json.dumps(summary))

        if args.push_to_hub:
            trainer.push_to_hub()
            print(f"Pushed to https://huggingface.co/{args.out}")
            if probe_dir.exists():
                from huggingface_hub import HfApi

                HfApi().upload_folder(
                    repo_id=args.out,
                    folder_path=str(probe_dir),
                    path_in_repo="probe",
                )
                print(
                    "Probe drawings, before and after, at "
                    f"https://huggingface.co/{args.out}/tree/main/probe"
                )
        else:
            print(
                "NOTE: --push-to-hub was not passed, so this checkpoint dies "
                "with the Job and only the numbers above survive."
            )
        if not args.no_tracking:
            space = args.trackio_space or args.out
            print(f"Curves: https://huggingface.co/spaces/{space}")
    finally:
        provider.stop()
        if judge_provider is not None:
            judge_provider.stop()


if __name__ == "__main__":
    main()
