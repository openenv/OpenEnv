"""Score frontier models on the pelican SVG environment via HF Inference Providers.

Each model is asked for the drawing, the reply goes straight to the environment,
and the environment does all the scoring. The script never computes a reward
itself, which is the point: swap in your own model list and you get numbers
produced by the same code path everyone else's numbers came from.

Read the numbers as a demonstration that the pipeline works end to end, not as a
model ranking. Simon Willison, who invented this prompt, is explicit that "the
correlation between pelican performance and actual model quality has been mostly
severed now" (https://simonwillison.net/2026/Jul/16/kimi-k3/). Frontier models
saturate the canonical task.

Example:
    # Against a deployed Space
    python examples/pelican_svg_eval.py \\
        --env-url https://sergiopaniego-pelican-svg-env.hf.space --samples 20

    # Against a local server
    PYTHONPATH=src:envs uvicorn pelican_svg_env.server.app:app --port 8000 &
    python examples/pelican_svg_eval.py --env-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from envs.pelican_svg_env import PelicanSvgAction, PelicanSvgEnv
from envs.pelican_svg_env.server.svg_source import extract_svg, SvgParseError
from huggingface_hub import AsyncInferenceClient, get_token

# Open weights, served by the Inference Providers router.
# Closed models, reached through their own APIs rather than the router, which
# serves open weights only. Both vendors reject `temperature` on these models:
# Anthropic deprecated it for the Claude 5 family and GPT-5.6 accepts only its
# default, so these cannot be sampled at the same temperature as the open ones.
CLOSED_MODELS = {
    "claude-fable-5": "anthropic",
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-5.6-luna": "openai",
}

ANTHROPIC_KEY_VAR = "PELICAN_ANTHROPIC_API_KEY"

DEFAULT_MODELS = [
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.2",
    "Qwen/Qwen3.6-27B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
]

# The original prompt. Everything else in the catalogue is a variation of ours,
# useful for checking the scorer is not overfitted but not part of the headline.
DEFAULT_TASK = "pelican_bicycle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score models on the pelican SVG environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-url",
        default="http://127.0.0.1:8000",
        help="Base URL of a running pelican_svg_env server or Space.",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, help="Model ids to evaluate."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task id to pin.")
    parser.add_argument("--samples", type=int, default=20, help="Samples per model.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Zero gives one answer repeated, which says "
        "nothing about variance.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12000,
        help="Generation cap. Too low truncates the SVG mid-document, which the "
        "environment reports as truncated_svg rather than as a refusal.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=6, help="Concurrent generations."
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Request the whole completion in one response. Streaming is the "
        "default because the router's gateway cuts a non-streaming request off "
        "at 120 seconds, and a reasoning model on a busy provider needs longer: "
        "Kimi K3 failed 19 of 20 non-streaming samples with a 504 at exactly "
        "120s, and completed in 305s once streamed.",
    )
    parser.add_argument("--out", type=Path, default=Path("pelican_eval_results.json"))
    parser.add_argument(
        "--save-svgs",
        type=Path,
        default=None,
        help="Directory to write each submission's SVG to. Worth using: reading "
        "the numbers without ever looking at the drawings is how a scorer bug "
        "survives a whole benchmark run.",
    )
    return parser.parse_args()


async def _stream(client, model, prompt, seed, args) -> tuple[str, str]:
    """Collect a streamed completion, keeping the answer and the thinking apart.

    Reasoning models put their chain of thought in `delta.reasoning_content` and
    the answer in `delta.content`. Reading only `content` looks like the model
    returned nothing, when in fact it spent the whole budget thinking.
    """
    answer: list[str] = []
    thinking: list[str] = []
    stream = await client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=seed,
        stream=True,
    )
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        if getattr(delta, "content", None):
            answer.append(delta.content)
        for attribute in ("reasoning_content", "reasoning"):
            chunk = getattr(delta, attribute, None)
            if chunk:
                thinking.append(chunk)
    return "".join(answer), "".join(thinking)


async def _generate_anthropic(model, prompt, args) -> dict:
    """Anthropic Messages API. No temperature: the Claude 5 family rejects it."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ[ANTHROPIC_KEY_VAR])
    # Streamed, not because the text is needed incrementally but because the SDK
    # refuses a non-streaming request whose max_tokens could take over ten
    # minutes, which any budget large enough for a detailed SVG does.
    async with client.messages.stream(
        model=model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = await stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    thinking = "".join(
        getattr(b, "thinking", "") for b in message.content if b.type == "thinking"
    )
    return {
        "reply": text,
        "reasoning_chars": len(thinking),
        "prompt_tokens": message.usage.input_tokens,
        "completion_tokens": message.usage.output_tokens,
    }


async def _generate_openai(model, prompt, args) -> dict:
    """OpenAI chat completions. `max_completion_tokens`, and no temperature."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model=model,
        max_completion_tokens=args.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    return {
        "reply": response.choices[0].message.content or "",
        # Reported in tokens, not characters: the API does not return the text of
        # its reasoning, only how much of it there was.
        "reasoning_chars": 0,
        "reasoning_tokens": reasoning_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


async def generate(client, semaphore, model, prompt, seed, args):
    """Ask one model for one drawing, returning the reply and its token usage."""
    async with semaphore:
        for attempt in range(3):
            started = time.monotonic()
            try:
                provider = CLOSED_MODELS.get(model)
                if provider is not None:
                    out = await (
                        _generate_anthropic(model, prompt, args)
                        if provider == "anthropic"
                        else _generate_openai(model, prompt, args)
                    )
                    return {
                        "reply_chars": len(out["reply"]),
                        "latency_s": round(time.monotonic() - started, 2),
                        "error": None,
                        **out,
                    }
                if args.no_stream:
                    response = await client.chat_completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        seed=seed,
                    )
                    usage = response.usage
                    reply = response.choices[0].message.content or ""
                    thinking = ""
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                else:
                    reply, thinking = await _stream(client, model, prompt, seed, args)
                    # Streamed responses carry no usage block, so report
                    # characters and let the caller convert if it cares.
                    prompt_tokens = 0
                    completion_tokens = 0

                error = None
                if not reply and thinking:
                    error = (
                        f"budget spent reasoning without emitting an answer "
                        f"({len(thinking)} chars of thinking). Raise --max-tokens."
                    )
                return {
                    "reply": reply,
                    "reasoning_chars": len(thinking),
                    "reply_chars": len(reply),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_s": round(time.monotonic() - started, 2),
                    "error": error,
                }
            except Exception as exc:
                failure = exc
                if attempt < 2:
                    await asyncio.sleep(4 * (attempt + 1))
        return {
            "reply": "",
            "reasoning_chars": 0,
            "reply_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": round(time.monotonic() - started, 2),
            "error": f"{type(failure).__name__}: {str(failure)[:160]}",
        }


def fetch_prompt(env_url: str, task: str) -> str:
    """Open a short-lived connection just to read the prompt."""
    with PelicanSvgEnv(base_url=env_url) as env:
        return env.reset(task_id=task).observation.prompt


async def run(args: argparse.Namespace) -> list[dict]:
    client = AsyncInferenceClient(api_key=get_token(), timeout=600)
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    if args.save_svgs:
        args.save_svgs.mkdir(parents=True, exist_ok=True)

    # Generate first, with no environment connection open. Holding an idle
    # WebSocket across a long generation phase gets it closed underneath us
    # ("ConnectionClosedOK"), and a reasoning model on a rate-limited provider
    # can take tens of minutes. The connection is only needed for scoring.
    prompt = fetch_prompt(args.env_url, args.task)
    print(f"task: {args.task}\nprompt: {prompt.splitlines()[0]}\n")

    jobs = [(m, i) for m in args.models for i in range(args.samples)]
    print(f"generating {len(jobs)} samples at temperature {args.temperature}...")
    started = time.monotonic()
    outputs = await asyncio.gather(
        *(generate(client, semaphore, m, prompt, 1000 + i, args) for m, i in jobs)
    )
    print(f"generation finished in {time.monotonic() - started:.0f}s\n")

    with PelicanSvgEnv(base_url=args.env_url) as env:
        for (model, sample), out in zip(jobs, outputs):
            row = {"model": model, "sample": sample, "task": args.task, **out}
            row.pop("reply", None)
            if out["error"]:
                rows.append({**row, "reward": None})
                print(f"  GEN FAIL {model} #{sample}: {out['error'][:80]}")
                continue

            # Scoring happens inside the environment, never here.
            env.reset(task_id=args.task)
            result = env.step(PelicanSvgAction(response=out["reply"]))
            observation = result.observation
            judge = observation.breakdown.get("judge") or {}

            svg_path = None
            if args.save_svgs:
                slug = f"{model.replace('/', '__')}__{args.task}__{sample:03d}"
                try:
                    svg_path = args.save_svgs / f"{slug}.svg"
                    svg_path.write_text(extract_svg(out["reply"]))
                except SvgParseError:
                    # Nothing extractable, so keep the raw reply for diagnosis.
                    svg_path = args.save_svgs / f"{slug}.raw.txt"
                    svg_path.write_text(out["reply"])
            rows.append(
                {
                    **row,
                    "reward": result.reward,
                    "gate_passed": observation.gate_passed,
                    "violations": observation.violations,
                    "structure": observation.structure_score,
                    "semantic": observation.semantic_score,
                    "judged": observation.judged,
                    "judge_model": judge.get("model", ""),
                    "caption": judge.get("caption", ""),
                    "svg_path": str(svg_path) if svg_path else None,
                }
            )
            print(
                f"  {model.split('/')[-1][:30]:<30} #{sample:<3} "
                f"reward={result.reward:.3f} gate={observation.gate_passed} "
                f"out_tokens={out['completion_tokens']:<6} "
                f"{','.join(observation.violations)}"
            )
    return rows


def summarise(rows: list[dict], models: list[str]) -> None:
    scored = [r for r in rows if r.get("reward") is not None]
    judges = {r.get("judge_model") for r in scored if r.get("judge_model")}
    print("\n" + "=" * 96)
    print(f"judge: {', '.join(sorted(judges)) or 'none, deterministic scoring only'}")
    print("=" * 96)
    header = (
        f"{'model':<34} {'n':>3} {'mean':>7} {'sd':>6} {'best':>6} "
        f"{'struct':>7} {'sem':>6} {'ans ch':>8} {'rejected':>9}"
    )
    print(header)
    for model in models:
        mine = [r for r in scored if r["model"] == model]
        if not mine:
            print(f"{model.split('/')[-1][:33]:<34} {'no data':>3}")
            continue
        rewards = [r["reward"] for r in mine]
        print(
            f"{model.split('/')[-1][:33]:<34} {len(mine):>3} "
            f"{statistics.mean(rewards):>7.3f} {statistics.pstdev(rewards):>6.3f} "
            f"{max(rewards):>6.3f} "
            f"{statistics.mean(r['structure'] for r in mine):>7.3f} "
            f"{statistics.mean(r['semantic'] for r in mine):>6.3f} "
            f"{statistics.mean(r['reply_chars'] for r in mine):>8.0f} "
            f"{sum(1 for r in mine if not r['gate_passed']):>4}/{len(mine):<4}"
        )

    total_tokens = sum(r.get("completion_tokens", 0) for r in scored)
    total_reasoning = sum(r.get("reasoning_chars", 0) for r in scored)
    total_answer = sum(r.get("reply_chars", 0) for r in scored)
    print(
        f"\noutput volume: {total_answer:,} answer chars, "
        f"{total_reasoning:,} reasoning chars"
        + (f", {total_tokens:,} completion tokens" if total_tokens else "")
        + ". Reasoning is billed and is where the cost goes: Simon's single Kimi "
        "K3 pelican cost 25 cents."
    )
    perfect = [r for r in scored if r["reward"] == 1.0]
    if perfect:
        print(
            f"{len(perfect)} of {len(scored)} samples scored a perfect 1.000. "
            "Saturation on the canonical task is expected and is itself the "
            "argument against using it to rank models."
        )
    failures = [r for r in scored if not r["gate_passed"]]
    if failures:
        print("\nrejected before scoring:")
        for r in failures:
            print(
                f"  {r['model'].split('/')[-1][:30]:<30} #{r['sample']:<3} {r['violations']}"
            )


def main() -> None:
    args = parse_args()
    rows = asyncio.run(run(args))
    summarise(rows, args.models)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
