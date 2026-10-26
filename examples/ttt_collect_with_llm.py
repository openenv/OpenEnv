"""Collect Tic-Tac-Toe rollouts with a real LLM as the teacher.

Provider-agnostic via ``openenv.core.llm_client``: works with OpenAI,
Anthropic, and any OpenAI-compatible endpoint (vLLM, TGI, Ollama,
HuggingFace Inference, Together, Groq, Fireworks, ...) without any
code change.

The OpenSpiel server can be local (Docker) or a Hugging Face Space —
pass the URL via ``--base-url``. The LLM endpoint is independent:
``--llm-endpoint`` + ``--llm-port`` for OpenAI-compatible servers, or
``--provider openai|anthropic`` for hosted APIs.

Examples:
    # Hosted OpenAI against a HF Space:
    OPENAI_API_KEY=sk-... python examples/ttt_collect_with_llm.py \\
        --base-url https://<user>-<repo>.hf.space \\
        --provider openai --model gpt-5-mini \\
        --num-episodes 20 \\
        --output-dir /tmp/ttt-openai

    # Local vLLM (OpenAI-compatible) with Qwen:
    python examples/ttt_collect_with_llm.py \\
        --base-url http://localhost:8000 \\
        --llm-endpoint http://localhost:8001 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --num-episodes 20

    # Anthropic against a HF Space:
    ANTHROPIC_API_KEY=sk-ant-... python examples/ttt_collect_with_llm.py \\
        --base-url https://<user>-<repo>.hf.space \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --num-episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter
from openenv.core.harness.collect import (
    CollectRunner,
    RolloutSerializer,
    build_model_step,
)
from openenv.core.llm_client import LLMClient, OpenAIClient, create_llm_client

from openspiel_env.client import OpenSpielEnv
from openspiel_env.harness import OpenSpielSessionFactory

SYSTEM_PROMPT = """You are playing tic-tac-toe against a random opponent.

For each turn, call the `play_move` tool with an ``action_id`` drawn from
the ``legal_actions`` list shown in the latest observation. Numbers on
the board are empty cells; `X` and `O` are played cells. You play as X.
Choose moves that win or block the opponent.
"""


def _env_api_key(provider: str) -> str:
    mapping = {
        "openai": ["OPENAI_API_KEY", "API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "API_KEY"],
    }
    for name in mapping.get(provider, []):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        f"No API key found for provider={provider!r}. "
        f"Set one of: {mapping.get(provider, [])}"
    )


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    """Pick the right LLMClient subclass based on CLI flags."""
    if args.llm_endpoint:
        # OpenAI-compatible self-hosted endpoint (vLLM, TGI, Ollama, etc.).
        return OpenAIClient(
            endpoint=args.llm_endpoint,
            port=args.llm_port,
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY") or "not-needed",
            system_prompt=SYSTEM_PROMPT,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Hosted provider.
    return create_llm_client(
        provider=args.provider,
        model=args.model,
        api_key=_env_api_key(args.provider),
        system_prompt=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenSpiel server URL (local Docker or HF Space).",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ttt-collect-llm"))
    parser.add_argument(
        "--episode-id-prefix",
        default="ttt",
        help="Prefix for serialized episode ids (stable across resumes).",
    )

    # LLM selection: either a hosted provider or a self-hosted endpoint.
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Hosted provider (ignored when --llm-endpoint is set).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model name to pass to the LLM API.",
    )
    parser.add_argument(
        "--llm-endpoint",
        default=None,
        help=(
            "Base URL of an OpenAI-compatible server (e.g. http://localhost:8001 "
            "for vLLM). /v1 is appended when the URL has no path."
        ),
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=None,
        help=(
            "Port appended to --llm-endpoint when the URL does not include one. "
            "No default: earlier versions assumed 8000."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=200)

    parser.add_argument(
        "--keep-losses",
        action="store_true",
        help="Keep losing rollouts. Default keeps only wins + draws (reward >= 0).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm_client = build_llm_client(args)
    teacher_label = (
        f"{llm_client.base_url}/{args.model}"
        if args.llm_endpoint
        else f"{args.provider}/{args.model}"
    )

    factory = OpenSpielSessionFactory(
        lambda: OpenSpielEnv(base_url=args.base_url),
        game_name="tic_tac_toe",
    )
    serializer = RolloutSerializer(args.output_dir)
    serializer.write_metadata(
        {
            "env_id": "tic_tac_toe",
            "env_base_url": args.base_url,
            "teacher": teacher_label,
            "num_episodes_requested": args.num_episodes,
            "temperature": args.temperature,
            "keep_losses": args.keep_losses,
        }
    )

    runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=args.max_turns),
    )

    model_step = build_model_step(llm_client, system_prompt=SYSTEM_PROMPT)

    should_keep = None if args.keep_losses else (lambda record: record.reward >= 0.0)

    print(f"Env server:  {args.base_url}")
    print(f"Teacher:     {teacher_label}")
    print(f"Output dir:  {args.output_dir}")
    print()

    result = runner.run(
        model_step=model_step,
        num_episodes=args.num_episodes,
        episode_id_prefix=args.episode_id_prefix,
        should_keep=should_keep,
    )

    print(
        f"Collected: {result.num_collected}  "
        f"Skipped: {result.num_skipped}  "
        f"Dropped: {result.num_dropped}"
    )
    print(
        f"Avg reward: {result.avg_reward:.3f}  "
        f"Success rate: {result.success_rate:.0%}"
    )
    print()
    print(f"results.jsonl:  {serializer.results_path}")
    print(f"metadata.json:  {serializer.metadata_path}")

    if serializer.results_path.exists():
        with serializer.results_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if first_line:
            payload = json.loads(first_line)
            print()
            print(
                f"First episode: id={payload['episode_id']} "
                f"reward={payload['reward']} turns={payload['metrics'].get('turns')}"
            )


if __name__ == "__main__":
    main()
