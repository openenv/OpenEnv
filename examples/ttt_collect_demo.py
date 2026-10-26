"""Smoke-test demo for the harness collect pipeline on Tic-Tac-Toe.

Two run modes:

1. Scripted mode (default, no setup): a scripted OpenSpiel client and a
   deterministic teacher exercise the full pipeline (session → harness →
   verify → serialize) and leave a real ``results.jsonl`` +
   ``metadata.json`` on disk so you can inspect the schema downstream
   tools will consume.

2. Remote mode (``--base-url``): talks to a real OpenSpiel server —
   either a local Docker container or a deployed Hugging Face Space.
   Requires ``OPENSPIEL_GAME=tic_tac_toe`` set on the server.

Usage:
    # Scripted (no Docker, no API):
    PYTHONPATH=src:envs uv run python examples/ttt_collect_demo.py \\
        --num-episodes 10 --output-dir /tmp/ttt-collect-demo

    # Against a Hugging Face Space:
    PYTHONPATH=src:envs uv run python examples/ttt_collect_demo.py \\
        --base-url https://<user>-<repo>.hf.space \\
        --num-episodes 10 --output-dir /tmp/ttt-collect-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from openenv.core.client_types import StepResult
from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
)
from openenv.core.harness.collect import CollectRunner, RolloutSerializer
from openenv.core.llm_client import LLMResponse, ToolCall

from openspiel_env.client import OpenSpielEnv
from openspiel_env.harness import OpenSpielSessionFactory
from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState


class ScriptedTTTClient:
    """A tic_tac_toe client that ends after `turns_to_win` plays."""

    def __init__(self, turns_to_win: int = 1, winning: bool = True):
        self._turns_to_win = turns_to_win
        self._winning = winning
        self._played = 0

    def reset(self, **kwargs: Any) -> StepResult[OpenSpielObservation]:
        self._played = 0
        return StepResult(
            observation=OpenSpielObservation(
                info_state=[1.0] * 9 + [0.0] * 18,
                legal_actions=list(range(9)),
                current_player_id=0,
                game_phase="playing",
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: OpenSpielAction) -> StepResult[OpenSpielObservation]:
        self._played += 1
        done = self._played >= self._turns_to_win
        reward = 1.0 if (done and self._winning) else 0.0
        info_state = [0.0] * 27
        info_state[9] = 1.0
        for i in range(1, 9):
            info_state[i] = 1.0
        return StepResult(
            observation=OpenSpielObservation(
                info_state=info_state,
                legal_actions=[] if done else list(range(1, 9)),
                current_player_id=0,
                game_phase="terminal" if done else "playing",
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> OpenSpielState:
        return OpenSpielState(step_count=self._played, game_name="tic_tac_toe")

    def close(self) -> None:
        return None


def _extract_legal_actions(messages: list[dict[str, Any]]) -> list[int]:
    """Scan messages for the most recent ``legal_actions`` list.

    The prompt and tool results both surface ``legal_actions`` — we look
    at both so the teacher stays legal across turns.
    """
    import json as _json
    import re as _re

    for message in reversed(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue

        # Tool results serialize as JSON strings containing legal_actions.
        try:
            payload = _json.loads(content)
        except (_json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and "legal_actions" in payload:
            legal = payload["legal_actions"]
            if isinstance(legal, list):
                return [int(a) for a in legal]

        # The initial prompt uses the textual form ``Legal actions: [0, 1, ...]``.
        match = _re.search(r"Legal actions:\s*\[([^\]]*)\]", content)
        if match:
            raw = match.group(1).strip()
            if not raw:
                return []
            return [int(x) for x in raw.split(",")]

    return []


def scripted_teacher(messages, tools, sampling):
    """Pick the first legal action seen in the conversation so far."""
    del tools, sampling
    legal = _extract_legal_actions(messages)
    action_id = legal[0] if legal else 0
    return ModelStepResult(
        response=LLMResponse(
            content=f"Playing cell {action_id} (first legal).",
            tool_calls=[
                ToolCall(
                    id=f"t-{action_id}",
                    name="play_move",
                    args={"action_id": action_id},
                )
            ],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ttt-collect-demo"))
    parser.add_argument("--max-turns", type=int, default=9)
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenSpiel server URL (local Docker or Hugging Face Space). "
            "When omitted, runs against the built-in scripted client."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.base_url:
        print(f"Using remote OpenSpiel server: {args.base_url}")
        client_factory = lambda: OpenSpielEnv(base_url=args.base_url)  # noqa: E731
        teacher_name = "scripted-legal-action"
    else:
        print("Using built-in scripted client (no Docker required)")
        client_factory = lambda: ScriptedTTTClient(  # noqa: E731
            turns_to_win=1, winning=True
        )
        teacher_name = "scripted-demo"

    factory = OpenSpielSessionFactory(client_factory, game_name="tic_tac_toe")
    serializer = RolloutSerializer(args.output_dir)
    serializer.write_metadata(
        {
            "env_id": "tic_tac_toe",
            "teacher_model": teacher_name,
            "num_episodes": args.num_episodes,
            "base_url": args.base_url,
        }
    )

    runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=args.max_turns),
    )

    result = runner.run(
        model_step=scripted_teacher,
        num_episodes=args.num_episodes,
        episode_id_prefix="ttt",
    )

    print(
        f"Collected: {result.num_collected}  "
        f"Skipped: {result.num_skipped}  "
        f"Dropped: {result.num_dropped}"
    )
    print(f"Avg reward: {result.avg_reward:.3f}  Success rate: {result.success_rate:.0%}")
    print()
    print(f"results.jsonl:  {serializer.results_path}")
    print(f"metadata.json:  {serializer.metadata_path}")
    print()

    with serializer.results_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    print("First episode (pretty):")
    print(json.dumps(json.loads(first_line), indent=2)[:1500])


if __name__ == "__main__":
    main()
