#!/usr/bin/env python3
"""Play QED Math with an LLM via any OpenAI-compatible API.

QED Math is a mathematical reasoning benchmark that evaluates LLMs on their
ability to solve proof-based and answer-based math problems (AMC/AIME/Olympiad
level). The agent interacts via three MCP tools: ``get_problem``,
``get_grading_guidelines``, and ``submit_proof``.

Prerequisites
-------------
1. Build the QED Math Docker image::

       docker build -f envs/qed_math_env/server/Dockerfile -t qed-math-env:latest .

2. Run the container::

       docker run -p 8000:8000 \\
         -e JUDGE_API_BASE_URL=https://router.huggingface.co/v1 \\
         -e JUDGE_API_KEY=$HF_TOKEN \\
         qed-math-env:latest

3. Set your API credentials and run::

       export API_BASE_URL=https://router.huggingface.co/v1
       export API_KEY=$HF_TOKEN
       export MODEL=openai/gpt-oss-120b:novita
       python examples/qed_math_inference.py

Environment variables
---------------------
API_BASE_URL  OpenAI-compatible base URL (default: HuggingFace router)
API_KEY       API key (falls back to HF_TOKEN)
MODEL         Model identifier (default: openai/gpt-oss-120b:novita)
QED_MATH_URL  Base URL of the running QED Math server (default: http://localhost:8000)
MAX_EPISODES  Number of episodes to play (default: 3)
MAX_TOKENS    Max tokens per LLM completion (default: 4096)
VERBOSE       Set to "0" to suppress step-level output (default: 1)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b:novita")

QED_MATH_URL = os.getenv("QED_MATH_URL", "http://localhost:8000")

MAX_EPISODES = int(os.getenv("MAX_EPISODES", "3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
VERBOSE = os.getenv("VERBOSE", "1") != "0"

SYSTEM_PROMPT = """You are an expert mathematician. You will be given a mathematical problem to solve.

Your workflow:
1. Call get_problem to retrieve the problem statement.
2. Optionally call get_grading_guidelines to understand how your proof will be evaluated.
3. Reason carefully, then call submit_proof with your complete solution.

For answer-based problems (AMC/AIME style):
- Wrap your final numerical answer in \\boxed{} inside the proof text.
- Example: "The answer is \\boxed{42}."

For proof-based problems:
- Write a rigorous, step-by-step proof.
- State every non-trivial claim and justify it.

For multi-step problems:
- If submit_proof returns done=false, continue refining and submit again.

Be concise but complete.
"""


def _tools_to_openai_format(tools: list) -> list[dict[str, Any]]:
    """Convert MCP tool descriptors to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        input_schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {}
        )
        if input_schema and "properties" in input_schema:
            for name, schema in input_schema["properties"].items():
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", ""),
                }
            required = input_schema.get("required", [])

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return openai_tools


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    env: Any,
    client: OpenAI,
    tools: list[dict[str, Any]],
    episode_num: int,
) -> dict[str, Any]:
    """Run a single QED Math episode and return a result summary dict."""
    tool_names = {t["function"]["name"] for t in tools}

    reset_result = await env.reset()
    obs = reset_result.observation

    problem_id = obs.problem_id
    problem_type = obs.problem_type
    # The problem text is available after calling get_problem via MCP tool;
    # we prime the chat with a bare user message so the LLM calls get_problem first.
    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(
            f"Episode {episode_num}  |  problem_id={problem_id or '(random)'}  |  type={problem_type}"
        )
        print(f"{'=' * 60}")

    chat_history: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Please solve the math problem."},
    ]

    step_count = 0
    final_reward: float = 0.0
    final_score: int = 0
    done = False
    # Accumulate output tokens across all LLM turns so reward shaping
    # (discount factor + length penalty) matches QED-Nano training semantics.
    total_output_tokens = 0

    while not done:
        step_count += 1
        if VERBOSE:
            print(f"\n--- Step {step_count} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=cast(Any, chat_history),
            tools=cast(Any, tools),
            tool_choice="required",
            max_completion_tokens=MAX_TOKENS,
        )

        message = response.choices[0].message

        # Extract tool call (fall back to submit_proof if the model returns text only)
        if message.tool_calls:
            tool_call_obj = cast(Any, message.tool_calls[0])
            function_payload = getattr(tool_call_obj, "function", None)

            if function_payload is not None:
                tool_name = str(getattr(function_payload, "name", "submit_proof"))
                raw_arguments = str(getattr(function_payload, "arguments", "{}"))
                try:
                    tool_args = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    tool_args = {"proof": raw_arguments}
            else:
                # Custom/unknown tool-call shape: fall back to submit_proof.
                tool_name = "submit_proof"
                raw_input = getattr(tool_call_obj, "input", "")
                tool_args = {"proof": str(raw_input)}
                raw_arguments = json.dumps(tool_args)

            tool_call_id = str(getattr(tool_call_obj, "id", "fallback"))
        else:
            tool_name = "submit_proof"
            tool_args = {"proof": message.content or ""}
            tool_call_id = "fallback"
            raw_arguments = json.dumps(tool_args)

        # Track token usage for reward shaping
        if response.usage:
            total_output_tokens += response.usage.completion_tokens

        if VERBOSE:
            preview = json.dumps(tool_args)[:120]
            print(f"Tool: {tool_name}({preview})")

        # Append the assistant turn
        chat_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": raw_arguments,
                        },
                    }
                ],
            }
        )

        # Guard against hallucinated tool names
        if tool_name not in tool_names:
            if VERBOSE:
                print(
                    f"  [warn] Unknown tool '{tool_name}' — falling back to submit_proof"
                )
            tool_name = "submit_proof"
            tool_args = {"proof": tool_args.get("proof", str(tool_args))}

        # Pass cumulative output token count to submit_proof so the server can
        # apply discount-factor and length-penalty reward shaping (QED-Nano training
        # semantics). Other tools ignore this extra kwarg via **kwargs.
        call_kwargs = dict(tool_args)
        if tool_name == "submit_proof":
            call_kwargs.setdefault("output_length_tokens", total_output_tokens)

        # Dispatch to the environment
        step_result = await env.call_tool(tool_name, **call_kwargs)

        # Normalise result to a text string for the LLM
        if hasattr(step_result, "model_dump"):
            result_dict = step_result.model_dump()
        elif isinstance(step_result, dict):
            result_dict = step_result
        else:
            result_dict = {"result": str(step_result)}

        result_text = json.dumps(result_dict)

        if VERBOSE:
            preview = result_text[:200]
            print(f"Result: {preview}{'...' if len(result_text) > 200 else ''}")

        # Check if episode ended (submit_proof may be multi-attempt for
        # multi-step problems).
        if tool_name == "submit_proof":
            final_reward = float(result_dict.get("reward") or 0.0)
            final_score = int(result_dict.get("score") or 0)
            done = bool(result_dict.get("done", True))
            if VERBOSE:
                if done:
                    outcome = (
                        "CORRECT" if result_dict.get("is_correct") else "INCORRECT"
                    )
                    print(
                        f"\nOutcome: {outcome}  score={final_score}/7  reward={final_reward:.3f}"
                    )
                else:
                    attempts_remaining = int(result_dict.get("attempts_remaining", 0))
                    print(
                        "\nSubmission graded but episode continues  "
                        f"score={final_score}/7  reward={final_reward:.3f}  "
                        f"attempts_remaining={attempts_remaining}"
                    )
                feedback = result_dict.get("feedback", "")
                if feedback and not result_dict.get("is_correct"):
                    print(f"Feedback: {feedback[:300]}")
        else:
            done = bool(result_dict.get("done", False))

        if not done:
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                }
            )

    return {
        "episode": episode_num,
        "problem_id": problem_id,
        "problem_type": problem_type,
        "score": final_score,
        "reward": final_reward,
        "steps": step_count,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def async_main() -> None:
    if not API_KEY:
        raise SystemExit(
            "API_KEY (or HF_TOKEN) must be set.\n"
            "  export API_KEY=your_key_here\n"
            "  # or\n"
            "  export HF_TOKEN=your_hf_token"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    qed_math_env_module = importlib.import_module("qed_math_env.client")
    QEDMathEnv = qed_math_env_module.QEDMathEnv

    async with QEDMathEnv(base_url=QED_MATH_URL) as env:
        # Discover tools and convert to OpenAI format
        mcp_tools = await env.list_tools()
        tools = _tools_to_openai_format(mcp_tools)

        if VERBOSE:
            tool_names = [t["function"]["name"] for t in tools]
            print(f"QED Math server: {QED_MATH_URL}")
            print(f"API:   {API_BASE_URL}")
            print(f"Model: {MODEL}")
            print(f"Tools: {tool_names}")
            print(f"Episodes: {MAX_EPISODES}")

        results = []
        for episode_num in range(1, MAX_EPISODES + 1):
            result = await run_episode(env, client, tools, episode_num)
            results.append(result)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    correct = sum(1 for r in results if r["score"] == 7)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_tokens = sum(r["output_tokens"] for r in results) / len(results)

    print(f"Episodes:      {len(results)}")
    print(
        f"Fully correct: {correct}/{len(results)} ({100 * correct / len(results):.1f}%)"
    )
    print(f"Avg reward:    {avg_reward:.3f}")
    print(f"Avg steps:     {avg_steps:.1f}")
    print(f"Avg tokens:    {avg_tokens:.0f}")
    print()
    for r in results:
        print(
            f"  ep={r['episode']:>2}  problem_id={r['problem_id'] or '?':<20}  "
            f"score={r['score']}/7  reward={r['reward']:.3f}  "
            f"steps={r['steps']}  tokens={r['output_tokens']}"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
