# SPDX-License-Identifier: BSD-3-Clause

"""
An LLM agent that plays one `posttrain_env` episode through its MCP tools.

The agent reads the task prompt, writes and submits training scripts to HF
Jobs through the environment's broker, checks them with `dev_evaluate`, and
calls `submit`, which ends the episode with the hidden-split score as reward.
The agent never holds the broker's token: only the server does.

Usage:

    # Terminal 1: the server, with the broker token (see envs/posttrain_env/README.md)
    docker build -t posttrain-env -f envs/posttrain_env/server/Dockerfile envs/posttrain_env
    docker run --rm -p 8000:8000 -e HF_TOKEN -e POSTTRAIN_TEST_SEED posttrain-env

    # Terminal 2: the agent, with any OpenAI-compatible endpoint.
    # The default is Hugging Face Inference Providers, with an HF token
    # allowed to call them; keep it separate from the broker's token.
    export OPENAI_API_KEY=...
    python examples/posttrain_env_agent.py --task t0_arithmetic_format \
        --model Qwen/Qwen3.8-2.4T-A95B --transcript episode.json
"""

import argparse
import json
import os

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction
from posttrain_env import PostTrainEnv

SYSTEM_PROMPT = (
    "You are an ML engineer working through tools. Each tool call is one step "
    "of a limited episode, and jobs spend a limited budget, so plan before you "
    "submit a job and read its logs when it fails. Calling `submit` ends the "
    "episode."
)
MAX_RESULT_CHARS = 20_000


def tool_specs(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def call(env, name, arguments):
    """Run one tool call; return the text for the model and the step result."""
    result = env.step(CallToolAction(tool_name=name, arguments=arguments))
    obs = result.observation
    if obs.error is not None:
        return f"Error: {obs.error.message}", result
    data = obs.result
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return json.dumps(data)[:MAX_RESULT_CHARS], result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="t0_artifact")
    parser.add_argument("--model", default="Qwen/Qwen3.8-2.4T-A95B")
    parser.add_argument(
        "--base-url", default=os.environ.get("POSTTRAIN_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ENDPOINT_URL", "https://router.huggingface.co/v1"),
    )
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--transcript", help="write the messages and outcome as JSON")
    args = parser.parse_args()

    # Episodes run for an hour or more, so ride out provider rate limits.
    llm = OpenAI(
        base_url=args.endpoint, api_key=os.environ["OPENAI_API_KEY"], max_retries=10
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # `submit` blocks while the score job runs, so the client waits for the
    # server's longest tool call instead of the default message timeout.
    with PostTrainEnv(base_url=args.base_url, message_timeout_s=3600).sync() as env:
        reset = env.reset(task_id=args.task)
        tools = tool_specs(env.list_tools())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": reset.observation.metadata["prompt"]},
        ]
        result = reset
        for turn in range(1, args.max_turns + 1):
            response = llm.chat.completions.create(
                model=args.model, messages=messages, tools=tools, max_tokens=8192
            )
            usage["prompt_tokens"] += response.usage.prompt_tokens
            usage["completion_tokens"] += response.usage.completion_tokens
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                print(f"[{turn}] text: {(message.content or '')[:300]}")
                messages.append(
                    {
                        "role": "user",
                        "content": "Use a tool, or call `submit` to finish.",
                    }
                )
                continue
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    text = f"Error: arguments are not valid JSON ({e})"
                else:
                    print(f"[{turn}] {name} {json.dumps(arguments)[:300]}")
                    text, result = call(env, name, arguments)
                print(f"    -> {text[:300]}")
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": text}
                )
                if result.done:
                    break
            if result.done:
                break

        metadata = result.observation.metadata
        outcome = {
            "task": args.task,
            "model": args.model,
            "turns": turn,
            "done": result.done,
            "reward": result.reward,
            "episode_end": metadata.get("episode_end"),
            "service_spend": metadata.get("service_spend"),
            "llm_usage": usage,
        }
        print(json.dumps(outcome, indent=2))
        if args.transcript:
            with open(args.transcript, "w") as f:
                json.dump({"outcome": outcome, "messages": messages}, f, indent=2)


if __name__ == "__main__":
    main()
