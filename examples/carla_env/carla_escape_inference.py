#!/usr/bin/env python3
"""
Inference script for fine-tuned models on the CARLA trolley escape scenario.

Uses the same prompt and tool definitions as the GRPO training script (carla.py in TRL)
to ensure consistent behavior with the fine-tuned model.

Usage:
    # Run with a fine-tuned model from the Hub
    python carla_escape_inference.py \
        --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
        --base-url https://sergiopaniego-carla-env.hf.space

    # Save camera images
    python carla_escape_inference.py \
        --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
        --base-url https://sergiopaniego-carla-env.hf.space \
        --save-images

    # Run multiple episodes
    python carla_escape_inference.py \
        --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
        --base-url https://sergiopaniego-carla-env.hf.space \
        --episodes 5
"""

import argparse
import asyncio
import base64
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "envs"))

from carla_env import CarlaEnv, CarlaAction
from llm_clients import LocalClient

# Same prompt used during GRPO training in TRL's carla.py
TRAINING_PROMPT = """You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""

# Same tools the model was trained with (OpenAI format, used by the chat template)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Advance time and get a new observation of the scene.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_stop",
            "description": "Apply maximum braking to stop the vehicle.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lane_change",
            "description": "Change lane to the left or right. Direction must be 'left' or 'right'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction to change lane",
                    }
                },
                "required": ["direction"],
            },
        },
    },
]

SCENARIO_NAME = "trolley_micro_escape_exists"


def _describe(obs) -> str:
    """Build a text description from the observation fields (same as training)."""
    parts = []
    parts.append(f"Speed: {obs.speed_kmh:.1f} km/h.")
    if obs.nearby_actors:
        for actor in obs.nearby_actors:
            parts.append(f"- {actor.get('type', 'actor')} at {actor.get('distance', '?')}m")
    else:
        parts.append("No nearby actors detected.")
    if obs.collision_detected:
        parts.append(f"COLLISION detected with {obs.collided_with or 'unknown'}!")
    return "\n".join(parts)


async def run_episode(llm, base_url, verbose=True, save_images=False, output_dir="llm_images", max_turns=10):
    """Run one escape episode with the fine-tuned model.

    Replicates the exact message flow from GRPO training:
    1. Initial prompt has the observation concatenated (no newline), matching
       ``prompt[-1]["content"] += observation`` from the trainer.
    2. The model's tool calls are appended as an assistant message with ``tool_calls``.
    3. Tool results are appended as ``{"role": "tool", "name": ..., "content": ...}``.
    4. The tokenizer's ``apply_chat_template(tools=TOOLS)`` renders the conversation
       in the same format the model was trained on.
    """
    env = CarlaEnv(base_url=base_url)
    async with env:
        result = await env.reset(scenario_name=SCENARIO_NAME)
        reward = 0.0
        env_description = _describe(result.observation)

        if verbose:
            print(f"\n   Initial observation:\n   {env_description}\n")

        if save_images:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Build initial prompt exactly as the trainer does:
        # prompt[-1]["content"] += observation  (concatenated, no separator)
        messages = [{"role": "user", "content": TRAINING_PROMPT + env_description}]

        for turn in range(max_turns):
            response = llm.chat(messages, TOOLS, max_tokens=512)

            if verbose:
                if response["text"]:
                    print(f"   [{turn+1}] Model text: {response['text'][:200]}")

            if not response["tool_calls"]:
                if verbose:
                    print(f"   [{turn+1}] No tool call — ending episode")
                break

            tool_call = response["tool_calls"][0]
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            if verbose:
                args_str = f"({tool_args})" if tool_args else ""
                print(f"   [{turn+1}] Action: {tool_name}{args_str}")

            # Execute action on the environment (the server advances time per tool)
            if tool_name == "emergency_stop":
                step_result = await env.step(CarlaAction(action_type="emergency_stop"))
            elif tool_name == "lane_change":
                direction = tool_args.get("direction", "left")
                step_result = await env.step(CarlaAction(action_type="lane_change", lane_direction=direction))
            else:
                step_result = await env.step(CarlaAction(action_type="observe"))

            reward = step_result.observation.rubric_reward or 0.0
            env_description = _describe(step_result.observation)

            if verbose:
                print(f"           Result: {env_description}")
                print(f"           Reward: {reward}")

            # Save image if requested
            if save_images:
                img_result = await env.step(CarlaAction(action_type="capture_image"))
                if img_result.observation.camera_image:
                    image_data = base64.b64decode(img_result.observation.camera_image)
                    image_file = output_path / f"turn_{turn+1:02d}_{tool_name}.jpg"
                    image_file.write_bytes(image_data)

            # Build messages exactly as the trainer does:
            # 1. Assistant message with tool_calls (formatted by chat template)
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": tool_name, "arguments": tool_args}}
                ],
            })
            # 2. Tool result message (role: "tool", not "user")
            messages.append({"role": "tool", "name": tool_name, "content": env_description})

            if step_result.done:
                if verbose:
                    print(f"\n   Episode done after {turn+1} turns.")
                break

        return reward, turn + 1


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned model on the CARLA trolley escape scenario",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sergiopaniego/Qwen3-0.6B-carla-trolley-escape",
        help="Hugging Face model ID to load locally with transformers",
    )
    parser.add_argument(
        "--base-url",
        default="https://sergiopaniego-carla-env.hf.space",
        help="CARLA environment base URL",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per episode (default: 10)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save camera images at each turn",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_images",
        help="Directory to save images (default: llm_images)",
    )
    args = parser.parse_args()

    # Load model once
    print(f"Loading model: {args.model}")
    llm = LocalClient(args.model)

    results = []
    for ep in range(args.episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'='*70}")

        reward, turns = asyncio.run(
            run_episode(
                llm,
                args.base_url,
                save_images=args.save_images,
                output_dir=args.output_dir,
                max_turns=args.max_turns,
            )
        )
        results.append((reward, turns))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for i, (reward, turns) in enumerate(results, 1):
        print(f"  Episode {i}: reward={reward:.2f}, turns={turns}")
    if len(results) > 1:
        avg_reward = sum(r for r, _ in results) / len(results)
        print(f"\n  Average reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
