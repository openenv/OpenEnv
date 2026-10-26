#!/usr/bin/env python3
"""
LLM Inference: Maze Navigation

Reproduces examples from sinatras blog post:
5. GPT-5.2 on maze (200 turns, 62.7m traveled - best result)
6. GPT-4.1-mini on maze (4.1m traveled - worst result)

Usage:
    python maze_navigation.py --model gpt-5.2
    python maze_navigation.py --model gpt-4.1-mini
    python maze_navigation.py --model gpt-5.2 --save-images  # Save camera images
    python maze_navigation.py --run-all-blog-examples
"""

import argparse
import asyncio
import math
import sys
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add repo src/ and envs/ to path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "envs"))

from carla_env import CarlaEnv, CarlaAction
from config import MODELS, MAZE_SCENARIOS, BLOG_EXAMPLES
from llm_clients import create_client, build_vision_message

@dataclass
class MazeResult:
    """Result of maze navigation episode."""
    model: str
    scenario: str
    distance_traveled: float
    goal_distance: float
    success: bool
    steps: int
    decisions: List[str]

def run_maze_episode(
    model_key: str,
    scenario_key: str = "maze-1",
    base_url: str = "http://localhost:8000",
    max_steps: int = 75,
    verbose: bool = True,
    save_images: bool = False,
    output_dir: str = "llm_images",
    image_interval: int = 5,
    vision: bool = False,
    vision_interval: int = 1,
    scenario_config_overrides: Optional[Dict] = None,
) -> MazeResult:
    """Run maze navigation episode with LLM decision-making.

    Args:
        model_key: Model identifier
        scenario_key: Scenario identifier
        base_url: Environment URL
        max_steps: Maximum navigation steps
        verbose: Print progress
        save_images: Save camera images during navigation
        output_dir: Directory to save images
        image_interval: Capture image every N steps (default: 5)
        vision: Send camera images to the LLM for visual reasoning
        vision_interval: Capture image for LLM every N steps (default: 1)
    """

    # Get configs
    model_config = MODELS[model_key]
    scenario_config = MAZE_SCENARIOS[scenario_key]

    if vision and not model_config.supports_vision:
        import warnings
        warnings.warn(
            f"Model '{model_key}' does not support vision. "
            "Images will not be sent to the LLM. Use a vision-capable model "
            "(e.g. claude-sonnet-4.5, gpt-5.2) or remove --vision.",
            stacklevel=2,
        )
        vision = False

    # Setup output directory if saving images
    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*70)
        print(f"Model: {model_config.name}")
        print(f"Scenario: {scenario_config.description}")
        print(f"Goal: Navigate to destination (~153m)")
        print(f"Max steps: {max_steps}")
        print("="*70)

    # Create LLM client
    llm = create_client(model_config.provider, model_config.model_id)

    async def _run():
        env = CarlaEnv(base_url=base_url)
        async with env:
            # Reset to maze scenario
            reset_kwargs = {"scenario_name": scenario_config.scenario_name}
            merged_overrides = {**(scenario_config.overrides or {}), **(scenario_config_overrides or {})}
            if merged_overrides:
                reset_kwargs["scenario_config"] = merged_overrides
            result = await env.reset(**reset_kwargs)
            obs = result.observation

            initial_distance = obs.goal_distance if getattr(obs, 'goal_distance', None) is not None else 153.0
            distance_traveled = 0.0
            prev_location = getattr(obs, 'location', None)
            decisions = []

            if verbose:
                print(f"\n📍 Starting Position:")
                print(f"{obs.scene_description[:200]}...\n")

            # Capture initial image if requested
            if save_images:
                result_img = await env.step(CarlaAction(action_type="capture_image"))
                if result_img.observation.camera_image:
                    image_data = base64.b64decode(result_img.observation.camera_image)
                    image_file = output_path / f"{model_key}_maze_step_000_start.jpg"
                    image_file.write_bytes(image_data)
                    if verbose:
                        print(f"📸 Saved initial image: {image_file.name}\n")

            # Define navigation tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_goal_info",
                        "description": "Get information about the goal location (distance and direction)",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "control_vehicle",
                        "description": "Control the vehicle with throttle and steering",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "throttle": {
                                    "type": "number",
                                    "description": "Throttle value (0.0 to 1.0)",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "steer": {
                                    "type": "number",
                                    "description": "Steering value (-1.0 left to 1.0 right)",
                                    "minimum": -1.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["throttle", "steer"]
                        }
                    }
                }
            ]

            # Track action history for context (keep last 10 actions)
            action_history = []

            # Multi-step navigation loop with stateless calls
            for step in range(max_steps):
                # Build fresh context for each call with history summary
                history_summary = "\n".join(action_history[-10:]) if action_history else "No actions yet."

                # Build current state description
                if getattr(obs, 'goal_distance', None) is not None:
                    goal_info = f"Goal: {obs.goal_distance:.1f}m {getattr(obs, 'goal_direction', 'unknown')}"
                else:
                    goal_info = "Goal: Unknown"

                current_state = f"Speed: {obs.speed_kmh:.1f} km/h, {goal_info}"

                # Create fresh prompt for this step
                prompt = f"""You are navigating a vehicle to a goal location.

Current state:
{current_state}

Recent actions (last 10):
{history_summary}

Available tools:
- get_goal_info(): Returns distance and direction to goal
- control_vehicle(throttle, steer): Move the car (throttle 0-1, steer -1 to 1 for left/right)

Instructions:
- Make ONE tool call to proceed with navigation
- Alternate between checking goal location and steering toward it
- Use small steering adjustments (-0.3 to 0.3 recommended)
- Maintain steady throttle (0.4-0.6 recommended)

Make your tool call now (no explanations)."""

                # Build message — include camera image if vision is enabled
                if vision and (step % vision_interval == 0):
                    img_result = await env.step(CarlaAction(action_type="capture_image"))
                    image_b64 = img_result.observation.camera_image
                    if image_b64:
                        messages = [build_vision_message(prompt, image_b64, model_config.provider)]
                    else:
                        messages = [{"role": "user", "content": prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]

                # Get LLM decision
                response = llm.chat(messages, tools, max_tokens=256)

                if not response["tool_calls"]:
                    if verbose:
                        print(f"Step {step+1}: No tool call from LLM")
                        if response.get("text"):
                            print(f"   Model said: {response['text'][:150]}")
                    break

                tool_call = response["tool_calls"][0]
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                decisions.append(tool_name)

                # Execute tool
                if tool_name == "get_goal_info":
                    action = CarlaAction(action_type="observe")
                    result = await env.step(action)
                    obs = result.observation
                    if getattr(obs, 'goal_distance', None) is not None:
                        direction = getattr(obs, 'goal_direction', 'unknown')
                        action_history.append(f"Step {step+1}: get_goal_info() → {obs.goal_distance:.1f}m {direction}")
                    else:
                        action_history.append(f"Step {step+1}: get_goal_info() → unavailable")

                elif tool_name == "control_vehicle":
                    throttle = tool_args.get("throttle", 0.5)
                    steer = tool_args.get("steer", 0.0)

                    action = CarlaAction(
                        action_type="control",
                        throttle=throttle,
                        steer=steer,
                        brake=0.0
                    )
                    result = await env.step(action)
                    obs = result.observation

                    # Update distance traveled from location coordinates
                    cur_location = getattr(obs, 'location', None)
                    if cur_location and prev_location:
                        dx = cur_location[0] - prev_location[0]
                        dy = cur_location[1] - prev_location[1]
                        distance_traveled += math.sqrt(dx * dx + dy * dy)
                        prev_location = cur_location

                    action_history.append(f"Step {step+1}: control(t={throttle:.2f}, s={steer:.2f}) → {obs.speed_kmh:.1f} km/h")

                else:
                    action_history.append(f"Step {step+1}: unknown tool")

                # Capture image periodically if requested
                if save_images and (step + 1) % image_interval == 0:
                    result_img = await env.step(CarlaAction(action_type="capture_image"))
                    if result_img.observation.camera_image:
                        image_data = base64.b64decode(result_img.observation.camera_image)
                        image_file = output_path / f"{model_key}_maze_step_{step+1:03d}.jpg"
                        image_file.write_bytes(image_data)
                        if verbose:
                            print(f"📸 Saved image at step {step+1}: {image_file.name}")

                if verbose and (step + 1) % 10 == 0:
                    print(f"Step {step+1}/{max_steps}: {distance_traveled:.1f}m traveled, "
                          f"{len([d for d in decisions if d == 'control_vehicle'])} controls, "
                          f"{len([d for d in decisions if d == 'get_goal_info'])} observations")

                # Check if done
                if result.done:
                    if verbose:
                        print(f"\n✓ Episode ended at step {step+1}")
                    break

            # Capture final image if requested
            if save_images:
                result_img = await env.step(CarlaAction(action_type="capture_image"))
                if result_img.observation.camera_image:
                    image_data = base64.b64decode(result_img.observation.camera_image)
                    image_file = output_path / f"{model_key}_maze_step_{step+1:03d}_final.jpg"
                    image_file.write_bytes(image_data)
                    if verbose:
                        print(f"📸 Saved final image: {image_file.name}")

            # Final stats
            final_distance = obs.goal_distance if getattr(obs, 'goal_distance', None) is not None else initial_distance
            success = final_distance < 10.0  # Within 10m of goal

            if verbose:
                print(f"\n📊 Final Results:")
                print(f"   Distance traveled: {distance_traveled:.1f}m")
                print(f"   Goal distance: {final_distance:.1f}m")
                print(f"   Success: {success}")
                print(f"   Total steps: {step+1}")
                print(f"   Control actions: {len([d for d in decisions if d == 'control_vehicle'])}")
                print(f"   Observations: {len([d for d in decisions if d == 'get_goal_info'])}\n")

            return MazeResult(
                model=model_config.name,
                scenario=scenario_config.description,
                distance_traveled=distance_traveled,
                goal_distance=final_distance,
                success=success,
                steps=step+1,
                decisions=decisions
            )

    try:
        return asyncio.run(_run())
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference on maze navigation scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific model
  python maze_navigation.py --model gpt-5.2

  # Run all blog examples
  python maze_navigation.py --run-all-blog-examples

  # Use HuggingFace Space
  python maze_navigation.py --model gpt-4.1-mini \\
    --base-url https://sergiopaniego-carla-env.hf.space
        """
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--run-all-blog-examples",
        action="store_true",
        help="Run all maze examples from sinatras blog post"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="CARLA environment base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=75,
        help="Maximum steps per episode (default: 75)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save camera images during navigation"
    )
    parser.add_argument(
        "--output-dir",
        default="llm_images",
        help="Directory to save images (default: llm_images)"
    )
    parser.add_argument(
        "--image-interval",
        type=int,
        default=5,
        help="Capture image every N steps (default: 5)"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Send camera images to the LLM for visual reasoning (requires vision-capable model)"
    )
    parser.add_argument(
        "--vision-interval",
        type=int,
        default=1,
        help="Send image to LLM every N steps when --vision is enabled (default: 1)"
    )
    parser.add_argument("--camera-width", type=int, default=None, help="Camera image width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=None, help="Camera image height (default: 360)")
    parser.add_argument("--camera-fov", type=int, default=None, help="Camera field of view (default: 90)")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG compression quality (default: 75)")

    args = parser.parse_args()

    # Build scenario_config overrides from CLI flags
    overrides = {}
    if args.camera_width is not None:
        overrides["camera_width"] = args.camera_width
    if args.camera_height is not None:
        overrides["camera_height"] = args.camera_height
    if args.camera_fov is not None:
        overrides["camera_fov"] = args.camera_fov
    if args.jpeg_quality is not None:
        overrides["jpeg_quality"] = args.jpeg_quality
    scenario_config_overrides = overrides or None

    if args.run_all_blog_examples:
        # Run all maze examples from blog
        maze_examples = [
            model
            for model, scenario in BLOG_EXAMPLES
            if scenario in MAZE_SCENARIOS
        ]

        print(f"\n🚀 Running {len(maze_examples)} maze navigation examples from blog...\n")

        results = []
        for i, model_key in enumerate(maze_examples, 1):
            print(f"[{i}/{len(maze_examples)}]")
            try:
                result = run_maze_episode(
                    model_key, "maze-1", args.base_url, args.max_steps,
                    save_images=args.save_images,
                    output_dir=args.output_dir,
                    image_interval=args.image_interval,
                    vision=args.vision,
                    vision_interval=args.vision_interval,
                    scenario_config_overrides=scenario_config_overrides,
                )
                results.append(result)
            except Exception as e:
                print(f"Failed: {e}")
                results.append(None)

            if i < len(maze_examples):
                print("\n" + "-"*70 + "\n")

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY: Maze Navigation Examples from Blog")
        print("="*70)
        for i, r in enumerate(results, 1):
            if r:
                print(f"\n{i}. {r.model}:")
                print(f"   Distance traveled: {r.distance_traveled:.1f}m ({r.distance_traveled/153*100:.1f}% of goal)")
                print(f"   Success: {r.success}")
                print(f"   Steps: {r.steps}")
            else:
                print(f"\n{i}. Failed")

    elif args.model:
        # Run single example
        result = run_maze_episode(
            args.model, "maze-1", args.base_url, args.max_steps,
            save_images=args.save_images,
            output_dir=args.output_dir,
            image_interval=args.image_interval,
            vision=args.vision,
            vision_interval=args.vision_interval,
            scenario_config_overrides=scenario_config_overrides,
        )

        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        print(f"Model: {result.model}")
        print(f"Distance traveled: {result.distance_traveled:.1f}m")
        print(f"Goal distance: {result.goal_distance:.1f}m" if result.goal_distance is not None else "Goal distance: Unknown")
        print(f"Success: {result.success}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
