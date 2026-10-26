#!/usr/bin/env python3
"""
LLM Inference: Free-Roam Autonomous Driving

Navigate to a random goal in open traffic with configurable NPC vehicles
and pedestrians. Supports map selection and continuous reward.

Usage:
    python free_roam_navigation.py --model gpt-5.2
    python free_roam_navigation.py --model claude-sonnet-4.5 --scenario free-roam-traffic
    python free_roam_navigation.py --model gpt-5.2 --save-images
    python free_roam_navigation.py --run-all
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
from config import MODELS, FREE_ROAM_SCENARIOS
from llm_clients import create_client, build_vision_message


@dataclass
class FreeRoamResult:
    """Result of free-roam navigation episode."""
    model: str
    scenario: str
    distance_traveled: float
    goal_distance: float
    success: bool
    steps: int
    total_reward: float
    collisions: int
    decisions: List[str]


def run_free_roam_episode(
    model_key: str,
    scenario_key: str = "free-roam-default",
    base_url: str = "http://localhost:8000",
    max_steps: int = 200,
    verbose: bool = True,
    save_images: bool = False,
    output_dir: str = "llm_images",
    image_interval: int = 10,
    scenario_config_overrides: Optional[Dict] = None,
    vision: bool = False,
    vision_interval: int = 1,
) -> FreeRoamResult:
    """Run free-roam navigation episode with LLM decision-making.

    Args:
        model_key: Model identifier
        scenario_key: Scenario identifier from FREE_ROAM_SCENARIOS
        base_url: Environment URL
        max_steps: Maximum navigation steps
        verbose: Print progress
        save_images: Save camera images during navigation
        output_dir: Directory to save images
        image_interval: Capture image every N steps
        vision: Send camera images to the LLM for visual reasoning
        vision_interval: Send image to LLM every N steps (default: 1)
    """
    model_config = MODELS[model_key]
    scenario_config = FREE_ROAM_SCENARIOS[scenario_key]

    if vision and not model_config.supports_vision:
        import warnings
        warnings.warn(
            f"Model '{model_key}' does not support vision. "
            "Images will not be sent to the LLM. Use a vision-capable model "
            "(e.g. claude-sonnet-4.5, gpt-5.2) or remove --vision.",
            stacklevel=2,
        )
        vision = False

    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print(f"Model: {model_config.name}")
        print(f"Scenario: {scenario_config.description}")
        print(f"Max steps: {max_steps}")
        print("=" * 70)

    llm = create_client(model_config.provider, model_config.model_id)

    async def _run():
        env = CarlaEnv(base_url=base_url)
        async with env:
            reset_kwargs = {"scenario_name": scenario_config.scenario_name}
            # Merge config-defined overrides with CLI overrides (CLI wins)
            merged_overrides = {**(scenario_config.overrides or {}), **(scenario_config_overrides or {})}
            if merged_overrides:
                reset_kwargs["scenario_config"] = merged_overrides
            result = await env.reset(**reset_kwargs)
            obs = result.observation

            initial_distance = obs.goal_distance if hasattr(obs, "goal_distance") and obs.goal_distance else 200.0
            distance_traveled = 0.0
            prev_location = getattr(obs, 'location', None)
            total_reward = 0.0
            collision_count = 0
            decisions = []

            if verbose:
                goal_dist = f"{obs.goal_distance:.1f}m" if obs.goal_distance else "unknown"
                print(f"\nStarting - Goal distance: {goal_dist}")
                print(f"{obs.scene_description[:200]}...\n")

            # Capture initial image
            if save_images:
                result_img = await env.step(CarlaAction(action_type="capture_image"))
                if result_img.observation.camera_image:
                    image_data = base64.b64decode(result_img.observation.camera_image)
                    image_file = output_path / f"{model_key}_freeroam_step_000_start.jpg"
                    image_file.write_bytes(image_data)
                    if verbose:
                        print(f"Saved initial image: {image_file.name}\n")

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_goal_info",
                        "description": "Get distance and direction to the goal",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
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
                                    "description": "Throttle (0.0 to 1.0)",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "steer": {
                                    "type": "number",
                                    "description": "Steering (-1.0 left to 1.0 right)",
                                    "minimum": -1.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["throttle", "steer"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "emergency_stop",
                        "description": "Emergency brake - use to avoid collisions",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            ]

            action_history = []

            for step in range(max_steps):
                history_summary = "\n".join(action_history[-10:]) if action_history else "No actions yet."

                goal_info = "Goal: Unknown"
                if hasattr(obs, "goal_distance") and obs.goal_distance:
                    direction = getattr(obs, "goal_direction", "unknown")
                    goal_info = f"Goal: {obs.goal_distance:.1f}m {direction}"

                collision_info = ""
                if obs.collision_detected:
                    collision_info = "\nWARNING: Collision detected! Adjust your driving."

                nearby_info = ""
                if obs.nearby_actors:
                    actors = [f"  - {a['type']} {a['distance']:.0f}m {a['position']}" for a in obs.nearby_actors[:5]]
                    nearby_info = "\nNearby actors:\n" + "\n".join(actors)

                prompt = f"""You are driving a vehicle in an open city environment with traffic.

Current state:
Speed: {obs.speed_kmh:.1f} km/h, {goal_info}{collision_info}{nearby_info}

Recent actions (last 10):
{history_summary}

Available tools:
- get_goal_info(): Get distance and direction to goal
- control_vehicle(throttle, steer): Drive (throttle 0-1, steer -1 to 1)
- emergency_stop(): Emergency brake to avoid collisions

Instructions:
- Navigate to the goal while avoiding collisions with traffic
- Watch for nearby actors and adjust speed/steering accordingly
- Use emergency_stop() if collision is imminent
- Maintain moderate speed (30-50 km/h) in traffic areas

Make ONE tool call now."""

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

                response = llm.chat(messages, tools, max_tokens=256)

                if not response["tool_calls"]:
                    if verbose:
                        print(f"Step {step + 1}: No tool call from LLM")
                    break

                tool_call = response["tool_calls"][0]
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                decisions.append(tool_name)

                if tool_name == "get_goal_info":
                    action = CarlaAction(action_type="observe")
                    result = await env.step(action)
                    obs = result.observation
                    if hasattr(obs, "goal_distance") and obs.goal_distance:
                        direction = getattr(obs, "goal_direction", "unknown")
                        action_history.append(
                            f"Step {step + 1}: get_goal_info() -> {obs.goal_distance:.1f}m {direction}"
                        )
                    else:
                        action_history.append(f"Step {step + 1}: get_goal_info() -> unavailable")

                elif tool_name == "control_vehicle":
                    throttle = tool_args.get("throttle", 0.5)
                    steer = tool_args.get("steer", 0.0)
                    action = CarlaAction(
                        action_type="control",
                        throttle=throttle,
                        steer=steer,
                        brake=0.0,
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

                    action_history.append(
                        f"Step {step + 1}: control(t={throttle:.2f}, s={steer:.2f}) -> {obs.speed_kmh:.1f} km/h"
                    )

                elif tool_name == "emergency_stop":
                    action = CarlaAction(action_type="emergency_stop")
                    result = await env.step(action)
                    obs = result.observation
                    action_history.append(
                        f"Step {step + 1}: emergency_stop() -> {obs.speed_kmh:.1f} km/h"
                    )

                else:
                    action_history.append(f"Step {step + 1}: unknown tool")

                # Track reward and collisions
                if result.observation.reward:
                    total_reward += result.observation.reward
                if obs.collision_detected:
                    collision_count += 1

                # Save image periodically
                if save_images and (step + 1) % image_interval == 0:
                    result_img = await env.step(CarlaAction(action_type="capture_image"))
                    if result_img.observation.camera_image:
                        image_data = base64.b64decode(result_img.observation.camera_image)
                        image_file = output_path / f"{model_key}_freeroam_step_{step + 1:03d}.jpg"
                        image_file.write_bytes(image_data)
                        if verbose:
                            print(f"Saved image at step {step + 1}: {image_file.name}")

                if verbose and (step + 1) % 20 == 0:
                    goal_d = obs.goal_distance if hasattr(obs, "goal_distance") and obs.goal_distance else float("inf")
                    print(
                        f"Step {step + 1}/{max_steps}: {distance_traveled:.1f}m traveled, "
                        f"goal: {goal_d:.1f}m, reward: {total_reward:.2f}, collisions: {collision_count}"
                    )

                if result.done:
                    if verbose:
                        print(f"\nEpisode ended at step {step + 1}")
                    break

            # Final image
            if save_images:
                result_img = await env.step(CarlaAction(action_type="capture_image"))
                if result_img.observation.camera_image:
                    image_data = base64.b64decode(result_img.observation.camera_image)
                    image_file = output_path / f"{model_key}_freeroam_step_{step + 1:03d}_final.jpg"
                    image_file.write_bytes(image_data)

            final_distance = obs.goal_distance if hasattr(obs, "goal_distance") and obs.goal_distance else initial_distance
            success = final_distance < 10.0

            if verbose:
                print(f"\nFinal Results:")
                print(f"   Distance traveled: {distance_traveled:.1f}m")
                print(f"   Goal distance: {final_distance:.1f}m")
                print(f"   Success: {success}")
                print(f"   Total steps: {step + 1}")
                print(f"   Total reward: {total_reward:.2f}")
                print(f"   Collisions: {collision_count}")
                print(f"   Control actions: {len([d for d in decisions if d == 'control_vehicle'])}")
                print(f"   Observations: {len([d for d in decisions if d == 'get_goal_info'])}")
                print(f"   Emergency stops: {len([d for d in decisions if d == 'emergency_stop'])}\n")

            return FreeRoamResult(
                model=model_config.name,
                scenario=scenario_config.description,
                distance_traveled=distance_traveled,
                goal_distance=final_distance,
                success=success,
                steps=step + 1,
                total_reward=total_reward,
                collisions=collision_count,
                decisions=decisions,
            )

    try:
        return asyncio.run(_run())
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference on free-roam driving scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default free-roam (current map, no traffic)
  python free_roam_navigation.py --model gpt-5.2

  # Specific map with traffic
  python free_roam_navigation.py --model claude-sonnet-4.5 --scenario free-roam-traffic

  # Save camera images
  python free_roam_navigation.py --model gpt-5.2 --save-images

  # Use HuggingFace Space
  python free_roam_navigation.py --model gpt-5.2 \\
    --base-url https://sergiopaniego-carla-env.hf.space
        """,
    )
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument(
        "--scenario",
        choices=list(FREE_ROAM_SCENARIOS.keys()),
        default="free-roam-default",
        help="Scenario variant (default: free-roam-default)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all free-roam scenario variants with the specified model",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="CARLA environment base URL",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--save-images", action="store_true", help="Save camera images")
    parser.add_argument("--output-dir", default="llm_images", help="Image output directory")
    parser.add_argument("--image-interval", type=int, default=10, help="Capture image every N steps")
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
    parser.add_argument("--weather", type=str, default=None, help="Weather preset (e.g. HardRainNoon)")
    parser.add_argument("--max-steps-scenario", type=int, default=None, help="Override scenario max_steps")
    parser.add_argument("--route-min", type=float, default=None, help="Min random goal distance (m)")
    parser.add_argument("--route-max", type=float, default=None, help="Max random goal distance (m)")
    parser.add_argument("--camera-width", type=int, default=None, help="Camera image width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=None, help="Camera image height (default: 360)")
    parser.add_argument("--camera-fov", type=int, default=None, help="Camera field of view (default: 90)")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG compression quality (default: 75)")

    args = parser.parse_args()

    # Build scenario_config overrides from CLI flags
    overrides = {}
    if args.weather:
        overrides["weather"] = args.weather
    if args.max_steps_scenario is not None:
        overrides["max_steps"] = args.max_steps_scenario
    if args.route_min is not None:
        overrides["route_distance_min"] = args.route_min
    if args.route_max is not None:
        overrides["route_distance_max"] = args.route_max
    if args.camera_width is not None:
        overrides["camera_width"] = args.camera_width
    if args.camera_height is not None:
        overrides["camera_height"] = args.camera_height
    if args.camera_fov is not None:
        overrides["camera_fov"] = args.camera_fov
    if args.jpeg_quality is not None:
        overrides["jpeg_quality"] = args.jpeg_quality
    scenario_config_overrides = overrides or None

    if args.run_all:
        if not args.model:
            print("Error: --model required with --run-all")
            return

        print(f"\nRunning {len(FREE_ROAM_SCENARIOS)} free-roam variants with {args.model}...\n")

        results = []
        for i, scenario_key in enumerate(FREE_ROAM_SCENARIOS, 1):
            print(f"[{i}/{len(FREE_ROAM_SCENARIOS)}]")
            try:
                result = run_free_roam_episode(
                    args.model,
                    scenario_key,
                    args.base_url,
                    args.max_steps,
                    save_images=args.save_images,
                    output_dir=args.output_dir,
                    image_interval=args.image_interval,
                    scenario_config_overrides=scenario_config_overrides,
                    vision=args.vision,
                    vision_interval=args.vision_interval,
                )
                results.append((scenario_key, result))
            except Exception as e:
                print(f"Failed: {e}")
                results.append((scenario_key, None))

            if i < len(FREE_ROAM_SCENARIOS):
                print("\n" + "-" * 70 + "\n")

        print("\n" + "=" * 70)
        print("SUMMARY: Free-Roam Navigation")
        print("=" * 70)
        for scenario_key, r in results:
            if r:
                print(f"\n  {scenario_key}:")
                print(f"    Distance: {r.distance_traveled:.1f}m, Goal: {r.goal_distance:.1f}m")
                print(f"    Reward: {r.total_reward:.2f}, Collisions: {r.collisions}, Success: {r.success}")
            else:
                print(f"\n  {scenario_key}: Failed")

    elif args.model:
        run_free_roam_episode(
            args.model,
            args.scenario,
            args.base_url,
            args.max_steps,
            save_images=args.save_images,
            output_dir=args.output_dir,
            image_interval=args.image_interval,
            scenario_config_overrides=scenario_config_overrides,
            vision=args.vision,
            vision_interval=args.vision_interval,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
