#!/usr/bin/env python3
"""
Autopilot Navigation (no LLM required).

Uses CARLA's built-in navigation agents (BasicAgent / BehaviorAgent) to
drive to the goal autonomously.  Useful as a baseline to compare against
LLM-driven navigation.

Usage:
    # Maze with default (normal) behavior
    python autopilot_navigation.py --scenario maze-1

    # Free-roam with cautious driving
    python autopilot_navigation.py --scenario free-roam-default --behavior cautious

    # Aggressive driving, save images
    python autopilot_navigation.py --scenario free-roam-traffic \
        --behavior aggressive --save-images

    # Custom server
    python autopilot_navigation.py --scenario maze-1 \
        --base-url https://sergiopaniego-carla-env.hf.space
"""

import argparse
import asyncio
import base64
import math
import sys
from pathlib import Path
from typing import Dict, Optional

# Add repo src/ and envs/ to path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "envs"))

from carla_env import CarlaEnv, CarlaAction
from config import MAZE_SCENARIOS, FREE_ROAM_SCENARIOS

ALL_SCENARIOS = {**MAZE_SCENARIOS, **FREE_ROAM_SCENARIOS}


def run_autopilot(
    scenario_key: str,
    base_url: str = "http://localhost:8000",
    behavior: str = "normal",
    route_steps: int = 5,
    max_steps: int = 200,
    verbose: bool = True,
    save_images: bool = False,
    output_dir: str = "llm_images",
    image_interval: int = 10,
    scenario_config_overrides: Optional[Dict] = None,
):
    """Run a scenario using CARLA's built-in navigation agent.

    Args:
        scenario_key: Scenario identifier from MAZE_SCENARIOS or FREE_ROAM_SCENARIOS.
        base_url: Environment server URL.
        behavior: Navigation behavior: "cautious", "normal", or "aggressive".
        route_steps: Simulation ticks per follow_route action.
        max_steps: Maximum high-level steps before stopping.
        verbose: Print progress.
        save_images: Save camera images to disk.
        output_dir: Directory for saved images.
        image_interval: Save an image every N steps.
    """
    scenario_config = ALL_SCENARIOS[scenario_key]

    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print(f"Autopilot Navigation ({behavior})")
        print(f"Scenario: {scenario_config.description}")
        print(f"Max steps: {max_steps}, route_steps/tick: {route_steps}")
        print("=" * 70)

    async def _run():
        env = CarlaEnv(base_url=base_url)
        async with env:
            # 1. Reset
            reset_kwargs = {"scenario_name": scenario_config.scenario_name}
            merged_overrides = {**(scenario_config.overrides or {}), **(scenario_config_overrides or {})}
            if merged_overrides:
                reset_kwargs["scenario_config"] = merged_overrides
            result = await env.reset(**reset_kwargs)
            obs = result.observation
            prev_location = getattr(obs, "location", None)
            distance_traveled = 0.0

            if verbose:
                print(f"\nStarting: {obs.scene_description[:200]}...\n")

            # 2. Initialize navigation agent
            await env.step(CarlaAction(
                action_type="init_navigation_agent",
                navigation_behavior=behavior,
            ))
            if verbose:
                print(f"Navigation agent initialized (behavior={behavior})")

            # 3. Set destination (use goal from observation if available)
            goal_loc = getattr(obs, "goal_location", None)
            if goal_loc:
                await env.step(CarlaAction(
                    action_type="set_destination",
                    destination_x=goal_loc[0],
                    destination_y=goal_loc[1],
                    destination_z=goal_loc[2] if len(goal_loc) > 2 else 0.0,
                ))
                if verbose:
                    print(f"Destination set: {goal_loc}")
            else:
                if verbose:
                    print("No explicit goal_location on observation — "
                          "agent will use scenario's built-in goal if available.")

            # 4. Follow route loop
            step = 0
            for step in range(max_steps):
                result = await env.step(CarlaAction(
                    action_type="follow_route",
                    route_steps=route_steps,
                ))
                obs = result.observation

                # Track distance
                cur_location = getattr(obs, "location", None)
                if cur_location and prev_location:
                    dx = cur_location[0] - prev_location[0]
                    dy = cur_location[1] - prev_location[1]
                    distance_traveled += math.sqrt(dx * dx + dy * dy)
                    prev_location = cur_location

                # Save image
                if save_images and (step + 1) % image_interval == 0:
                    img_result = await env.step(CarlaAction(action_type="capture_image"))
                    if img_result.observation.camera_image:
                        image_data = base64.b64decode(img_result.observation.camera_image)
                        fname = f"autopilot_{scenario_key}_step_{step+1:03d}.jpg"
                        (output_path / fname).write_bytes(image_data)
                        if verbose:
                            print(f"  Saved {fname}")

                # Progress
                if verbose and (step + 1) % 20 == 0:
                    goal_d = obs.goal_distance if obs.goal_distance else "?"
                    print(f"Step {step+1}/{max_steps}: "
                          f"{distance_traveled:.1f}m traveled, "
                          f"speed {obs.speed_kmh:.1f} km/h, "
                          f"goal {goal_d}")

                if result.done:
                    if verbose:
                        print(f"\nEpisode ended at step {step + 1}")
                    break

            # Final stats
            goal_d = obs.goal_distance if obs.goal_distance else "unknown"
            if verbose:
                print(f"\nResults:")
                print(f"  Distance traveled: {distance_traveled:.1f}m")
                print(f"  Goal distance: {goal_d}")
                print(f"  Steps: {step + 1}")
                print(f"  Speed: {obs.speed_kmh:.1f} km/h")

    try:
        return asyncio.run(_run())
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run CARLA autopilot navigation (no LLM needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autopilot_navigation.py --scenario maze-1
  python autopilot_navigation.py --scenario free-roam-default --behavior cautious
  python autopilot_navigation.py --scenario maze-1 \\
    --base-url https://sergiopaniego-carla-env.hf.space
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()),
        required=True,
        help="Scenario to run",
    )
    parser.add_argument(
        "--behavior",
        choices=["cautious", "normal", "aggressive"],
        default="normal",
        help="Navigation agent behavior (default: normal)",
    )
    parser.add_argument(
        "--route-steps",
        type=int,
        default=5,
        help="Simulation ticks per follow_route call (default: 5)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="CARLA environment base URL",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps")
    parser.add_argument("--save-images", action="store_true", help="Save camera images")
    parser.add_argument("--output-dir", default="llm_images", help="Image output directory")
    parser.add_argument("--image-interval", type=int, default=10, help="Save image every N steps")
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

    run_autopilot(
        scenario_key=args.scenario,
        base_url=args.base_url,
        behavior=args.behavior,
        route_steps=args.route_steps,
        max_steps=args.max_steps,
        save_images=args.save_images,
        output_dir=args.output_dir,
        image_interval=args.image_interval,
        scenario_config_overrides=scenario_config_overrides,
    )


if __name__ == "__main__":
    main()
