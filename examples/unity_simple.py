#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""
Unity ML-Agents Environment Example Usage

This script demonstrates how to use the Unity ML-Agents environment
through the OpenEnv interface, with support for direct mode, server mode,
and Docker-based deployment.

=============================================================================
USAGE EXAMPLES (run from the OpenEnv repository root)
=============================================================================

1. DIRECT MODE (Recommended for quick testing - no server required)
   ----------------------------------------------------------------
   Runs the Unity environment directly in-process.
   This is the simplest way to get started.

    # Run with graphics (default: 1280x720 window)
    python examples/unity_simple.py --direct

    # Run with custom window size
    python examples/unity_simple.py --direct --width 1920 --height 1080

    # Run headless (no graphics, faster for training)
    python examples/unity_simple.py --direct --no-graphics --time-scale 20

    # Run 3DBall environment for 5 episodes
    python examples/unity_simple.py --direct --env 3DBall --episodes 5

    # Run alternating between PushBlock and 3DBall
    python examples/unity_simple.py --direct --env both --episodes 6


2. SERVER MODE (For client-server architecture)
   ---------------------------------------------
   First, start the server in one terminal, then connect with this script.

   Step 1: Start the server (in Terminal 1):
    cd envs/unity_env
    uvicorn server.app:app --host 0.0.0.0 --port 8000

   Or with environment variables for custom settings:
    UNITY_WIDTH=1920 UNITY_HEIGHT=1080 uvicorn server.app:app --port 8000
    UNITY_NO_GRAPHICS=1 UNITY_TIME_SCALE=20 uvicorn server.app:app --port 8000

   Step 2: Run this script (in Terminal 2, from repo root):
    python examples/unity_simple.py --url http://localhost:8000
    python examples/unity_simple.py --url http://localhost:8000 --env 3DBall --episodes 5


3. DOCKER MODE (For containerized deployment)
   -------------------------------------------
   Automatically starts a Docker container and connects to it.

   First, build the Docker image:
    cd envs/unity_env
    docker build -f server/Dockerfile -t unity-env:latest .

   Then run (from repo root):
    python examples/unity_simple.py --docker
    python examples/unity_simple.py --docker --width 1280 --height 720
    python examples/unity_simple.py --docker --no-graphics --time-scale 20
    python examples/unity_simple.py --docker --env 3DBall --episodes 10

=============================================================================

The first run will download Unity environment binaries (~500MB).
Subsequent runs use cached binaries from ~/.mlagents-cache/
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.unity_env.client import UnityEnv
from envs.unity_env.models import UnityAction


def run_pushblock_episode(
    client: UnityEnv,
    max_steps: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Run a single episode of PushBlock with random actions.

    Args:
        client: Connected UnityEnv client
        max_steps: Maximum steps per episode
        verbose: Print progress information

    Returns:
        Dictionary with episode statistics
    """
    # Reset to PushBlock environment
    result = client.reset(env_id="PushBlock")

    if verbose:
        print(f"Environment: PushBlock")
        print(f"Behavior: {result.observation.behavior_name}")
        print(f"Vector obs dims: {len(result.observation.vector_observations)}")
        action_spec = result.observation.action_spec_info
        print(f"Action spec: {action_spec}")
        print()

    episode_reward = 0.0
    step_count = 0

    while not result.done and step_count < max_steps:
        # PushBlock has 7 discrete actions:
        # 0=noop, 1=forward, 2=backward, 3=rotate_left,
        # 4=rotate_right, 5=strafe_left, 6=strafe_right
        action_idx = random.randint(0, 6)
        action = UnityAction(discrete_actions=[action_idx])

        result = client.step(action)
        episode_reward += result.reward or 0.0
        step_count += 1

        if verbose and step_count % 100 == 0:
            print(f"  Step {step_count}: cumulative reward = {episode_reward:.2f}")

    return {
        "steps": step_count,
        "reward": episode_reward,
        "done": result.done,
    }


def run_3dball_episode(
    client: UnityEnv,
    max_steps: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Run a single episode of 3DBall with random actions.

    Args:
        client: Connected UnityEnv client
        max_steps: Maximum steps per episode
        verbose: Print progress information

    Returns:
        Dictionary with episode statistics
    """
    # Reset to 3DBall environment
    result = client.reset(env_id="3DBall")

    if verbose:
        print(f"Environment: 3DBall")
        print(f"Behavior: {result.observation.behavior_name}")
        print(f"Vector obs dims: {len(result.observation.vector_observations)}")
        action_spec = result.observation.action_spec_info
        print(f"Action spec: {action_spec}")
        print()

    episode_reward = 0.0
    step_count = 0

    while not result.done and step_count < max_steps:
        # 3DBall has 2 continuous actions for X and Z rotation
        action = UnityAction(
            continuous_actions=[
                random.uniform(-1, 1),  # X rotation
                random.uniform(-1, 1),  # Z rotation
            ]
        )

        result = client.step(action)
        episode_reward += result.reward or 0.0
        step_count += 1

        if verbose and step_count % 100 == 0:
            print(f"  Step {step_count}: cumulative reward = {episode_reward:.2f}")

    return {
        "steps": step_count,
        "reward": episode_reward,
        "done": result.done,
    }


def run_episodes(
    client: UnityEnv,
    env_name: str,
    episodes: int,
    max_steps: int,
    verbose: bool,
) -> list:
    """Run multiple episodes and collect results."""
    all_results = []

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

        if env_name == "PushBlock":
            result = run_pushblock_episode(
                client,
                max_steps=max_steps,
                verbose=verbose,
            )
        elif env_name == "3DBall":
            result = run_3dball_episode(
                client,
                max_steps=max_steps,
                verbose=verbose,
            )
        else:  # both
            if episode % 2 == 0:
                result = run_pushblock_episode(
                    client,
                    max_steps=max_steps,
                    verbose=verbose,
                )
            else:
                result = run_3dball_episode(
                    client,
                    max_steps=max_steps,
                    verbose=verbose,
                )

        all_results.append(result)
        print(
            f"Episode {episode + 1}: {result['steps']} steps, "
            f"reward: {result['reward']:.2f}"
        )

    return all_results


def print_summary(all_results: list) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_steps = sum(r["steps"] for r in all_results)
    avg_reward = sum(r["reward"] for r in all_results) / len(all_results)
    max_reward = max(r["reward"] for r in all_results)
    min_reward = min(r["reward"] for r in all_results)
    print(f"Total episodes: {len(all_results)}")
    print(f"Total steps: {total_steps}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print("=" * 60)


def run_with_server(args) -> None:
    """Run using a connection to an existing server."""
    print("=" * 60)
    print("Unity ML-Agents Environment - Server Mode")
    print("=" * 60)
    print(f"\nConnecting to: {args.url}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print()

    # Connect to the environment server
    with UnityEnv(base_url=args.url) as client:
        all_results = run_episodes(
            client,
            env_name=args.env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=not args.quiet,
        )
        print_summary(all_results)


def run_with_docker(args) -> None:
    """Run using Docker (automatically starts container)."""
    print("=" * 60)
    print("Unity ML-Agents Environment - Docker Mode")
    print("=" * 60)
    print(f"\nDocker image: {args.docker_image}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Window size: {args.width}x{args.height}")
    print(f"Graphics: {'Disabled (headless)' if args.no_graphics else 'Enabled'}")
    print()

    # Build environment variables for Docker
    env_vars = {
        "UNITY_NO_GRAPHICS": "1" if args.no_graphics else "0",
        "UNITY_WIDTH": str(args.width),
        "UNITY_HEIGHT": str(args.height),
        "UNITY_TIME_SCALE": str(args.time_scale),
        "UNITY_QUALITY_LEVEL": str(args.quality_level),
    }

    print("Starting Docker container...")
    print(f"  Environment variables: {env_vars}")
    print()

    try:
        # Use from_docker_image to automatically start and connect
        client = UnityEnv.from_docker_image(
            args.docker_image,
            environment=env_vars,
        )

        try:
            all_results = run_episodes(
                client,
                env_name=args.env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                verbose=not args.quiet,
            )
            print_summary(all_results)
        finally:
            print("\nClosing Docker container...")
            client.close()

    except Exception as e:
        print(f"\nError running with Docker: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Docker is running")
        print("  2. Build the image first:")
        print(f"     docker build -f server/Dockerfile -t {args.docker_image} .")
        print("  3. Or use server mode instead:")
        print("     python examples/unity_simple.py --url http://localhost:8000")
        sys.exit(1)


def run_direct(args) -> None:
    """
    Run Unity environment in direct mode (local server started automatically).

    This mode starts an embedded local server and connects to it, providing
    the convenience of direct execution while maintaining client-server separation.
    Useful for quick testing and debugging. For production, use server or Docker mode.
    """
    print("=" * 60)
    print("Unity ML-Agents Environment - Direct Mode")
    print("=" * 60)
    print(f"\nEnvironment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Window size: {args.width}x{args.height}")
    print(f"Graphics: {'Disabled (headless)' if args.no_graphics else 'Enabled'}")
    print(f"Time scale: {args.time_scale}x")
    print()

    print("Starting local Unity server...")
    print("(First run will download binaries - this may take a few minutes)")
    print()

    # Use from_direct() to start an embedded server and get a client
    client = UnityEnv.from_direct(
        env_id=args.env if args.env != "both" else "PushBlock",
        no_graphics=args.no_graphics,
        time_scale=args.time_scale,
        width=args.width,
        height=args.height,
        quality_level=args.quality_level,
    )

    try:
        all_results = []

        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

            # Determine which environment to use
            if args.env == "both":
                current_env = "PushBlock" if episode % 2 == 0 else "3DBall"
            else:
                current_env = args.env

            # Reset environment
            result = client.reset(env_id=current_env)

            if not args.quiet:
                print(f"Environment: {current_env}")
                print(f"Behavior: {result.observation.behavior_name}")
                print(f"Vector obs dims: {len(result.observation.vector_observations)}")
                print(f"Action spec: {result.observation.action_spec_info}")
                print()

            episode_reward = 0.0
            step_count = 0

            while not result.done and step_count < args.max_steps:
                # Generate action based on environment type
                if current_env == "3DBall":
                    action = UnityAction(
                        continuous_actions=[
                            random.uniform(-1, 1),
                            random.uniform(-1, 1),
                        ]
                    )
                else:
                    action = UnityAction(discrete_actions=[random.randint(0, 6)])

                result = client.step(action)
                episode_reward += result.reward or 0.0
                step_count += 1

                if not args.quiet and step_count % 100 == 0:
                    print(
                        f"  Step {step_count}: cumulative reward = {episode_reward:.2f}"
                    )

            episode_result = {
                "steps": step_count,
                "reward": episode_reward,
                "done": result.done,
            }
            all_results.append(episode_result)
            print(
                f"Episode {episode + 1}: {episode_result['steps']} steps, "
                f"reward: {episode_result['reward']:.2f}"
            )

        print_summary(all_results)

    finally:
        print("\nClosing Unity environment...")
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run Unity ML-Agents environment examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to running server (default)
  %(prog)s --url http://localhost:8000

  # Run via Docker
  %(prog)s --docker

  # Run directly without server (for testing)
  %(prog)s --direct

  # With graphics window (800x600 default)
  %(prog)s --direct --width 1280 --height 720

  # Headless mode (faster training)
  %(prog)s --direct --no-graphics --time-scale 20

  # Run 3DBall environment
  %(prog)s --direct --env 3DBall --episodes 5
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--docker",
        action="store_true",
        help="Run via Docker (automatically starts container)",
    )
    mode_group.add_argument(
        "--direct",
        action="store_true",
        help="Run Unity environment directly without server",
    )

    # Connection settings
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the Unity environment server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--docker-image",
        default="unity-env:latest",
        help="Docker image to use (default: unity-env:latest)",
    )

    # Environment settings
    parser.add_argument(
        "--env",
        choices=["PushBlock", "3DBall", "both"],
        default="PushBlock",
        help="Which environment to run (default: PushBlock)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )

    # Graphics settings
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width in pixels (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height in pixels (default: 600)",
    )
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Run in headless mode without graphics (faster training)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Simulation speed multiplier (default: 1.0, use 20.0 for fast training)",
    )
    parser.add_argument(
        "--quality-level",
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
        help="Graphics quality level 0-5 (default: 5)",
    )

    # Output settings
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Run in appropriate mode
    if args.docker:
        run_with_docker(args)
    elif args.direct:
        run_direct(args)
    else:
        run_with_server(args)


if __name__ == "__main__":
    main()
