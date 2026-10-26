#!/usr/bin/env python3
"""
Rubric Reward Demo (no LLM required).

Demonstrates how `rubric_reward` works alongside raw `reward` during
autopilot navigation.  Uses CARLA's built-in BehaviorAgent to drive
toward the goal while printing both reward signals each step.

Usage:
    # Free-roam with rubric tracking
    python rubric_autopilot_example.py --scenario free-roam-default

    # Maze scenario, 50 steps max
    python rubric_autopilot_example.py --scenario maze-1 --max-steps 50

    # Custom server
    python rubric_autopilot_example.py --scenario free-roam-default \
        --base-url https://sergiopaniego-carla-env.hf.space
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo src/ and envs/ to path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "envs"))

from carla_env import CarlaEnv, CarlaAction
from config import MAZE_SCENARIOS, FREE_ROAM_SCENARIOS

ALL_SCENARIOS = {**MAZE_SCENARIOS, **FREE_ROAM_SCENARIOS}


def run_rubric_demo(
    scenario_key: str,
    base_url: str = "http://localhost:8000",
    behavior: str = "normal",
    route_steps: int = 5,
    max_steps: int = 100,
):
    """Run autopilot navigation while tracking rubric vs raw rewards."""
    scenario_config = ALL_SCENARIOS[scenario_key]

    print("=" * 70)
    print("Rubric Reward Demo")
    print(f"Scenario: {scenario_config.description}")
    print(f"Max steps: {max_steps}, route_steps/tick: {route_steps}")
    print("=" * 70)

    async def _run():
        env = CarlaEnv(base_url=base_url)
        async with env:
            # 1. Reset
            reset_kwargs = {"scenario_name": scenario_config.scenario_name}
            if scenario_config.overrides:
                reset_kwargs["scenario_config"] = scenario_config.overrides
            result = await env.reset(**reset_kwargs)
            obs = result.observation

            print(f"\nStarting: {obs.scene_description[:200]}...\n")

            # 2. Initialize navigation agent
            await env.step(CarlaAction(
                action_type="init_navigation_agent",
                navigation_behavior=behavior,
            ))
            print(f"Navigation agent initialized (behavior={behavior})")

            # 3. Set destination
            goal_loc = getattr(obs, "goal_location", None)
            if goal_loc:
                await env.step(CarlaAction(
                    action_type="set_destination",
                    destination_x=goal_loc[0],
                    destination_y=goal_loc[1],
                    destination_z=goal_loc[2] if len(goal_loc) > 2 else 0.0,
                ))
                print(f"Destination set: {goal_loc}")

            # 4. Follow route loop — track both reward signals
            total_raw = 0.0
            total_rubric = 0.0
            step = 0

            print(f"\n{'Step':>5}  {'Raw':>8}  {'Rubric':>8}  {'Speed':>8}  {'Goal Dist':>10}")
            print("-" * 50)

            for step in range(max_steps):
                result = await env.step(CarlaAction(
                    action_type="follow_route",
                    route_steps=route_steps,
                ))
                obs = result.observation

                raw_r = obs.reward
                rubric_r = obs.rubric_reward
                total_raw += raw_r
                total_rubric += rubric_r

                goal_d = f"{obs.goal_distance:.1f}" if obs.goal_distance else "?"
                print(f"{step+1:>5}  {raw_r:>8.3f}  {rubric_r:>8.3f}  "
                      f"{obs.speed_kmh:>7.1f}  {goal_d:>10}")

                if result.done:
                    print(f"\nEpisode ended at step {step + 1}: {obs.done_reason}")
                    break

            # 5. Summary
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)
            print(f"  Steps:              {step + 1}")
            print(f"  Total raw reward:   {total_raw:.3f}")
            print(f"  Total rubric reward:{total_rubric:.3f}")
            print(f"  Goal reached:       {obs.done_reason == 'goal_reached' if obs.done_reason else False}")
            print(f"  Final goal dist:    {obs.goal_distance or '?'}")

    try:
        asyncio.run(_run())
    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Rubric reward demo — compare raw vs rubric rewards during autopilot navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rubric_autopilot_example.py --scenario free-roam-default
  python rubric_autopilot_example.py --scenario maze-1 --max-steps 50
  python rubric_autopilot_example.py --scenario free-roam-default \\
    --base-url https://sergiopaniego-carla-env.hf.space
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()),
        default="free-roam-default",
        help="Scenario to run (default: free-roam-default)",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps (default: 100)",
    )

    args = parser.parse_args()

    run_rubric_demo(
        scenario_key=args.scenario,
        base_url=args.base_url,
        behavior=args.behavior,
        route_steps=args.route_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
