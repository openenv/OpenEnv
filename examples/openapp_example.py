#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""
Example usage of the OpenApp Environment.

This script demonstrates how to use the OpenApp environment with OpenEnv.
It can be run in two modes:
1. With Docker: Uses the Docker image to run the environment
2. Local: Directly uses the OpenAppEnvironment class (requires OpenApps installed)

Usage:
    # Run with Docker (recommended)
    python examples/openapp_example.py --mode docker

    # Run locally without Docker
    python examples/openapp_example.py --mode local

    # Run with custom number of steps
    python examples/openapp_example.py --mode docker --num-steps 20

Visualization Options:
    # To SEE the browser window and watch agent interactions in real-time:
    #
    # Terminal 1: Start OpenApps server with visible browser
    cd OpenApps
    python OpenApps/launch.py browsergym_env_args.headless=False

    # Terminal 2: Run your agent code
    export OPENAPPS_URL=http://localhost:5001
    python examples/openapp_example.py --mode local

    # Or access the web interface directly in your browser:
    # - OpenApps: http://localhost:5001
    # - Calendar: http://localhost:5001/calendar
    # - Todo: http://localhost:5001/todo
    # - Messenger: http://localhost:5001/messages
    # - Maps: http://localhost:5001/maps

    # Docker mode web interface
    # - Web UI: http://localhost:8000/web
    # - API docs: http://localhost:8000/docs

Important:
    The browser visualization is controlled by the OpenApps SERVER, not the client.
    You must launch the server with 'browsergym_env_args.headless=False' to see
    the browser window.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_with_docker(num_steps: int = 15, headless: bool = True):
    """Run OpenApp environment using Docker container."""
    from openapp_env import OpenAppAction, OpenAppEnv

    print("=" * 70)
    print("Starting OpenApp environment with Docker...")
    print(f"Headless mode: {headless}")
    print("=" * 70)

    try:
        # Create environment from Docker image
        env = OpenAppEnv.from_docker_image("openapp-env:latest")

        # Reset to start a new session
        print("\n[1/4] Resetting environment...")
        result = env.reset()
        print(f"✓ Environment reset")
        print(f"  Starting URL: {result.observation.url}")
        print(f"  Open pages: {len(result.observation.open_pages_urls)}")
        print(f"  HTML length: {len(result.observation.html)} characters")

        # Example actions to demonstrate different action types
        actions = [
            {
                "description": "Navigate to calendar app",
                "action": OpenAppAction(
                    action_type="goto", url="http://localhost:5001/calendar"
                ),
            },
            {
                "description": "Scroll down to see more content",
                "action": OpenAppAction(action_type="scroll", direction="down"),
            },
            {
                "description": "Navigate to todo app",
                "action": OpenAppAction(
                    action_type="goto", url="http://localhost:5001/todo"
                ),
            },
            {
                "description": "Navigate to messenger app",
                "action": OpenAppAction(
                    action_type="goto", url="http://localhost:5001/messenger"
                ),
            },
            {
                "description": "Navigate to maps app",
                "action": OpenAppAction(
                    action_type="goto", url="http://localhost:5001/maps"
                ),
            },
            {
                "description": "Navigate back to home",
                "action": OpenAppAction(
                    action_type="goto", url="http://localhost:5001"
                ),
            },
        ]

        # Run demonstration steps
        print(f"\n[2/4] Running {min(num_steps, len(actions))} demonstration steps...")
        for i, action_info in enumerate(actions[:num_steps]):
            print(f"\nStep {i+1}: {action_info['description']}")
            print(f"  Action type: {action_info['action'].action_type}")

            result = env.step(action_info["action"])

            print(f"  ✓ Action executed")
            print(f"    Current URL: {result.observation.url}")
            print(f"    Reward: {result.reward}")
            print(f"    Done: {result.done}")

            if result.observation.last_action_error:
                print(f"    ⚠️  Error: {result.observation.last_action_error}")

            # Check app state if available
            if result.observation.app_state:
                print(
                    f"    App state keys: {list(result.observation.app_state.keys())}"
                )

            # Small delay for readability
            time.sleep(0.5)

            if result.done:
                print(f"\n✓ Episode finished at step {i+1}!")
                break

        # Get final state
        print(f"\n[3/4] Getting final environment state...")
        state = env.state()
        print(f"✓ Final state:")
        print(f"  Episode ID: {state.episode_id}")
        print(f"  Total steps: {state.step_count}")

        # Summary
        print(f"\n[4/4] Session Summary:")
        print(f"  Steps taken: {state.step_count}")
        print(f"  Final URL: {result.observation.url}")
        print(f"  Episode complete: {result.done}")

        # Web interface info
        print(f"\n" + "=" * 70)
        print(f"💡 TIP: Access the web interface at http://localhost:8000/web")
        print(f"    - Interactive UI for manual testing")
        print(f"    - API documentation at http://localhost:8000/docs")
        print(f"=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\n[Cleanup] Closing environment...")
        env.close()
        print("✓ Environment closed")

    return 0


def run_local(num_steps: int = 15, headless: bool = True):
    """Run OpenApp environment locally without Docker."""

    # Check if OPENAPPS_URL is set
    if not os.environ.get("OPENAPPS_URL"):
        print("=" * 70)
        print("❌ ERROR: OPENAPPS_URL not set")
        print("=" * 70)
        print("\nLocal mode requires OpenApps server to be running.")
        print("\nPlease follow these steps:")
        print("\n1. Start OpenApps server in a separate terminal:")
        print("   cd /path/to/OpenApps")
        print("   uv run launch.py")
        print("\n2. Set the OPENAPPS_URL environment variable:")
        print("   export OPENAPPS_URL=http://localhost:5001")
        print("\n3. Run this script again:")
        print("   python examples/openapp_example.py --mode local")
        print("\nAlternatively, use Docker mode (recommended):")
        print("   python examples/openapp_example.py --mode docker")
        print("=" * 70)
        return 1

    try:
        # NOTE: This example imports from the server module directly for local development.
        # This is intentional for local testing/debugging where no HTTP server is involved.
        # In production, use the client API (OpenAppEnv) which communicates over HTTP/WebSocket.
        # See run_with_docker() for the recommended production pattern.
        from openapp_env.models import OpenAppAction
        from openapp_env.server.openapp_environment import OpenAppEnvironment
    except ImportError as e:
        print(f"❌ Error importing local modules: {e}")
        print("\nMake sure you have installed the environment:")
        print("  cd envs/openapp_env")
        print("  pip install -e .")
        return 1

    print("=" * 70)
    print("Starting OpenApp environment locally...")
    print(f"Using OpenApps server at: {os.environ.get('OPENAPPS_URL')}")
    print(f"Headless mode: {headless}")
    print("=" * 70)

    try:
        # Create environment locally
        print("\n[1/4] Initializing local environment...")
        env = OpenAppEnvironment(
            openapps_url=os.environ.get("OPENAPPS_URL"),
            headless=headless,
            max_steps=50,
        )

        # Reset environment
        print("\n[2/4] Resetting environment...")
        result = env.reset()
        print(f"✓ Environment reset")
        print(f"  Starting URL: {result.url}")
        print(f"  HTML length: {len(result.html)} characters")

        # Take some example steps
        print(f"\n[3/4] Running {num_steps} steps...")
        for i in range(num_steps):
            # Simple actions for demonstration
            actions = [
                OpenAppAction(
                    action_type="goto", url=f"{os.environ.get('OPENAPPS_URL')}/calendar"
                ),
                OpenAppAction(action_type="scroll", direction="down"),
                OpenAppAction(
                    action_type="goto", url=f"{os.environ.get('OPENAPPS_URL')}/todo"
                ),
                OpenAppAction(action_type="noop"),
            ]

            action = actions[i % len(actions)]
            result = env.step(action)

            if i % 5 == 0 or result.done:
                print(f"Step {i+1}:")
                print(f"  Action: {action.action_type}")
                print(f"  URL: {result.url}")
                print(f"  Reward: {result.reward}")
                print(f"  Done: {result.done}")

            if result.done:
                print(f"\n✓ Episode finished at step {i+1}!")
                break

        # Get final state
        print(f"\n[4/4] Final state:")
        print(f"  Episode ID: {env.state.episode_id}")
        print(f"  Total steps: {env.state.step_count}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\n[Cleanup] Closing environment...")
        env.close()
        print("✓ Environment closed")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OpenApp Environment Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Docker (recommended)
  python examples/openapp_example.py --mode docker

  # Run locally without Docker
  python examples/openapp_example.py --mode local

  # Show browser window to visualize agent actions
  python examples/openapp_example.py --mode local --show-browser

  # Run with custom number of steps
  python examples/openapp_example.py --mode docker --num-steps 20

Visualization:
  - Use --show-browser to see the browser window and watch agent interactions
  - Access OpenApps web interface at http://localhost:5001 (when server is running)
  - Docker mode web interface: http://localhost:8000/web

Note:
  - Docker mode requires: docker build -t openapp-env:latest -f envs/openapp_env/server/Dockerfile envs/openapp_env
  - Local mode requires: pip install -e envs/openapp_env && playwright install chromium
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["docker", "local"],
        default="docker",
        help="Run mode: 'docker' (recommended) or 'local'",
    )
    parser.add_argument(
        "--num-steps", type=int, default=15, help="Number of steps to run (default: 15)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (no visible window)",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        default=False,
        help="Show browser window (opposite of --headless, easier to remember)",
    )

    args = parser.parse_args()

    # Determine headless mode: default to True unless --show-browser is used
    headless = not args.show_browser if args.show_browser else args.headless

    print("\n" + "=" * 70)
    print("OpenApp Environment Example")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.num_steps}")
    print(f"Headless: {headless}")

    if args.mode == "docker":
        return run_with_docker(args.num_steps, headless)
    else:
        return run_local(args.num_steps, headless)


if __name__ == "__main__":
    sys.exit(main())
