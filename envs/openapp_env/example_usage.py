#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of OpenApp Environment.

This script demonstrates how to use the OpenApp environment with OpenEnv.

For a complete runnable example, see: examples/openapp_example.py

Visualization Options:
    To see the browser window and watch agent interactions:

    Terminal 1: Start OpenApps server with visible browser
        cd OpenApps
        python OpenApps/launch.py browsergym_env_args.headless=False

    Terminal 2: Run your agent code
        export OPENAPPS_URL=http://localhost:5001
        python examples/openapp_example.py --mode local

    Or access OpenApps web interface at http://localhost:5001
    Docker mode web interface at http://localhost:8000/web

Important:
    Browser visualization is controlled by the OpenApps SERVER, not the client.
    Launch the server with 'browsergym_env_args.headless=False' to see the browser.
"""

import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from envs.openapp_env import OpenAppAction, OpenAppEnv


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("OpenApp Environment - Basic Usage Example")
    print("=" * 60)

    # Option 1: Connect to a running server
    print("\nOption 1: Connect to running server")
    print("client = OpenAppEnv(base_url='http://localhost:8000')")

    # Option 2: Start from Docker image (recommended)
    print("\nOption 2: Start from Docker image")
    print("client = OpenAppEnv.from_docker_image('openapp-env:latest')")

    print("\n" + "-" * 60)


def example_actions():
    """Example of different action types."""
    print("\nExample Actions")
    print("-" * 60)

    # Navigate to a page
    print("\n1. Navigate to calendar app:")
    print("action = OpenAppAction(")
    print("    action_type='goto',")
    print("    url='http://localhost:5001/calendar'")
    print(")")
    print("result = client.step(action)")

    # Click on an element
    print("\n2. Click on a button:")
    print("action = OpenAppAction(")
    print("    action_type='click',")
    print("    bid='add-event-btn'  # BrowserGym element ID")
    print(")")
    print("result = client.step(action)")

    # Fill a form field
    print("\n3. Fill in text input:")
    print("action = OpenAppAction(")
    print("    action_type='fill',")
    print("    bid='event-title-input',")
    print("    text='Team Meeting'")
    print(")")
    print("result = client.step(action)")

    # Select from dropdown
    print("\n4. Select from dropdown:")
    print("action = OpenAppAction(")
    print("    action_type='select_option',")
    print("    bid='time-select',")
    print("    value='14:00'")
    print(")")
    print("result = client.step(action)")

    # Scroll the page
    print("\n5. Scroll down:")
    print("action = OpenAppAction(")
    print("    action_type='scroll',")
    print("    direction='down'")
    print(")")
    print("result = client.step(action)")

    # No operation
    print("\n6. No operation (useful for observation):")
    print("action = OpenAppAction(action_type='noop')")
    print("result = client.step(action)")


def example_observations():
    """Example of observation structure."""
    print("\n\nObservation Structure")
    print("-" * 60)

    print("\nAfter reset() or step(), you receive:")
    print("result.observation.html          # Current page HTML")
    print("result.observation.url           # Current URL")
    print("result.observation.open_pages_urls  # All open pages")
    print("result.observation.axtree_txt    # Accessibility tree")
    print("result.observation.app_state     # App states (calendar, todo, etc.)")
    print("result.observation.task_info     # Task information (if using tasks)")
    print("result.observation.screenshot    # Page screenshot (base64)")
    print("result.observation.last_action_error  # Error from last action")
    print("result.reward                    # Step reward")
    print("result.done                      # Episode done flag")


def example_complete_workflow():
    """Complete workflow example."""
    print("\n\nComplete Workflow Example")
    print("=" * 60)

    example_code = """
from envs.openapp_env import OpenAppAction, OpenAppEnv

# Create client (starts Docker container)
client = OpenAppEnv.from_docker_image("openapp-env:latest")

try:
    # Reset environment
    result = client.reset()
    print(f"Starting at: {result.observation.url}")

    # Navigate to calendar
    result = client.step(OpenAppAction(
        action_type="goto",
        url="http://localhost:5001/calendar"
    ))

    # Click to add new event
    result = client.step(OpenAppAction(
        action_type="click",
        bid="new-event-button"
    ))

    # Fill event title
    result = client.step(OpenAppAction(
        action_type="fill",
        bid="title-input",
        text="Project Review Meeting"
    ))

    # Fill event date
    result = client.step(OpenAppAction(
        action_type="fill",
        bid="date-input",
        text="2025-12-15"
    ))

    # Submit form
    result = client.step(OpenAppAction(
        action_type="click",
        bid="submit-button"
    ))

    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
    print(f"App State: {result.observation.app_state}")

finally:
    # Always cleanup
    client.close()
"""

    print(example_code)


def example_with_tasks():
    """Example using OpenApps tasks."""
    print("\n\nUsing Tasks (Task-Based RL)")
    print("=" * 60)

    example_code = """
# Environment can be configured with specific tasks
# Tasks define goals and automatic reward calculation

from envs.openapp_env.server.openapp_environment import OpenAppEnvironment

env = OpenAppEnvironment(
    openapps_url="http://localhost:5001",  # OpenApps server URL
    task_name="add_meeting_with_dennis",   # Optional task name
    headless=False,  # Set to False to watch the browser
    max_steps=50,
)

obs = env.reset()
# Now the environment has a goal: add a meeting with Dennis
# Rewards will be based on progress toward this goal

# Agent loop
done = False
while not done:
    action = agent.get_action(obs)  # Your agent
    obs = env.step(action)
    done = obs.done

print(f"Task completed! Reward: {obs.reward}")
env.close()
"""

    print(example_code)


def example_visualization():
    """Example of visualization options."""
    print("\n\nVisualization Options")
    print("=" * 60)

    example_code = """
# Option 1: Show browser window (watch agent in real-time)
from envs.openapp_env.server.openapp_environment import OpenAppEnvironment

env = OpenAppEnvironment(
    openapps_url="http://localhost:5001",
    headless=False,  # Show browser window
)

obs = env.reset()
# You'll see a browser window open!

# Option 2: Access web interface manually
# While OpenApps server is running, open in your browser:
# - Main: http://localhost:5001
# - Calendar: http://localhost:5001/calendar
# - Todo: http://localhost:5001/todo
# - Messenger: http://localhost:5001/messenger
# - Maps: http://localhost:5001/maps

# Option 3: Use the example script with --show-browser
# python examples/openapp_example.py --mode local --show-browser
"""

    print(example_code)


def main():
    """Run all examples."""
    example_basic_usage()
    example_actions()
    example_observations()
    example_complete_workflow()
    example_with_tasks()
    example_visualization()

    print("\n" + "=" * 60)
    print("For a complete runnable example:")
    print("  python examples/openapp_example.py --mode local --show-browser")
    print("\nFor more information, see:")
    print("- README.md in this directory")
    print("- OpenApps docs: https://facebookresearch.github.io/OpenApps/")
    print("- OpenEnv docs: https://huggingface.co/docs/openenv")
    print("=" * 60)


if __name__ == "__main__":
    main()
