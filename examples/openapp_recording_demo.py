#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo script optimized for recording videos of OpenApp environment interactions.

This script provides slower, more visible agent interactions suitable for video recording.

SETUP FOR RECORDING:

Terminal 1 - Start OpenApps server with visible browser:
    cd OpenApps
    python launch.py browsergym_env_args.headless=False

Terminal 2 - Run this script:
    export OPENAPPS_URL=http://localhost:5001
    python examples/openapp_recording_demo.py

Then use screen recording software to capture the browser window.

Options:
    --scenario: Choose demo scenario (calendar, todo, messages, shopping, all)
    --delay: Seconds to wait between actions (default: 2)
    --verbose: Print detailed information during recording
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openapp_env.models import OpenAppAction
from openapp_env.server.openapp_environment import OpenAppEnvironment


class RecordingDemo:
    """Demo scenarios optimized for video recording."""

    def __init__(self, openapps_url: str, delay: float = 2.0, verbose: bool = False):
        """
        Initialize recording demo.

        Args:
            openapps_url: URL of OpenApps server
            delay: Seconds to wait between actions
            verbose: Print detailed information
        """
        self.openapps_url = openapps_url
        self.delay = delay
        self.verbose = verbose
        self.env = None

    def setup(self):
        """Set up the environment."""
        print("=" * 70)
        print("OpenApp Recording Demo")
        print("=" * 70)
        print(f"OpenApps server: {self.openapps_url}")
        print(f"Action delay: {self.delay}s")
        print(f"Verbose mode: {self.verbose}")
        print()
        print("⏺️  START YOUR SCREEN RECORDING NOW!")
        print("   Recording the browser window that appears...")
        print()

        # Give user time to start recording
        for i in range(3, 0, -1):
            print(f"   Starting in {i}...", end='\r')
            time.sleep(1)
        print("\n" + "=" * 70)
        print()

        # Create environment
        self.env = OpenAppEnvironment(
            openapps_url=self.openapps_url,
            headless=False,  # Visible browser
            max_steps=100,
        )

    def wait(self, message: str = None):
        """Wait between actions with optional message."""
        if message and self.verbose:
            print(f"   {message}")
        time.sleep(self.delay)

    def step(self, action: OpenAppAction, description: str):
        """Execute action with description."""
        print(f"🎬 {description}")
        if self.verbose:
            print(f"   Action: {action.action_type}")
            if hasattr(action, 'url') and action.url:
                print(f"   URL: {action.url}")
            if hasattr(action, 'bid') and action.bid:
                print(f"   Element: {action.bid}")
            if hasattr(action, 'text') and action.text:
                print(f"   Text: {action.text}")

        result = self.env.step(action)

        if self.verbose:
            print(f"   ✓ Current URL: {result.url}")
            if result.last_action_error:
                print(f"   ⚠️  Error: {result.last_action_error}")

        self.wait()
        return result

    def calendar_scenario(self):
        """Demonstrate calendar interactions with meaningful actions."""
        print("\n" + "=" * 70)
        print("SCENARIO: Calendar Management")
        print("=" * 70)
        print("Demonstrating: View calendar, switch views, navigate months, view events\n")

        self.wait("Resetting environment...")
        self.env.reset()

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/10 - Navigate to home page"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar"),
            "2/10 - Open calendar application (Calendar view)"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "3/10 - Scroll down to view calendar grid"
        )

        # Switch to Agenda view
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar?view=agenda"),
            "4/10 - Switch to Agenda view"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "5/10 - Browse agenda events list"
        )

        # Switch back to Calendar view
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar?view=calendar"),
            "6/10 - Switch back to Calendar view"
        )

        # Navigate to next month
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar/calendar_content/2026/1?view=calendar"),
            "7/10 - Navigate to next month (January 2026)"
        )

        # Navigate to previous month
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar/calendar_content/2025/12?view=calendar"),
            "8/10 - Navigate back to current month (December 2025)"
        )

        # View a specific event (if available)
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar/event/1"),
            "9/10 - View event details"
        )

        # Return to calendar
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar"),
            "10/10 - Return to calendar main view"
        )

        print("✓ Calendar scenario complete!\n")

    def todo_scenario(self):
        """Demonstrate todo list interactions with meaningful actions."""
        print("\n" + "=" * 70)
        print("SCENARIO: Todo List Management")
        print("=" * 70)
        print("Demonstrating: Browse tasks, view edit interface, navigate todo items\n")

        self.wait("Resetting environment...")
        self.env.reset()

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/11 - Navigate to home page"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo"),
            "2/11 - Open todo list application"
        )

        # Give viewers time to see the initial todo list
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "3/11 - Browse through todo items"
        )

        # View edit interface for first todo
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo/edit/0"),
            "4/11 - Open edit interface for first task"
        )

        # Return to main todo view
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo"),
            "5/11 - Return to todo list"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "6/11 - Scroll through more tasks"
        )

        # View edit interface for another todo
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo/edit/5"),
            "7/11 - Open edit interface for another task"
        )

        # Return to main todo view
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo"),
            "8/11 - Return to todo list"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "9/11 - Final browse through todo items"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="up"),
            "10/11 - Scroll back to top"
        )

        # Return to home
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "11/11 - Return to home page"
        )

        print("✓ Todo scenario complete!\n")

    def messages_scenario(self):
        """Demonstrate messenger with actual message sending."""
        print("\n" + "=" * 70)
        print("SCENARIO: Messenger - Send Message")
        print("=" * 70)
        print("Demonstrating: Browse conversations, send interactive message to Alice\n")

        self.wait("Resetting environment...")
        self.env.reset()

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/14 - Navigate to home page"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages"),
            "2/14 - Open messenger application"
        )

        # View conversations list
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "3/14 - Browse conversation list"
        )
        # time.sleep(3)
        # Open conversation with Alice
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages/Alice"),
            "4/14 - Open conversation with Alice"
        )

        # Scroll to view conversation history
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "5/14 - Scroll through message history with Alice"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="up"),
            "6/14 - Scroll back to message input"
        )

        # Click on the message input field to focus it
        print("        → Clicking message input field to focus it...")
        result = self.step(
            OpenAppAction(
                action_type="click",
                bid="msg-input"
            ),
            "7/14 - Click message input field"
        )
        if hasattr(result, 'last_action_error') and result.last_action_error:
            print(f"        ⚠️  Error clicking input: {result.last_action_error}")

        # Type the message using fill with CSS selector (directly targets the input element by HTML ID)
        print("        → Typing message using fill with CSS selector...")
        result = self.step(
            OpenAppAction(
                action_type="fill",
                bid="#msg-input",
                text="Hi, We are submitting the OpenApp for OpenEnv Hackathon and we are super excited about this!"
            ),
            "8/14 - Type message: 'Hi, We are submitting the OpenApp for OpenEnv Hackathon...'"
        )
        if hasattr(result, 'last_action_error') and result.last_action_error:
            print(f"        ⚠️  Error typing message: {result.last_action_error}")

        # Pause to let viewers see the typed message
        self.step(
            OpenAppAction(action_type="noop"),
            "9/14 - View typed message before sending"
        )

        # Send the message by pressing Enter
        print("        → Pressing Enter to send message...")
        result = self.step(
            OpenAppAction(
                action_type="send_keys",
                text="\n"
            ),
            "10/14 - Press Enter to send message"
        )
        if hasattr(result, 'last_action_error') and result.last_action_error:
            print(f"        ⚠️  Error sending message: {result.last_action_error}")

        # Pause to see Alice's response
        self.step(
            OpenAppAction(action_type="noop"),
            "11/14 - View Alice's response"
        )

        # Go back to messages list
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages"),
            "12/14 - Return to conversations list"
        )

        # Return to home
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "13/14 - Return to home page"
        )

        # Final pause
        self.step(
            OpenAppAction(action_type="noop"),
            "14/14 - Demo complete"
        )

        print("✓ Messenger scenario complete!\n")

    def codeeditor_scenario(self):
        """Demonstrate code editor by typing a PyTorch training loop."""
        print("\n" + "=" * 70)
        print("SCENARIO: Code Editor - PyTorch Training Loop")
        print("=" * 70)
        print("Demonstrating: Create file, type code interactively, save file\n")

        self.wait("Resetting environment...")
        self.env.reset()

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/15 - Navigate to home page"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/codeeditor"),
            "2/15 - Open code editor application"
        )

        # Give viewers time to see the code editor interface
        self.step(
            OpenAppAction(action_type="noop"),
            "3/15 - View code editor interface with file tree"
        )

        # Create a new file
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/codeeditor/train.py"),
            "4/15 - Create new file: train.py"
        )

        # The PyTorch training loop code to type
        pytorch_code = '''import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
'''

        # Type the code line by line for dramatic effect
        lines = pytorch_code.strip().split('\n')

        # Type first few lines with imports
        current_text = ""
        for i, line in enumerate(lines[:4]):  # imports
            current_text += line + "\n"

        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#editor",
                text=current_text.strip()
            ),
            "5/15 - Type imports: torch, nn, optim"
        )

        # Add class definition
        for i, line in enumerate(lines[4:10]):  # class header
            current_text += line + "\n"

        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#editor",
                text=current_text.strip()
            ),
            "6/15 - Define SimpleNet class with layers"
        )

        # Add forward method
        for i, line in enumerate(lines[10:15]):  # forward method
            current_text += line + "\n"

        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#editor",
                text=current_text.strip()
            ),
            "7/15 - Add forward() method"
        )

        # Add model initialization
        for i, line in enumerate(lines[15:20]):  # model init
            current_text += line + "\n"

        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#editor",
                text=current_text.strip()
            ),
            "8/15 - Initialize model, loss function, optimizer"
        )

        # Add training loop
        for i, line in enumerate(lines[20:]):  # training loop
            current_text += line + "\n"

        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#editor",
                text=current_text.strip()
            ),
            "9/15 - Add training loop with forward/backward pass"
        )

        # Pause to let viewers see the complete code
        self.step(
            OpenAppAction(action_type="noop"),
            "10/15 - View complete PyTorch training code"
        )

        # Save the file by clicking the save button
        self.step(
            OpenAppAction(action_type="click", bid="Save"),
            "13/15 - Save the file"
        )

        # Return to code editor index
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/codeeditor"),
            "14/15 - Return to code editor to see saved file"
        )

        # Return to home
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "15/15 - Return to home page"
        )

        print("✓ Code Editor scenario complete!\n")

    def maps_scenario(self):
        """Demonstrate maps with search and landmark exploration."""
        print("\n" + "=" * 70)
        print("SCENARIO: Maps Exploration with Search")
        print("=" * 70)
        print("Demonstrating: Search locations, view landmarks, explore map areas\n")

        self.wait("Resetting environment...")
        self.env.reset()

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/12 - Navigate to home page"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps"),
            "2/12 - Open maps application"
        )

        # Give viewers time to see the initial map with landmarks
        self.step(
            OpenAppAction(action_type="noop"),
            "3/12 - View map with landmarks and search interface"
        )

        # Scroll to explore different parts of the interface
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "4/12 - Scroll to view route planning interface"
        )

        # Pause to show the route planning form
        self.step(
            OpenAppAction(action_type="noop"),
            "5/12 - View route planning controls (From/To location fields)"
        )

        # Scroll back up to the map
        self.step(
            OpenAppAction(action_type="scroll", direction="up"),
            "6/12 - Scroll back to main map view"
        )

        # Use the search API to find a location (this demonstrates the search feature)
        # Search for "San Francisco" to pan the map
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps/where?q=San+Francisco"),
            "7/12 - Search for location: San Francisco"
        )

        # Return to maps to see the result
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps"),
            "8/12 - Return to map view"
        )

        # Explore the map by scrolling
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "9/12 - Pan map to explore different areas"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="up"),
            "10/12 - Pan map back"
        )

        self.step(
            OpenAppAction(action_type="noop"),
            "11/12 - Final view of map with all features"
        )

        # Return to home
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "12/12 - Return to home page"
        )

        print("✓ Maps scenario complete!\n")

    def app_tour_scenario(self):
        """Tour through all applications with meaningful interactions."""
        print("\n" + "=" * 70)
        print("SCENARIO: Complete App Tour")
        print("=" * 70)
        print("Demonstrating: All OpenApps applications with diverse interactions\n")

        self.wait("Starting tour...")
        self.env.reset()

        # Home page
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "1/21 - Start at home page"
        )

        # === Calendar ===
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar"),
            "2/21 - Calendar: View calendar application"
        )
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "3/21 - Browse calendar events"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar?view=agenda"),
            "4/21 - Switch to Agenda view"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/calendar/event/1"),
            "5/21 - View event details"
        )

        # === Todo ===
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo"),
            "6/21 - Todo: Open task manager"
        )
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "7/21 - Browse todo items"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo/edit/0"),
            "8/21 - View task edit interface"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/todo"),
            "9/21 - Return to todo list"
        )

        # === Messenger ===
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages"),
            "10/26 - Messenger: View conversations list"
        )
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "11/26 - Browse conversations"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages/Alice"),
            "12/26 - Open conversation with Alice"
        )
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "13/26 - View message history"
        )
        # Send a message to Alice
        self.step(
            OpenAppAction(
                action_type="fill",
                bid="#msg-input",
                text="Hi, We are submitting the OpenApp for OpenEnv Hackathon and we are super excited about this!"
            ),
            "14/26 - Type hackathon message to Alice"
        )
        self.step(
            OpenAppAction(action_type="send_keys", text="\n"),
            "15/26 - Send message (press Enter)"
        )
        self.step(
            OpenAppAction(action_type="noop"),
            "16/26 - View Alice's response"
        )
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/messages"),
            "17/26 - Return to conversations list"
        )

        # === Maps ===
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps"),
            "18/26 - Maps: Open navigation app"
        )
        self.step(
            OpenAppAction(action_type="noop"),
            "19/26 - View maps with landmarks"
        )
        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "20/26 - View route planning interface"
        )

        # Search for a location
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps/where?q=Golden+Gate+Park"),
            "21/26 - Search: Golden Gate Park"
        )

        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/maps"),
            "22/26 - Return to map view"
        )

        self.step(
            OpenAppAction(action_type="scroll", direction="down"),
            "23/26 - Explore map areas"
        )

        # Return home
        self.step(
            OpenAppAction(action_type="goto", url=f"{self.openapps_url}/"),
            "24/26 - Return to home page"
        )

        print("✓ App tour complete!\n")

    def cleanup(self):
        """Clean up and close environment."""
        print("\n" + "=" * 70)
        print("🎬 Recording Demo Complete!")
        print("=" * 70)
        print()
        print("⏹️  STOP YOUR SCREEN RECORDING NOW!")
        print()
        print("Waiting 1 seconds before cleanup...")
        time.sleep(1)

        if self.env:
            self.env.close()
            print("✓ Environment closed")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="OpenApp Recording Demo - Optimized for video recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tour all apps (recommended for comprehensive demo)
  python examples/openapp_recording_demo.py --scenario all

  # Record calendar interactions only
  python examples/openapp_recording_demo.py --scenario calendar

  # Record todo list interactions
  python examples/openapp_recording_demo.py --scenario todo

  # Record messenger interactions
  python examples/openapp_recording_demo.py --scenario messages

  # Record maps interactions
  python examples/openapp_recording_demo.py --scenario maps

  # Record code editor with PyTorch training loop
  python examples/openapp_recording_demo.py --scenario codeeditor

  # Slower pacing for detailed recording (3 seconds between actions)
  python examples/openapp_recording_demo.py --scenario calendar --delay 3

  # Verbose output with detailed logs
  python examples/openapp_recording_demo.py --scenario all --verbose

Before running:
  1. Terminal 1: cd OpenApps && python launch.py browsergym_env_args.headless=False
  2. Terminal 2: export OPENAPPS_URL=http://localhost:5001
  3. Start your screen recording software
  4. Run this script
        """,
    )

    parser.add_argument(
        "--scenario",
        choices=["calendar", "todo", "messages", "maps", "codeeditor", "all"],
        default="all",
        help="Demo scenario to record (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between actions (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during recording",
    )

    args = parser.parse_args()

    # Check if OPENAPPS_URL is set
    openapps_url = os.environ.get("OPENAPPS_URL")
    if not openapps_url:
        print("=" * 70)
        print("❌ ERROR: OPENAPPS_URL not set")
        print("=" * 70)
        print()
        print("Please set up OpenApps server first:")
        print()
        print("Terminal 1 - Start OpenApps server with visible browser:")
        print("    cd OpenApps")
        print("    python launch.py browsergym_env_args.headless=False")
        print()
        print("Terminal 2 - Set URL and run this script:")
        print("    export OPENAPPS_URL=http://localhost:5001")
        print("    python examples/openapp_recording_demo.py")
        print()
        print("=" * 70)
        return 1

    # Create demo
    demo = RecordingDemo(
        openapps_url=openapps_url,
        delay=args.delay,
        verbose=args.verbose,
    )

    try:
        # Setup
        demo.setup()

        # Run selected scenario
        if args.scenario == "all":
            demo.app_tour_scenario()
        elif args.scenario == "calendar":
            demo.calendar_scenario()
        elif args.scenario == "todo":
            demo.todo_scenario()
        elif args.scenario == "messages":
            demo.messages_scenario()
        elif args.scenario == "maps":
            demo.maps_scenario()
        elif args.scenario == "codeeditor":
            demo.codeeditor_scenario()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        demo.cleanup()


if __name__ == "__main__":
    sys.exit(main())
