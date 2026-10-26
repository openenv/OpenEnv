#!/usr/bin/env python3
"""Minimal Echo environment client example.

Start the Echo server first:

    PYTHONPATH=src:envs uv run python -m echo_env.server.app

Then run this example:

    PYTHONPATH=src:envs uv run python examples/local_echo_env.py
"""

import os

from echo_env import EchoEnv


def main():
    """Exercise the MCP Echo tools through the Python client."""
    base_url = os.getenv("ECHO_BASE_URL", "http://localhost:8000")

    print("=" * 60)
    print("EchoEnv local client example")
    print("=" * 60)
    print()

    try:
        print("Connecting to Echo server...")
        print(f"  EchoEnv(base_url={base_url!r})")
        print()

        client = EchoEnv(base_url=base_url).sync()

        print("Client created.\n")

        # Now use it like any other client
        print("Testing the environment:")
        print("-" * 60)

        # Reset
        print("\n1. Reset:")
        result = client.reset()
        print(f"   Status: {result.observation.metadata.get('status')}")
        print(f"   Message: {result.observation.metadata.get('message')}")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        print("\n2. List tools:")
        tools = client.list_tools()
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")

        # Send some messages
        print("\n3. Send messages:")

        messages = [
            "Hello, World!",
            "Testing echo environment",
            "One more message",
        ]

        for i, msg in enumerate(messages, 1):
            echoed = client.call_tool("echo_message", message=msg)
            with_length = client.call_tool("echo_with_length", message=msg)
            print(f"   {i}. '{msg}'")
            print(f"      Echoed: '{echoed}'")
            print(f"      Length: {with_length['length']}")

        print("\n" + "-" * 60)
        print("\nAll operations successful.")
        print()

        print("Cleaning up...")
        client.close()
        print("Client closed.")
        print()

        print("=" * 60)
        print("Test completed successfully.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
