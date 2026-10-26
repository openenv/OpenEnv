#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Simple test showing how users will use CodingEnv.from_docker_image().

This is the simplest possible usage
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_env import CodeAction, CodingEnv


async def main():
    """Test CodingEnv.from_docker_image()."""
    print("=" * 60)
    print("CodingEnv.from_docker_image() Test")
    print("=" * 60)
    print()

    client = None
    try:
        # This is what users will do - just one line!
        print("Creating client from Docker image...")
        print("  CodingEnv.from_docker_image('coding-env:latest')")
        print()

        client = await CodingEnv.from_docker_image("coding-env:latest")

        print("✓ Client created and container started!\n")

        # Now use it like any other client
        print("Testing the environment:")
        print("-" * 60)

        # Reset
        print("\n1. Reset:")
        result = await client.reset()
        print(f"   stdout: {result.observation.stdout}")
        print(f"   stderr: {result.observation.stderr}")
        print(f"   exit_code: {result.observation.exit_code}")

        # Get initial state
        state = await client.state()
        print(f"   State: episode_id={state.episode_id}, step_count={state.step_count}")

        # Execute some Python code
        print("\n2. Execute Python code:")

        code_samples = [
            "print('Hello, World!')",
            "x = 5 + 3\nprint(f'Result: {x}')",
            "import math\nprint(f'Pi is approximately {math.pi:.4f}')",
            "# Multi-line calculation\nfor i in range(1, 4):\n    print(f'{i} squared is {i**2}')",
        ]

        for i, code in enumerate(code_samples, 1):
            result = await client.step(CodeAction(code=code))
            display_code = code.replace("\n", "\\n")[:50]
            print(f"   {i}. Code: {display_code}...")
            print(f"      → stdout: {result.observation.stdout.strip()}")
            print(f"      → exit_code: {result.observation.exit_code}")
            if result.observation.stderr:
                print(f"      → stderr: {result.observation.stderr}")

        # Test error scenarios
        print("\n3. Test error scenarios:")

        error_samples = [
            ("Division by zero", "x = 1 / 0\nprint('Should not reach here')"),
            ("Undefined variable", "print(undefined_variable)"),
            ("Syntax error", "print('Hello'"),
        ]

        for i, (description, code) in enumerate(error_samples, 1):
            result = await client.step(CodeAction(code=code))
            display_code = code.replace("\n", "\\n")[:40]
            print(f"   {i}. {description}")
            print(f"      Code: {display_code}...")
            print(f"      → exit_code: {result.observation.exit_code}")
            if result.observation.stderr:
                # Truncate long error messages
                error_msg = result.observation.stderr[:100]
                if len(result.observation.stderr) > 100:
                    error_msg += "..."
                print(f"      → stderr: {error_msg}")

        # Check final state
        print("\n4. Check final state:")
        state = await client.state()
        print(f"   episode_id: {state.episode_id}")
        print(f"   step_count: {state.step_count}")
        print(f"   last_exit_code: {state.last_exit_code}")

        print("\n" + "-" * 60)
        print("\n✓ All operations successful!")
        print()

        print("Cleaning up...")
        await client.close()
        client = None
        print("✓ Container stopped and removed")
        print()

        print("=" * 60)
        print("Test completed successfully! 🎉")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
