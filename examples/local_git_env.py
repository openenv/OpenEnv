#!/usr/bin/env python3
"""
Simple test showing how users will use GitEnv.from_docker_image().

This is the simplest possible usage.

Prerequisites:
    1. .env file configured (copy from .env.example)
    2. Shared Gitea running: ./scripts/setup_shared_gitea.sh
    3. OpenEnv repo migrated to Gitea (see README)
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from git_env import GitAction, GitEnv


def main():
    """Test GitEnv.from_docker_image()."""
    print("=" * 60)
    print("GitEnv.from_docker_image() Test")
    print("=" * 60)
    print()

    try:
        # Pass environment variables from .env to container
        env_vars = {
            "GITEA_URL": os.getenv("GITEA_URL"),
            "GITEA_USERNAME": os.getenv("GITEA_USERNAME"),
            "GITEA_PASSWORD": os.getenv("GITEA_PASSWORD"),
        }

        # Verify env vars are loaded
        if not all(env_vars.values()):
            print("❌ Error: Required environment variables not found in .env")
            print("   Make sure .env file exists (copy from .env.example)")
            return False

        print("Creating client from Docker image with .env credentials...")
        print("  Using GitEnv.from_docker_image() factory method")
        print()

        # Create client using from_docker_image factory method
        client = GitEnv.from_docker_image(
            "git-env:latest", env_vars=env_vars
        ).sync()

        print("✓ Client created and container started!\n")

        # Now use it like any other client
        print("Testing the environment:")
        print("-" * 60)

        # Reset
        print("\n1. Reset:")
        result = client.reset()
        print(f"   Message: {result.observation.message}")
        print(f"   Success: {result.observation.success}")

        # Get initial state
        state = client.state()
        print(f"   State: episode_id={state.episode_id}, step_count={state.step_count}")
        print(f"   Gitea ready: {state.gitea_ready}")

        # List repositories
        print("\n2. List repositories:")
        result = client.step(GitAction(action_type="list_repos"))
        print(f"   Success: {result.observation.success}")
        print(f"   Found {len(result.observation.repos)} repositories")
        for repo in result.observation.repos:
            print(f"     - {repo['name']}")

        # Clone repository
        print("\n3. Clone repository:")
        result = client.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))
        print(f"   Success: {result.observation.success}")
        print(f"   Message: {result.observation.message}")
        print(f"   Output: {result.observation.output}")

        # Execute git commands
        print("\n4. Execute git commands:")

        git_commands = [
            "status",
            "log --oneline -5",
            "branch -a",
        ]

        for cmd in git_commands:
            result = client.step(
                GitAction(action_type="execute_git_command", command=cmd, working_dir="OpenEnv")
            )
            print(f"\n   git {cmd}:")
            print(f"   Success: {result.observation.success}")
            if result.observation.output:
                # Show first few lines
                lines = result.observation.output.strip().split("\n")[:5]
                for line in lines:
                    print(f"     {line}")
                if len(result.observation.output.strip().split("\n")) > 5:
                    print("     ...")

        # Check final state
        print("\n5. Check final state:")
        state = client.state()
        print(f"   episode_id: {state.episode_id}")
        print(f"   step_count: {state.step_count}")
        print(f"   gitea_ready: {state.gitea_ready}")

        print("\n" + "-" * 60)
        print("\n✓ All operations successful!")
        print()

        print("Cleaning up...")
        client.close()
        print("✓ Container stopped and removed")
        print()

        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
