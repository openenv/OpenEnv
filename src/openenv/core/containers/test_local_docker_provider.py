#!/usr/bin/env python3
"""
End-to-end test for LocalDockerProvider.

This script tests the complete flow:
1. Start a container using LocalDockerProvider
2. Wait for it to be ready
3. Make HTTP requests to test the environment
4. Clean up the container
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
from openenv.core.containers.runtime import LocalDockerProvider


# TODO: Remove this test or make it a functional test since this will be tested in e2e test for echo env
def test_local_docker_provider():
    """Test LocalDockerProvider end-to-end."""
    print("=" * 60)
    print("LocalDockerProvider End-to-End Test")
    print("=" * 60)
    print()

    provider = None

    try:
        # Step 1: Create provider
        print("Step 1: Creating LocalDockerProvider...")
        provider = LocalDockerProvider()
        print("✓ Provider created\n")

        # Step 2: Start container
        print("Step 2: Starting echo-env container...")
        base_url = provider.start_container("echo-env:latest")
        print(f"✓ Container started at: {base_url}")
        if provider._container_id:
            print(f"  Container ID: {provider._container_id[:12]}...")
        if provider._container_name:
            print(f"  Container name: {provider._container_name}\n")

        # Step 3: Wait for ready
        print("Step 3: Waiting for container to be ready...")
        provider.wait_for_ready(base_url, timeout_s=30.0)
        print("✓ Container is ready!\n")

        # Step 4: Test health endpoint
        print("Step 4: Testing /health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✓ Health check passed\n")

        # Step 5: Test reset endpoint
        print("Step 5: Testing /reset endpoint...")
        response = requests.post(
            f"{base_url}/reset",
            json={},
            headers={"Content-Type": "application/json"},
        )
        print(f"  Status: {response.status_code}")
        data = response.json()
        print(f"  Message: {data['observation']['echoed_message']}")
        print(f"  Reward: {data['reward']}")
        print(f"  Done: {data['done']}")
        assert response.status_code == 200
        assert data["observation"]["echoed_message"] == "Echo environment ready!"
        print("✓ Reset test passed\n")

        # Step 6: Test step endpoint
        print("Step 6: Testing /step endpoint...")
        response = requests.post(
            f"{base_url}/step",
            json={"action": {"message": "Hello from LocalDockerProvider!"}},
            headers={"Content-Type": "application/json"},
        )
        print(f"  Status: {response.status_code}")
        data = response.json()
        print(f"  Echoed: {data['observation']['echoed_message']}")
        print(f"  Length: {data['observation']['message_length']}")
        print(f"  Reward: {data['reward']}")
        assert response.status_code == 200
        assert (
            data["observation"]["echoed_message"] == "Hello from LocalDockerProvider!"
        )
        assert data["observation"]["message_length"] == 31
        print("✓ Step test passed\n")

        # Step 7: Test state endpoint
        print("Step 7: Testing /state endpoint...")
        response = requests.get(f"{base_url}/state")
        print(f"  Status: {response.status_code}")
        data = response.json()
        print(f"  Episode ID: {data['episode_id']}")
        print(f"  Step count: {data['step_count']}")
        assert response.status_code == 200
        assert data["step_count"] == 1  # One step from above
        print("✓ State test passed\n")

        # Step 8: Multiple steps
        print("Step 8: Testing multiple steps...")
        for i in range(3):
            response = requests.post(
                f"{base_url}/step",
                json={"action": {"message": f"Message {i + 1}"}},
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200
            print(f"  Step {i + 1}: ✓")

        # Check state updated
        response = requests.get(f"{base_url}/state")
        data = response.json()
        assert data["step_count"] == 4  # 1 + 3 more steps
        print(f"  Final step count: {data['step_count']}")
        print("✓ Multiple steps test passed\n")

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Step 9: Cleanup
        if provider is not None:
            print("\nStep 9: Cleaning up container...")
            try:
                provider.stop_container()
                print("✓ Container stopped and removed\n")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}\n")


def test_provider_with_custom_port():
    """Test provider with custom port."""
    print("=" * 60)
    print("LocalDockerProvider with Custom Port Test")
    print("=" * 60)
    print()

    provider = None

    try:
        provider = LocalDockerProvider()

        print("Starting container on custom port 8123...")
        base_url = provider.start_container("echo-env:latest", port=8123)
        print(f"✓ Started at: {base_url}")
        assert ":8123" in base_url

        print("Waiting for ready...")
        provider.wait_for_ready(base_url)
        print("✓ Ready!")

        print("Testing health...")
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        print("✓ Health check passed")

        print("\n✓ Custom port test passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

    finally:
        if provider is not None:
            provider.stop_container()
            print("✓ Cleaned up\n")


def test_provider_with_env_vars():
    """Test provider with environment variables."""
    print("=" * 60)
    print("LocalDockerProvider with Environment Variables Test")
    print("=" * 60)
    print()

    provider = None

    try:
        provider = LocalDockerProvider()

        print("Starting container with environment variables...")
        base_url = provider.start_container(
            "echo-env:latest", env_vars={"DEBUG": "true", "LOG_LEVEL": "info"}
        )
        print(f"✓ Started at: {base_url}")

        print("Waiting for ready...")
        provider.wait_for_ready(base_url)
        print("✓ Ready!")

        print("Testing health...")
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        print("✓ Health check passed")

        print("\n✓ Environment variables test passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

    finally:
        if provider is not None:
            provider.stop_container()
            print("✓ Cleaned up\n")


if __name__ == "__main__":
    print()
    print("🐳 LocalDockerProvider Test Suite")
    print()

    results = []

    # Run basic test
    results.append(("Basic End-to-End", test_local_docker_provider()))

    # Run custom port test
    results.append(("Custom Port", test_provider_with_custom_port()))

    # Run environment variables test
    results.append(("Environment Variables", test_provider_with_env_vars()))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25} {status}")
    print("=" * 60)

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
