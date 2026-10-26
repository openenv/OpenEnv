# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for Unity ML-Agents environment server.

=============================================================================
HOW TO RUN THESE TESTS
=============================================================================

Running Tests:
    # From the OpenEnv repository root directory:

    # Run all Unity environment tests
    pytest tests/envs/test_unity_environment.py -v

    # Run with longer timeout (recommended for first run - downloads ~500MB binaries)
    pytest tests/envs/test_unity_environment.py -v --timeout=300

    # Run with print output visible
    pytest tests/envs/test_unity_environment.py -v -s

=============================================================================
"""

import os
import subprocess
import sys
import time

import pytest
import requests

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Check if mlagents-envs is installed
try:
    import mlagents_envs  # noqa: F401

    MLAGENTS_INSTALLED = True
except ImportError:
    MLAGENTS_INSTALLED = False

# Skip all tests if mlagents-envs is not installed
pytestmark = pytest.mark.skipif(
    not MLAGENTS_INSTALLED, reason="mlagents-envs not installed"
)

from envs.unity_env.client import UnityEnv
from envs.unity_env.models import UnityAction, UnityObservation, UnityState


@pytest.fixture(scope="module")
def server():
    """Starts the Unity environment server as a background process.

    Note: Unity environments can take 30-120 seconds to initialize on first run
    due to binary downloads (~500MB). Subsequent runs use cached binaries.
    """
    # Define paths for subprocess environment
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SRC_PATH = os.path.join(ROOT_DIR, "src")
    PORT = 8011  # Use a unique port to avoid conflicts
    localhost = f"http://localhost:{PORT}"

    print(f"\n--- Starting Unity ML-Agents server on port {PORT} ---")

    server_env = {
        **os.environ,
        "PYTHONPATH": f"{SRC_PATH}:{ROOT_DIR}",
        "UNITY_NO_GRAPHICS": "1",  # Run headless for testing
        "UNITY_TIME_SCALE": "20",  # Speed up for faster tests
        # Bypass proxy for localhost
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }

    # Use uvicorn directly instead of gunicorn for simpler setup
    uvicorn_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "envs.unity_env.server.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]

    # Create a log file for server output
    log_file = os.path.join(ROOT_DIR, "tests", "unity_server_test.log")
    log_handle = open(log_file, "w")

    server_process = subprocess.Popen(
        uvicorn_command,
        env=server_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=ROOT_DIR,
    )

    # Wait for server to become healthy
    # Note: Initial startup is quick, but first reset() will download binaries
    print("\n--- Waiting for server to become healthy... ---")
    time.sleep(2)  # Give server time to fully initialize

    # Bypass proxy for localhost requests
    no_proxy = {"http": None, "https": None}

    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5, proxies=no_proxy)
            if response.status_code == 200:
                is_healthy = True
                print("✅ Server is running and healthy!")
                break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i + 1}/12: Server not ready ({e}), waiting 5 seconds...")
            time.sleep(5)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        server_process.kill()
        log_handle.close()
        with open(log_file, "r") as f:
            print(f.read())
        pytest.skip("Server failed to start")

    yield localhost

    # Cleanup
    print("\n--- Cleaning up server ---")
    try:
        server_process.terminate()
        server_process.wait(timeout=10)
        print("✅ Server process terminated")
    except subprocess.TimeoutExpired:
        server_process.kill()
        print("✅ Server process killed")
    except ProcessLookupError:
        print("✅ Server process was already terminated")
    finally:
        log_handle.close()


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_endpoint_returns_200(self, server):
        """Test that the health endpoint returns 200 OK."""
        response = requests.get(
            f"{server}/health", proxies={"http": None, "https": None}
        )
        assert response.status_code == 200

    def test_health_endpoint_returns_status(self, server):
        """Test that the health endpoint returns status field."""
        response = requests.get(
            f"{server}/health", proxies={"http": None, "https": None}
        )
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestUnityEnvClient:
    """Tests for the UnityEnv client."""

    # Note: This test may take up to 3 minutes on first run (binary download)
    def test_reset_returns_valid_observation(self, server):
        """Test that reset() returns a valid observation."""
        with UnityEnv(base_url=server) as env:
            result = env.reset(env_id="PushBlock")

            assert result is not None
            assert result.observation is not None
            assert isinstance(result.observation, UnityObservation)
            assert hasattr(result.observation, "vector_observations")
            assert hasattr(result.observation, "behavior_name")
            assert hasattr(result.observation, "action_spec_info")
            assert result.observation.done is False

    def test_reset_with_different_environments(self, server):
        """Test that reset() can switch between environments."""
        with UnityEnv(base_url=server) as env:
            # Reset to PushBlock
            result1 = env.reset(env_id="PushBlock")
            assert result1.observation.behavior_name is not None
            assert "Push" in result1.observation.behavior_name

            # Reset to 3DBall
            result2 = env.reset(env_id="3DBall")
            assert result2.observation.behavior_name is not None
            assert "3DBall" in result2.observation.behavior_name

    def test_step_discrete_action(self, server):
        """Test that step() works with discrete actions (PushBlock)."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")

            # PushBlock has 7 discrete actions (0-6)
            action = UnityAction(discrete_actions=[1])  # Move forward
            result = env.step(action)

            assert result is not None
            assert result.observation is not None
            assert isinstance(result.reward, (int, float)) or result.reward is None
            assert isinstance(result.done, bool)

    def test_step_continuous_action(self, server):
        """Test that step() works with continuous actions (3DBall)."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="3DBall")

            # 3DBall has 2 continuous actions
            action = UnityAction(continuous_actions=[0.5, -0.3])
            result = env.step(action)

            assert result is not None
            assert result.observation is not None
            assert isinstance(result.reward, (int, float)) or result.reward is None
            assert isinstance(result.done, bool)

    def test_step_multiple_times(self, server):
        """Test that step() can be called multiple times."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")

            for i in range(10):
                action = UnityAction(discrete_actions=[i % 7])
                result = env.step(action)
                assert result.observation is not None

    def test_state_endpoint(self, server):
        """Test that state() returns valid state information."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")

            state = env.state()

            assert state is not None
            assert isinstance(state, UnityState)
            assert hasattr(state, "env_id")
            assert hasattr(state, "episode_id")
            assert hasattr(state, "step_count")
            assert hasattr(state, "behavior_name")
            assert hasattr(state, "action_spec")
            assert state.env_id == "PushBlock"

    def test_step_count_increments(self, server):
        """Test that step count increments correctly."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")

            state1 = env.state()
            assert state1.step_count == 0

            action = UnityAction(discrete_actions=[1])
            env.step(action)

            state2 = env.state()
            assert state2.step_count == 1

            env.step(action)

            state3 = env.state()
            assert state3.step_count == 2

    def test_reset_resets_step_count(self, server):
        """Test that reset() resets the step count."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")

            # Take some steps
            action = UnityAction(discrete_actions=[1])
            for _ in range(5):
                env.step(action)

            state1 = env.state()
            assert state1.step_count == 5

            # Reset
            env.reset(env_id="PushBlock")

            state2 = env.state()
            assert state2.step_count == 0

    def test_episode_id_changes_on_reset(self, server):
        """Test that episode ID changes on each reset."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")
            state1 = env.state()

            env.reset(env_id="PushBlock")
            state2 = env.state()

            assert state1.episode_id != state2.episode_id

    def test_action_spec_info(self, server):
        """Test that action spec info is provided correctly."""
        with UnityEnv(base_url=server) as env:
            # PushBlock - discrete actions
            result = env.reset(env_id="PushBlock")
            action_spec = result.observation.action_spec_info

            assert action_spec is not None
            assert action_spec.get("is_discrete") is True
            assert action_spec.get("discrete_size") == 1
            assert len(action_spec.get("discrete_branches", [])) > 0

            # 3DBall - continuous actions
            result = env.reset(env_id="3DBall")
            action_spec = result.observation.action_spec_info

            assert action_spec is not None
            assert action_spec.get("is_continuous") is True
            assert action_spec.get("continuous_size") == 2


class TestUnityEnvModels:
    """Tests for Unity environment models."""

    def test_unity_action_discrete(self):
        """Test creating a discrete UnityAction."""
        action = UnityAction(discrete_actions=[1, 2, 3])
        assert action.discrete_actions == [1, 2, 3]
        assert action.continuous_actions is None

    def test_unity_action_continuous(self):
        """Test creating a continuous UnityAction."""
        action = UnityAction(continuous_actions=[0.5, -0.3, 1.0])
        assert action.continuous_actions == [0.5, -0.3, 1.0]
        assert action.discrete_actions is None

    def test_unity_action_with_metadata(self):
        """Test creating a UnityAction with metadata."""
        action = UnityAction(
            discrete_actions=[1], metadata={"test": "value", "number": 42}
        )
        assert action.discrete_actions == [1]
        assert action.metadata == {"test": "value", "number": 42}

    def test_unity_observation_creation(self):
        """Test creating a UnityObservation."""
        obs = UnityObservation(
            vector_observations=[1.0, 2.0, 3.0],
            behavior_name="TestBehavior",
            done=False,
            reward=0.5,
            action_spec_info={"is_discrete": True},
            observation_spec_info={"count": 1},
        )
        assert obs.vector_observations == [1.0, 2.0, 3.0]
        assert obs.behavior_name == "TestBehavior"
        assert obs.done is False
        assert obs.reward == 0.5

    def test_unity_state_creation(self):
        """Test creating a UnityState."""
        state = UnityState(
            episode_id="test-episode-123",
            step_count=10,
            env_id="PushBlock",
            behavior_name="PushBlockBehavior",
            action_spec={"is_discrete": True},
            observation_spec={"count": 1},
            available_envs=["PushBlock", "3DBall"],
        )
        assert state.episode_id == "test-episode-123"
        assert state.step_count == 10
        assert state.env_id == "PushBlock"
        assert state.available_envs == ["PushBlock", "3DBall"]


class TestAvailableEnvironments:
    """Tests for available environments functionality."""

    def test_available_environments_static_method(self):
        """Test the static available_environments method."""
        envs = UnityEnv.available_environments()
        assert isinstance(envs, list)
        assert "PushBlock" in envs
        assert "3DBall" in envs

    def test_available_envs_from_state(self, server):
        """Test getting available environments from state."""
        with UnityEnv(base_url=server) as env:
            env.reset(env_id="PushBlock")
            state = env.state()

            assert state.available_envs is not None
            assert isinstance(state.available_envs, list)
            assert len(state.available_envs) > 0
            assert "PushBlock" in state.available_envs
            assert "3DBall" in state.available_envs
