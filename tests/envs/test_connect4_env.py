# SPDX-License-Identifier: BSD-3-Clause

"""Test Connect4 environment client and server integration.

NOTE: This is a legacy test file using unittest patterns with manual server lifecycle.
For comprehensive Connect4 tests, see test_websockets.py::TestConnect4Environment.
"""

import os
import sys

import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import signal
import subprocess
import time
import unittest

import requests
from envs.connect4_env import Connect4Action, Connect4Env, Connect4Observation


# Skip this legacy test file - comprehensive tests in test_websockets.py
pytestmark = pytest.mark.skip(
    reason="Legacy test file - see test_websockets.py for comprehensive Connect4 tests"
)


class TestConnect4(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        self.client = None

        self.actions = []
        super().__init__(methodName)

    def test_setup_server(self):
        self.server_process = subprocess.Popen(
            ["python", "-m", "envs.connect4_env.server.app"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give it a few seconds to start
        time.sleep(3)

    def check_server_running(self):
        try:
            # Attempt to ping the server
            response = requests.get(
                "http://127.0.0.1:8000/health"
            )  # or "/" depending on your app
            self.assertEqual(response.status_code, 200)

        except requests.ConnectionError:
            self.fail("Server did not start or is unreachable")

    def test_connect4_env_client(self):
        self.test_setup_server()
        self.check_server_running()

        self.client = Connect4Env(base_url="http://127.0.0.1:8000")

        assert isinstance(self.client, Connect4Env)

    def test_connect4_initial_state(self):
        self.test_connect4_env_client()

        result = self.client.reset()

        observation = result.observation

        assert isinstance(observation, Connect4Observation)

        assert isinstance(observation.board, list)
        assert isinstance(observation.legal_actions, list)
        assert isinstance(observation.done, bool)
        assert isinstance(observation.reward, float)

        assert len(observation.board) == 6  # 6 rows
        assert all(len(row) == 7 for row in observation.board)  # 7 columns
        assert (
            len(observation.legal_actions) == 7
        )  # All columns should be legal at start
        assert not observation.done
        assert observation.reward == 0.0

        if isinstance(observation.legal_actions, float):
            self.actions = observation.legal_actions

    def check_valid_action(self, action):
        legal_actions = self.actions

        if self.assertIn(
            action, legal_actions, f"Action {action} is not legal in the current state."
        ):
            return True

        return False

    def step_action(self, column):
        valid = self.check_valid_action(column)

        assert isinstance(valid, bool)

        if valid:
            action = Connect4Action(column=column)

            result = self.client.step(action)

            assert isinstance(result, object)

            observation = result.observation
            assert isinstance(observation, Connect4Observation)
            assert isinstance(observation.board, list)
            assert isinstance(observation.legal_actions, list)
            assert isinstance(observation.done, bool)
            assert isinstance(observation.reward, float)

        return result

    def tearDown(self):
        if self.server_process:
            # Try terminating the process gracefully
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(self.server_process.pid, signal.SIGKILL)

            # Close the pipes to avoid ResourceWarnings
            for stream in [
                self.server_process.stdin,
                self.server_process.stdout,
                self.server_process.stderr,
            ]:
                if stream and not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
