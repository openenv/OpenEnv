"""Unit tests for BrowserGym cleanup behavior."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

pytest.importorskip("gymnasium", reason="gymnasium is not installed")

from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment


class _BrokenGymEnv:
    def close(self) -> None:
        raise RuntimeError("close failed")


def test_browsergym_close_swallows_cleanup_errors() -> None:
    """HTTP cleanup should not mask a successful reset/step result."""
    env = BrowserGymEnvironment.__new__(BrowserGymEnvironment)
    env.gym_env = _BrokenGymEnv()

    BrowserGymEnvironment.close(env)
