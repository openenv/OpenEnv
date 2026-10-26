# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
envs/julia_env/julia_transforms.py
--------------------------------
Safety and quality transforms for Julia code.
"""

import re

# Support both in-repo and standalone imports
try:
    # In-repo imports
    from openenv.core.env_server.interfaces import Transform

    from ..models import JuliaObservation
except ImportError:
    from models import JuliaObservation

    # Standalone imports
    from openenv.core.env_server.interfaces import Transform


# -------------------------
# Safety Transform
# -------------------------
class JuliaSafetyTransform(Transform):
    """Detects dangerous Julia operations and penalizes them with a negative reward."""

    def __init__(self, penalty: float = -3.0):
        self.penalty = penalty
        self.dangerous_patterns = [
            r"run\(",
            r"read\(",
            r"write\(",
            r"unsafe_",
            r"ccall\(",
            r"Base\.exit",
            r"Base\.kill",
            r"rm\(",  # file deletion
            r"download\(",  # downloading
        ]

    def __call__(self, observation):
        # Only act on JuliaObservation objects
        if not isinstance(observation, JuliaObservation):
            return observation

        # Extract executed code from metadata (core_code + test_code)
        if observation.metadata:
            code = (
                observation.metadata.get("core_code", "")
                + "\n"
                + observation.metadata.get("test_code", "")
            )
        else:
            code = ""

        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                # Apply penalty and record violation
                observation.reward = (observation.reward or 0.0) + self.penalty
                observation.metadata = observation.metadata or {}
                observation.metadata["safety_violation"] = pattern
                return observation

        # Safe code gets neutral reward
        observation.reward = observation.reward or 0.0
        return observation


# -------------------------
# Factory
# -------------------------
def create_safe_julia_transform():
    """Creates safety transform for Julia code."""
    return JuliaSafetyTransform()
