# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transforms specific to coding environments."""

import ast
import re

from openenv.core.env_server.base_transforms import CompositeTransform
from openenv.core.env_server.interfaces import Transform
from openenv.core.env_server.types import Observation

from ..models import CodeObservation


class CodeSafetyTransform(Transform):
    """Evaluates code safety and assigns penalties for dangerous patterns."""

    def __init__(self, penalty: float = -1.0):
        self.penalty = penalty
        # \b stops a match inside a longer name such as my_open( or model.eval(
        # style names; (?<![\w.]) keeps eval( and exec( from matching inside
        # identifiers or method calls. popen/Popen are listed because \bopen(
        # no longer catches them.
        self.dangerous_patterns = [
            r"\bimport\s+os\b",
            r"\bimport\s+subprocess\b",
            r"(?<![\w.])eval\(",
            r"(?<![\w.])exec\(",
            r"__import__",
            r"\bopen\(",
            r"\bpopen\(",
            r"\bPopen\(",
        ]

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        if "last_code" in observation.metadata:
            code = observation.metadata["last_code"]
            for pattern in self.dangerous_patterns:
                if re.search(pattern, code):
                    observation.reward = self.penalty
                    observation.metadata["safety_violation"] = pattern
                    break
            else:
                if observation.reward is None:
                    observation.reward = 0.0

        return observation


class CodeQualityTransform(Transform):
    """Evaluates and rewards code quality metrics."""

    def __init__(
        self,
        concise_bonus: float = 0.1,
        max_length_threshold: int = 100,
        syntax_penalty: float = -0.2,
    ):
        self.concise_bonus = concise_bonus
        self.max_length_threshold = max_length_threshold
        self.syntax_penalty = syntax_penalty

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        quality_score = 0.0

        if "last_code" in observation.metadata:
            code = observation.metadata["last_code"]

            # Reward concise code
            if len(code.strip()) <= self.max_length_threshold:
                quality_score += self.concise_bonus

            # Check syntax (redundant but useful for quality assessment)
            try:
                ast.parse(code)
            except SyntaxError:
                quality_score += self.syntax_penalty

        # Add to existing reward
        if observation.reward is None:
            observation.reward = quality_score
        else:
            observation.reward += quality_score

        return observation


def create_safe_coding_transform() -> CompositeTransform:
    """Create a transform focused on safe coding practices and quality."""
    return CompositeTransform([CodeSafetyTransform(), CodeQualityTransform()])
