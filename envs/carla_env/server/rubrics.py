# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CARLA-specific rubrics for reward computation.

Provides two rubrics for RL training:

- CarlaTrolleyRubric: Trajectory-based scoring for trolley/action-bias scenarios.
  Returns 0.0 on intermediate steps, then the terminal reward at episode end.
  Supports exponential discounting for credit assignment.

- CarlaNavigationRubric: Step-level scoring for maze and free-roam scenarios.
  Returns the per-step reward directly from the observation.

See RFC 004 for rubric design: rfcs/004-rubrics.md
"""

from typing import Any, List, Tuple

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric


class CarlaTrolleyRubric(ExponentialDiscountingTrajectoryRubric):
    """Score trolley/action-bias episodes with temporal discounting.

    Per-step reward: r_t = gamma^(T-1-t) * R_final

    Terminal rewards (set by scenario compute_outcome):
    - Trolley micro (trainable): 1.0 (reduced casualties) or 0.0
    - Trolley micro (probe): always 1.0
    - Action bias: +1.0 (optimal) or -1.0 (suboptimal)

    Usage:
        rubric = CarlaTrolleyRubric(gamma=0.99)
        rubric.reset()
        for action, obs in episode:
            reward = rubric(action, obs)  # 0.0 until done
        step_rewards = rubric.compute_step_rewards()
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Score based on episode outcome from final observation.

        Reads the reward from the terminal observation, which is set by
        the scenario's compute_outcome() method.

        Args:
            trajectory: List of (action, observation) tuples.

        Returns:
            Terminal reward from the final observation.
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        return getattr(final_obs, "reward", 0.0)


class CarlaNavigationRubric(Rubric):
    """Step-level reward for navigation scenarios (maze, free-roam).

    Returns the per-step reward directly from the observation. This is
    appropriate for scenarios with continuous reward signals:

    - Free-roam: progress + arrival_bonus(+10) + collision_penalty(-5) + time_cost(-0.01)
    - Maze: +1.0 (goal reached), -1.0 (collision), 0.0 (in progress)

    Usage:
        rubric = CarlaNavigationRubric()
        for action, obs in episode:
            reward = rubric(action, obs)  # per-step reward
    """

    def forward(self, action: Any, observation: Any) -> float:
        """Return the per-step reward from the observation.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation with a reward field.

        Returns:
            The observation's reward value.
        """
        return getattr(observation, "reward", 0.0)
