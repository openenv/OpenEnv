# Type definitions for EnvTorch
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

# Generic type for observations
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


@dataclass
class StepResult(Generic[ObsT]):
    """
    Represents the result of one environment step.

    Attributes:
        observation:
            The environment's observation after the action.
        reward (`float`, *optional*):
            Scalar reward for this step.
        done (`bool`, *optional*, defaults to `False`):
            Whether the episode is finished.
        metadata (`dict`, *optional*):
            Additional metadata returned alongside the observation.
    """

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None
