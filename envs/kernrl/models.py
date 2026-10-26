# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the kernrl GPU kernel optimization environment.

The kernrl environment enables training LLMs to write optimized CUDA/Triton
kernels by providing real hardware feedback.
"""

from typing import Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation, State


class KernelAction(Action):
    """Action for the kernrl environment - kernel code submission."""

    code: str = Field(
        ..., min_length=1, description="The CUDA/Triton kernel code to evaluate"
    )


class KernelObservation(Observation):
    """Observation from the kernrl environment - evaluation results."""

    problem_id: str = Field(..., description="Unique identifier for the problem")
    problem_description: str = Field(
        ..., description="Full problem description with requirements"
    )
    reference_code: str = Field(..., description="PyTorch reference implementation")
    gpu_info: str = Field(..., description="GPU device information")
    turn: int = Field(default=0, ge=0, description="Current turn number")
    max_turns: int = Field(default=10, ge=1, description="Maximum turns allowed")
    feedback: str = Field(default="", description="Evaluation feedback for the agent")
    # Evaluation results
    compilation_success: bool = Field(
        default=False, description="Whether the code compiled successfully"
    )
    compilation_error: Optional[str] = Field(
        default=None, description="Compilation error message if failed"
    )
    correctness_pass: Optional[bool] = Field(
        default=None, description="Whether output matches reference"
    )
    max_diff: Optional[float] = Field(
        default=None, description="Maximum difference from reference output"
    )
    speedup: Optional[float] = Field(
        default=None, description="Speedup factor vs PyTorch baseline"
    )


class KernelState(State):
    """State for the kernrl environment."""

    problem_id: Optional[str] = Field(default=None, description="Current problem ID")
    turn: int = Field(default=0, ge=0, description="Current turn number")
    max_turns: int = Field(default=10, ge=1, description="Maximum turns allowed")
    best_speedup: float = Field(
        default=0.0, ge=0.0, description="Best speedup achieved so far"
    )
    solved: bool = Field(
        default=False, description="Whether the problem has been solved"
    )
