# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the REPL Environment.

The REPL environment provides a Python REPL for training language models
on code execution tasks, based on the Recursive Language Models (RLM) paradigm.

Supports two finalization patterns:
1. RLM-style: print('FINAL(answer)') or print('FINAL_VAR(var_name)')
2. Prime Intellect style: answer = {"content": "...", "ready": True}
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class REPLAction(Action):
    """Action containing Python code to execute in the REPL.

    Supports multiple finalization patterns:
    1. RLM-style: print('FINAL(answer)') or print('FINAL_VAR(var_name)') in code
    2. Prime Intellect style: answer = {"content": "...", "ready": True} in namespace
    3. Explicit: Set is_final=True with final_answer
    """

    code: str = Field(default="", description="Python code to execute")
    is_final: bool = Field(
        default=False,
        description="Whether this action signals the final answer",
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final answer if is_final=True"
    )


class CodeBlockResult(BaseModel):
    """Result of executing a single code block."""

    stdout: str = Field(default="", description="Standard output from execution")
    stderr: str = Field(default="", description="Standard error from execution")
    locals_snapshot: Dict[str, str] = Field(
        default_factory=dict,
        description="String representations of new/modified variables",
    )
    execution_time: float = Field(
        default=0.0, ge=0, description="Execution time in seconds"
    )
    success: bool = Field(default=True, description="Whether execution succeeded")
    exception: Optional[str] = Field(
        default=None, description="Exception message if execution failed"
    )


class REPLObservation(Observation):
    """Observation returned after code execution in the REPL."""

    result: CodeBlockResult = Field(
        default_factory=CodeBlockResult, description="Result of code execution"
    )
    context_preview: Optional[str] = Field(
        default=None,
        description="Preview of the context (first N chars) if context is loaded",
    )
    context_length: int = Field(
        default=0, ge=0, description="Total length of context in characters"
    )
    available_variables: List[str] = Field(
        default_factory=list,
        description="List of variable names available in the namespace",
    )
    iteration: int = Field(default=0, ge=0, description="Current iteration number")
    max_iterations: int = Field(
        default=30, ge=1, description="Maximum allowed iterations"
    )


class REPLState(State):
    """Extended state for REPL environment."""

    context: Optional[str] = Field(
        default=None, description="The context/problem to work with"
    )
    task_prompt: Optional[str] = Field(
        default=None, description="The task description to solve"
    )
    iteration: int = Field(default=0, ge=0, description="Current iteration number")
    max_iterations: int = Field(
        default=30, ge=1, description="Max iterations before termination"
    )
    namespace_keys: List[str] = Field(
        default_factory=list, description="Variables currently in namespace"
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final answer if episode is complete"
    )
    total_execution_time: float = Field(
        default=0.0, ge=0, description="Total code execution time in seconds"
    )
