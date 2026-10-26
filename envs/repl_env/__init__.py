# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment for OpenEnv.

A Python REPL environment for training language models on code execution tasks,
based on the Recursive Language Models (RLM) paradigm.

This environment allows language models to:
- Execute Python code in a sandboxed REPL
- Work with large contexts loaded as variables
- Finalize answers via FINAL(), FINAL_VAR(), or answer dict pattern
- Optionally make recursive LLM calls via llm_query() / llm_query_batched()

Example:
    >>> from repl_env import LocalREPLEnv, REPLAction
    >>>
    >>> # Local in-process helper
    >>> env = LocalREPLEnv()
    >>>
    >>> # Reset with context
    >>> result = env.reset(context="Hello World", task_prompt="Count characters")
    >>>
    >>> # Execute code
    >>> result = env.execute("count = len(context)")
    >>> result = env.execute("print(f'FINAL({count})')")
    >>>
    >>> # Check result
    >>> print(f"Done: {result.done}, Answer: {result.observation.metadata['final_answer']}")
    >>>
    >>> env.close()

References:
    - RLM Paper: https://arxiv.org/abs/2512.24601
    - Prime Intellect Blog: https://www.primeintellect.ai/blog/rlm
    - Alex Zhang Blog: https://alexzhang13.github.io/blog/2025/rlm/
"""

from .client import REPLEnv
from .local import LocalREPLEnv
from .models import CodeBlockResult, REPLAction, REPLObservation, REPLState
from .prompts import (
    build_rlm_system_prompt,
    build_user_prompt,
    # Parsing utilities
    extract_code_blocks,
    format_observations,
    # Prompt building
    QueryMetadata,
    # System prompts
    RLM_SYSTEM_PROMPT,
    RLM_SYSTEM_PROMPT_QWEN,
)
from .rubrics import (
    CodeExecutionRubric,
    CustomMetricRubric,
    ExactMatchRubric,
    FuzzyMatchRubric,
    REPLRubric,
)
from .runner import LocalRLMRunner, RLMRunResult

__all__ = [
    # Models
    "REPLAction",
    "REPLObservation",
    "REPLState",
    "CodeBlockResult",
    # Client
    "REPLEnv",
    "LocalREPLEnv",
    "LocalRLMRunner",
    "RLMRunResult",
    # System prompts
    "RLM_SYSTEM_PROMPT",
    "RLM_SYSTEM_PROMPT_QWEN",
    # Prompt building
    "QueryMetadata",
    "build_rlm_system_prompt",
    "build_user_prompt",
    # Parsing utilities
    "extract_code_blocks",
    "format_observations",
    # Rubrics
    "REPLRubric",
    "ExactMatchRubric",
    "FuzzyMatchRubric",
    "CustomMetricRubric",
    "CodeExecutionRubric",
]
