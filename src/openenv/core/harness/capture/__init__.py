"""Token-level capture for agentic rollouts.

An OpenAI-spec proxy that sits between a coding agent and an inference engine, recording the exact
token ids and logprobs of every model call so a rollout can be trained on.

The core idea is that nothing is ever tokenised locally. The engine returns `prompt_token_ids`, so
turn k+1's prompt IS the canonical tokenisation of everything before it, and turns are linked by
exact token prefix. Re-rendering a prompt offline drifts from what the model actually saw, and a
drifted prompt silently fragments one long conversation into several short ones.

Four wire dialects are supported (chat-completions, Responses, Anthropic Messages, Google
generateContent) because coding agents did not agree on one. Validated across 16 harnesses, each
cross-checked against the harness's own trace.
"""

from .contract import (
    measure_retokenization_skew,
    to_trace_entries,
    to_training_trace,
    to_turn_records,
)
from .detection import APIType, detect
from .graph import RolloutGraph, TurnNode
from .runner import CaptureServer
from .upstream import InferenceClient, UpstreamError
from .validate_llm import LLMReport, require_llm, validate_llm

__all__ = [
    "CaptureServer",
    "to_turn_records",
    "to_trace_entries",
    "to_training_trace",
    "measure_retokenization_skew",
    "APIType",
    "detect",
    "RolloutGraph",
    "TurnNode",
    "InferenceClient",
    "UpstreamError",
    "LLMReport",
    "validate_llm",
    "require_llm",
]
