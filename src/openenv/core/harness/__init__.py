# SPDX-License-Identifier: BSD-3-Clause

"""Harness integration helpers for training, evaluation, and wrapping agents.

This package hosts two complementary layers:

1. **Trainer-side rollout API** (``openenv.core.harness.rollout``): a harness
   drives an entire episode in one call (``run_white_box``/``run_black_box``)
   against a resource session. Used by ``openenv collect`` and the training
   tutorials.
2. **Turn-based agentic harness API** (RFC 005): an external harness such as
   OpenClaw or Claude Code runs inside the environment container, and each
   ``step()`` is one conversational turn. See
   [`~openenv.core.harness.environment.HarnessEnvironment`] and
   [`~openenv.core.harness.adapter.AgenticHarnessAdapter`].

Both layers are importable from ``openenv.core.harness``.
"""

from .adapter import (
    AgenticHarnessAdapter,
    HarnessError,
    HarnessNotRunningError,
    HarnessStartupError,
    HarnessTurnTimeoutError,
)
from .bridge import build_bridge_server, HarnessMCPBridge
from .config import HarnessConfig, HarnessTransport
from .environment import HarnessAction, HarnessEnvironment
from .events import (
    events_to_metadata,
    HarnessClientMessage,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
)
from .process import HarnessProcess
from .rollout import (  # noqa: F401  (_resolve_env_reward: private back-compat re-export)
    _resolve_env_reward,
    build_harness_rollout_func,
    CLIHarnessAdapter,
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    LoopOwningSession,
    MCPHarnessAdapter,
    Message,
    ModelStep,
    ModelStepResult,
    RESERVED_TOOL_NAMES,
    ResourceSession,
    ResourceSessionFactory,
    RolloutEvent,
    SessionMCPBridge,
    StepEnvSessionAdapter,
    ToolResult,
    ToolTraceEntry,
    TraceEntry,
    TrainableSession,
    VerifyResult,
)
from .tools import resolve_tool_conflicts
from .training import TrainingTrace, TrainingTurn

__all__ = [
    # Trainer-side rollout API (openenv.core.harness.rollout)
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "LoopOwningSession",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "TraceEntry",
    "TrainableSession",
    "TrainingTrace",
    "TrainingTurn",
    "VerifyResult",
    "build_harness_rollout_func",
    # Turn-based agentic harness API (RFC 005)
    "AgenticHarnessAdapter",
    "HarnessAction",
    "HarnessClientMessage",
    "HarnessConfig",
    "HarnessEnvironment",
    "HarnessError",
    "HarnessEvent",
    "HarnessEventType",
    "HarnessMCPBridge",
    "HarnessNotRunningError",
    "HarnessProcess",
    "HarnessResponse",
    "HarnessStartupError",
    "HarnessTransport",
    "HarnessTurnTimeoutError",
    "build_bridge_server",
    "events_to_metadata",
    "resolve_tool_conflicts",
]
