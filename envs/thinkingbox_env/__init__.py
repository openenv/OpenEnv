"""Expose the public ThinkingBox wire types and client.

The client is loaded lazily so importing server modules never crosses the
OpenEnv client-server boundary.
"""

from typing import Any

from openenv.core import ListToolsAction

from .models import (
    CallToolAction,
    SubmitMessageAction,
    SubmittedToolCall,
    ThinkingBoxAction,
    ThinkingBoxExecutionProvenance,
    ThinkingBoxObservation,
    ThinkingBoxState,
    ToolCallResult,
)

__all__ = [
    "CallToolAction",
    "ListToolsAction",
    "SubmitMessageAction",
    "SubmittedToolCall",
    "ThinkingBoxAction",
    "ThinkingBoxEnv",
    "ThinkingBoxExecutionProvenance",
    "ThinkingBoxObservation",
    "ThinkingBoxState",
    "ToolCallResult",
]


def __getattr__(name: str) -> Any:
    """Load optional package exports without importing client code on servers."""
    if name == "ThinkingBoxEnv":
        from .client import ThinkingBoxEnv

        globals()[name] = ThinkingBoxEnv
        return ThinkingBoxEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return package exports, including lazily loaded attributes."""
    return sorted(set(globals()) | set(__all__))
