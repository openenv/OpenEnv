# SPDX-License-Identifier: BSD-3-Clause

"""Configuration types for external agentic harnesses (RFC 005)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HarnessTransport(str, Enum):
    """How a harness exposes its interface to the adapter."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "http"
    MCP = "mcp"


class HarnessConfig(BaseModel):
    """
    Configuration for an external agentic harness process.

    Describes how to start and configure a harness (e.g. OpenClaw, Claude
    Code) that runs as a subprocess inside an OpenEnv container. Consumed by
    [`~openenv.core.harness.adapter.AgenticHarnessAdapter`] implementations.

    Args:
        name (`str`):
            Harness identity, e.g. `"openclaw"` or `"claude-code"`.
        command (`list[str]`):
            Command line used to start the harness process. Must be non-empty.
        working_directory (`str`, *optional*, defaults to `"/workspace"`):
            Directory the harness operates in inside the container.
        env_vars (`dict[str, str]`, *optional*):
            Additional environment variables for the harness process.
        transport (`HarnessTransport`, *optional*, defaults to `HarnessTransport.STDIO`):
            How the harness exposes its interface.
        mcp_config_path (`str`, *optional*):
            Where to write the MCP configuration the harness reads at startup.
            When `None`, the adapter picks a harness-specific default.
        startup_timeout_s (`float`, *optional*, defaults to `30.0`):
            Maximum wall-clock time to wait for the harness to become ready.
        session_timeout_s (`float`, *optional*, defaults to `600.0`):
            Maximum wall-clock time a single conversational turn may take.
            Bounds one `step()`; it is not an episode-wide limit.
        model (`str`, *optional*):
            Override for the harness's default LLM model.
        api_key_env_var (`str`, *optional*):
            Name of the environment variable holding the LLM API key.

    Examples:

    ```python
    config = HarnessConfig(name="openclaw", command=["openclaw", "run"])
    ```
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    command: list[str]
    working_directory: str = "/workspace"
    env_vars: dict[str, str] = Field(default_factory=dict)
    transport: HarnessTransport = HarnessTransport.STDIO
    mcp_config_path: Optional[str] = None
    startup_timeout_s: float = 30.0
    session_timeout_s: float = 600.0
    model: Optional[str] = None
    api_key_env_var: Optional[str] = None

    @field_validator("command")
    @classmethod
    def _command_must_be_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("command must contain at least one element")
        return value


__all__ = ["HarnessConfig", "HarnessTransport"]
