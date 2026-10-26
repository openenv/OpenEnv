# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


# Type aliases
Scalar = Union[int, float, bool]


# =============================================================================
# Enums for Type Safety
# =============================================================================


class ServerMode(str, Enum):
    """Server operation mode."""

    SIMULATION = "simulation"
    PRODUCTION = "production"


class HealthStatus(str, Enum):
    """Server health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class WSErrorCode(str, Enum):
    """WebSocket error codes for structured error handling."""

    INVALID_JSON = "INVALID_JSON"
    UNKNOWN_TYPE = "UNKNOWN_TYPE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    CAPACITY_REACHED = "CAPACITY_REACHED"
    FACTORY_ERROR = "FACTORY_ERROR"
    SESSION_ERROR = "SESSION_ERROR"


# =============================================================================
# Core Types
# =============================================================================


class Action(BaseModel):
    """Base class for all environment actions.

    All action subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field assignment
        arbitrary_types_allowed=True,  # Allow numpy arrays, torch tensors, etc.
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the action"
    )


class Observation(BaseModel):
    """Base class for all environment observations.

    All observation subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: bool | int | float | None = Field(
        default=None, description="Reward signal from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the observation"
    )


class ResetRequest(BaseModel):
    """Request model for environment reset."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for custom reset parameters
        json_schema_extra={"examples": [{"seed": 42, "episode_id": "episode-001"}, {}]},
    )

    seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducible episodes"
    )
    episode_id: Optional[str] = Field(
        default=None, max_length=255, description="Custom episode identifier"
    )


class ResetResponse(BaseModel):
    """Response model for environment reset."""

    model_config = ConfigDict(extra="forbid")

    observation: Dict[str, Any] = Field(
        ..., description="Initial observation from the environment"
    )
    reward: Optional[float] = Field(
        default=None, description="Initial reward (typically None at reset)"
    )
    done: bool = Field(
        default=False, description="Whether episode is already done (typically False)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the environment"
    )


class StepRequest(BaseModel):
    """Request model for environment step."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for custom step parameters
        json_schema_extra={
            "examples": [
                {"action": {"value": 1}, "timeout_s": 30.0},
                {"action": {"value": 1}, "render": True, "verbose": False},
            ]
        },
    )

    action: Dict[str, Any] = Field(
        ...,
        description="Action to execute, must conform to environment's action schema",
    )
    timeout_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional timeout in seconds for action execution",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional request identifier for tracking",
    )


class StepResponse(BaseModel):
    """Response model for environment step."""

    model_config = ConfigDict(extra="forbid")

    observation: Dict[str, Any] = Field(
        ..., description="Observation resulting from the action"
    )
    reward: Optional[float] = Field(
        default=None, description="Reward signal from the action"
    )
    done: bool = Field(default=False, description="Whether the episode has terminated")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the environment"
    )


class BaseMessage(BaseModel):
    """Base class for WebSocket messages with shared configuration."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class State(BaseModel):
    """Base class for environment state.

    Represents internal environment state, separate from observations.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    episode_id: Optional[str] = Field(
        default=None, description="Unique identifier for the current episode"
    )
    step_count: int = Field(
        default=0,
        ge=0,  # Greater than or equal to 0
        description="Number of steps taken in the current episode",
    )


class CodeExecResult(BaseMessage):
    """Result of code execution containing stdout, stderr, and exit code."""

    stdout: str = Field(description="Standard output from code execution")
    stderr: str = Field(description="Standard error from code execution")
    exit_code: int = Field(description="Exit code from code execution")


class EnvironmentMetadata(BaseMessage):
    """Metadata about an environment for documentation and UI purposes."""

    name: str = Field(description="Name of the environment")
    description: str = Field(description="Description of what the environment does")
    readme_content: Optional[str] = Field(
        default=None, description="Content of the README file for the environment"
    )
    version: Optional[str] = Field(
        default=None, description="Version of the environment"
    )
    author: Optional[str] = Field(default=None, description="Author of the environment")
    documentation_url: Optional[str] = Field(
        default=None, description="URL to the environment's documentation"
    )


class SchemaResponse(BaseMessage):
    """Response model for the combined schema endpoint."""

    action: Dict[str, Any] = Field(
        description="JSON schema for actions accepted by this environment"
    )
    observation: Dict[str, Any] = Field(
        description="JSON schema for observations returned by this environment"
    )
    state: Dict[str, Any] = Field(
        description="JSON schema for environment state objects"
    )
    reset_observation: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema for observations returned by reset"
    )

    @model_serializer(mode="wrap")
    def _omit_undeclared_reset_schema(self, handler):
        data = handler(self)
        if self.reset_observation is None:
            data.pop("reset_observation", None)
        return data


class HealthResponse(BaseMessage):
    """Response model for health check endpoint."""

    status: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Health status of the environment server",
    )


class ListTasksRequest(BaseMessage):
    """Request model for ORS-compatible task listing."""

    split: str = Field(description="Task split name")


class NumTasksRequest(BaseMessage):
    """Request model for ORS-compatible task counts."""

    split: str = Field(description="Task split name")


class GetTaskRequest(BaseMessage):
    """Request model for ORS-compatible task lookup."""

    split: str = Field(description="Task split name")
    index: int = Field(description="Task index within the split")


class GetTaskRangeRequest(BaseMessage):
    """Request model for ORS-compatible task range lookup."""

    split: str = Field(description="Task split name")
    start: Optional[int] = Field(default=None, description="Inclusive start index")
    stop: Optional[int] = Field(default=None, description="Exclusive stop index")


class WSResetMessage(BaseMessage):
    """WebSocket message to reset the environment."""

    type: Literal["reset"] = Field(default="reset", description="Message type")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional reset parameters (seed, episode_id, etc.)",
    )


class WSStepMessage(BaseMessage):
    """WebSocket message to execute a step."""

    type: Literal["step"] = Field(default="step", description="Message type")
    data: Dict[str, Any] = Field(
        ..., description="Action data conforming to environment's action schema"
    )


class WSStateMessage(BaseMessage):
    """WebSocket message to request current state."""

    type: Literal["state"] = Field(default="state", description="Message type")


class WSCloseMessage(BaseMessage):
    """WebSocket message to close the session."""

    type: Literal["close"] = Field(default="close", description="Message type")


# Discriminated union for incoming WebSocket messages
# Note: WSMCPMessage is defined in mcp_types.py to avoid circular imports
# The union here covers the core message types; MCP messages are handled separately
WSIncomingMessage = Annotated[
    WSResetMessage | WSStepMessage | WSStateMessage | WSCloseMessage,
    Field(discriminator="type"),
]


class WSObservationResponse(BaseModel):
    """WebSocket response containing an observation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["observation"] = Field(
        default="observation", description="Response type"
    )
    data: Dict[str, Any] = Field(description="Observation data")


class WSStateResponse(BaseModel):
    """WebSocket response containing environment state."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["state"] = Field(default="state", description="Response type")
    data: Dict[str, Any] = Field(description="State data")


class WSErrorResponse(BaseModel):
    """WebSocket response for errors."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = Field(default="error", description="Response type")
    data: Dict[str, Any] = Field(description="Error details including message and code")


class ConcurrencyConfig(BaseMessage):
    """Configuration for concurrent environment sessions."""

    max_concurrent_envs: int = Field(
        default=1,
        ge=1,
        description="Maximum number of concurrent WebSocket sessions allowed",
    )
    session_timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout in seconds for inactive sessions. None means no timeout.",
    )


class ServerCapacityStatus(BaseMessage):
    """Status of server capacity for concurrent sessions."""

    active_sessions: int = Field(
        ge=0,
        description="Number of currently active sessions",
    )
    max_sessions: int = Field(
        ge=1,
        description="Maximum number of allowed sessions",
    )

    @model_validator(mode="after")
    def check_capacity_bounds(self) -> "ServerCapacityStatus":
        if self.active_sessions > self.max_sessions:
            raise ValueError(
                f"active_sessions ({self.active_sessions}) cannot exceed "
                f"max_sessions ({self.max_sessions})"
            )
        return self

    @property
    def available_slots(self) -> int:
        """Number of available session slots."""
        return self.max_sessions - self.active_sessions

    @property
    def is_at_capacity(self) -> bool:
        """Whether the server has reached maximum capacity."""
        return self.available_slots == 0

    @classmethod
    def from_counts(cls, active: int, max_sessions: int) -> "ServerCapacityStatus":
        """Create status from active and max session counts."""
        return cls(
            active_sessions=active,
            max_sessions=max_sessions,
        )


class SessionInfo(BaseMessage):
    """Information about an active session."""

    session_id: str = Field(description="Unique identifier for the session")
    created_at: float = Field(description="Unix timestamp when the session was created")
    last_activity_at: float = Field(
        description="Unix timestamp of the last activity in the session"
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps executed in this session",
    )
    environment_type: str = Field(
        description="Environment type for this session (e.g. `CodingEnv`)"
    )
