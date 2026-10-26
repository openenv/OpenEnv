"""
AWM-specific Pydantic models for action and observation types.
"""

from typing import Annotated, Any

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field, field_validator, TypeAdapter


_AWMActionUnion = Annotated[
    ListToolsAction | CallToolAction,
    Field(discriminator="type"),
]
_awm_action_adapter = TypeAdapter(_AWMActionUnion)


class AWMAction(Action):
    """Discriminated union action type for AWM.

    model_validate() returns the concrete ListToolsAction or CallToolAction
    (not an AWMAction instance), which is what AWMEnvironment.step() expects.
    """

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Action:  # type: ignore[override]
        return _awm_action_adapter.validate_python(obj)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return _awm_action_adapter.json_schema(**kwargs)


class AWMObservation(Observation):
    """
    Observation with AWM-specific fields promoted to top level.
    model_dump() excludes None-valued fields by default so that keys like
    ``tool_name=None`` do not appear in the wire payload.
    This is because the generic MCPToolClient._parse_result() routes observations based on key presence (e.g. ``"tool_name" in obs_data``). We may need to modify the MCPToolClient in the future. Currently, I try to avoid modifying any openenv code.
    """

    model_config = ConfigDict(extra="forbid")

    reward_type: str | None = Field(
        default=None,
        description="Reward classification label for this step/episode outcome",
    )
    scenario: str | None = Field(default=None, description="Current scenario name")
    task: str | None = Field(default=None, description="Current task description")
    task_idx: int | None = Field(default=None, description="Current task index")
    has_verifier: dict | bool | None = Field(
        default=None,
        description="Verifier support info: {sql: bool, code: bool} or legacy bool",
    )

    @field_validator("has_verifier", mode="before")
    @classmethod
    def _convert_bool_to_dict(cls, v: Any) -> dict | None:
        """Convert legacy bool format to new dict format."""
        if v is None:
            return None
        if isinstance(v, bool):
            # Legacy format: True means both modes available (conservative assumption)
            return {"sql": v, "code": v} if v else None
        return v

    num_tools: int | None = Field(
        default=None, description="Number of tools discovered"
    )
    tool_name: str | None = Field(default=None, description="Name of the tool called")
    tool_result: Any = Field(default=None, description="Result from the tool call")
    error: str | None = Field(default=None, description="Error message if any")
    warning: str | None = Field(default=None, description="Warning message if any")
    verify_result: dict | None = Field(
        default=None, description="Verifier output on episode end"
    )
    steps_taken: int | None = Field(
        default=None, description="Steps taken in this episode"
    )
    scenarios: list | None = Field(
        default=None, description="List of all scenarios (from __list_scenarios__)"
    )
    total: int | None = Field(default=None, description="Total number of scenarios")
    trajectory_path: str | None = Field(
        default=None, description="Path to saved trajectory JSON file"
    )
    session_dir: str | None = Field(
        default=None, description="Session directory path (when keep_session=True)"
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class AWMListToolsObservation(ListToolsObservation):
    """ListToolsObservation with AWM error field promoted to top level."""

    model_config = ConfigDict(extra="forbid")

    error: str | None = Field(default=None, description="Error message if any")
