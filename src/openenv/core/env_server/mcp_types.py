# SPDX-License-Identifier: BSD-3-Clause

"""
MCP (Model Context Protocol) type definitions for OpenEnv.

This module defines strongly typed models for MCP tool discovery and invocation,
following RFC 003. These types map MCP's REST-like API (tools/list, tools/call)
to Gym-style action types.

Key design decisions:
- Tool discovery (list_tools) does NOT require reset() first
- Reserved tool names (reset, step, state, close) are prohibited
- Both step() and WebSocket /mcp paths are supported
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .types import Action, BaseMessage, Observation


# =============================================================================
# JSON-RPC 2.0 Types
# =============================================================================


class JsonRpcErrorCode(int, Enum):
    """
    Standard JSON-RPC 2.0 error codes.

    See: https://www.jsonrpc.org/specification#error_object
    """

    # Standard JSON-RPC errors
    PARSE_ERROR = -32700  # Invalid JSON was received
    INVALID_REQUEST = -32600  # JSON is not a valid Request object
    METHOD_NOT_FOUND = -32601  # Method does not exist / is not available
    INVALID_PARAMS = -32602  # Invalid method parameter(s)
    INTERNAL_ERROR = -32603  # Internal JSON-RPC error

    # Server errors (reserved for implementation-defined errors)
    SERVER_ERROR = -32000  # Generic server error


class McpMethod(str, Enum):
    """Supported MCP method names."""

    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"


class JsonRpcError(BaseModel):
    """
    JSON-RPC 2.0 error object.

    See: https://www.jsonrpc.org/specification#error_object
    """

    model_config = ConfigDict(extra="forbid")

    code: int = Field(description="Error code indicating the error type")
    message: str = Field(description="Short description of the error")
    data: Optional[Any] = Field(
        default=None, description="Additional error information"
    )

    @classmethod
    def from_code(
        cls, code: JsonRpcErrorCode, message: Optional[str] = None, data: Any = None
    ) -> "JsonRpcError":
        """Create an error from a standard error code."""
        default_messages = {
            JsonRpcErrorCode.PARSE_ERROR: "Parse error",
            JsonRpcErrorCode.INVALID_REQUEST: "Invalid Request",
            JsonRpcErrorCode.METHOD_NOT_FOUND: "Method not found",
            JsonRpcErrorCode.INVALID_PARAMS: "Invalid params",
            JsonRpcErrorCode.INTERNAL_ERROR: "Internal error",
            JsonRpcErrorCode.SERVER_ERROR: "Server error",
        }
        return cls(
            code=code.value,
            message=message or default_messages.get(code, "Unknown error"),
            data=data,
        )


class JsonRpcRequest(BaseModel):
    """
    JSON-RPC 2.0 request object.

    See: https://www.jsonrpc.org/specification#request_object
    """

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = Field(description="JSON-RPC version, must be '2.0'")
    method: str = Field(description="Name of the method to be invoked")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameter values for the method"
    )
    id: Optional[Union[str, int]] = Field(
        default=None, description="Request identifier established by the client"
    )


class JsonRpcResponse(BaseModel):
    """
    JSON-RPC 2.0 response object.

    Per JSON-RPC 2.0 spec, a response has either 'result' or 'error', not both.
    This model excludes None values during serialization to comply with the spec.

    See: https://www.jsonrpc.org/specification#response_object
    """

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC version")
    result: Optional[Any] = Field(
        default=None, description="Result of the method invocation"
    )
    error: Optional[JsonRpcError] = Field(
        default=None, description="Error object if method invocation failed"
    )
    id: Optional[Union[str, int]] = Field(
        default=None, description="Request identifier from the request"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Serialize to dict, excluding result or error when None (JSON-RPC compliance)."""
        # Always include jsonrpc and id, but only include result OR error
        data: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            data["error"] = (
                self.error.model_dump()
                if hasattr(self.error, "model_dump")
                else self.error
            )
        else:
            # Only include result if there's no error
            data["result"] = self.result
        return data

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string, excluding result or error when None (JSON-RPC compliance)."""
        import json

        return json.dumps(self.model_dump())

    @classmethod
    def success(
        cls, result: Any, request_id: Optional[Union[str, int]] = None
    ) -> "JsonRpcResponse":
        """Create a success response."""
        return cls(result=result, id=request_id)

    @classmethod
    def error_response(
        cls,
        code: JsonRpcErrorCode,
        message: Optional[str] = None,
        data: Any = None,
        request_id: Optional[Union[str, int]] = None,
    ) -> "JsonRpcResponse":
        """Create an error response from a standard error code."""
        return cls(
            error=JsonRpcError.from_code(code, message, data),
            id=request_id,
        )


# =============================================================================
# MCP Tool Types
# =============================================================================


class Tool(BaseModel):
    """
    Strongly typed MCP tool specification.

    Follows the MCP ToolSpec format for tool discovery.
    See: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(
        description="Human-readable description of what the tool does"
    )
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema for the tool's input parameters"
    )


class ToolErrorType(str, Enum):
    """Types of errors that can occur during tool execution."""

    EXECUTION_ERROR = "execution_error"  # Tool ran but failed
    INVALID_ARGS = "invalid_args"  # Invalid arguments provided
    TRANSPORT_ERROR = "transport_error"  # Communication failure
    TOOL_NOT_FOUND = "tool_not_found"  # Tool doesn't exist
    TIMEOUT = "timeout"  # Operation timed out


class ToolError(BaseModel):
    """
    Structured error for tool execution failures.

    This is used for transport/framework errors, NOT for errors returned
    by the tool itself (those go in the result field).
    """

    model_config = ConfigDict(extra="forbid")

    error_type: ToolErrorType = Field(description="Category of the error")
    message: str = Field(description="Human-readable error message")


# --- MCP Actions ---


class ListToolsAction(Action):
    """
    Request list of available tools from the environment.

    This action triggers MCP's tools/list operation and returns
    all available tools with their schemas. Does NOT require reset()
    to be called first.
    """

    type: Literal["list_tools"] = Field(
        default="list_tools", description="Action type discriminator"
    )


class CallToolAction(Action):
    """
    Call a specific tool via MCP.

    This action triggers MCP's tools/call operation with the
    specified tool name and arguments.
    """

    type: Literal["call_tool"] = Field(
        default="call_tool", description="Action type discriminator"
    )
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


# --- MCP Observations ---


class ListToolsObservation(Observation):
    """
    Response containing available tools.

    Returned when processing a ListToolsAction.
    """

    tools: List[Tool] = Field(description="List of available tools with their schemas")


class CallToolObservation(Observation):
    """
    Response from tool execution.

    Contains the tool's result or an error if the call failed.
    Tool-specific errors (from the tool itself) are included in the result.
    Transport/framework errors use the error field.
    """

    tool_name: str = Field(description="Name of the tool that was called")
    result: Any = Field(
        default=None, description="Tool-specific result (may include tool errors)"
    )
    error: Optional[ToolError] = Field(
        default=None, description="Transport/framework error if call failed"
    )


# --- WebSocket Message Types for MCP ---


class WSMCPMessage(BaseMessage):
    """
    WebSocket message for MCP JSON-RPC requests.

    Allows direct MCP access via WebSocket for production inference,
    bypassing the step() API.
    """

    type: Literal["mcp"] = Field(default="mcp", description="Message type")
    data: Dict[str, Any] = Field(description="JSON-RPC payload (method, params, id)")


class WSMCPResponse(BaseModel):
    """
    WebSocket response for MCP JSON-RPC.

    Contains the JSON-RPC response from the MCP server.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="mcp", description="Response type")
    data: Dict[str, Any] = Field(description="JSON-RPC response payload")


# Reserved tool names that cannot be used (protects dual API boundary)
RESERVED_TOOL_NAMES = frozenset(["reset", "step", "state", "close"])
