# SPDX-License-Identifier: BSD-3-Clause

"""
MCP Environment base class for OpenEnv.

This module provides the MCPEnvironment base class that integrates FastMCP servers
with OpenEnv's Gym-style Environment interface. It handles MCP tool discovery
and invocation through the step() API, following RFC 003.

Key features:
- Automatic routing of ListToolsAction and CallToolAction to MCP server
- Reserved tool name validation (reset, step, state, close are protected)
- Timeout handling for tool calls
- Proper error categorization (tool not found, execution errors, timeouts)
- Mode-aware tool registration (production vs simulation)
- Code mode support via get_callables() and execute_code()

Examples:

    ```python
    from fastmcp import FastMCP
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    class MyMCPEnv(MCPEnvironment):
        def __init__(self):
            mcp = FastMCP("my-server")

            # Register mode-specific tools
            @self.tool(mode="production")
            def my_tool(arg: str) -> str:
                return f"Production: {arg}"

            @self.tool(mode="simulation")
            def my_tool(arg: str) -> str:
                return f"Simulation: {arg}"

            super().__init__(mcp)

        def reset(self, seed=None, episode_id=None, **kwargs):
            # Reset logic here
            ...

        def _step_impl(self, action):
            # Handle non-MCP actions
            ...

        @property
        def state(self):
            # Return current state
            ...
    ```
"""

import asyncio
import inspect
from abc import abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

from ..utils import run_async_safely
from .interfaces import Environment
from .mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
    Tool,
    ToolError,
    ToolErrorType,
)
from .types import Action, Observation


# Default timeout for MCP tool calls in seconds
MCP_TOOL_CALL_TIMEOUT = 30.0

# Valid modes for tool registration
VALID_MODES = {"production", "simulation"}


def get_server_tools(mcp_server: Any) -> Dict[str, Any]:
    """
    Get tools from a FastMCP server, compatible with both 2.x and 3.x.

    Returns:
        Dictionary mapping tool names to tool objects.
    """
    # FastMCP 2.x: get_tools() returns dict {name: Tool}
    if hasattr(mcp_server, "get_tools"):
        result = run_async_safely(mcp_server.get_tools())
        if isinstance(result, dict):
            return result
    # FastMCP 3.x: list_tools() returns list of Tool objects
    if hasattr(mcp_server, "list_tools"):
        tools_list = run_async_safely(mcp_server.list_tools())
        return {t.name: t for t in tools_list}
    return {}


def _tool_from_server_tool(tool: Any) -> Tool:
    """Convert a FastMCP tool object into OpenEnv's Tool model."""
    return Tool(
        name=tool.name,
        description=tool.description or "",
        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
    )


def _tool_from_mode_schema(schema: Dict[str, Any]) -> Tool:
    """Convert a stored mode-aware schema into OpenEnv's Tool model."""
    return Tool(
        name=schema["name"],
        description=schema["description"],
        input_schema=schema["input_schema"],
    )


def _schema_for_mode(
    mode_schemas: Dict[Optional[str], Dict[str, Any]], current_mode: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Return the schema visible for the current mode, if any."""
    if None in mode_schemas:
        return mode_schemas[None]
    return mode_schemas.get(current_mode)


class MCPEnvironment(Environment):
    """
    Base class for environments that expose tools via MCP (Model Context Protocol).

    MCPEnvironment bridges FastMCP servers with OpenEnv's Gym-style API, allowing
    agents to discover and invoke MCP tools through the standard step() interface.

    The class automatically handles:
    - ListToolsAction: Returns available tools from the MCP server
    - CallToolAction: Invokes a specific tool with arguments

    All other actions are delegated to the abstract _step_impl() method,
    which subclasses must implement.

    Args:
        mcp_server: A FastMCP server instance containing tool definitions.
            The server's tools will be validated against reserved names.
        transform: Optional transform to apply to observations (inherited from Environment).

    Raises:
        ValueError: If any tool in the MCP server uses a reserved name
            (reset, step, state, close).

    Examples:

        ```python
        from fastmcp import FastMCP

        mcp = FastMCP("calculator")

        @mcp.tool()
        def add(a: int, b: int) -> int:
            return a + b

        env = MyMCPEnvironment(mcp)
        obs = env.step(ListToolsAction())
        obs.tools[0].name  # 'add'
        ```
    """

    def __init__(self, mcp_server: Any, transform: Optional[Any] = None) -> None:
        """
        Initialize the MCP environment.

        Args:
            mcp_server: A FastMCP server instance with tool definitions.
            transform: Optional transform to apply to observations.

        Raises:
            ValueError: If any tool uses a reserved name (reset, step, state, close).
        """
        super().__init__(transform=transform)

        # Validate tool names before storing
        self._validate_tool_names(mcp_server)

        self.mcp_server = mcp_server
        self.mcp_client = Client(mcp_server)

        # Track mode-specific tools: {tool_name: {mode: func}}
        # mode can be "production", "simulation", or None (available in all modes)
        self._mode_tools = defaultdict(dict)

        # Track tool schemas for list_tools: {tool_name: {mode: schema}}
        self._mode_tool_schemas = defaultdict(dict)

    def _require_mcp_client(self) -> Any:
        """Return MCP client or raise if environment has been closed."""
        if self.mcp_client is None:
            raise RuntimeError("MCP client is not available; environment is closed")
        return self.mcp_client

    def _require_mcp_server(self) -> Any:
        """Return MCP server or raise if environment has been closed."""
        if self.mcp_server is None:
            raise RuntimeError("MCP server is not available; environment is closed")
        return self.mcp_server

    @asynccontextmanager
    async def mcp_session(self):
        """
        Context manager for MCP client sessions.

        This wrapper serves two purposes:

        1. **Null guard** — raises a clear error if ``close()`` has already
           been called (``mcp_client`` is ``None``).

        2. **AsyncExitStack adapter** — FastMCP's ``Client.__aenter__``
           creates a background ``asyncio.Task`` for session management.
           When entered directly via ``AsyncExitStack`` in the HTTP session
           path (``_create_session``), this task can be cancelled by ASGI
           harnesses (e.g. Starlette ``TestClient``) between requests,
           corrupting session state.  Wrapping in an ``asynccontextmanager``
           generator isolates the task lifecycle: the generator frame keeps
           ``async with client:`` suspended at ``yield``, so cleanup only
           runs when the stack explicitly closes the generator — not when
           the event loop cancels orphaned tasks.

        Delegates to FastMCP's ``Client`` context manager which is
        reentrant: the first entry opens the transport and subsequent
        (nested) entries simply increment an internal reference counter.
        The transport is closed only when the outermost context exits.

        No external lock is needed because ``Client._connect`` /
        ``Client._disconnect`` already serialise connection state changes
        through their own ``anyio.Lock``.
        """
        client = self._require_mcp_client()
        async with client:
            yield client

    @property
    def supports_code_mode(self) -> bool:
        """Check if this environment supports code mode (execute_code)."""
        return True

    def _get_server_tools(self, mcp_server: Any) -> Dict[str, Any]:
        """
        Get tools from a FastMCP server, compatible with both 2.x and 3.x.

        Returns:
            Dictionary mapping tool names to tool objects.
        """
        return get_server_tools(mcp_server)

    def get_callables(self) -> Dict[str, Callable]:
        """
        Get callable functions for code mode.

        Returns tool functions as direct Python callables, enabling code mode
        where agents write Python code that calls tools directly (no JSON-RPC
        overhead). Mode-specific tools are filtered by the current mode.

        Returns:
            Dictionary mapping tool names to callables.
        """
        callables: Dict[str, Callable] = {}
        current_mode = getattr(self, "_mode", None)

        # Extract callables from FastMCP server using public API
        for tool_name, tool in self._get_server_tools(self.mcp_server).items():
            if hasattr(tool, "fn") and callable(tool.fn):
                callables[tool_name] = tool.fn

        # Add mode-specific tools available in current mode
        for tool_name, mode_funcs in self._mode_tools.items():
            if None in mode_funcs:
                # Tool available in all modes (already in FastMCP if registered there)
                if tool_name not in callables:
                    callables[tool_name] = mode_funcs[None]
            elif current_mode in mode_funcs:
                # Tool available in current mode only
                callables[tool_name] = mode_funcs[current_mode]

        return callables

    def execute_code(self, code: str) -> Observation:
        """
        Execute Python code with tools available as callables.

        This enables the CodeAct pattern where agents write Python code
        that calls tools directly as functions, avoiding JSON-RPC overhead.

        Args:
            code: Python code to execute. Tools are available as functions
                in the execution namespace. Set a variable named 'result'
                to capture the return value.

        Returns:
            Observation with result in metadata["result"] or error in
            metadata["error"].
        """
        namespace = self.get_callables()

        result_dict: Dict[str, Any] = {}
        try:
            exec(code, namespace, result_dict)
            result = result_dict.get("result")
            return Observation(done=False, reward=0.0, metadata={"result": result})
        except SyntaxError as e:
            return Observation(
                done=False, reward=0.0, metadata={"error": f"Syntax error: {str(e)}"}
            )
        except Exception as e:
            return Observation(done=False, reward=0.0, metadata={"error": str(e)})

    def _validate_tool_names(self, mcp_server: Any) -> None:
        """
        Validate that no tools use reserved names.

        Reserved names (reset, step, state, close) are protected to maintain
        the dual API boundary between infrastructure and agent APIs.

        Args:
            mcp_server: The FastMCP server to validate.

        Raises:
            ValueError: If any tool uses a reserved name.
        """
        tools_dict = self._get_server_tools(mcp_server)
        if tools_dict:
            tool_names = set(tools_dict.keys())
            conflicts = tool_names & RESERVED_TOOL_NAMES
            if conflicts:
                raise ValueError(
                    f"MCP tools cannot use reserved names: {sorted(conflicts)}. "
                    f"Reserved names are: {sorted(RESERVED_TOOL_NAMES)}"
                )

    def tool(self, mode: Optional[str] = None) -> Callable:
        """
        Decorator for registering mode-aware tools.

        Args:
            mode: Optional mode for the tool ("production" or "simulation").
                If None, tool is available in all modes.

        Returns:
            A decorator function for registering tools.

        Raises:
            ValueError: If mode is not None, "production", or "simulation".
        """
        if mode is not None and mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Mode must be 'production', 'simulation', or None."
            )

        def decorator(func: Callable) -> Callable:
            tool_name = func.__name__
            # Validate tool name is not reserved
            if tool_name in RESERVED_TOOL_NAMES:
                raise ValueError(
                    f"Tool name '{tool_name}' is reserved and cannot be used. "
                    f"Reserved names are: {sorted(RESERVED_TOOL_NAMES)}"
                )

            # If mode is None, register with FastMCP as usual
            if mode is None:
                mcp_server = self._require_mcp_server()
                decorated_func = mcp_server.tool()(func)
                self._mode_tools[tool_name][None] = func
                return decorated_func

            # For mode-specific tools, don't register with FastMCP
            # Instead, track them ourselves
            self._mode_tools[tool_name][mode] = func

            # Extract schema information from function signature
            sig = inspect.signature(func)
            schema = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param_name, param in sig.parameters.items():
                # Get type annotation
                param_type = param.annotation
                json_type = "string"  # default
                if param_type in (int, "int"):
                    json_type = "integer"
                elif param_type in (float, "float"):
                    json_type = "number"
                elif param_type in (bool, "bool"):
                    json_type = "boolean"

                schema["properties"][param_name] = {"type": json_type}

                # If no default value, it's required
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)

            # Store the schema for this mode-specific tool
            self._mode_tool_schemas[tool_name][mode] = {
                "name": tool_name,
                "description": func.__doc__ or "",
                "input_schema": schema,
            }

            return func

        return decorator

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute an action in the environment.

        This method routes MCP-specific actions (ListToolsAction, CallToolAction)
        to the appropriate handlers, while delegating all other actions to
        the subclass's _step_impl() method.

        Args:
            action (`Action`):
                The action to execute. `ListToolsAction` returns available MCP tools,
                `CallToolAction` invokes a specific MCP tool, and any other action
                is delegated to _step_impl().
            timeout_s (`float`, *optional*):
                Timeout in seconds for the action. Defaults to MCP_TOOL_CALL_TIMEOUT
                (30s) for MCP actions.
            **kwargs (`Any`):
                Additional arguments passed to handlers.

        Returns:
            `Observation`: `ListToolsObservation` for `ListToolsAction`,
            `CallToolObservation` for `CallToolAction`, or a subclass-defined
            Observation for other actions.
        """
        if isinstance(action, ListToolsAction):
            return self._handle_list_tools()
        elif isinstance(action, CallToolAction):
            return self._handle_call_tool(action, timeout_s=timeout_s)
        else:
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _handle_list_tools(self) -> ListToolsObservation:
        """Sync wrapper — delegates to the canonical async implementation."""
        return run_async_safely(self._async_handle_list_tools())

    async def _async_list_tools(self) -> list:
        """
        Async helper to list tools from the MCP client.

        Returns:
            List of tool objects from the MCP server.
        """
        async with self.mcp_session() as client:
            return await client.list_tools()

    def _handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: Optional[float] = None,
    ) -> CallToolObservation:
        """Sync wrapper — delegates to the canonical async implementation."""
        return run_async_safely(
            self._async_handle_call_tool(action, timeout_s=timeout_s)
        )

    async def _async_call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Async helper to call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            The result from the tool execution.
        """
        async with self.mcp_session() as client:
            return await client.call_tool(tool_name, arguments)

    async def _async_handle_list_tools(self) -> ListToolsObservation:
        """Async version of _handle_list_tools — avoids run_async_safely."""
        try:
            current_mode = getattr(self, "_mode", None)
            tools_result = await self._async_list_tools()
            tools = []
            for tool in tools_result:
                if tool.name not in self._mode_tool_schemas:
                    tools.append(_tool_from_server_tool(tool))
            for mode_schemas in self._mode_tool_schemas.values():
                schema = _schema_for_mode(mode_schemas, current_mode)
                if schema is not None:
                    tools.append(_tool_from_mode_schema(schema))
            return ListToolsObservation(tools=tools)
        except Exception as e:
            return ListToolsObservation(
                tools=[],
                metadata={"error": str(e), "error_type": "list_tools_failed"},
            )

    async def _async_handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: Optional[float] = None,
    ) -> CallToolObservation:
        """Async version of _handle_call_tool — avoids run_async_safely."""
        timeout = timeout_s if timeout_s is not None else MCP_TOOL_CALL_TIMEOUT
        tool_name = action.tool_name
        current_mode = getattr(self, "_mode", None)

        if tool_name in self._mode_tools:
            mode_info = self._mode_tools[tool_name]
            if None in mode_info:
                func = mode_info[None]
            elif current_mode in mode_info:
                func = mode_info[current_mode]
            else:
                return CallToolObservation(
                    tool_name=tool_name,
                    result=None,
                    error=ToolError(
                        error_type=ToolErrorType.TOOL_NOT_FOUND,
                        message=f"Tool '{tool_name}' not available in {current_mode} mode",
                    ),
                )
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(**action.arguments)
                else:
                    result = func(**action.arguments)
                return CallToolObservation(
                    tool_name=tool_name,
                    result=CallToolResult(
                        content=[TextContent(type="text", text=str(result))],
                        structured_content={"result": result},
                        meta=None,
                        data=result,
                        is_error=False,
                    ),
                )
            except Exception as e:
                return CallToolObservation(
                    tool_name=tool_name,
                    result=None,
                    error=ToolError(
                        error_type=ToolErrorType.EXECUTION_ERROR,
                        message=str(e),
                    ),
                )

        try:
            result = await asyncio.wait_for(
                self._async_call_tool(action.tool_name, action.arguments),
                timeout=timeout,
            )
            return CallToolObservation(tool_name=action.tool_name, result=result)
        except asyncio.TimeoutError:
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=ToolErrorType.TIMEOUT,
                    message=f"Tool '{action.tool_name}' timed out after {timeout} seconds",
                ),
            )
        except Exception as e:
            error_message = str(e)
            if (
                "not found" in error_message.lower()
                or "unknown tool" in error_message.lower()
            ):
                error_type = ToolErrorType.TOOL_NOT_FOUND
            elif (
                "invalid" in error_message.lower()
                or "argument" in error_message.lower()
            ):
                error_type = ToolErrorType.INVALID_ARGS
            else:
                error_type = ToolErrorType.EXECUTION_ERROR
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(error_type=error_type, message=error_message),
            )

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Async step that routes MCP actions without going through run_async_safely.

        The WebSocket handler calls this directly on the outer event loop, where
        the MCP session is already open, avoiding the thread/event-loop deadlock
        that occurs when the sync step() path is used via run_in_executor.
        """
        if isinstance(action, ListToolsAction):
            return await self._async_handle_list_tools()
        elif isinstance(action, CallToolAction):
            return await self._async_handle_call_tool(action, timeout_s=timeout_s)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._step_impl(action, timeout_s=timeout_s, **kwargs)
            )

    @abstractmethod
    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions in the environment.

        Subclasses must implement this method to handle any actions that are
        not ListToolsAction or CallToolAction. This is where environment-specific
        action processing should occur.

        Args:
            action: The action to execute (guaranteed not to be an MCP action).
            timeout_s: Optional timeout in seconds.
            **kwargs: Additional arguments.

        Returns:
            An Observation appropriate for the action.
        """
        pass

    def close(self) -> None:
        """
        Clean up resources used by the environment.

        This method cleans up the MCP client and any other resources.
        Subclasses should call super().close() if they override this method.
        """
        # The MCP client uses async context manager, so cleanup happens
        # automatically when the context exits. We just clear references.
        self.mcp_client = None
        self.mcp_server = None
