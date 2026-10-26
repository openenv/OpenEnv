"""Client for the Jupyter environment.

Connects to a running jupyter_env server and exposes notebook tools through the
MCP tool-calling interface.

Example:
    >>> with JupyterEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['add_and_execute_code_cell', 'edit_and_execute_current_cell',
    ...     #  'execute_shell_command', 'get_notebook_state']
    ...
    ...     # Execute code
    ...     result = env.call_tool("add_and_execute_code_cell", code="import pandas as pd; pd.__version__")
    ...     print(result)
    ...
    ...     # Fix a cell
    ...     result = env.call_tool("edit_and_execute_current_cell", code="import pandas; pandas.__version__")
    ...
    ...     # Run shell command
    ...     result = env.call_tool("execute_shell_command", command="pip install polars")
    ...
    ...     # Inspect history
    ...     state = env.call_tool("get_notebook_state")
    ...     print(state)

Example with Docker:
    >>> env = JupyterEnv.from_docker_image("jupyter-env:latest")
    >>> try:
    ...     env.reset()
    ...     result = env.call_tool("add_and_execute_code_cell", code="print('hello')")
    ... finally:
    ...     env.close()
"""

from openenv.core.mcp_client import MCPToolClient


class JupyterEnv(MCPToolClient):
    """Client for the Jupyter environment.

    Inherits all MCP tool-calling functionality from MCPToolClient:
    - ``list_tools()``         — discover available notebook tools
    - ``call_tool(name, **kwargs)`` — call a tool by name
    - ``reset(**kwargs)``      — start a new episode (creates fresh E2B sandbox)
    - ``step(action)``         — low-level action execution

    Available tools (exposed by server):
    - ``add_and_execute_code_cell(code)``    — execute Python in notebook
    - ``edit_and_execute_current_cell(code)`` — replace + re-run last cell
    - ``execute_shell_command(command)``     — run shell in sandbox
    - ``get_notebook_state()``               — return cell history summary
    """

    pass  # MCPToolClient provides all needed functionality
