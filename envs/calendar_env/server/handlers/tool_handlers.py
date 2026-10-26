"""
MCP Tool Handlers - Generic API Caller for Calendar API

This module provides a generic handler that dynamically calls internal Calendar API endpoints
based on tool configuration. This approach is much more scalable and maintainable.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import httpx
from dotenv import load_dotenv

load_dotenv()


def log_tool_response(tool_name, tool_input, result, database_id):
    """Log tool responses to file for debugging"""
    try:
        # Only log during test runs (when database_id looks like test ID)
        if not database_id or "test" not in database_id.lower():
            return

        # Create logs directory
        logs_dir = Path("tests/tool_responses")
        logs_dir.mkdir(exist_ok=True)

        # Create timestamped log entry
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds
        log_entry = {
            "timestamp": timestamp,
            "tool_name": tool_name,
            "database_id": database_id,
            "input": tool_input,
            "result": result,
        }

        # Append to daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"tool_responses_{date_str}.json"

        # Read existing logs or create new
        logs = []
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except:
                logs = []

        # Add new entry
        logs.append(log_entry)

        # Write back to file
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    except Exception:
        # Don't let logging errors break the tool execution
        pass


def get_query_parameters_for_tool(tool_name: str) -> set:
    """
    Automatically determine which parameters should be query parameters
    by inspecting the FastAPI route dependencies and parameters
    """
    import inspect

    from apis.acl.router import router as acl_router
    from apis.calendarList.router import router as calendar_list_router

    # Import routers to inspect their endpoints
    from apis.calendars.router import router as calendars_router
    from apis.colors.router import router as colors_router
    from apis.events.router import router as events_router
    from apis.settings.router import router as settings_router
    from apis.users.router import router as users_router
    from fastapi.params import Query as QueryParam

    # Check all routes in all routers
    routers = [
        calendars_router,
        calendar_list_router,
        events_router,
        colors_router,
        users_router,
        settings_router,
        acl_router,
    ]

    for router in routers:
        for route in router.routes:
            if hasattr(route, "endpoint") and route.endpoint.__name__ == tool_name:
                query_params = set()

                # Method 1: Check route's dependant (FastAPI's internal parameter analysis)
                if hasattr(route, "dependant") and route.dependant:
                    for dep in route.dependant.query_params:
                        query_params.add(dep.alias or dep.name)

                # Method 2: Fallback - manual inspection of function signature
                if not query_params:
                    sig = inspect.signature(route.endpoint)
                    for param_name, param in sig.parameters.items():
                        # Skip obvious path/header params
                        if param_name in ["calendarId", "eventId", "x_database_id"]:
                            continue

                        # Check if it has Query() as default or in annotation
                        if param.default is not inspect.Parameter.empty:
                            # Check if default is a Query instance
                            if isinstance(param.default, QueryParam):
                                query_params.add(param_name)
                            # Check string representation for Query()
                            elif "Query(" in str(param.default):
                                query_params.add(param_name)

                return query_params

    return set()  # Return empty set if tool not found


# Internal API base URL (same server) - configurable via environment or .env
API_PORT = os.getenv("API_PORT", "8004")
API_BASE_URL = f"http://localhost:{API_PORT}"

# Import tools from the modular structure
from calendar_mcp.tools import MCP_TOOLS


# Dynamic endpoint mapping based on FastAPI router inspection
def get_api_endpoint_for_tool(tool_name: str) -> tuple:
    """
    Dynamically determine API endpoint by matching tool name to router function names.
    This eliminates the need for static mapping and ensures consistency.
    """
    from apis.acl.router import router as acl_router
    from apis.calendarList.router import router as calendar_list_router

    # Import routers to inspect their endpoints
    from apis.calendars.router import router as calendars_router
    from apis.colors.router import router as colors_router
    from apis.events.router import router as events_router
    from apis.freebusy.router import router as freebusy_router
    from apis.settings.router import router as settings_router
    from apis.users.router import router as users_router

    # Check all routes in all routers
    routers = [
        calendars_router,
        calendar_list_router,
        events_router,
        colors_router,
        users_router,
        settings_router,
        acl_router,
        freebusy_router,
    ]

    for router in routers:
        for route in router.routes:
            if hasattr(route, "endpoint") and route.endpoint.__name__ == tool_name:
                # Extract method and path (path already includes router prefix)
                methods = list(route.methods)
                method = methods[0] if methods else "GET"
                full_path = route.path
                return (method, full_path)

    # Return None if not found
    return None


async def execute_tool_generic(
    tool_name: str, arguments: Dict, database_id: str, user_id: str
) -> Dict:
    """
    Generic tool executor that dynamically calls API endpoints based on tool configuration.
    This eliminates the need for individual handler functions.
    Supports path parameters for endpoints like /calendars/{calendarId}
    """
    try:
        # Find the tool configuration
        tool_config = None
        for tool in MCP_TOOLS:
            if tool["name"] == tool_name:
                tool_config = tool
                break

        if not tool_config:
            result = {
                "text": f"Tool configuration not found: {tool_name}",
                "isError": True,
            }
            log_tool_response(tool_name, arguments, result, database_id)
            return result

        # Get API endpoint dynamically from router inspection
        endpoint_info = get_api_endpoint_for_tool(tool_name)
        if not endpoint_info:
            result = {
                "text": f"API endpoint not found for tool: {tool_name}. Check that router function name matches tool name.",
                "isError": True,
            }
            log_tool_response(tool_name, arguments, result, database_id)
            return result

        method, api_endpoint = endpoint_info

        # Handle path parameters (like {calendarId})
        final_endpoint = substitute_path_parameters(api_endpoint, arguments)

        # Check for missing path parameters
        if final_endpoint.startswith("ERROR_MISSING_PARAM_"):
            missing_param = final_endpoint.replace("ERROR_MISSING_PARAM_", "")
            result = {
                "text": f"The parameter '{missing_param}' is required and cannot be empty.",
                "isError": True,
            }
            log_tool_response(tool_name, arguments, result, database_id)
            return result

        # Build full URL
        full_url = f"{API_BASE_URL}{final_endpoint}"

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-database-id": database_id,
            "x-user-id": user_id,
        }

        # Add user_id header if present in arguments
        # if "user_id" in arguments:
        #     headers["x-user-id"] = arguments["user_id"]

        # Prepare request data (separate body and query parameters)
        # RL Gym style:
        # - Allow nested 'body' field to carry JSON body for non-GET/DELETE methods
        body_data, query_data = prepare_request_data(arguments, api_endpoint, tool_name)

        # Make HTTP request
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                # For GET requests, use query parameters (combine body_data and query_data)
                all_query_params = {**body_data, **query_data}
                response = await client.get(
                    full_url, headers=headers, params=all_query_params
                )
            elif method.upper() == "DELETE":
                # DELETE requests typically don't have body, but may have query params
                response = await client.delete(
                    full_url, headers=headers, params=query_data
                )
            else:
                # POST, PUT, PATCH requests with JSON body and optional query params
                response = await client.request(
                    method.upper(),
                    full_url,
                    headers=headers,
                    json=body_data,
                    params=query_data,
                )

        # Handle response
        if response.status_code == 204:
            # No content responses (like DELETE)
            result = {
                "text": f"Operation completed successfully",
                "status_code": response.status_code,
                "isError": False,
            }
        elif 200 <= response.status_code < 300:
            # Success responses
            try:
                response_data = response.json()
                result = {
                    "text": json.dumps(response_data, indent=2),
                    "status_code": response.status_code,
                    "isError": False,
                }
            except:
                result = {
                    "text": f"Operation completed successfully\nStatus: {response.status_code}",
                    "status_code": response.status_code,
                    "isError": False,
                }
        else:
            # Error responses: prefer a friendly, contextual message
            friendly_text = None
            error_data = None
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    # Prefer FastAPI 'detail' string
                    if isinstance(error_data.get("detail"), str):
                        friendly_text = f"{error_data['detail']}"
                    # Handle FastAPI/Pydantic validation error list
                    elif isinstance(error_data.get("detail"), list):
                        details = error_data.get("detail")
                        lines = []
                        for item in details:
                            if not isinstance(item, dict):
                                continue
                            msg = item.get("msg") or item.get("message")
                            loc = item.get("loc") or []
                            # Build a full path like defaultReminders[0].method, skipping generic roots
                            path_parts = []
                            if isinstance(loc, list):
                                for part in loc:
                                    if part in ("body", "query", "path"):
                                        continue
                                    if isinstance(part, int):
                                        # attach index to previous token if exists
                                        if path_parts:
                                            path_parts[-1] = f"{path_parts[-1]}[{part}]"
                                        else:
                                            path_parts.append(f"[{part}]")
                                    else:
                                        path_parts.append(str(part))
                            path_str = ".".join(path_parts) if path_parts else ""

                            # Rewrite common messages to friendlier phrasing
                            if isinstance(msg, str):
                                lower = msg.lower()
                                if lower.startswith("string should have at least"):
                                    msg = msg.replace(
                                        "String should have", "must have"
                                    ).replace("string should have", "must have")
                                elif lower == "field required":
                                    msg = "is required"
                                elif lower.startswith("value error"):
                                    msg = msg.replace("Value error, ", "")

                            if path_str:
                                lines.append(f"- {path_str}: {msg}")
                            else:
                                lines.append(f"- {msg}")

                        if lines:
                            friendly_text = "Validation errors:\n" + "\n".join(lines)
                    # Fallback to Google-style error.message
                    if (
                        friendly_text is None
                        and isinstance(error_data.get("error"), dict)
                        and isinstance(error_data["error"].get("message"), str)
                    ):
                        friendly_text = f"{error_data['error']['message']}"
            except Exception:
                # Non-JSON error body
                pass

            # Calendars-specific enhancement: synthesize message if missing
            if not friendly_text and "/calendars" in final_endpoint:
                # For 404s, attempt to compose a clear message with user and calendarId
                if response.status_code == 404:
                    cal_id = arguments.get("calendarId") or arguments.get("id")
                    if cal_id:
                        friendly_text = f"User '{user_id}' has no calendar '{cal_id}'"

            if friendly_text is not None:
                result = {
                    "text": friendly_text,
                    "status_code": response.status_code,
                    "isError": True,
                }
            else:
                # Fall back to structured error output
                try:
                    if error_data is None:
                        error_data = response.json()
                    result = {
                        "text": f"API Error: {json.dumps(error_data, indent=2)}",
                        "status_code": response.status_code,
                        "isError": True,
                    }
                except Exception:
                    result = {
                        "text": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "isError": True,
                    }

        # Log the tool response for debugging
        log_tool_response(tool_name, arguments, result, database_id)
        return result

    except httpx.RequestError as e:
        result = {
            "text": f"Request Error: {str(e)}\nFull URL: {full_url}",
            "isError": True,
        }
        log_tool_response(tool_name, arguments, result, database_id)
        return result
    except Exception as e:
        result = {"text": f"Unexpected Error: {str(e)}", "isError": True}
        log_tool_response(tool_name, arguments, result, database_id)
        return result


def extract_api_endpoint(description: str) -> str:
    """Extract API endpoint from tool description"""
    import re

    # Look for pattern like "API Endpoint: POST /calendars"
    match = re.search(r"API Endpoint:\s*[A-Z]+\s*(/[^\s\n]+)", description)
    if match:
        return match.group(1)

    return ""


def extract_http_method(description: str) -> str:
    """Extract HTTP method from tool description"""
    import re

    # Look for pattern like "API Endpoint: POST /calendars"
    match = re.search(r"API Endpoint:\s*([A-Z]+)\s*/", description)
    if match:
        return match.group(1)

    return "GET"  # Default to GET


def substitute_path_parameters(endpoint: str, arguments: Dict) -> str:
    """Replace path parameters like {calendarId} with actual values"""
    import re

    # Find all path parameters in the format {paramName}
    path_params = re.findall(r"\{(\w+)\}", endpoint)

    result_endpoint = endpoint
    for param in path_params:
        param_value = None

        if param in arguments and arguments[param]:
            param_value = str(arguments[param])
        elif param + "Id" in arguments and arguments[param + "Id"]:
            # Handle cases like {calendarId} mapped to calendarId argument
            param_value = str(arguments[param + "Id"])
        elif (
            param.replace("Id", "") in arguments and arguments[param.replace("Id", "")]
        ):
            # Handle cases like {calendarId} mapped to calendar argument
            base_param = param.replace("Id", "")
            param_value = str(arguments[base_param])

        if param_value:
            result_endpoint = result_endpoint.replace(f"{{{param}}}", param_value)
        else:
            # If path parameter is missing or empty, return error indicator
            return f"ERROR_MISSING_PARAM_{param}"

    return result_endpoint


def prepare_request_data(
    arguments: Dict, endpoint: str, tool_name: str
) -> tuple[Dict, Dict]:
    """Prepare request data by separating body and query parameters"""
    import re

    # Find all path parameters
    path_params = set(re.findall(r"\{(\w+)\}", endpoint))

    # Dynamically get query parameters for this tool by inspecting router
    query_params_for_tool = get_query_parameters_for_tool(tool_name)

    body_data = {}
    query_data = {}

    for key, value in arguments.items():
        # Skip path parameters and their variations
        if (
            key in path_params
            or key + "Id" in path_params
            or key.replace("Id", "") in path_params
        ):
            continue

        # Decide if this should be query parameter or body parameter
        if key in query_params_for_tool:
            query_data[key] = value
        else:
            body_data[key] = value

    return body_data, query_data


# Create the MCP_TOOLS_LIST and TOOL_HANDLERS for compatibility with the MCP handler
MCP_TOOLS_LIST = MCP_TOOLS

# All tools use the same generic handler
TOOL_HANDLERS = {tool["name"]: execute_tool_generic for tool in MCP_TOOLS}
