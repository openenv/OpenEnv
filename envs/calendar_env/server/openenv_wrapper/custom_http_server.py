"""
Generic HTTP Environment Server for MCP Integration.

This module provides a custom HTTP server that wraps any MCP environment
and exposes OpenEnv-compliant HTTP endpoints (/reset, /step, /state).

The server is fully generic and works with any MCP integration.
All MCP-specific configuration is loaded from config.py.
"""

import asyncio
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Query, Request
from openenv.core.env_server.http_server import HTTPEnvServer

from .config import (
    get_seed_data_function,
    get_session_manager_class,
    HTTP_HEADER_DEFAULTS,
    HTTP_HEADERS,
    MCP_NAME,
)

logger = logging.getLogger(__name__)


class MCPHTTPEnvServer(HTTPEnvServer):
    """
    Generic HTTP Environment Server for any MCP integration.

    This server wraps any MCP environment and provides HTTP endpoints
    for OpenEnv integration. It's fully generic and works with any MCP.

    HTTP headers are configured via config.py:HTTP_HEADERS, making it easy
    to adapt to different MCP authentication and multi-tenancy patterns.
    """

    def __init__(self, env, action_cls, observation_cls):
        """Initialize custom HTTP server with MCP session manager."""
        # Store classes before calling super().__init__()
        self.action_cls = action_cls
        self.observation_cls = observation_cls

        # Call parent init
        super().__init__(
            env=env, action_cls=action_cls, observation_cls=observation_cls
        )

        # Create a persistent environment instance for HTTP endpoints
        # The parent class stores env_factory in self._env_factory
        # We create one instance for the HTTP endpoints (not WebSocket)
        if callable(self._env_factory):
            self.env = self._env_factory()
        else:
            self.env = self._env_factory

        # Dynamically load the session manager from config
        SessionManagerClass = get_session_manager_class()
        self.session_manager = SessionManagerClass()

    def _get_header_value(self, headers: dict, header_key: str) -> Optional[str]:
        """
        Get header value using configured header name.

        Args:
            headers: Request headers dictionary
            header_key: Key in HTTP_HEADERS config (e.g., "database_id", "access_token")

        Returns:
            Header value or default from config
        """
        header_name = HTTP_HEADERS.get(header_key)
        if not header_name:
            logger.warning(f"Header key '{header_key}' not configured in HTTP_HEADERS")
            return HTTP_HEADER_DEFAULTS.get(header_key)

        value = headers.get(header_name)
        if value is None:
            value = HTTP_HEADER_DEFAULTS.get(header_key)

        return value

    def register_routes(self, app: Any) -> None:
        if not isinstance(app, FastAPI):
            raise TypeError("app must be a FastAPI instance")

        # Register custom reset endpoint
        @app.post("/reset")
        async def reset_with_database_refresh(
            request: Request, body: Optional[Dict[str, Any]] = Body(default=None)
        ) -> Dict[str, Any]:
            """
            Reset the environment and optionally reset the database.

            The database_id can be provided via:
            1. Request body: {"database_id": "my_db", "sql_content": "INSERT INTO..."}
            2. HTTP header: x-database-id
            3. Default value if neither provided

            Args (in request body, all optional):
                database_id: Database identifier for multi-tenancy
                sql_content: Custom SQL content to use for seeding instead of default

            Returns:
                Observation with reset status and database reset result
            """
            headers = dict(request.headers)

            # Get database_id from body first, then header, then default
            body = body or {}
            database_id = body.get("database_id") or self._get_header_value(
                headers, "database_id"
            )
            access_token = self._get_header_value(headers, "access_token")

            # Get optional sql_content from body
            sql_content = body.get("sql_content")

            logger.info(
                f"Reset request for database_id={database_id}, custom_sql={'yes' if sql_content else 'no'}"
            )

            # Reset database to original state (with optional custom SQL)
            db_reset_result = self._reset_database(database_id, sql_content=sql_content)

            # Set request context in environment
            self.env.set_request_context(
                database_id=database_id, access_token=access_token
            )

            # Execute reset in thread pool (environments may use sync code)
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(None, self.env.reset)

            # Serialize observation manually
            result = {
                "observation": observation.model_dump()
                if hasattr(observation, "model_dump")
                else observation.__dict__,
                "done": getattr(observation, "done", False),
                "reward": getattr(observation, "reward", 0.0),
            }

            # Add database reset info to observation metadata
            if isinstance(result, dict) and "observation" in result:
                if "metadata" not in result["observation"]:
                    result["observation"]["metadata"] = {}
                result["observation"]["metadata"]["database_reset_result"] = (
                    db_reset_result
                )

            logger.info(
                f"Environment reset completed for database {database_id}, DB refresh: {db_reset_result['success']}"
            )
            return result

        # Register custom step endpoint
        @app.post("/step")
        async def step_with_headers(
            request: Request, body: Dict[str, Any] = Body(...)
        ) -> Dict[str, Any]:
            # Extract headers using dynamic header names from config
            headers = dict(request.headers)
            database_id = self._get_header_value(headers, "database_id")
            access_token = self._get_header_value(headers, "access_token")

            # Debug logging to see what headers we're receiving
            logger.info(
                f"Step request - database_id: {database_id}, has_access_token: {bool(access_token)}"
            )
            if not access_token:
                logger.warning(
                    f"No access token found in headers. Available headers: {list(headers.keys())}"
                )

            # Set request context in environment
            self.env.set_request_context(
                database_id=database_id, access_token=access_token
            )

            logger.debug(
                f"Step request with database_id={database_id}, has_token={bool(access_token)}"
            )

            # Support both {"action": {...}} and direct action fields
            action_data = body.get("action", body)

            # Deserialize action manually using Pydantic
            try:
                action = self.action_cls(**action_data)
            except Exception as e:
                logger.error(f"Failed to deserialize action: {e}")
                return {
                    "observation": {
                        "success": False,
                        "error_message": f"Invalid action: {str(e)}",
                        "done": False,
                        "reward": -1.0,
                        "metadata": {},
                    },
                    "done": False,
                    "reward": -1.0,
                }

            # Execute step in thread pool (environments may use sync code)
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(None, self.env.step, action)

            # Serialize observation manually
            result = {
                "observation": observation.model_dump()
                if hasattr(observation, "model_dump")
                else observation.__dict__,
                "done": getattr(observation, "done", False),
                "reward": getattr(observation, "reward", 0.0),
            }

            return result

        # Register state endpoint
        @app.get("/state")
        async def get_state(
            request: Request, verify_queries: List[str] = Query(default=[])
        ) -> Dict[str, Any]:
            headers = dict(request.headers)
            database_id = headers.get("x-database-id", "default")

            state = self.env.state
            result = {
                "episode_id": state.episode_id,
                "step_count": state.step_count,
                "database_id": database_id,
            }

            db_path = self.session_manager.get_db_path(database_id)

            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if verify_queries:
                    result["verification_results"] = []

                    for query in verify_queries:
                        try:
                            cursor.execute(query)
                            rows = cursor.fetchall()
                            result["verification_results"].append(
                                {
                                    "query": query,
                                    "result": [dict(row) for row in rows],
                                    "success": True,
                                }
                            )
                        except Exception as query_error:
                            result["verification_results"].append(
                                {
                                    "query": query,
                                    "error": str(query_error),
                                    "success": False,
                                }
                            )

                conn.close()
            except Exception as e:
                result["db_error"] = str(e)

            return result

    def _reset_database(
        self, database_id: str, sql_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reset database to clean state with seed data.

        This method is generic and works with any MCP that follows
        the standard session manager interface.

        Args:
            database_id: Database identifier for multi-tenancy
            sql_content: Optional custom SQL content for seeding. If provided,
                        this will be used instead of the default seed data.

        Returns:
            Dictionary with reset status and details
        """
        try:
            # Dispose any cached engine connections to prevent stale connections
            # (Only if the SessionManager has this method - some MCPs may not have it)
            if hasattr(self.session_manager, "dispose_engine"):
                self.session_manager.dispose_engine(database_id)

            # Get database path using session manager
            db_path = self.session_manager.get_db_path(database_id)

            # Drop all existing tables using sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            conn.commit()
            conn.close()

            # Recreate tables using session manager
            self.session_manager.init_database(database_id, create_tables=True)

            # Use custom SQL content if provided, otherwise use default seed data
            if sql_content:
                logger.info(f"Using custom SQL content for database {database_id}")
                seed_sql = sql_content
                used_custom_sql = True
            else:
                seed_data_fn = get_seed_data_function()
                seed_sql = seed_data_fn()
                used_custom_sql = False

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Parse and execute SQL statements
            statements = []
            for line in seed_sql.split("\n"):
                line = line.strip()
                if line and not line.startswith("--"):
                    statements.append(line)

            full_sql = " ".join(statements)
            individual_statements = [
                stmt.strip() for stmt in full_sql.split(";") if stmt.strip()
            ]

            executed_count = 0
            for statement in individual_statements:
                try:
                    if not statement.strip():
                        continue
                    cursor.execute(statement)
                    executed_count += 1
                except Exception as e:
                    logger.error(
                        f"Error executing statement during seeding: {statement[:100]}..."
                    )
                    logger.error(f"Error details: {e}")
                    raise e

            conn.commit()
            conn.close()

            seed_source = "custom SQL" if used_custom_sql else "default seed data"
            logger.info(
                f"Database {database_id} reset and seeded with {seed_source} ({executed_count} statements)"
            )

            return {
                "success": True,
                "message": f"Database reset to clean state and seeded with {seed_source}",
                "database_id": database_id,
                "seeded": True,
                "used_custom_sql": used_custom_sql,
                "statements_executed": executed_count,
            }

        except Exception as e:
            logger.error(f"Error resetting database {database_id}: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Database reset failed: {str(e)}",
                "database_id": database_id,
                "seeded": False,
                "used_custom_sql": sql_content is not None,
            }
