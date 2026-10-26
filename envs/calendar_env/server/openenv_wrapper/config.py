"""
OpenEnv Wrapper Configuration for MCP Integration

This file contains MCP-specific configuration that needs to be customized
when copying openenv_wrapper to a different MCP project.

INSTRUCTIONS FOR NEW MCP:
1. Copy the entire openenv_wrapper folder to your MCP project
2. Update this config.py file with your MCP-specific values:
   - MCP_NAME: Your MCP name (e.g., "Slack", "GitHub", "Jira")
   - SESSION_MANAGER_CLASS: Import path to your SessionManager
   - MCP_TOOLS_MODULE: Import path to your MCP tools
   - SEED_DATA_FUNCTION: Import path to your seed data function
3. Ensure your database session manager follows the expected interface
4. That's it! The rest of the openenv_wrapper files are fully generic
"""

# ============================================================================
# MCP-SPECIFIC CONFIGURATION - CUSTOMIZE THIS SECTION FOR YOUR MCP
# ============================================================================

# Name of your MCP service (used in logs and class names)
MCP_NAME = "Calendar"

# Import path for your SessionManager class
# Your SessionManager should have these methods:
#   - get_db_path(db_id: str) -> str
#   - init_database(db_id: str, create_tables: bool = False)
#   - dispose_engine(db_id: str)
SESSION_MANAGER_MODULE = "database.session_manager"
SESSION_MANAGER_CLASS = "CalendarSessionManager"

# Import path for your MCP tools list
# Should export: MCP_TOOLS (list of tool schemas)
MCP_TOOLS_MODULE = "calendar_mcp.tools"
MCP_TOOLS_EXPORT = "MCP_TOOLS"

# Import path for your tool handlers
# Should export: MCP_TOOLS_LIST and TOOL_HANDLERS
TOOL_HANDLERS_MODULE = "handlers.tool_handlers"

# Import path for your seed data function
# Function should return SQL content as a string
SEED_DATA_MODULE = "data.multi_user_sample"
SEED_DATA_FUNCTION = "get_multi_user_sql"

# Import path for your UserManager (for access token validation)
# Your UserManager should have these methods:
#   - get_first_user_token() -> Optional[str]
#   - get_user_by_static_token(static_token: str) -> Optional[Dict]
USER_MANAGER_MODULE = "database.managers.user_manager"
USER_MANAGER_CLASS = "UserManager"

# Database directory name (relative to project root)
DATABASE_DIR = "mcp_databases"

# Database file prefix (e.g., "Calendar_" results in "Calendar_db_id.sqlite")
DATABASE_PREFIX = f"{MCP_NAME}_"

# HTTP Headers configuration
# These headers are used in the OpenEnv HTTP endpoints (/reset, /step, /state)
# Customize these based on your MCP's authentication and database selection needs
#
# EXAMPLES FOR DIFFERENT MCPs:
#
# For Calendar MCP (current):
#   HTTP_HEADERS = {
#       "database_id": "x-database-id",
#       "access_token": "x-access-token",
#   }
#
# For Teams MCP:
#   HTTP_HEADERS = {
#       "database_id": "x-database-id",
#       "access_token": "x-teams-access-token",
#   }
#
# For Slack MCP:
#   HTTP_HEADERS = {
#       "database_id": "x-database-id",
#       "access_token": "x-slack-token",
#   }
#
HTTP_HEADERS = {
    # Header name for database/session ID
    # This header is used to specify which database instance to use (multi-tenancy)
    "database_id": "x-database-id",
    # Header name for access token/authentication
    # This header is used to authenticate the user making the request
    # Calendar MCP uses "x-access-token" for authentication
    "access_token": "x-access-token",
}

# Default values for headers (used when header is not provided)
HTTP_HEADER_DEFAULTS = {
    "database_id": "default",
    "access_token": None,
}

# ============================================================================
# HELPER FUNCTIONS - DO NOT MODIFY UNLESS NECESSARY
# ============================================================================


def get_session_manager_class():
    """Dynamically import and return the SessionManager class"""
    import importlib

    module = importlib.import_module(SESSION_MANAGER_MODULE)
    return getattr(module, SESSION_MANAGER_CLASS)


def get_mcp_tools():
    """Dynamically import and return the MCP tools list"""
    import importlib

    module = importlib.import_module(MCP_TOOLS_MODULE)
    return getattr(module, MCP_TOOLS_EXPORT)


def get_tool_handlers():
    """Dynamically import and return tool handlers"""
    import importlib

    module = importlib.import_module(TOOL_HANDLERS_MODULE)
    return {
        "MCP_TOOLS_LIST": getattr(module, "MCP_TOOLS_LIST"),
        "TOOL_HANDLERS": getattr(module, "TOOL_HANDLERS"),
    }


def get_seed_data_function():
    """Dynamically import and return the seed data function"""
    import importlib

    module = importlib.import_module(SEED_DATA_MODULE)
    return getattr(module, SEED_DATA_FUNCTION)


def get_user_manager_class():
    """
    Dynamically import and return the UserManager class with compatibility wrappers.

    This function adds method aliases to handle different method names across MCPs.
    Calendar MCP uses different method names than Teams MCP:
    - get_first_user_from_db() → get_first_user_token()
    - get_user_by_access_token() → get_user_by_static_token()
    """
    import importlib

    module = importlib.import_module(USER_MANAGER_MODULE)
    user_mgr_cls = getattr(module, USER_MANAGER_CLASS)

    # Add compatibility wrapper for get_first_user_token
    # Calendar uses: get_first_user_from_db()
    # Generic code expects: get_first_user_token(db_id)
    if hasattr(user_mgr_cls, "get_first_user_from_db") and not hasattr(
        user_mgr_cls, "get_first_user_token"
    ):

        def get_first_user_token_wrapper(self, db_id: str):
            """Wrapper to get first user's token - Calendar compatibility"""
            user_dict = self.get_first_user_from_db()
            if user_dict:
                # Return the static_token from the user dict
                return user_dict.get("static_token")
            return None

        user_mgr_cls.get_first_user_token = get_first_user_token_wrapper

    # Add compatibility wrapper for get_user_by_static_token
    # Calendar uses: get_user_by_access_token(static_token)
    # Generic code expects: get_user_by_static_token(token, db_id=None)
    # Note: Calendar's method doesn't need db_id since UserManager is already initialized with it
    if hasattr(user_mgr_cls, "get_user_by_access_token") and not hasattr(
        user_mgr_cls, "get_user_by_static_token"
    ):

        def get_user_by_static_token_wrapper(self, token: str, db_id: str = None):
            """Wrapper to get user by static token - Calendar compatibility"""
            return self.get_user_by_access_token(token)

        user_mgr_cls.get_user_by_static_token = get_user_by_static_token_wrapper

    return user_mgr_cls
