"""
Calendar API Clone - FastAPI Application
Complete implementation of Google Calendar APIs
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

load_dotenv()

from apis.acl.router import router as acl_router
from apis.calendarList.router import router as calendar_list_router
from apis.calendars.router import router as calendars_router
from apis.colors.router import router as colors_router

# Import API routers
from apis.core_apis import router as core_router
from apis.database_router import router as database_router
from apis.events.router import router as events_router
from apis.freebusy.router import router as freebusy_router
from apis.mcp.router import router as mcp_router
from apis.settings.router import router as settings_router
from apis.users.router import api_router as user_api_router, router as users_router
from database.seed_database import init_seed_database

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Import OpenEnv modules
try:
    from openenv_wrapper.config import MCP_NAME
    from openenv_wrapper.custom_http_server import MCPHTTPEnvServer
    from openenv_wrapper.data_models import MCPAction, MCPObservation
    from openenv_wrapper.mcp_env_environment import MCPEnvironment

    OPENENV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenEnv modules not available: {e}")
    OPENENV_AVAILABLE = False


def create_calendar_environment():
    """
    Factory function for creating Calendar environment with config.

    This function is called for each WebSocket session to create an isolated
    environment instance.
    """
    database_id = os.getenv("DEFAULT_DATABASE_ID", "default")
    auth_token = os.getenv("DEFAULT_AUTH_TOKEN")
    return MCPEnvironment(database_id=database_id, auth_token=auth_token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Generic lifespan event handler for OpenEnv integration.

    This lifespan function is fully generic and can be copied to any MCP project.
    It dynamically loads the MCP configuration from openenv_wrapper/config.py.

    To use in a different MCP:
    1. Copy the entire openenv_wrapper folder to your MCP project
    2. Update openenv_wrapper/config.py with your MCP-specific settings
    3. Copy this lifespan function and the imports above to your main.py
    4. Pass lifespan=lifespan to FastAPI(...)
    """
    # Startup
    logger.info("Starting Calendar API Backend...")

    # Initialize separate seed storage database
    init_seed_database()
    logger.info("Seed database initialized successfully")

    if OPENENV_AVAILABLE:
        logger.info(f"Initializing {MCP_NAME} OpenEnv environment...")
        try:
            # Pass the factory function (not an instance!)
            # The new OpenEnv expects a callable (class or factory function)
            # It will call this function to create environment instances as needed
            http_server = MCPHTTPEnvServer(
                env=create_calendar_environment,  # Pass function, don't call it!
                action_cls=MCPAction,
                observation_cls=MCPObservation,
            )

            # Register all custom routes (reset, step, state)
            http_server.register_routes(app)

            logger.info(f"{MCP_NAME} OpenEnv environment initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {MCP_NAME} OpenEnv: {e}", exc_info=True)
            # Continue without OpenEnv routes if initialization fails
    else:
        logger.warning("OpenEnv routes not registered - modules not available")

    yield

    # Shutdown
    if OPENENV_AVAILABLE:
        logger.info(f"Shutting down {MCP_NAME} OpenEnv environment...")
    logger.info("Shutting down Calendar API Backend...")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Calendar API Clone",
    description="""
    Google Calendar API implementation with FastAPI

    ## Features

    * **Calendar Management**: Create, read, update, delete calendars
    * **Event Management**: Full event lifecycle management
    * **Database APIs**: Database management and inspection
    * **Official APIs**: Google Calendar API v3 endpoints implemented

    ## API Categories

    * **Core APIs**: Health check and system status
    * **Calendar APIs**: Calendar CRUD operations (SQLAlchemy-based)
    * **CalendarList APIs**: User calendar list management and settings
    * **Event APIs**: Event management and scheduling with 11 endpoints
    * **ACL APIs**: Access control list management with 7 endpoints
    * **Colors APIs**: Static color definitions for calendars and events
    * **Database APIs**: Database management and inspection
    * **MCP APIs**: Model Context Protocol support

    ## Getting Started

    1. Check API health using `GET /health`
    2. Initialize database using database APIs
    3. Start creating calendars and events

    ## Google Calendar APIs

    This implementation includes the major Google Calendar API v3 endpoints for:
    - Calendar management (6 endpoints)
    - Calendar list management (7 endpoints)
    - Event management (11 endpoints)
    - Access control (7 endpoints)
    - Colors (1 endpoint)
    - Free/busy queries (1 endpoint)
    """,
    version="1.0.0",
    contact={
        "name": "Calendar API Clone",
        "url": "https://github.com/calendar-api-clone",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include MCP router for Model Context Protocol support (no prefix)
app.include_router(mcp_router, tags=["MCP"])

# Include API routers
app.include_router(core_router)
app.include_router(database_router)
app.include_router(calendars_router)
app.include_router(calendar_list_router)
app.include_router(events_router)
app.include_router(colors_router)
app.include_router(users_router)
app.include_router(user_api_router)
app.include_router(settings_router)
app.include_router(acl_router)
app.include_router(freebusy_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/api", include_in_schema=False)
async def api_root():
    """API root endpoint"""
    return RedirectResponse(url="/docs")


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    from fastapi.responses import JSONResponse

    logger.error(f"Internal server error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "support": "Check logs for more details",
        },
    )
