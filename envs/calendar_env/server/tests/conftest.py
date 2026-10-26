"""
Pytest configuration and shared fixtures for Calendar API tests
"""

import sys
import uuid
from pathlib import Path

import pytest
from apis.calendarList.router import router as calendar_list_router
from apis.calendars.router import router as calendars_router
from apis.mcp.router import router as mcp_router
from data.multi_user_sample import get_multi_user_sql
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture
def app():
    """Create a FastAPI app with the MCP router for testing"""
    app = FastAPI()
    app.include_router(mcp_router)
    return app


@pytest.fixture
def client(app):
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def calendar_app():
    """Create a FastAPI app with the Calendar router for testing"""
    app = FastAPI()
    app.include_router(calendars_router)
    return app


@pytest.fixture
def calendar_client(calendar_app):
    """Create a test client for calendar endpoints"""
    return TestClient(calendar_app)


@pytest.fixture
def calendar_list_app():
    """Create a FastAPI app with the CalendarList router for testing"""
    app = FastAPI()
    app.include_router(calendar_list_router)
    return app


@pytest.fixture
def calendar_list_client(calendar_list_app):
    """Create a test client for calendarList endpoints"""
    return TestClient(calendar_list_app)


@pytest.fixture
def full_app():
    """Create a complete FastAPI app with all routers (like main.py)"""
    from main import app

    return app


@pytest.fixture
def full_client(full_app):
    """Create a test client for the complete application"""
    return TestClient(full_app)


@pytest.fixture(scope="function")
def test_database_id():
    """Generate unique test database ID for each test"""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def seeded_database(full_client, test_database_id):
    """Setup a seeded test database"""
    # Get sample SQL data
    sql_content = get_multi_user_sql()

    # Seed the database
    seed_response = full_client.post(
        "/api/seed-database",
        json={"database_id": test_database_id, "sql_content": sql_content},
    )

    assert seed_response.status_code == 200, (
        f"Failed to seed database: {seed_response.text}"
    )

    return {
        "client": full_client,
        "database_id": test_database_id,
        "users": {
            "alice": "alice_manager",
            "bob": "bob_developer",
            "carol": "carol_designer",
            "dave": "dave_sales",
        },
    }


@pytest.fixture
def mcp_request_helper():
    """Helper function to create MCP requests"""

    def create_mcp_request(method_name, arguments, request_id=1):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": method_name, "arguments": arguments},
        }

    return create_mcp_request


@pytest.fixture
def api_headers():
    """Helper function to create API headers"""

    def create_headers(database_id, user_id):
        return {"x-database-id": database_id, "x-user-id": user_id}

    return create_headers
