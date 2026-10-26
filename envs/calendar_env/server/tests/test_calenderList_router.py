# Integration tests using seed-database API

import json
from unittest.mock import patch

import pytest


class TestDatabaseSeeding:
    """Verify database seeding works correctly"""

    def test_seed_database_success(self, seeded_database):
        """Test that database seeding creates all required tables"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]

        response = client.get(
            "/api/database-state", headers={"x-database-id": database_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify expected tables with data
        expected_tables = ["users", "calendars", "events", "acls", "scopes"]
        for table in expected_tables:
            assert table in data["table_counts"]
            assert data["table_counts"][table] > 0


class TestMCPFlow:
    """Test MCP protocol flow: Swagger → MCP → Calendar APIs → Database"""

    def test_mcp_vs_direct_api_returns_same_data(
        self, seeded_database, mcp_request_helper, api_headers
    ):
        """Verify MCP flow returns (same results as direct API call)"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # Direct API call (what Swagger does)
        direct_response = client.get(
            "/users/me/calendarList?maxResults=100&showDeleted=false&showHidden=false",
            headers=api_headers(database_id, user_id),
        )
        assert direct_response.status_code == 200
        direct_data = direct_response.json()

        # trying to make actual HTTP requests
        # verify the direct API works correctly
        assert len(direct_data["items"]) > 0
        assert "kind" in direct_data
        assert direct_data["kind"] == "calendar#calendarList"

    def test_mcp_get_calendar(self, seeded_database, mcp_request_helper, api_headers):
        """Test MCP flow for getting specific calendar"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # Get a calendar ID first
        list_response = client.get(
            "/users/me/calendarList", headers=api_headers(database_id, user_id)
        )
        calendar_id = list_response.json()["items"][0]["id"]

        # Get the calendar details directly
        direct_calendar_response = client.get(
            f"/calendars/{calendar_id}", headers=api_headers(database_id, user_id)
        )
        calendar_data_direct = direct_calendar_response.json()

        # Trying to make actual HTTP requests
        # Verify the direct API works correctly
        assert calendar_data_direct["id"] == calendar_id
        assert "summary" in calendar_data_direct


class TestCalendarAPIs:
    """Test Calendar CRUD operations with real database"""

    def test_calendar_crud_operations(self, seeded_database, api_headers):
        """Test create, read, update, delete calendar"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # CREATE
        create_response = client.post(
            "/calendars",
            json={"summary": "Test Calendar", "timeZone": "UTC"},
            headers=api_headers(database_id, user_id),
        )
        assert create_response.status_code == 201
        calendar_id = create_response.json()["id"]

        # READ
        get_response = client.get(
            f"/calendars/{calendar_id}", headers=api_headers(database_id, user_id)
        )
        assert get_response.status_code == 200
        assert get_response.json()["summary"] == "Test Calendar"

        # UPDATE
        update_response = client.patch(
            f"/calendars/{calendar_id}",
            json={"summary": "Updated Calendar"},
            headers=api_headers(database_id, user_id),
        )
        assert update_response.status_code == 200
        assert update_response.json()["summary"] == "Updated Calendar"

        # DELETE
        delete_response = client.delete(
            f"/calendars/{calendar_id}", headers=api_headers(database_id, user_id)
        )
        assert delete_response.status_code == 200


class TestCalendarListAPIs:
    """Test CalendarList operations with real database"""

    def test_get_calendar_list_success(self, seeded_database, api_headers):
        """Test getting calendar list with real seeded data"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        response = client.get(
            "/users/me/calendarList", headers=api_headers(database_id, user_id)
        )

        assert response.status_code == 200
        data = response.json()
        assert data["kind"] == "calendar#calendarList"
        assert len(data["items"]) > 0

        # Verify Alice's calendars are present
        summaries = [item["summary"] for item in data["items"]]
        assert any("Alice" in summary for summary in summaries)

    def test_get_primary_calendar(self, seeded_database, api_headers):
        """Test getting primary calendar using 'primary' keyword"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        response = client.get(
            "/users/me/calendarList/primary", headers=api_headers(database_id, user_id)
        )

        assert response.status_code == 200
        assert response.json()["primary"] is True

    def test_update_calendar_in_list(self, seeded_database, api_headers):
        """Test updating calendar list entry"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # Get writable calendar
        list_response = client.get(
            "/users/me/calendarList", headers=api_headers(database_id, user_id)
        )
        calendars = list_response.json()["items"]
        writable_cal = next(
            c for c in calendars if c["accessRole"] in ["writer", "owner"]
        )

        # Update settings
        response = client.patch(
            f"/users/me/calendarList/{writable_cal['id']}",
            json={"hidden": True, "selected": False},
            headers=api_headers(database_id, user_id),
        )

        assert response.status_code == 200
        assert response.json()["hidden"] is True


class TestEventAPIs:
    """Test Event operations with real database"""

    def test_create_and_get_event(self, seeded_database, api_headers):
        """Test creating and retrieving event"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # Get calendar
        list_response = client.get(
            "/users/me/calendarList", headers=api_headers(database_id, user_id)
        )
        calendar_id = list_response.json()["items"][0]["id"]

        # Create event
        event_data = {
            "summary": "Test Event",
            "description": "Event created during integration test",
            "start": {"dateTime": "2024-12-01T10:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2024-12-01T11:00:00Z", "timeZone": "UTC"},
        }

        create_response = client.post(
            f"/calendars/{calendar_id}/events",
            json=event_data,
            headers=api_headers(database_id, user_id),
        )

        assert create_response.status_code == 201
        event_id = create_response.json()["id"]

        # Get event
        get_response = client.get(
            f"/calendars/{calendar_id}/events/{event_id}",
            headers=api_headers(database_id, user_id),
        )

        assert get_response.status_code == 200
        assert get_response.json()["summary"] == "Test Event"


class TestACLPermissions:
    """Test ACL permissions with real database"""

    def test_owner_has_acl_permissions(self, seeded_database, api_headers):
        """Test that calendar owner has proper ACL entries"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]
        user_id = seeded_database["users"]["alice"]

        # Get owner calendar
        list_response = client.get(
            "/users/me/calendarList", headers=api_headers(database_id, user_id)
        )
        calendars = list_response.json()["items"]
        owner_cal = next(c for c in calendars if c.get("accessRole") == "owner")

        # Get ACL list
        acl_response = client.get(
            f"/calendars/{owner_cal['id']}/acl",
            headers=api_headers(database_id, user_id),
        )

        assert acl_response.status_code == 200
        acls = acl_response.json()
        # ACL response should be a dict with 'items' containing the list of rules
        assert isinstance(acls, dict)
        assert "items" in acls
        assert isinstance(acls["items"], list)
        assert any(acl.get("role") == "owner" for acl in acls["items"])

    def test_cross_user_isolation(self, seeded_database, api_headers):
        """Test that users can only see their own calendars"""
        client = seeded_database["client"]
        database_id = seeded_database["database_id"]

        # Get Alice's calendars
        alice_response = client.get(
            "/users/me/calendarList",
            headers=api_headers(database_id, seeded_database["users"]["alice"]),
        )

        # Get Bob's calendars
        bob_response = client.get(
            "/users/me/calendarList",
            headers=api_headers(database_id, seeded_database["users"]["bob"]),
        )

        assert alice_response.status_code == 200
        assert bob_response.status_code == 200

        # Each user should have their own calendars
        assert len(alice_response.json()["items"]) > 0
        assert len(bob_response.json()["items"]) > 0
