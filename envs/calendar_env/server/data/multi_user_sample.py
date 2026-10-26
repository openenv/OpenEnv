"""
Multi-User Sample Data for Calendar Application
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from .enhanced_event_seed_data import generate_enhanced_event_sql
from .watch_channel_seed_data import get_watch_channel_sql


def get_multi_user_sample_data() -> Dict[str, Any]:
    """
    Generate sample data for multiple users demonstrating multi-user scenarios
    """

    # Sample users
    users = [
        {
            "user_id": "alice_manager",
            "email": "alice.manager@techcorp.com",
            "name": "Alice Johnson",
            "given_name": "Alice",
            "family_name": "Johnson",
            "static_token": "ya29.A0ARrdaM-k9Vq7GzY2pL4mQf8sN1xT0bR3uHcJWv5yKzP6eF2.qwErTyUIopASDfGhJkLzXcVbNm12_34-56",
            "timezone": "America/New_York",
            "role": "Project Manager",
        },
        {
            "user_id": "bob_developer",
            "email": "bob.smith@techcorp.com",
            "name": "Bob Smith",
            "given_name": "Bob",
            "family_name": "Smith",
            "static_token": "ya29.A0ARrdaM-Zx8Nw3Q4pVb6Ls9R1mT0cG2uF5yH7kJd8sA1Lq2.wErtYuIoPaSdFgHjKlZxCvBnM987_65-43",
            "timezone": "America/Los_Angeles",
            "role": "Senior Developer",
        },
        {
            "user_id": "carol_designer",
            "email": "carol.white@techcorp.com",
            "name": "Carol White",
            "given_name": "Carol",
            "family_name": "White",
            "static_token": "ya29.A0ARrdaM-b7Hc5Vn2Qm8R1sT4pL0xY9wK3uF6jZ2eRc1.QaWsEdRfTgHyJuIkOlPzXcVbNmKjHgf_21-098",
            "timezone": "Europe/London",
            "role": "UX Designer",
        },
        {
            "user_id": "dave_sales",
            "email": "dave.brown@techcorp.com",
            "name": "Dave Brown",
            "given_name": "Dave",
            "family_name": "Brown",
            "static_token": "ya29.A0ARrdaM-p3Lk9Vb6Qw2Zx8N1sT4mH7gF5yR0uJc2ePq.ZxCvBnMlKjHgFfDsaQwErTyUiOpAsDfGhJk_77-11",
            "timezone": "America/Chicago",
            "role": "Sales Director",
        },
    ]

    # Alice's calendars (Project Manager)
    alice_calendars = [
        {
            "calendar_id": "alice-primary",
            "user_id": "alice_manager",
            "summary": "Alice Johnson",
            "description": "Primary calendar for Alice Johnson - Project Manager",
            "time_zone": "America/New_York",
            "is_primary": True,
            "color_id": "1",
        },
        {
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "summary": "Project Management",
            "description": "Project meetings, deadlines, and milestones",
            "time_zone": "America/New_York",
            "is_primary": False,
            "color_id": "7",
        },
        {
            "calendar_id": "alice-team",
            "user_id": "alice_manager",
            "summary": "Team Coordination",
            "description": "Team meetings, 1-on-1s, and team events",
            "time_zone": "America/New_York",
            "is_primary": False,
            "color_id": "11",
        },
    ]

    # Bob's calendars (Developer)
    bob_calendars = [
        {
            "calendar_id": "bob-primary",
            "user_id": "bob_developer",
            "summary": "Bob Smith",
            "description": "Primary calendar for Bob Smith - Senior Developer",
            "time_zone": "America/Los_Angeles",
            "is_primary": True,
            "color_id": "2",
        },
        {
            "calendar_id": "bob-development",
            "user_id": "bob_developer",
            "summary": "Development Schedule",
            "description": "Sprint planning, code reviews, and development tasks",
            "time_zone": "America/Los_Angeles",
            "is_primary": False,
            "color_id": "9",
        },
        {
            "calendar_id": "bob-personal",
            "user_id": "bob_developer",
            "summary": "Personal Time",
            "description": "Personal appointments and time off",
            "time_zone": "America/Los_Angeles",
            "is_primary": False,
            "color_id": "14",
        },
    ]

    # Carol's calendars (Designer)
    carol_calendars = [
        {
            "calendar_id": "carol-primary",
            "user_id": "carol_designer",
            "summary": "Carol White",
            "description": "Primary calendar for Carol White - UX Designer",
            "time_zone": "Europe/London",
            "is_primary": True,
            "color_id": "4",
        },
        {
            "calendar_id": "carol-design",
            "user_id": "carol_designer",
            "summary": "Design Work",
            "description": "Design sessions, user research, and creative time",
            "time_zone": "Europe/London",
            "is_primary": False,
            "color_id": "16",
        },
    ]

    # Dave's calendars (Sales)
    dave_calendars = [
        {
            "calendar_id": "dave-primary",
            "user_id": "dave_sales",
            "summary": "Dave Brown",
            "description": "Primary calendar for Dave Brown - Sales Director",
            "time_zone": "America/Chicago",
            "is_primary": True,
            "color_id": "6",
        },
        {
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "summary": "Sales Activities",
            "description": "Client meetings, sales calls, and deals",
            "time_zone": "America/Chicago",
            "is_primary": False,
            "color_id": "23",
        },
    ]

    # Sample events for each user
    base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    alice_events = [
        {
            "event_id": "alice-event-1",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "summary": "Sprint Planning Meeting",
            "description": "Plan upcoming sprint with development team",
            "location": "Conference Room A",
            "start_datetime": base_date,
            "end_datetime": base_date + timedelta(hours=1),
            "status": "confirmed",
        },
        {
            "event_id": "alice-event-2",
            "calendar_id": "alice-team",
            "user_id": "alice_manager",
            "summary": "1-on-1 with Bob",
            "description": "Weekly check-in with Bob Smith",
            "location": "Alice's Office",
            "start_datetime": base_date + timedelta(days=1, hours=2),
            "end_datetime": base_date + timedelta(days=1, hours=2, minutes=30),
            "status": "confirmed",
        },
        {
            "event_id": "alice-event-3",
            "calendar_id": "alice-primary",
            "user_id": "alice_manager",
            "summary": "Board Meeting",
            "description": "Monthly board meeting presentation",
            "location": "Executive Conference Room",
            "start_datetime": base_date + timedelta(days=7),
            "end_datetime": base_date + timedelta(days=7, hours=2),
            "status": "confirmed",
        },
    ]

    bob_events = [
        {
            "event_id": "bob-event-1",
            "calendar_id": "bob-development",
            "user_id": "bob_developer",
            "summary": "Code Review Session",
            "description": "Review pull requests from junior developers",
            "location": "Development Room",
            "start_datetime": base_date + timedelta(hours=3),
            "end_datetime": base_date + timedelta(hours=4),
            "status": "confirmed",
        },
        {
            "event_id": "bob-event-2",
            "calendar_id": "bob-primary",
            "user_id": "bob_developer",
            "summary": "Architecture Discussion",
            "description": "Discuss system architecture for new feature",
            "location": "Video Call",
            "start_datetime": base_date + timedelta(days=2),
            "end_datetime": base_date + timedelta(days=2, hours=1, minutes=30),
            "status": "confirmed",
        },
        {
            "event_id": "bob-event-3",
            "calendar_id": "bob-personal",
            "user_id": "bob_developer",
            "summary": "Dentist Appointment",
            "description": "Annual dental checkup",
            "location": "Downtown Dental",
            "start_datetime": base_date + timedelta(days=5, hours=5),
            "end_datetime": base_date + timedelta(days=5, hours=6),
            "status": "confirmed",
        },
    ]

    carol_events = [
        {
            "event_id": "carol-event-1",
            "calendar_id": "carol-design",
            "user_id": "carol_designer",
            "summary": "User Research Session",
            "description": "Interview users about new feature requirements",
            "location": "Research Lab",
            "start_datetime": base_date + timedelta(hours=1),
            "end_datetime": base_date + timedelta(hours=3),
            "status": "confirmed",
        },
        {
            "event_id": "carol-event-2",
            "calendar_id": "carol-primary",
            "user_id": "carol_designer",
            "summary": "Design Review",
            "description": "Present mockups to stakeholders",
            "location": "Design Studio",
            "start_datetime": base_date + timedelta(days=3, hours=2),
            "end_datetime": base_date + timedelta(days=3, hours=3, minutes=30),
            "status": "confirmed",
        },
    ]

    dave_events = [
        {
            "event_id": "dave-event-1",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "summary": "Client Demo",
            "description": "Product demonstration for potential enterprise client",
            "location": "Client Office",
            "start_datetime": base_date + timedelta(days=1),
            "end_datetime": base_date + timedelta(days=1, hours=2),
            "status": "confirmed",
        },
        {
            "event_id": "dave-event-2",
            "calendar_id": "dave-primary",
            "user_id": "dave_sales",
            "summary": "Sales Team Meeting",
            "description": "Weekly sales team sync and pipeline review",
            "location": "Sales Conference Room",
            "start_datetime": base_date + timedelta(days=4),
            "end_datetime": base_date + timedelta(days=4, hours=1),
            "status": "confirmed",
        },
    ]

    settings = [
        {
            "id": "alice_timezone",
            "user_id": "alice_manager",
            "value": "America/New_York",
        },
        {
            "id": "bob_timezone",
            "user_id": "bob_developer",
            "value": "America/Los_Angeles",
        },
        {"id": "carol_timezone", "user_id": "carol_designer", "value": "Europe/London"},
        {"id": "dave_timezone", "user_id": "dave_sales", "value": "America/Chicago"},
    ]

    scopes = [
        {"id": "scope-alice", "type": "user", "value": "alice.manager@techcorp.com"},
        {"id": "scope-bob", "type": "user", "value": "bob.smith@techcorp.com"},
        {"id": "scope-carol", "type": "user", "value": "carol.white@techcorp.com"},
        {"id": "scope-dave", "type": "user", "value": "dave.brown@techcorp.com"},
        {"id": "scope-group", "type": "group", "value": "product-team@techcorp.com"},
        {"id": "scope-domain", "type": "domain", "value": "techcorp.com"},
        {"id": "scope-public", "type": "default", "value": "public"},
    ]

    acls = [
        # Alice's calendar ACLs (owner of her calendars)
        {
            "id": "acl-alice-primary",
            "calendar_id": "alice-primary",
            "user_id": "alice_manager",
            "role": "owner",
            "scope_id": "scope-alice",
            "etag": "etag-alice-primary",
        },
        {
            "id": "acl-alice-projects",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "role": "owner",
            "scope_id": "scope-alice",
            "etag": "etag-alice-projects",
        },
        {
            "id": "acl-alice-team",
            "calendar_id": "alice-team",
            "user_id": "alice_manager",
            "role": "owner",
            "scope_id": "scope-alice",
            "etag": "etag-alice-team",
        },
        # Bob's calendar ACLs (owner of his calendars)
        {
            "id": "acl-bob-primary",
            "calendar_id": "bob-primary",
            "user_id": "bob_developer",
            "role": "owner",
            "scope_id": "scope-bob",
            "etag": "etag-bob-primary",
        },
        {
            "id": "acl-bob-development",
            "calendar_id": "bob-development",
            "user_id": "bob_developer",
            "role": "owner",
            "scope_id": "scope-bob",
            "etag": "etag-bob-development",
        },
        {
            "id": "acl-bob-personal",
            "calendar_id": "bob-personal",
            "user_id": "bob_developer",
            "role": "owner",
            "scope_id": "scope-bob",
            "etag": "etag-bob-personal",
        },
        # Carol's calendar ACLs (owner of her calendars)
        {
            "id": "acl-carol-primary",
            "calendar_id": "carol-primary",
            "user_id": "carol_designer",
            "role": "owner",
            "scope_id": "scope-carol",
            "etag": "etag-carol-primary",
        },
        {
            "id": "acl-carol-design",
            "calendar_id": "carol-design",
            "user_id": "carol_designer",
            "role": "owner",
            "scope_id": "scope-carol",
            "etag": "etag-carol-design",
        },
        # Dave's calendar ACLs (owner of his calendars)
        {
            "id": "acl-dave-primary",
            "calendar_id": "dave-primary",
            "user_id": "dave_sales",
            "role": "owner",
            "scope_id": "scope-dave",
            "etag": "etag-dave-primary",
        },
        {
            "id": "acl-dave-sales",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "role": "owner",
            "scope_id": "scope-dave",
            "etag": "etag-dave-sales",
        },
        # Shared access examples
        {
            "id": "acl-shared-1",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "role": "writer",
            "scope_id": "scope-bob",
            "etag": "etag-shared-1",
        },
        {
            "id": "acl-shared-2",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "role": "reader",
            "scope_id": "scope-carol",
            "etag": "etag-shared-2",
        },
    ]

    return {
        "users": users,
        "calendars": alice_calendars + bob_calendars + carol_calendars + dave_calendars,
        "events": alice_events + bob_events + carol_events + dave_events,
        "settings": settings,
        "scopes": scopes,
        "acls": acls,
        "description": "Multi-user sample data with 4 users (Alice-PM, Bob-Dev, Carol-Design, Dave-Sales) demonstrating isolated data per user",
    }


def get_multi_user_sql(database_name: str = "multi_user_calendar") -> str:
    """
    Generate SQL statements for multi-user sample data
    """
    data = get_multi_user_sample_data()

    sql_statements = []

    # Header
    sql_statements.append(f"-- Multi-User Calendar Sample Data for {database_name}")
    sql_statements.append(f"-- Generated on: {datetime.now().isoformat()}")
    sql_statements.append(
        "-- Contains sample data for 4 users with isolated calendars and events"
    )
    sql_statements.append("")

    # Users
    sql_statements.append("-- Users")
    sql_statements.append(
        "INSERT INTO users (user_id, email, name, given_name, family_name, static_token, timezone, is_active, is_verified, created_at, updated_at) VALUES"
    )

    user_values = []
    for user in data["users"]:
        # Escape single quotes by doubling them
        name = user["name"].replace("'", "''")
        given_name = user.get("given_name", "").replace("'", "''")
        family_name = user.get("family_name", "").replace("'", "''")

        user_values.append(
            f"('{user['user_id']}', '{user['email']}', '{name}', "
            f"'{given_name}', '{family_name}', "
            f"'{user['static_token']}', '{user['timezone']}', 1, 1, "
            f"datetime('now'), datetime('now'))"
        )

    sql_statements.append(",\n".join(user_values) + ";")
    sql_statements.append("")

    # Calendars
    sql_statements.append("-- Calendars")
    sql_statements.append(
        "INSERT INTO calendars (calendar_id, user_id, summary, description, time_zone, is_primary, color_id, hidden, selected, deleted, created_at, updated_at) VALUES"
    )

    calendar_values = []
    for calendar in data["calendars"]:
        # Escape single quotes by doubling them
        summary = calendar["summary"].replace("'", "''")
        description = calendar.get("description", "").replace("'", "''")

        calendar_values.append(
            f"('{calendar['calendar_id']}', '{calendar['user_id']}', '{summary}', "
            f"'{description}', '{calendar['time_zone']}', "
            f"{1 if calendar.get('is_primary') else 0}, '{calendar.get('color_id', '1')}', "
            f"0, 1, 0, datetime('now'), datetime('now'))"
        )

    sql_statements.append(",\n".join(calendar_values) + ";")
    sql_statements.append("")

    # Events
    sql_statements = generate_enhanced_event_sql(sql_statements)

    # Colors (shared across users)
    sql_statements.append("-- Google Calendar Colors (shared across all users)")
    from data.google_colors import GOOGLE_CALENDAR_COLORS

    sql_statements.append(
        "INSERT INTO colors (color_id, color_type, background, foreground, created_at, updated_at) VALUES"
    )

    color_values = []
    for color in GOOGLE_CALENDAR_COLORS:
        color_values.append(
            f"('{color['color_id']}', '{color['color_type'].upper()}', '{color['background']}', "
            f"'{color['foreground']}', datetime('now'), datetime('now'))"
        )

    sql_statements.append(",\n".join(color_values) + ";")
    sql_statements.append("")

    # Settings
    sql_statements.append("-- Calendar Settings")
    sql_statements.append("INSERT INTO settings (id, user_id, value) VALUES")
    setting_values = []
    for setting in data["settings"]:
        setting_values.append(
            f"('{setting['id']}', '{setting['user_id']}', '{setting['value']}')"
        )
    sql_statements.append(",\n".join(setting_values) + ";\n")
    sql_statements.append("")

    # Scopes
    sql_statements.append("-- ACL Scopes")
    sql_statements.append("INSERT INTO scopes (id, type, value) VALUES")
    scope_values = []
    for scope in data["scopes"]:
        scope_values.append(f"('{scope['id']}', '{scope['type']}', '{scope['value']}')")
    sql_statements.append(",\n".join(scope_values) + ";\n")
    sql_statements.append("")

    # ACLs
    sql_statements.append("-- Access Control Rules")
    sql_statements.append(
        "INSERT INTO acls (id, calendar_id, user_id, role, scope_id, etag) VALUES"
    )
    acl_values = []
    for acl in data["acls"]:
        acl_values.append(
            f"('{acl['id']}', '{acl['calendar_id']}', '{acl['user_id']}', "
            f"'{acl['role']}', '{acl['scope_id']}', '{acl['etag']}')"
        )
    sql_statements.append(",\n".join(acl_values) + ";\n")
    sql_statements.append("")

    sql_statements = get_watch_channel_sql(sql_statements)

    return "\n".join(sql_statements)
