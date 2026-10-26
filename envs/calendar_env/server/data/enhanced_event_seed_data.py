"""
Enhanced Event Seed Data with all new fields and related tables
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def get_enhanced_event_seed_data() -> Dict[str, Any]:
    """
    Generate comprehensive seed data demonstrating all Event model features including:
    - All new Event fields (conference_data, reminders, extended_properties, etc.)
    - Attendees with various statuses and roles
    - Attachments (file URLs)
    - Working location properties for different work modes
    """

    # Base datetime for consistent relative dates
    base_date = datetime.now(timezone.utc).replace(
        hour=9, minute=0, second=0, microsecond=0
    )

    # Office locations - required for working location properties
    office_locations_data = [
        {
            "id": "office-building-1",
            "buildingId": "building-1",
            "deskId": None,
            "floorId": None,
            "floorSectionId": None,
            "label": "TechCorp Main Campus - Building 1",
        },
        {
            "id": "office-building-2-floor-3",
            "buildingId": "building-2",
            "deskId": "desk-W3-45",
            "floorId": "floor-3",
            "floorSectionId": "west-wing",
            "label": "TechCorp Main Campus - Building 2, Floor 3, West Wing, Desk W3-45",
        },
        {
            "id": "office-meeting-room-a",
            "buildingId": "building-1",
            "deskId": None,
            "floorId": "floor-1",
            "floorSectionId": "conference-area",
            "label": "Conference Room A - Building 1",
        },
    ]

    # Events data -  to match model exactly
    events_data = [
        {
            "event_id": "event-corrected-001",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "organizer_id": "alice_manager",
            "organizer_email": "alice.johnson@techcorp.com",
            "organizer_display_name": "Alice Johnson",
            "organizer_self": True,
            "summary": "Sprint Planning & Architecture Review",
            "description": "Detailed sprint planning session with architecture discussion for Q4 features. We'll review user stories, estimate effort, and plan the technical approach.",
            "location": "Conference Room A, Building 1",
            "start_datetime": base_date,
            "end_datetime": base_date + timedelta(hours=2),
            "start_timezone": "America/New_York",
            "end_timezone": "America/New_York",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date,
            "originalStartTime_timeZone": "America/New_York",
            "recurrence": None,
            "status": "confirmed",
            "visibility": "default",
            "color_id": "7",
            "eventType": "default",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": True,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 1,
            "iCalUID": "event-corrected-001@gmail.com",
            "source": {
                "title": "Sprint Planning Board",
                "url": "https://jira.techcorp.com/sprint-planning-q4",
            },
        },
        {
            "event_id": "event-corrected-002",
            "calendar_id": "bob-development",
            "user_id": "bob_developer",
            "organizer_id": "bob_developer",
            "organizer_email": "bob.smith@techcorp.com",
            "organizer_display_name": "Bob Smith",
            "organizer_self": True,
            "summary": "Deep Work: Core Algorithm Implementation",
            "description": "Focused development time for implementing the new search algorithm. No interruptions please.",
            "location": "Developer Workspace, Building 2",
            "start_datetime": base_date + timedelta(days=1, hours=1),
            "end_datetime": base_date + timedelta(days=1, hours=4),
            "start_timezone": "America/Los_Angeles",
            "end_timezone": "America/Los_Angeles",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=1, hours=1),
            "originalStartTime_timeZone": "America/Los_Angeles",
            "recurrence": None,
            "status": "confirmed",
            "visibility": "private",
            "color_id": "9",
            "eventType": "focusTime",
            "focusTimeProperties": {
                "autoDeclineMode": "declineNone",
                "declineMessage": "I'm in focus time. Please reschedule or reach out via Slack for urgent matters.",
                "chatStatus": "doNotDisturb",
            },
            "guestsCanInviteOthers": False,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": False,
            "outOfOfficeProperties": None,
            "sequence": 0,
            "source": None,
            "iCalUID": "event-corrected-002@gmail.com",
        },
        {
            "event_id": "event-corrected-003",
            "calendar_id": "carol-primary",
            "user_id": "carol_designer",
            "organizer_id": "carol_designer",
            "organizer_email": "carol.white@techcorp.com",
            "organizer_display_name": "Carol White",
            "organizer_self": True,
            "summary": "Annual Leave - Family Vacation",
            "description": "Taking time off for family vacation. Will have limited access to email.",
            "location": "Bali, Indonesia",
            "start_datetime": base_date + timedelta(days=14),
            "end_datetime": base_date + timedelta(days=21),
            "start_timezone": "Asia/Makassar",
            "end_timezone": "Asia/Makassar",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=14),
            "originalStartTime_timeZone": "Asia/Makassar",
            "recurrence": None,
            "status": "confirmed",
            "visibility": "public",
            "color_id": "4",
            "eventType": "outOfOffice",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": False,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": {
                "autoDeclineMode": "declineAllConflictingInvitations",
                "declineMessage": "I'm currently on vacation and won't be available. For urgent design matters, please contact Sarah (sarah@techcorp.com). I'll respond to messages when I return.",
                "autoDeclineEventTypes": ["default", "focusTime", "workingLocation"],
            },
            "sequence": 0,
            "source": None,
            "iCalUID": "event-corrected-003@gmail.com",
        },
        {
            "event_id": "event-corrected-004",
            "calendar_id": "bob-primary",
            "user_id": "bob_developer",
            "organizer_id": "bob_developer",
            "organizer_email": "bob.smith@techcorp.com",
            "organizer_display_name": "Bob Smith",
            "organizer_self": True,
            "summary": "Office Day - Collaboration Sessions",
            "description": "In office today for team collaboration and pair programming sessions.",
            "location": None,
            "start_datetime": base_date + timedelta(days=2),
            "end_datetime": base_date + timedelta(days=2, hours=8),
            "start_timezone": "America/Los_Angeles",
            "end_timezone": "America/Los_Angeles",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=2),
            "originalStartTime_timeZone": "America/Los_Angeles",
            "recurrence": None,
            "status": "confirmed",
            "visibility": "public",
            "color_id": "2",
            "eventType": "workingLocation",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": True,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 0,
            "source": None,
            "iCalUID": "event-corrected-004@gmail.com",
        },
        {
            "event_id": "event-corrected-005",
            "calendar_id": "dave-primary",
            "user_id": "dave_sales",
            "organizer_id": "dave_sales",
            "organizer_email": "dave.brown@techcorp.com",
            "organizer_display_name": "Dave Brown",
            "organizer_self": True,
            "summary": "Dave's Birthday",
            "description": "Happy Birthday Dave!",
            "location": None,
            "start_datetime": base_date + timedelta(days=30),
            "end_datetime": base_date + timedelta(days=30, hours=1),
            "start_timezone": None,
            "end_timezone": None,
            "originalStartTime_date": (base_date + timedelta(days=30)).date(),
            "originalStartTime_dateTime": None,
            "originalStartTime_timeZone": "UTC",
            "recurrence": None,
            "status": "confirmed",
            "visibility": "public",
            "color_id": "6",
            "eventType": "birthday",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": True,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 0,
            "source": None,
            "iCalUID": "event-corrected-005@gmail.com",
        },
        {
            "event_id": "event-corrected-006",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "organizer_id": "dave_sales",
            "organizer_email": "dave.brown@techcorp.com",
            "organizer_display_name": "Dave Brown",
            "organizer_self": True,
            "recurring_event_id": "rec-event-001",
            "summary": "Enterprise Client Demo - TechCorp Solutions",
            "description": "Product demonstration for MegaCorp Inc. Focus on enterprise features, security, and scalability. Bring pricing sheets and technical specs.",
            "location": "MegaCorp Headquarters, 123 Business Ave, New York, NY",
            "start_datetime": base_date + timedelta(days=3, hours=2),
            "end_datetime": base_date + timedelta(days=3, hours=4),
            "start_timezone": "America/New_York",
            "end_timezone": "America/New_York",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=3, hours=2),
            "originalStartTime_timeZone": "America/New_York",
            "recurrence": ["RRULE:COUNT=2"],
            "status": "confirmed",
            "visibility": "default",
            "color_id": "1",
            "eventType": "default",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": False,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 2,
            "source": {
                "title": "CRM System - MegaCorp Deal",
                "url": "https://crm.techcorp.com/deals/megacorp-2024",
            },
            "iCalUID": "event-icalid-001@gmail.com",
        },
        {
            "event_id": "event-corrected-007",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "organizer_id": "dave_sales",
            "organizer_email": "dave.brown@techcorp.com",
            "organizer_display_name": "Dave Brown",
            "organizer_self": True,
            "recurring_event_id": "rec-event-001",
            "summary": "Enterprise Client Demo - TechCorp Solutions",
            "description": "Product demonstration for MegaCorp Inc. Focus on enterprise features, security, and scalability. Bring pricing sheets and technical specs.",
            "location": "MegaCorp Headquarters, 123 Business Ave, New York, NY",
            "start_datetime": base_date + timedelta(days=4, hours=2),
            "end_datetime": base_date + timedelta(days=4, hours=4),
            "start_timezone": "America/New_York",
            "end_timezone": "America/New_York",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=4, hours=2),
            "originalStartTime_timeZone": "America/New_York",
            "recurrence": ["RRULE:COUNT=2"],
            "status": "confirmed",
            "visibility": "default",
            "color_id": "1",
            "eventType": "default",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": False,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 2,
            "source": {
                "title": "CRM System - MegaCorp Deal",
                "url": "https://crm.techcorp.com/deals/megacorp-2024",
            },
            "iCalUID": "event-icalid-001@gmail.com",
        },
        {
            "event_id": "event-corrected-008",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "organizer_id": "dave_sales",
            "organizer_email": "dave.brown@techcorp.com",
            "organizer_display_name": "Dave Brown",
            "organizer_self": True,
            "recurring_event_id": "rec-event-001",
            "summary": "Enterprise Client Demo - TechCorp Solutions",
            "description": "Product demonstration for MegaCorp Inc. Focus on enterprise features, security, and scalability. Bring pricing sheets and technical specs.",
            "location": "MegaCorp Headquarters, 123 Business Ave, New York, NY",
            "start_datetime": base_date + timedelta(days=5, hours=2),
            "end_datetime": base_date + timedelta(days=5, hours=4),
            "start_timezone": "America/New_York",
            "end_timezone": "America/New_York",
            "originalStartTime_date": None,
            "originalStartTime_dateTime": base_date + timedelta(days=5, hours=2),
            "originalStartTime_timeZone": "America/New_York",
            "recurrence": ["RRULE:COUNT=2"],
            "status": "confirmed",
            "visibility": "default",
            "color_id": "1",
            "eventType": "default",
            "focusTimeProperties": None,
            "guestsCanInviteOthers": False,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "outOfOfficeProperties": None,
            "sequence": 2,
            "source": {
                "title": "CRM System - MegaCorp Deal",
                "url": "https://crm.techcorp.com/deals/megacorp-2024",
            },
            "iCalUID": "event-icalid-001@gmail.com",
        },
    ]

    recurring_event_data = [
        {
            "recurring_event_id": "rec-event-001",
            "original_recurrence": ["RRULE:COUNT=2"],
        }
    ]

    # ConferenceData - separate table (correct relationship name)
    conference_data = [
        {
            "id": "conf-corrected-001",
            "event_id": "event-corrected-001",
            "request_id": "req-sprint-planning-001",
            "solution_type": "hangoutsMeet",
            "status_code": "success",
            "meeting_uri": "https://meet.google.com/abc-defg-hij",
            "label": "Sprint Planning Meet",
        },
        {
            "id": "conf-corrected-002",
            "event_id": "event-corrected-006",
            "request_id": "req-client-demo-001",
            "solution_type": "hangoutsMeet",
            "status_code": "success",
            "meeting_uri": "https://meet.google.com/enterprise-demo-xyz",
            "label": "Client Demo Backup",
        },
    ]

    # BirthdayProperties - separate table (correct relationship name)
    birthday_properties = [
        {
            "id": "birthday-corrected-001",
            "event_id": "event-corrected-005",
            "type": "birthday",
        }
    ]

    # ExtendedProperties - separate table with scope enum (correct relationship name)
    extended_properties = [
        {
            "id": "ext-corrected-001",
            "event_id": "event-corrected-001",
            "scope": "private",
            "properties": {
                "departmentBudget": "engineering",
                "projectCode": "PROJ-2024-Q4",
            },
        },
        {
            "id": "ext-corrected-002",
            "event_id": "event-corrected-001",
            "scope": "shared",
            "properties": {"meetingType": "sprint_planning", "priority": "high"},
        },
        {
            "id": "ext-corrected-003",
            "event_id": "event-corrected-002",
            "scope": "private",
            "properties": {"taskType": "development", "estimatedComplexity": "high"},
        },
        {
            "id": "ext-corrected-004",
            "event_id": "event-corrected-003",
            "scope": "shared",
            "properties": {
                "backupContact": "sarah@techcorp.com",
                "vacationType": "personal",
            },
        },
        {
            "id": "ext-corrected-005",
            "event_id": "event-corrected-004",
            "scope": "private",
            "properties": {"commute_reminder": "Leave by 8:00 AM to avoid traffic"},
        },
        {
            "id": "ext-corrected-006",
            "event_id": "event-corrected-006",
            "scope": "private",
            "properties": {
                "dealValue": "$250000",
                "clientPriority": "high",
                "preparationTime": "2 hours",
            },
        },
        {
            "id": "ext-corrected-007",
            "event_id": "event-corrected-006",
            "scope": "shared",
            "properties": {"meetingType": "client_demo", "department": "sales"},
        },
    ]

    # Reminders - separate table with method enum (correct relationship name)
    reminders_data = [
        {
            "id": "rem-corrected-001",
            "event_id": "event-corrected-001",
            "method": "email",
            "minutes": 1440,  # 1 day before
            "use_default": False,
        },
        {
            "id": "rem-corrected-002",
            "event_id": "event-corrected-001",
            "method": "popup",
            "minutes": 30,  # 30 minutes before
            "use_default": False,
        },
        {
            "id": "rem-corrected-003",
            "event_id": "event-corrected-002",
            "method": "popup",
            "minutes": 15,  # 15 minutes before
            "use_default": False,
        },
        {
            "id": "rem-corrected-004",
            "event_id": "event-corrected-003",
            "method": "email",
            "minutes": 10080,  # 1 week before
            "use_default": False,
        },
        {
            "id": "rem-corrected-005",
            "event_id": "event-corrected-005",
            "method": "popup",
            "minutes": 10,  # Day of reminder
            "use_default": False,
        },
        {
            "id": "rem-corrected-006",
            "event_id": "event-corrected-006",
            "method": "email",
            "minutes": 2880,  # 2 days before
            "use_default": False,
        },
        {
            "id": "rem-corrected-007",
            "event_id": "event-corrected-006",
            "method": "popup",
            "minutes": 60,  # 1 hour before
            "use_default": False,
        },
        {
            "id": "rem-corrected-008",
            "event_id": "event-corrected-006",
            "method": "popup",
            "minutes": 15,  # 15 minutes before
            "use_default": False,
        },
    ]

    # Attendees data -  to match model exactly
    attendees_data = [
        # Sprint Planning attendees
        {
            "attendees_id": "att-corrected-001",
            "event_id": "event-corrected-001",
            "user_id": "alice_manager",
            "comment": "Looking forward to planning Q4!",
            "displayName": "Alice Johnson",
            "additionalGuests": 0,
            "optional": False,
            "resource": False,
            "responseStatus": "accepted",
        },
        {
            "attendees_id": "att-corrected-002",
            "event_id": "event-corrected-001",
            "user_id": "bob_developer",
            "comment": None,
            "displayName": "Bob Smith",
            "additionalGuests": 0,
            "optional": False,
            "resource": False,
            "responseStatus": "accepted",
        },
        {
            "attendees_id": "att-corrected-003",
            "event_id": "event-corrected-001",
            "user_id": "carol_designer",
            "comment": "Will join if no conflicts with user research session",
            "displayName": "Carol White",
            "additionalGuests": 0,
            "optional": True,
            "resource": False,
            "responseStatus": "tentative",
        },
        {
            "attendees_id": "att-corrected-004",
            "event_id": "event-corrected-001",
            "user_id": None,  # Resource doesn't have user_id
            "comment": None,
            "displayName": "Conference Room A",
            "additionalGuests": 0,
            "optional": False,
            "resource": True,
            "responseStatus": "accepted",
        },
        # Client Demo attendees
        {
            "attendees_id": "att-corrected-005",
            "event_id": "event-corrected-006",
            "user_id": "dave_sales",
            "comment": "Prepared demo materials and pricing",
            "displayName": "Dave Brown",
            "additionalGuests": 0,
            "optional": False,
            "resource": False,
            "responseStatus": "accepted",
        },
        {
            "attendees_id": "att-corrected-006",
            "event_id": "event-corrected-006",
            "user_id": None,  # External attendee
            "comment": "Bringing 2 technical team members",
            "displayName": "John Doe",
            "additionalGuests": 2,
            "optional": False,
            "resource": False,
            "responseStatus": "accepted",
        },
        {
            "attendees_id": "att-corrected-007",
            "event_id": "event-corrected-006",
            "user_id": "sarah_tech",
            "comment": "Technical support for enterprise questions",
            "displayName": "Sarah Chen",
            "additionalGuests": 0,
            "optional": True,
            "resource": False,
            "responseStatus": "accepted",
        },
        {
            "attendees_id": "att-corrected-008",
            "event_id": "event-corrected-006",
            "user_id": "alice_manager",
            "comment": None,
            "displayName": "Alice Johnson",
            "additionalGuests": 0,
            "optional": True,
            "resource": False,
            "responseStatus": "needsAction",
        },
    ]

    # Attachments for events
    attachments_data = [
        {
            "attachment_id": "attach-corrected-001",
            "event_id": "event-corrected-001",
            "file_url": "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view",
        },
        {
            "attachment_id": "attach-corrected-002",
            "event_id": "event-corrected-001",
            "file_url": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
        },
        {
            "attachment_id": "attach-corrected-003",
            "event_id": "event-corrected-006",
            "file_url": "https://drive.google.com/file/d/enterprise-demo-deck-2024/view",
        },
        {
            "attachment_id": "attach-corrected-004",
            "event_id": "event-corrected-006",
            "file_url": "https://drive.google.com/file/d/pricing-sheet-enterprise-2024/view",
        },
        {
            "attachment_id": "attach-corrected-005",
            "event_id": "event-corrected-006",
            "file_url": "https://docs.google.com/document/d/technical-specs-enterprise/edit",
        },
    ]

    # Working location properties - corrected structure with proper office location foreign key
    working_location_data = [
        {
            "working_location_id": "wl-corrected-001",
            "event_id": "event-corrected-004",
            "type": "officeLocation",
            "homeOffice": False,
            "customLocationLabel": None,
            "officeLocationId": "office-building-2-floor-3",  #  - foreign key to office_locations
        }
    ]

    return {
        "office_locations": office_locations_data,
        "events": events_data,
        "recurring_events": recurring_event_data,
        "conference_data": conference_data,
        "birthday_properties": birthday_properties,
        "extended_properties": extended_properties,
        "reminders": reminders_data,
        "attendees": attendees_data,
        "attachments": attachments_data,
        "working_location_properties": working_location_data,
        "description": "Event seed data that exactly matches the updated event.py model structure with proper field names and relationships",
    }


def generate_enhanced_event_sql(sql_statements) -> str:
    """
    Generate SQL INSERT statements for enhanced event seed data
    """
    data = get_enhanced_event_seed_data()

    # Office Locations
    sql_statements.append("-- Office Locations")
    sql_statements.append("INSERT INTO office_locations (")
    sql_statements.append("    id, buildingId, deskId, floorId, floorSectionId, label")
    sql_statements.append(") VALUES")

    office_values = []
    for office in data["office_locations"]:
        building_id = (
            "NULL" if not office.get("buildingId") else f"'{office['buildingId']}'"
        )
        desk_id = "NULL" if not office.get("deskId") else f"'{office['deskId']}'"
        floor_id = "NULL" if not office.get("floorId") else f"'{office['floorId']}'"
        floor_section_id = (
            "NULL"
            if not office.get("floorSectionId")
            else f"'{office['floorSectionId']}'"
        )

        office_values.append(
            f"('{office['id']}', {building_id}, {desk_id}, {floor_id}, {floor_section_id}, '{office['label']}')"
        )

    sql_statements.append(",\n".join(office_values) + ";")
    sql_statements.append("")

    # Recurring Events
    sql_statements.append("-- Recurring Events")
    sql_statements.append("INSERT INTO recurring_events (")
    sql_statements.append("    recurring_event_id, original_recurrence")
    sql_statements.append(") VALUES")

    recurring_events = []
    for rec_event in data["recurring_events"]:
        recurring_event_id = (
            "NULL"
            if not rec_event.get("recurring_event_id")
            else f"'{rec_event['recurring_event_id']}'"
        )
        original_recurrence = (
            "NULL"
            if not rec_event.get("original_recurrence")
            else f"'{json.dumps(rec_event['original_recurrence']).replace(chr(39), chr(39) + chr(39))}'"
        )

        recurring_events.append(f"({recurring_event_id},{original_recurrence})")

    sql_statements.append(",\n".join(recurring_events) + ";")
    sql_statements.append("")

    # Events
    sql_statements.append("-- Events")
    sql_statements.append("INSERT INTO events (")
    sql_statements.append(
        "    event_id, calendar_id, user_id, organizer_id, organizer_email, organizer_display_name, organizer_self,"
    )
    sql_statements.append("    recurring_event_id, summary, description, location,")
    sql_statements.append(
        "    start_datetime, end_datetime, start_timezone, end_timezone, originalStartTime_date, originalStartTime_dateTime, originalStartTime_timeZone, recurrence,"
    )
    sql_statements.append(
        "    status, visibility, color_id, iCalUID, eventType, focusTimeProperties,"
    )
    sql_statements.append(
        "    guestsCanInviteOthers, guestsCanModify, guestsCanSeeOtherGuests,"
    )
    sql_statements.append(
        "    outOfOfficeProperties, sequence, source, created_at, updated_at"
    )
    sql_statements.append(") VALUES")

    event_values = []
    for event in data["events"]:
        # Handle optional fields
        description = (
            "NULL"
            if not event.get("description")
            else f"'{event['description'].replace(chr(39), chr(39) + chr(39))}'"
        )
        location = (
            "NULL"
            if not event.get("location")
            else f"'{event['location'].replace(chr(39), chr(39) + chr(39))}'"
        )
        start_tz = (
            "NULL"
            if not event.get("start_timezone")
            else f"'{event['start_timezone']}'"
        )
        end_tz = (
            "NULL" if not event.get("end_timezone") else f"'{event['end_timezone']}'"
        )

        # Handle originalStartTime fields
        original_start_date = "NULL"
        original_start_datetime = "NULL"
        original_start_timezone = "NULL"

        if event.get("originalStartTime_date"):
            original_start_date = f"'{event['originalStartTime_date'].isoformat()}'"
        if event.get("originalStartTime_dateTime"):
            original_start_datetime = (
                f"'{event['originalStartTime_dateTime'].isoformat()}'"
            )
        if event.get("originalStartTime_timeZone"):
            original_start_timezone = f"'{event['originalStartTime_timeZone']}'"

        # Handle recurrence field - it can be a list or None
        recurring_event_id = (
            "NULL"
            if not event.get("recurring_event_id")
            else event["recurring_event_id"]
        )

        recurrence = "NULL"
        if event.get("recurrence"):
            if isinstance(event["recurrence"], list):
                recurrence = f"'{json.dumps(event['recurrence']).replace(chr(39), chr(39) + chr(39))}'"
            else:
                recurrence = (
                    f"'{event['recurrence'].replace(chr(39), chr(39) + chr(39))}'"
                )

        color_id = "NULL" if not event.get("color_id") else f"'{event['color_id']}'"

        # Handle JSON fields - using
        focus_props = "NULL"
        if event.get("focusTimeProperties"):
            focus_props = f"'{json.dumps(event['focusTimeProperties']).replace(chr(39), chr(39) + chr(39))}'"

        ooo_props = "NULL"
        if event.get("outOfOfficeProperties"):
            ooo_props = f"'{json.dumps(event['outOfOfficeProperties']).replace(chr(39), chr(39) + chr(39))}'"

        source = "NULL"
        if event.get("source"):
            source = (
                f"'{json.dumps(event['source']).replace(chr(39), chr(39) + chr(39))}'"
            )

        # Handle organizer fields
        organizer_id = (
            "NULL" if not event.get("organizer_id") else f"'{event['organizer_id']}'"
        )
        organizer_email = (
            "NULL"
            if not event.get("organizer_email")
            else f"'{event['organizer_email']}'"
        )
        organizer_display_name = (
            "NULL"
            if not event.get("organizer_display_name")
            else f"'{event['organizer_display_name']}'"
        )
        organizer_self = 1 if event.get("organizer_self", False) else 0

        event_values.append(
            f"('{event['event_id']}', '{event['calendar_id']}', '{event['user_id']}', {organizer_id}, {organizer_email}, {organizer_display_name}, {organizer_self}, "
            f"'{recurring_event_id}', '{event['summary'].replace(chr(39), chr(39) + chr(39))}', {description}, {location}, "
            f"'{event['start_datetime'].isoformat()}', '{event['end_datetime'].isoformat()}', "
            f"{start_tz}, {end_tz}, {original_start_date}, {original_start_datetime}, {original_start_timezone}, {recurrence}, "
            f"'{event['status']}', '{event['visibility']}', {color_id}, '{event['iCalUID']}', '{event['eventType']}', {focus_props}, "
            f"{1 if event['guestsCanInviteOthers'] else 0}, "
            f"{1 if event['guestsCanModify'] else 0}, "
            f"{1 if event['guestsCanSeeOtherGuests'] else 0}, "
            f"{ooo_props}, {event['sequence']}, {source}, datetime('now'), datetime('now'))"
        )

    sql_statements.append(",\n".join(event_values) + ";")
    sql_statements.append("")

    # ConferenceData
    if data["conference_data"]:
        sql_statements.append("-- ConferenceData")
        sql_statements.append("INSERT INTO conference_data (")
        sql_statements.append(
            "    id, event_id, request_id, solution_type, status_code, meeting_uri, label"
        )
        sql_statements.append(") VALUES")

        conf_values = []
        for conf in data["conference_data"]:
            request_id = (
                "NULL" if not conf.get("request_id") else f"'{conf['request_id']}'"
            )
            solution_type = (
                "NULL"
                if not conf.get("solution_type")
                else f"'{conf['solution_type']}'"
            )
            status_code = (
                "NULL" if not conf.get("status_code") else f"'{conf['status_code']}'"
            )
            meeting_uri = (
                "NULL" if not conf.get("meeting_uri") else f"'{conf['meeting_uri']}'"
            )
            label = "NULL" if not conf.get("label") else f"'{conf['label']}'"

            conf_values.append(
                f"('{conf['id']}', '{conf['event_id']}', {request_id}, {solution_type}, "
                f"{status_code}, {meeting_uri}, {label})"
            )

        sql_statements.append(",\n".join(conf_values) + ";")
        sql_statements.append("")

    # BirthdayProperties
    if data["birthday_properties"]:
        sql_statements.append("-- BirthdayProperties")
        sql_statements.append(
            "INSERT INTO birthday_properties (id, event_id, type) VALUES"
        )

        birthday_values = []
        for birthday in data["birthday_properties"]:
            birthday_values.append(
                f"('{birthday['id']}', '{birthday['event_id']}', '{birthday['type']}')"
            )

        sql_statements.append(",\n".join(birthday_values) + ";")
        sql_statements.append("")

    # ExtendedProperties
    if data["extended_properties"]:
        sql_statements.append("-- ExtendedProperties")
        sql_statements.append(
            "INSERT INTO extended_properties (id, event_id, scope, properties) VALUES"
        )

        ext_values = []
        for ext in data["extended_properties"]:
            properties = json.dumps(ext["properties"]).replace(
                chr(39), chr(39) + chr(39)
            )
            ext_values.append(
                f"('{ext['id']}', '{ext['event_id']}', '{ext['scope']}', '{properties}')"
            )

        sql_statements.append(",\n".join(ext_values) + ";")
        sql_statements.append("")

    # Reminders
    if data["reminders"]:
        sql_statements.append("-- Reminders")
        sql_statements.append(
            "INSERT INTO reminders (id, event_id, method, minutes, use_default) VALUES"
        )

        reminder_values = []
        for reminder in data["reminders"]:
            reminder_values.append(
                f"('{reminder['id']}', '{reminder['event_id']}', '{reminder['method']}', "
                f"{reminder['minutes']}, {1 if reminder['use_default'] else 0})"
            )

        sql_statements.append(",\n".join(reminder_values) + ";")
        sql_statements.append("")

    # Attendees - with
    sql_statements.append("-- Attendees")
    sql_statements.append("INSERT INTO attendees (")
    sql_statements.append("    attendees_id, event_id, user_id, comment, displayName,")
    sql_statements.append("    additionalGuests, optional, resource, responseStatus")
    sql_statements.append(") VALUES")

    attendee_values = []
    for attendee in data["attendees"]:
        user_id = "NULL" if not attendee.get("user_id") else f"'{attendee['user_id']}'"
        comment = (
            "NULL"
            if not attendee.get("comment")
            else f"'{attendee['comment'].replace(chr(39), chr(39) + chr(39))}'"
        )
        display_name = (
            "NULL"
            if not attendee.get("displayName")
            else f"'{attendee['displayName']}'"
        )

        attendee_values.append(
            f"('{attendee['attendees_id']}', '{attendee['event_id']}', {user_id}, {comment}, {display_name}, "
            f"{attendee['additionalGuests']}, {1 if attendee['optional'] else 0}, "
            f"{1 if attendee['resource'] else 0}, '{attendee['responseStatus']}')"
        )

    sql_statements.append(",\n".join(attendee_values) + ";")
    sql_statements.append("")

    # Attachments
    sql_statements.append("-- Attachments")
    sql_statements.append(
        "INSERT INTO attachments (attachment_id, event_id, file_url) VALUES"
    )

    attachment_values = []
    for attachment in data["attachments"]:
        attachment_values.append(
            f"('{attachment['attachment_id']}', '{attachment['event_id']}', '{attachment['file_url']}')"
        )

    sql_statements.append(",\n".join(attachment_values) + ";")
    sql_statements.append("")

    # Working Location Properties
    sql_statements.append("-- Working Location Properties")
    sql_statements.append("INSERT INTO working_location_properties (")
    sql_statements.append(
        "    working_location_id, event_id, type, homeOffice, customLocationLabel, officeLocationId"
    )
    sql_statements.append(") VALUES")

    wl_values = []
    for wl in data["working_location_properties"]:
        custom_label = (
            "NULL"
            if not wl.get("customLocationLabel")
            else f"'{wl['customLocationLabel']}'"
        )
        office_location_id = (
            "NULL" if not wl.get("officeLocationId") else f"'{wl['officeLocationId']}'"
        )

        wl_values.append(
            f"('{wl['working_location_id']}', '{wl['event_id']}', '{wl['type']}', "
            f"{1 if wl['homeOffice'] else 0}, {custom_label}, {office_location_id})"
        )

    sql_statements.append(",\n".join(wl_values) + ";")
    sql_statements.append("")

    return sql_statements
