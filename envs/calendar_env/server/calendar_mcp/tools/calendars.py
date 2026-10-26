"""
Calendars Tools Module

This module contains tools related to calendar management.
Covers calendar CRUD operations, clearing, and listing functionality.
"""

CALENDARS_TOOLS = [
    {
        "name": "create_calendar",
        "description": """Create a new secondary calendar.

        Creates a new calendar following Google Calendar API v3 structure. This endpoint
        strictly creates a secondary calendar. It does not create or promote a primary calendar.
        The user must already exist; otherwise a 404 is returned.

        Request Body Requirements:
          - summary: Required. Calendar title (1-255 characters)
          - description: Optional. Calendar description (max 1000 characters)
          - location: Optional. Geographic location (max 500 characters)
          - timeZone: Optional. Calendar timezone in IANA format (default: UTC)
          - conferenceProperties: Optional. Conference solution settings

        Notes:
          - This operation cannot create a primary calendar. Use account provisioning or separate tooling to ensure a primary exists.

        Response Structure:
          - Returns the created calendar with Google Calendar API v3 format:
            * kind: "calendar#calendar"
            * etag: ETag of the resource
            * id: Unique calendar identifier (UUID)
            * summary: Calendar title
            * description: Calendar description (if provided)
            * location: Calendar location (if provided)
            * timeZone: Calendar timezone
            * conferenceProperties: Conference settings (if provided)

        Status Codes:
          - 201: Created - Calendar created successfully
          - 400: Bad Request - Invalid calendar data
          - 404: Not Found - User not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Calendar title (1-255 characters)",
                    "minLength": 1,
                    "maxLength": 255,
                },
                "description": {
                    "type": "string",
                    "description": "Calendar description (max 1000 characters)",
                    "maxLength": 1000,
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location (max 500 characters)",
                    "maxLength": 500,
                },
                "timeZone": {
                    "type": "string",
                    "description": "Calendar timezone in IANA format (default: UTC)",
                    "default": "UTC",
                },
                "conferenceProperties": {
                    "type": "object",
                    "description": "Conference solution settings",
                    "properties": {
                        "allowedConferenceSolutionTypes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "eventHangout",
                                    "eventNamedHangout",
                                    "hangoutsMeet",
                                ],
                            },
                            "description": "Allowed conference solution types",
                        }
                    },
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "get_calendar",
        "description": """Retrieve a specific calendar by its ID (supports 'primary').

        Returns calendar metadata following Google Calendar API v3 structure.
        Supports using the special keyword 'primary' as the calendar identifier to
        target the user's primary calendar.

        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or the keyword 'primary'

        Response Structure:
          - Returns calendar with Google Calendar API v3 format:
            * kind: "calendar#calendar"
            * etag: ETag of the resource
            * id: Unique calendar identifier
            * summary: Calendar title
            * description: Calendar description (if present)
            * location: Calendar location (if present)
            * timeZone: Calendar timezone
            * conferenceProperties: Conference settings (if present)

        Status Codes:
          - 200: Success - Calendar retrieved successfully
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID) or the keyword 'primary'",
                    "minLength": 1,
                }
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "patch_calendar",
        "description": """Partially update calendar metadata (cannot change which calendar is primary).

        Partially updates calendar metadata following Google Calendar API v3 structure.
        Only provided fields will be updated, others remain unchanged.
        You can update both primary and secondary calendars.
        Supports using the special keyword 'primary' as the calendar identifier.

        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or the keyword 'primary'

        Request Body (all optional):
          - summary: Calendar title (1-255 characters)
          - description: Calendar description (max 1000 characters)
          - location: Geographic location (max 500 characters)
          - timeZone: Calendar timezone in IANA format
          - conferenceProperties: Conference solution settings

        Restrictions:
          - Cannot modify which calendar is primary (the is_primary flag is immutable via PATCH)
          - At least one field must be provided for update
          - Primary status is automatically assigned and protected

        Response Structure:
          - Returns updated calendar with Google Calendar API v3 format

        Status Codes:
          - 200: Success - Calendar updated successfully
          - 400: Bad Request - No fields provided or attempt to modify primary status
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID) or the keyword 'primary'",
                    "minLength": 1,
                },
                "summary": {
                    "type": "string",
                    "description": "Calendar title (1-255 characters)",
                    "minLength": 1,
                    "maxLength": 255,
                },
                "description": {
                    "type": "string",
                    "description": "Calendar description (max 1000 characters)",
                    "maxLength": 1000,
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location (max 500 characters)",
                    "maxLength": 500,
                },
                "timeZone": {
                    "type": "string",
                    "description": "Calendar timezone in IANA format",
                },
                "conferenceProperties": {
                    "type": "object",
                    "description": "Conference solution settings",
                    "properties": {
                        "allowedConferenceSolutionTypes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "eventHangout",
                                    "eventNamedHangout",
                                    "hangoutsMeet",
                                ],
                            },
                            "description": "Allowed conference solution types",
                        }
                    },
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "update_calendar",
        "description": """Fully update calendar metadata (cannot change which calendar is primary).

        Completely updates calendar metadata following Google Calendar API v3 structure.
        All provided fields replace existing values. Optional fields omitted will remain unchanged
        in the current implementation. You can update both primary and secondary calendars.
        Primary calendar status (which calendar is primary) cannot be modified via this endpoint.
        Supports using the special keyword 'primary' as the calendar identifier.

        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or the keyword 'primary'

        Request Body (all optional; null clears for description/location/conferenceProperties):
          - summary: Calendar title (1-255 characters)
          - description: Calendar description (max 1000 characters) - null to clear
          - location: Geographic location (max 500 characters) - null to clear
          - timeZone: Calendar timezone in IANA format
          - conferenceProperties: Conference solution settings - null to clear

        Restrictions:
          - Cannot modify which calendar is primary (the is_primary flag is immutable via PUT)

        Response Structure:
          - Returns updated calendar with Google Calendar API v3 format

        Status Codes:
          - 200: Success - Calendar updated successfully
          - 400: Bad Request - Invalid update data or attempt to modify primary status
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID) or the keyword 'primary'",
                    "minLength": 1,
                },
                "summary": {
                    "type": "string",
                    "description": "Calendar title (1-255 characters)",
                    "minLength": 1,
                    "maxLength": 255,
                },
                "description": {
                    "type": "string",
                    "description": "Calendar description (max 1000 characters)",
                    "maxLength": 1000,
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location (max 500 characters)",
                    "maxLength": 500,
                },
                "timeZone": {
                    "type": "string",
                    "description": "Calendar timezone in IANA format",
                },
                "conferenceProperties": {
                    "type": "object",
                    "description": "Conference solution settings",
                    "properties": {
                        "allowedConferenceSolutionTypes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "eventHangout",
                                    "eventNamedHangout",
                                    "hangoutsMeet",
                                ],
                            },
                            "description": "Allowed conference solution types",
                        }
                    },
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "delete_calendar",
        "description": """Delete a secondary calendar (cannot delete primary calendar).

        Deletes a calendar following Google Calendar API v3 behavior.
        Primary calendars cannot be deleted - use clear operation instead.

        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID)

        Primary Calendar Protection:
          - Primary calendars cannot be deleted
          - Attempting to delete primary calendar returns 400 Bad Request
          - Use clear_calendar tool to remove events from primary calendar

        Cascade Behavior:
          - Deleting a calendar also deletes all associated events
          - This operation is irreversible

        Response Structure:
          - Returns 204 No Content on successful deletion
          - No response body as per Google Calendar API v3

        Status Codes:
          - 204: No Content - Calendar deleted successfully
          - 400: Bad Request - Attempted to delete primary calendar
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID)",
                    "minLength": 1,
                }
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "clear_calendar",
        "description": """Clear all events from a calendar (useful for primary calendars).

        Clears all events from a calendar following Google Calendar API v3 behavior.
        This is the recommended way to "reset" a primary calendar since primary calendars cannot be deleted.

        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID)

        Operation Details:
          - Removes all events from the specified calendar
          - Calendar metadata remains unchanged
          - Useful for primary calendars that cannot be deleted
          - Can be used on any calendar (primary or secondary)

        Response:
          - Returns 204 No Content on successful clear (no response body)

        Status Codes:
          - 204: No Content - Calendar cleared successfully
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID)",
                    "minLength": 1,
                }
            },
            "required": ["calendarId"],
        },
    },
]
