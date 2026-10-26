"""
FreeBusy MCP tools for Google Calendar API v3 compatibility
All FreeBusy API endpoints with clean tool definitions
"""

FREEBUSY_TOOLS = [
    {
        "name": "query_freebusy",
        "description": """Query free/busy information for a set of calendars.
        
        Returns free/busy information for specified calendars following Google Calendar API v3 structure.
        Shows busy time periods when calendars have confirmed events that block time.
        Essential for scheduling meetings and finding available time slots.
        
        Request Body Requirements:
          - timeMin: Required. Lower bound for the query (RFC3339 timestamp)
          - timeMax: Required. Upper bound for the query (RFC3339 timestamp)
          - items: Required. List of calendar identifiers to query
        
        Optional Parameters:
          - timeZone: Time zone for the query (default: UTC)
          - groupExpansionMax: Maximum number of calendars to expand for groups
          - calendarExpansionMax: Maximum number of events to expand for calendars
        
        Time Period Handling:
          - Only confirmed events block time (status = "confirmed")
          - Transparent events do not block time
          - Overlapping events are merged into continuous busy periods
          - Results are clipped to the requested time range
        
        Response Structure:
          - Returns FreeBusy resource with Google Calendar API v3 format:
            * kind: "calendar#freeBusy"
            * timeMin: Query start time
            * timeMax: Query end time
            * calendars: Object with calendar IDs as keys
            * Each calendar contains:
              - busy: Array of busy time periods
              - errors: Array of errors (if any)
        
        Status Codes:
          - 200: Success - FreeBusy information retrieved successfully
          - 400: Bad Request - Invalid query parameters
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeMin": {
                    "type": "string",
                    "description": "Lower bound for the query (RFC3339 timestamp)",
                },
                "timeMax": {
                    "type": "string",
                    "description": "Upper bound for the query (RFC3339 timestamp)",
                },
                "timeZone": {
                    "type": "string",
                    "description": "Time zone for the query (IANA timezone, default: UTC)",
                },
                "groupExpansionMax": {
                    "type": "integer",
                    "description": "Maximum number of calendars to expand for groups",
                },
                "calendarExpansionMax": {
                    "type": "integer",
                    "description": "Maximum number of events to expand for calendars",
                },
                "items": {
                    "type": "array",
                    "description": "List of calendar identifiers to query",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Calendar identifier",
                            }
                        },
                        "required": ["id"],
                    },
                },
            },
            "required": ["timeMin", "timeMax", "items"],
        },
    },
]


# Additional helper information for FreeBusy functionality
FREEBUSY_CONCEPTS = {
    "busy_time_calculation": {
        "description": "How busy times are calculated from calendar events",
        "rules": [
            "Only events with status='confirmed' block time",
            "Events with transparency='transparent' do not block time",
            "All-day events block the entire day",
            "Overlapping events are merged into continuous periods",
            "Event times are clipped to the query time range",
        ],
    },
    "time_zone_handling": {
        "description": "How timezones are handled in FreeBusy queries",
        "details": [
            "Query times should be in RFC3339 format",
            "All calculations are done in UTC internally",
            "Results are returned in the requested timezone",
            "Event times are converted from their native timezone",
            "Default timezone is UTC if not specified",
        ],
    },
    "error_handling": {
        "description": "How errors are handled in FreeBusy responses",
        "scenarios": [
            "Calendar not found: Returns error in calendar result",
            "No access to calendar: Returns error in calendar result",
            "Invalid time range: Returns 400 Bad Request",
            "Too many calendars: Returns 400 Bad Request",
            "Internal errors: Calendar marked with backend error",
        ],
    },
    "performance_considerations": {
        "description": "Performance aspects of FreeBusy queries",
        "guidelines": [
            "Limit time range to reasonable periods (max 366 days)",
            "Limit number of calendars per query (max 50)",
            "Use batch queries for multiple time ranges efficiently",
            "Consider caching for frequently accessed calendars",
        ],
    },
}


# Sample usage examples for documentation
FREEBUSY_EXAMPLES = {
    "simple_query": {
        "description": "Query busy times for a single calendar today",
        "request": {
            "timeMin": "2024-01-15T00:00:00Z",
            "timeMax": "2024-01-16T00:00:00Z",
            "timeZone": "UTC",
            "items": [{"id": "primary"}],
        },
    },
    "multiple_calendars": {
        "description": "Query multiple calendars for the next week",
        "request": {
            "timeMin": "2024-01-15T00:00:00Z",
            "timeMax": "2024-01-22T00:00:00Z",
            "timeZone": "America/New_York",
            "items": [
                {"id": "primary"},
                {"id": "work@company.com"},
                {"id": "team@company.com"},
            ],
        },
    },
}
