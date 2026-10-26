"""
Events MCP tools for Google Calendar API v3 compatibility
All 11 Events API endpoints with clean tool definitions
"""

EVENTS_TOOLS = [
    {
        "name": "list_events",
        "description": """List events on the specified calendar.
        
        Returns events from a calendar following Google Calendar API v3 structure.
        Supports filtering by time range, search terms, and other parameters to retrieve relevant events.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier to list events from
        
        Optional Parameters:
          - eventTypes: Event types to return (string: default, birthday, focusTime, fromGmail, outOfOffice, workingLocation)
          - iCalUID: Specifies an event ID in the iCalendar format to retrieve (string)
          - maxAttendees: Maximum number of attendees to include in response (integer)
          - maxResults: Maximum number of events returned (1-2500, default 250)
          - orderBy: Order of events returned (startTime or updated)
          - pageToken: Token for pagination
          - privateExtendedProperty: Extended properties constraint for private properties (string, format: propertyName=value)
          - q: Text search terms to find events (mutually exclusive with iCalUID)
          - sharedExtendedProperty: Extended properties constraint for shared properties (string, format: propertyName=value)
          - showDeleted: Include deleted events (boolean)
          - showHiddenInvitations: Whether to include hidden invitations (boolean)
          - singleEvents: Expand recurring events into instances (boolean)
          - syncToken: Token for incremental sync
          - timeMax: Upper bound for event start time (RFC3339 timestamp)
          - timeMin: Lower bound for event start time (RFC3339 timestamp)
          - timeZone: Time zone used in the response (string, IANA timezone)
          - updatedMin: Lower bound for last modification time (RFC3339 timestamp)
        
        Response Structure:
          - Returns events collection with Google Calendar API v3 format:
            * kind: "calendar#events"
            * etag: ETag of the collection
            * items: Array of Event objects
            * nextPageToken: Token for next page (if applicable)
            * nextSyncToken: Token for incremental sync (if applicable)
        
        Status Codes:
          - 200: Success - Events retrieved successfully
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventTypes": {
                    "type": "string",
                    "description": "Event types to return. Possible values are: 'default' - Events that don't match any of the events below, 'outOfOffice' - Out of office events, 'focusTime' - Focus time events, 'workingLocation' - Working location events, 'fromGmail' - Events from Gmail (deprecated), 'birthday' - Birthday events. Optional. Multiple event types can be provided using repeated parameter instances",
                },
                "iCalUID": {
                    "type": "string",
                    "description": "Specifies an event ID in the iCalendar format to be provided in the response. Optional. Use this if you want to search for an event by its iCalendar ID. Mutually exclusive with q. Optional.",
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "The maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
                },
                "maxResults": {
                    "type": "integer",
                    "description": "Maximum number of events returned on one result page. By default the value is 250 events. The page size can never be larger than 2500 events. Optional.",
                    "minimum": 1,
                    "maximum": 2500,
                },
                "orderBy": {
                    "type": "string",
                    "description": "The order of the events returned in the result. Optional. The default is an unspecified, stable order.",
                    "enum": ["startTime", "updated"],
                },
                "pageToken": {
                    "type": "string",
                    "description": "Token specifying which result page to return. Optional.",
                },
                "privateExtendedProperty": {
                    "type": "string",
                    "description": "Extended properties constraint specified as propertyName=value. Matches only private properties. This parameter might be repeated multiple times to return events that match all given constraints.",
                },
                "q": {
                    "type": "string",
                    "description": "Free text search terms to find events that match these terms in any field, except for extended properties. Optional.",
                },
                "sharedExtendedProperty": {
                    "type": "string",
                    "description": "Extended properties constraint specified as propertyName=value. Matches only shared properties. This parameter might be repeated multiple times to return events that match all given constraints.",
                },
                "showDeleted": {
                    "type": "boolean",
                    "description": "Whether to include deleted events (with status equals 'cancelled') in the result. Cancelled instances of recurring events (but not the underlying recurring event) will still be included if showDeleted and singleEvents are both False. If showDeleted and singleEvents are both True, only single instances of deleted events (but not the underlying recurring events) are returned. Optional. The default is False.",
                },
                "showHiddenInvitations": {
                    "type": "boolean",
                    "description": "Whether to include hidden invitations in the result. Optional. The default is False.",
                },
                "singleEvents": {
                    "type": "boolean",
                    "description": "Whether to expand recurring events into instances and only return single one-off events and instances of recurring events, but not the underlying recurring events themselves. Optional. The default is False.",
                },
                "syncToken": {
                    "type": "string",
                    "description": "Token obtained from the nextSyncToken field returned on the last page of results from the previous list request. It makes the result of this list request contain only entries that have changed since then. All events deleted since the previous list request will always be in the result set and it is not allowed to set showDeleted to False. There are several query parameters that cannot be specified together with nextSyncToken to ensure consistency of the client state.",
                },
                "timeMax": {
                    "type": "string",
                    "description": "Upper bound (exclusive) for an event's start time to filter by. Optional. The default is not to filter by start time. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00-07:00, 2011-06-03T10:00:00Z. Milliseconds may be provided but are ignored. If timeMin is set, timeMax must be greater than timeMin.",
                },
                "timeMin": {
                    "type": "string",
                    "description": "Lower bound (exclusive) for an event's end time to filter by. Optional. The default is not to filter by end time. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00-07:00, 2011-06-03T10:00:00Z. Milliseconds may be provided but are ignored. If timeMax is set, timeMin must be less than timeMax.",
                },
                "timeZone": {
                    "type": "string",
                    "description": "Time zone used in the response. Optional. The default is the time zone of the calendar.",
                },
                "updatedMin": {
                    "type": "string",
                    "description": "Lower bound for an event's last modification time (as a RFC3339 timestamp) to filter by. When specified, entries deleted since this time will always be included regardless of showDeleted. Optional. The default is not to filter by last modification time.",
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "create_event",
        "description": """Create a new event in the specified calendar following Google Calendar API v3 specification.
        
        Creates a new event with full Google Calendar API v3 compatibility. Supports all standard event properties
        including attendees, attachments, conference data, reminders, and event type-specific properties.
        
        Required Properties:
          - calendarId: Calendar identifier where event will be created
          - end: Event end time (dateTime/date object with optional timeZone)
          - start: Event start time (dateTime/date object with optional timeZone)
        
        Optional Properties (Google Calendar API v3 compliant):
          - attachments: File attachments for the event
          - attendees: List of event attendees with email, displayName, responseStatus
          - colorId: Color ID of the event (1-11 for event colors)
          - conferenceData: Conference/meeting data for video calls
          - description: Event description text
          - eventType: Event type (default, outOfOffice, focusTime, workingLocation)
          - extendedProperties: Private and shared extended properties
          - focusTimeProperties: Focus time properties for focusTime events
          - guestsCanInviteOthers: Whether guests can invite others
          - guestsCanModify: Whether guests can modify the event
          - guestsCanSeeOtherGuests: Whether guests can see other guests
          - hangoutLink: Hangout video call link
          - iCalUID: iCalendar UID for external integration
          - location: Geographic location of the event
          - locked: Whether the event is locked against changes
          - originalStartTime: Original start time for recurring event instances (must match start values)
          - outOfOfficeProperties: Out of office properties for outOfOffice events
          - privateCopy: Whether this is a private copy of the event
          - recurrence: List of RRULE, EXRULE, RDATE and EXDATE lines
          - reminders: Reminder settings with useDefault and overrides
          - sequence: iCalendar sequence number
          - source: Source from which the event was created
          - status: Event status (confirmed, tentative, cancelled)
          - summary: Event title/summary
          - transparency: Whether event blocks time (opaque, transparent)
          - visibility: Event visibility (default, public, private, confidential)
          - workingLocationProperties: Working location properties for workingLocation events
        
        Query Parameters:
          - conferenceDataVersion: Conference data version supported (0-1)
          - maxAttendees: Maximum number of attendees to include in response
          - sendUpdates: Guests who should receive notifications (all, externalOnly, none)
          - supportsAttachments: Whether client supports event attachments
        
        Response Structure:
          - Returns created event with complete Google Calendar API v3 format
          - Includes generated event ID, timestamps, and all provided properties
          - Attendees list may be limited by maxAttendees parameter
        
        Status Codes:
          - 201: Created - Event created successfully
          - 400: Bad Request - Invalid event data or missing required fields
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Calendar identifier where event will be created",
                },
                "end": {
                    "type": "object",
                    "description": "Event end time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "start": {
                    "type": "object",
                    "description": "Event start time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "attachments": {
                    "type": "array",
                    "description": "File attachments for the event",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fileUrl": {
                                "type": "string",
                                "description": "URL of attached file",
                            },
                        },
                        "required": ["fileUrl"],
                    },
                },
                "attendees": {
                    "type": "array",
                    "description": "List of event attendees",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Attendee email address",
                            },
                            "displayName": {
                                "type": "string",
                                "description": "Attendee display name",
                            },
                            "optional": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether attendee is optional",
                            },
                            "resource": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether attendee is a resource",
                            },
                            "responseStatus": {
                                "type": "string",
                                "description": "Response status: needsAction, declined, tentative, accepted",
                            },
                            "comment": {
                                "type": "string",
                                "description": "Attendee comment",
                            },
                            "additionalGuests": {
                                "type": "integer",
                                "description": "Number of additional guests",
                            },
                        },
                        "required": ["email"],
                    },
                },
                "colorId": {
                    "type": "string",
                    "description": "Color ID of the event (1-11 for event colors)",
                },
                "conferenceData": {
                    "type": "object",
                    "description": "Conference/meeting data for video calls",
                    "properties": {
                        "conferenceId": {
                            "type": "string",
                            "description": "Conference ID",
                        },
                        "conferenceSolution": {
                            "type": "object",
                            "description": "Conference solution details",
                            "properties": {
                                "iconUri": {
                                    "type": "string",
                                    "description": "Icon URI for the conference solution",
                                },
                                "key": {
                                    "type": "object",
                                    "description": "Conference solution key",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution type",
                                        }
                                    },
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name of the conference solution",
                                },
                            },
                        },
                        "createRequest": {
                            "type": "object",
                            "description": "Conference create request details",
                            "properties": {
                                "conferenceSolutionKey": {
                                    "type": "object",
                                    "description": "Conference solution for the create request",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution key type",
                                        }
                                    },
                                },
                                "requestId": {
                                    "type": "string",
                                    "description": "Request ID for creating the conference",
                                },
                            },
                        },
                        "entryPoints": {
                            "type": "array",
                            "description": "Conference entry points",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "accessCode": {
                                        "type": "string",
                                        "description": "Access code for the conference",
                                    },
                                    "entryPointType": {
                                        "type": "string",
                                        "description": "Type of entry point",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Label for the entry point",
                                    },
                                    "meetingCode": {
                                        "type": "string",
                                        "description": "Meeting code for the conference",
                                    },
                                    "passcode": {
                                        "type": "string",
                                        "description": "Passcode for the conference",
                                    },
                                    "password": {
                                        "type": "string",
                                        "description": "Password for the conference",
                                    },
                                    "pin": {
                                        "type": "string",
                                        "description": "PIN for the conference",
                                    },
                                    "uri": {
                                        "type": "string",
                                        "description": "URI for the conference entry point",
                                    },
                                },
                            },
                        },
                        "notes": {"type": "string", "description": "Conference notes"},
                        "signature": {
                            "type": "string",
                            "description": "Conference signature",
                        },
                    },
                },
                "description": {
                    "type": "string",
                    "description": "Event description text",
                },
                "birthdayProperties": {
                    "type": "object",
                    "description": "Birthday properties for birthday events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["birthday"],
                            "description": "Type of birthday event, must be 'birthday'. Cannot be changed after event creation.",
                        }
                    },
                    "required": ["type"],
                },
                "eventType": {
                    "type": "string",
                    "description": "Event type: default, birthday, outOfOffice, focusTime, workingLocation",
                    "enum": [
                        "default",
                        "birthday",
                        "outOfOffice",
                        "focusTime",
                        "workingLocation",
                    ],
                },
                "extendedProperties": {
                    "type": "object",
                    "description": "Extended properties",
                    "properties": {
                        "private": {
                            "type": "object",
                            "description": "Private extended properties",
                        },
                        "shared": {
                            "type": "object",
                            "description": "Shared extended properties",
                        },
                    },
                },
                "focusTimeProperties": {
                    "type": "object",
                    "description": "Focus time properties for focusTime events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "chatStatus": {
                            "type": "string",
                            "description": "The status to mark the user in Chat and related products",
                            "enum": ["available", "doNotDisturb"],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "guestsCanInviteOthers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can invite others",
                },
                "guestsCanModify": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether guests can modify the event",
                },
                "guestsCanSeeOtherGuests": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can see other guests",
                },
                "iCalUID": {
                    "type": "string",
                    "description": "iCalendar UID for external integration",
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location of the event",
                },
                "originalStartTime": {
                    "type": "object",
                    "description": "Original start time for recurring event instances (must match start field values)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "outOfOfficeProperties": {
                    "type": "object",
                    "description": "Out of office properties for outOfOffice events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "recurrence": {
                    "type": "array",
                    "description": """List of RRULE, EXRULE, RDATE and EXDATE lines for recurring events following RFC 5545 (iCalendar) standard.
                    
                    Supported Recurrence Types:
                    
                    RRULE (Recurrence Rule) - Defines the pattern for recurring events:
                    • FREQ: Frequency (DAILY, WEEKLY, MONTHLY, YEARLY)
                    • INTERVAL: Interval between occurrences (e.g., INTERVAL=2 for every 2 weeks)
                    • COUNT: Maximum number of occurrences
                    • UNTIL: End date (format: YYYYMMDDTHHMMSSZ)
                    • BYDAY: Days of week (MO, TU, WE, TH, FR, SA, SU)
                    • BYMONTHDAY: Days of month (1-31)
                    • BYMONTH: Months (1-12)
                    • BYSETPOS: Position in set (e.g., 1st, 2nd, -1 for last)
                    
                    EXDATE (Exception Dates) - Exclude specific occurrences:
                    • Format: EXDATE:YYYYMMDDTHHMMSSZ or EXDATE;VALUE=DATE:YYYYMMDD
                    • Use timezone format for timed events, date format for all-day events
                    
                    RDATE (Recurrence Dates) - Add specific occurrences:
                    • Format: RDATE:YYYYMMDDTHHMMSSZ or RDATE;VALUE=DATE:YYYYMMDD
                    
                    Common Examples:
                    • Daily: ["RRULE:FREQ=DAILY"]
                    • Every weekday: ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
                    • Weekly on specific days: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH"]
                    • Monthly on specific date: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=15"]
                    • First Monday of month: ["RRULE:FREQ=MONTHLY;BYDAY=1MO"]
                    • Last Friday of month: ["RRULE:FREQ=MONTHLY;BYDAY=-1FR"]
                    • Yearly: ["RRULE:FREQ=YEARLY"]
                    • Every 2 weeks: ["RRULE:FREQ=WEEKLY;INTERVAL=2"]
                    • 10 times only: ["RRULE:FREQ=WEEKLY;COUNT=10"]
                    • Until specific date: ["RRULE:FREQ=DAILY;UNTIL=20231231T235959Z"]
                    
                    Complex Examples:
                    • Weekly with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH", "EXDATE:20231024T100000Z"]
                    • Monthly with additional dates: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=1", "RDATE:20231215T100000Z"]
                    • Every other month on 2nd Tuesday: ["RRULE:FREQ=MONTHLY;INTERVAL=2;BYDAY=2TU"]
                    
                    All-Day Event Examples:
                    • Daily all-day: ["RRULE:FREQ=DAILY"]
                    • Weekly all-day with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=MO", "EXDATE;VALUE=DATE:20231030"]
                    
                    Note: For all-day events, use VALUE=DATE format for EXDATE/RDATE. For timed events, use full timestamp format.""",
                    "items": {"type": "string"},
                },
                "reminders": {
                    "type": "object",
                    "description": "Reminder settings",
                    "properties": {
                        "useDefault": {
                            "type": "boolean",
                            "description": "Whether to use default reminders",
                        },
                        "overrides": {
                            "type": "array",
                            "description": "Custom reminder overrides",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email", "popup"],
                                        "description": "Reminder method",
                                    },
                                    "minutes": {
                                        "type": "integer",
                                        "description": "Minutes before event",
                                    },
                                },
                                "required": ["method", "minutes"],
                            },
                        },
                    },
                },
                "sequence": {
                    "type": "integer",
                    "description": "iCalendar sequence number",
                },
                "source": {
                    "type": "object",
                    "description": "Source from which the event was created",
                    "properties": {
                        "url": {"type": "string", "description": "Source URL"},
                        "title": {"type": "string", "description": "Source title"},
                    },
                },
                "status": {
                    "type": "string",
                    "description": "Event status: confirmed, tentative, cancelled",
                    "enum": ["confirmed", "tentative", "cancelled"],
                },
                "summary": {"type": "string", "description": "Event title/summary"},
                "transparency": {
                    "type": "string",
                    "description": "Whether event blocks time: opaque, transparent",
                    "enum": ["opaque", "transparent"],
                },
                "visibility": {
                    "type": "string",
                    "description": "Event visibility: default, public, private, confidential",
                    "enum": ["default", "public", "private", "confidential"],
                },
                "workingLocationProperties": {
                    "type": "object",
                    "description": "Working location properties for workingLocation events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["homeOffice", "officeLocation", "customLocation"],
                            "description": "Type of the working location. Required when adding working location properties",
                        },
                        "customLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from a custom location",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "An optional extra label for additional information",
                                }
                            },
                        },
                        "homeOffice": {
                            "type": "object",
                            "description": "If present, specifies that the user is working at home",
                            "properties": {
                                "address_1": {
                                    "type": "string",
                                    "description": "Home Office address 1",
                                },
                                "address_2": {
                                    "type": "string",
                                    "description": "Home Office address 2",
                                },
                                "city": {
                                    "type": "string",
                                    "description": "City located",
                                },
                                "state": {
                                    "type": "string",
                                    "description": "State of the home office",
                                },
                                "postal_code": {
                                    "type": "string",
                                    "description": "Postal code of the home office",
                                },
                            },
                            "additionalProperties": True,
                        },
                        "officeLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from an office",
                            "properties": {
                                "buildingId": {
                                    "type": "string",
                                    "description": "An optional building identifier. This should reference a building ID in the organization's Resources database",
                                },
                                "deskId": {
                                    "type": "string",
                                    "description": "An optional desk identifier",
                                },
                                "floorId": {
                                    "type": "string",
                                    "description": "An optional floor identifier",
                                },
                                "floorSectionId": {
                                    "type": "string",
                                    "description": "An optional floor section identifier",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "The office name that's displayed in Calendar Web and Mobile clients. We recommend you reference a building name in the organization's Resources database",
                                },
                            },
                        },
                    },
                    "required": ["type"],
                },
                "conferenceDataVersion": {
                    "type": "integer",
                    "description": "Conference data version supported (0-1)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "Maximum number of attendees to include in response",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications: all, externalOnly, none",
                    "enum": ["all", "externalOnly", "none"],
                },
                "supportsAttachments": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether client supports event attachments",
                },
            },
            "required": ["calendarId", "end", "start"],
        },
    },
    {
        "name": "get_event",
        "description": """Retrieve a specific event by its ID from the specified calendar.
        
        Returns event details following Google Calendar API v3 structure.
        Provides complete event information including attendees, reminders, and recurrence settings.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier
          - eventId: Required. Event identifier
          - timeZone: Optional. Time zone for returned times (IANA timezone)
        
        Response Structure:
          - Returns event with Google Calendar API v3 format:
            * kind: "calendar#event"
            * etag: ETag of the resource
            * id: Unique event identifier
            * summary: Event title
            * description: Event description (if present)
            * location: Event location (if present)
            * start: Event start time with timezone
            * end: Event end time with timezone
            * recurrence: Recurrence rules (if recurring)
            * status: Event status
            * visibility: Event visibility
            * attendees: List of attendees (if present)
            * reminders: Reminder settings
        
        Status Codes:
          - 200: Success - Event retrieved successfully
          - 404: Not Found - Event or calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventId": {"type": "string", "description": "Event identifier"},
                "timeZone": {
                    "type": "string",
                    "description": "Time zone for returned times (IANA timezone)",
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "Maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
                },
            },
            "required": ["calendarId", "eventId"],
        },
    },
    {
        "name": "patch_event",
        "description": """Partially update an event (only specified fields are modified) following Google Calendar API v3.
        
        Updates an event with partial data using Google Calendar API v3 structure.
        Only provided fields will be updated, others remain unchanged.
        Use for incremental updates without affecting other event properties.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier
          - eventId: Required. Event identifier
        
        Optional Update Fields (Google Calendar API v3 compliant):
          - attachments: File attachments for the event
          - attendees: List of event attendees with email, displayName, responseStatus
          - colorId: Color ID of the event (1-11 for event colors)
          - conferenceData: Conference/meeting data for video calls
          - description: Event description text
          - end: Event end time (dateTime/date object with optional timeZone)
          - eventType: Event type (default, outOfOffice, focusTime, workingLocation)
          - extendedProperties: Private and shared extended properties
          - focusTimeProperties: Focus time properties for focusTime events
          - guestsCanInviteOthers: Whether guests can invite others
          - guestsCanModify: Whether guests can modify the event
          - guestsCanSeeOtherGuests: Whether guests can see other guests
          - hangoutLink: Hangout video call link
          - iCalUID: iCalendar UID for external integration
          - location: Geographic location of the event
          - locked: Whether the event is locked against changes
          - outOfOfficeProperties: Out of office properties for outOfOffice events
          - privateCopy: Whether this is a private copy of the event
          - recurrence: List of RRULE, EXRULE, RDATE and EXDATE lines
          - reminders: Reminder settings with useDefault and overrides
          - sequence: iCalendar sequence number
          - source: Source from which the event was created
          - start: Event start time (dateTime/date object with optional timeZone)
          - status: Event status (confirmed, tentative, cancelled)
          - summary: Event title/summary
          - transparency: Whether event blocks time (opaque, transparent)
          - visibility: Event visibility (default, public, private, confidential)
          - workingLocationProperties: Working location properties for workingLocation events
        
        Query Parameters:
          - sendUpdates: Guests who should receive notifications (all, externalOnly, none)
        
        Partial Update Behavior:
          - Only fields provided in request body are updated
          - Missing fields are left unchanged
          - At least one field should be provided for update
        
        Response Structure:
          - Returns updated event with Google Calendar API v3 format
        
        Status Codes:
          - 200: Success - Event updated successfully
          - 400: Bad Request - Invalid update data
          - 404: Not Found - Event or calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventId": {"type": "string", "description": "Event identifier"},
                "summary": {"type": "string", "description": "Event title"},
                "description": {"type": "string", "description": "Event description"},
                "location": {"type": "string", "description": "Event location"},
                "attachments": {
                    "type": "array",
                    "description": "File attachments for the event",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fileUrl": {
                                "type": "string",
                                "description": "URL of attached file",
                            },
                        },
                    },
                },
                "attendees": {
                    "type": "array",
                    "description": "List of event attendees",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Attendee email address",
                            },
                            "displayName": {
                                "type": "string",
                                "description": "Attendee display name",
                            },
                            "optional": {
                                "type": "boolean",
                                "description": "Whether attendee is optional",
                            },
                            "resource": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether attendee is a resource",
                            },
                            "responseStatus": {
                                "type": "string",
                                "description": "Response status",
                            },
                            "comment": {
                                "type": "string",
                                "description": "Attendee comment",
                            },
                            "additionalGuests": {
                                "type": "integer",
                                "description": "Number of additional guests",
                            },
                        },
                        "required": ["email"],
                    },
                },
                "colorId": {
                    "type": "string",
                    "description": "Color ID of the event(1-11 for event colors)",
                },
                "conferenceData": {
                    "type": "object",
                    "description": "Conference/meeting data for video calls",
                    "properties": {
                        "conferenceId": {
                            "type": "string",
                            "description": "Conference ID",
                        },
                        "conferenceSolution": {
                            "type": "object",
                            "description": "Conference solution details",
                            "properties": {
                                "iconUri": {
                                    "type": "string",
                                    "description": "Icon URI for the conference solution",
                                },
                                "key": {
                                    "type": "object",
                                    "description": "Conference solution key",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution type",
                                        }
                                    },
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name of the conference solution",
                                },
                            },
                        },
                        "createRequest": {
                            "type": "object",
                            "description": "Conference create request details",
                            "properties": {
                                "conferenceSolutionKey": {
                                    "type": "object",
                                    "description": "Conference solution for the create request",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution key type",
                                        }
                                    },
                                },
                                "requestId": {
                                    "type": "string",
                                    "description": "Request ID for creating the conference",
                                },
                            },
                        },
                        "entryPoints": {
                            "type": "array",
                            "description": "Conference entry points",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "accessCode": {
                                        "type": "string",
                                        "description": "Access code for the conference",
                                    },
                                    "entryPointType": {
                                        "type": "string",
                                        "description": "Type of entry point",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Label for the entry point",
                                    },
                                    "meetingCode": {
                                        "type": "string",
                                        "description": "Meeting code for the conference",
                                    },
                                    "passcode": {
                                        "type": "string",
                                        "description": "Passcode for the conference",
                                    },
                                    "password": {
                                        "type": "string",
                                        "description": "Password for the conference",
                                    },
                                    "pin": {
                                        "type": "string",
                                        "description": "PIN for the conference",
                                    },
                                    "uri": {
                                        "type": "string",
                                        "description": "URI for the conference entry point",
                                    },
                                },
                            },
                        },
                        "notes": {"type": "string", "description": "Conference notes"},
                        "signature": {
                            "type": "string",
                            "description": "Conference signature",
                        },
                    },
                },
                "birthdayProperties": {
                    "type": "object",
                    "description": "Birthday properties for birthday events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["birthday"],
                            "description": "Type of birthday event, must be 'birthday'. Cannot be changed after event creation.",
                        }
                    },
                    "required": ["type"],
                },
                "description": {"type": "string", "description": "Event description"},
                "end": {
                    "type": "object",
                    "description": "Event end time",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format",
                        },
                        "timeZone": {"type": "string", "description": "IANA timezone"},
                    },
                },
                "eventType": {
                    "type": "string",
                    "description": "Event type",
                    "enum": [
                        "default",
                        "birthday",
                        "outOfOffice",
                        "focusTime",
                        "workingLocation",
                    ],
                },
                "extendedProperties": {
                    "type": "object",
                    "description": "Extended properties",
                    "properties": {
                        "private": {
                            "type": "object",
                            "description": "Private extended properties",
                        },
                        "shared": {
                            "type": "object",
                            "description": "Shared extended properties",
                        },
                    },
                },
                "focusTimeProperties": {
                    "type": "object",
                    "description": "Focus time properties for focusTime events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "chatStatus": {
                            "type": "string",
                            "description": "The status to mark the user in Chat and related products",
                            "enum": ["available", "doNotDisturb"],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "guestsCanInviteOthers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can invite others",
                },
                "guestsCanModify": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether guests can modify the event",
                },
                "guestsCanSeeOtherGuests": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can see other guests",
                },
                "hangoutLink": {"type": "string", "description": "Hangout link"},
                "iCalUID": {"type": "string", "description": "iCalendar UID"},
                "location": {"type": "string", "description": "Event location"},
                "outOfOfficeProperties": {
                    "type": "object",
                    "description": "Out of office properties for outOfOffice events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "locked": {"type": "boolean", "description": "Whether event is locked"},
                "privateCopy": {
                    "type": "boolean",
                    "description": "Whether this is a private copy",
                },
                "recurrence": {
                    "type": "array",
                    "description": """List of RRULE, EXRULE, RDATE and EXDATE lines for recurring events following RFC 5545 (iCalendar) standard.
                    
                    Supported Recurrence Types:
                    
                    RRULE (Recurrence Rule) - Defines the pattern for recurring events:
                    • FREQ: Frequency (DAILY, WEEKLY, MONTHLY, YEARLY)
                    • INTERVAL: Interval between occurrences (e.g., INTERVAL=2 for every 2 weeks)
                    • COUNT: Maximum number of occurrences
                    • UNTIL: End date (format: YYYYMMDDTHHMMSSZ)
                    • BYDAY: Days of week (MO, TU, WE, TH, FR, SA, SU)
                    • BYMONTHDAY: Days of month (1-31)
                    • BYMONTH: Months (1-12)
                    • BYSETPOS: Position in set (e.g., 1st, 2nd, -1 for last)
                    
                    EXDATE (Exception Dates) - Exclude specific occurrences:
                    • Format: EXDATE:YYYYMMDDTHHMMSSZ or EXDATE;VALUE=DATE:YYYYMMDD
                    • Use timezone format for timed events, date format for all-day events
                    
                    RDATE (Recurrence Dates) - Add specific occurrences:
                    • Format: RDATE:YYYYMMDDTHHMMSSZ or RDATE;VALUE=DATE:YYYYMMDD
                    
                    Common Examples:
                    • Daily: ["RRULE:FREQ=DAILY"]
                    • Every weekday: ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
                    • Weekly on specific days: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH"]
                    • Monthly on specific date: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=15"]
                    • First Monday of month: ["RRULE:FREQ=MONTHLY;BYDAY=1MO"]
                    • Last Friday of month: ["RRULE:FREQ=MONTHLY;BYDAY=-1FR"]
                    • Yearly: ["RRULE:FREQ=YEARLY"]
                    • Every 2 weeks: ["RRULE:FREQ=WEEKLY;INTERVAL=2"]
                    • 10 times only: ["RRULE:FREQ=WEEKLY;COUNT=10"]
                    • Until specific date: ["RRULE:FREQ=DAILY;UNTIL=20231231T235959Z"]
                    
                    Complex Examples:
                    • Weekly with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH", "EXDATE:20231024T100000Z"]
                    • Monthly with additional dates: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=1", "RDATE:20231215T100000Z"]
                    • Every other month on 2nd Tuesday: ["RRULE:FREQ=MONTHLY;INTERVAL=2;BYDAY=2TU"]
                    
                    All-Day Event Examples:
                    • Daily all-day: ["RRULE:FREQ=DAILY"]
                    • Weekly all-day with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=MO", "EXDATE;VALUE=DATE:20231030"]
                    
                    Note: For all-day events, use VALUE=DATE format for EXDATE/RDATE. For timed events, use full timestamp format.""",
                    "items": {"type": "string"},
                },
                "reminders": {
                    "type": "object",
                    "description": "Reminder settings",
                    "properties": {
                        "useDefault": {
                            "type": "boolean",
                            "description": "Whether to use default reminders",
                        },
                        "overrides": {
                            "type": "array",
                            "description": "Custom reminder overrides",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email", "popup"],
                                        "description": "Reminder method",
                                    },
                                    "minutes": {
                                        "type": "integer",
                                        "description": "Minutes before event",
                                    },
                                },
                                "required": ["method", "minutes"],
                            },
                        },
                    },
                },
                "sequence": {"type": "integer", "description": "Sequence number"},
                "source": {
                    "type": "object",
                    "description": "Source from which the event was created",
                    "properties": {
                        "url": {"type": "string", "description": "Source URL"},
                        "title": {"type": "string", "description": "Source title"},
                    },
                },
                "start": {
                    "type": "object",
                    "description": "Event start time",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format",
                        },
                        "timeZone": {"type": "string", "description": "IANA timezone"},
                    },
                },
                "status": {
                    "type": "string",
                    "description": "Event status: confirmed, tentative, cancelled",
                    "enum": ["confirmed", "tentative", "cancelled"],
                },
                "summary": {"type": "string", "description": "Event title"},
                "transparency": {
                    "type": "string",
                    "description": "Event transparency",
                    "enum": ["opaque", "transparent"],
                },
                "visibility": {
                    "type": "string",
                    "description": "Event visibility",
                    "enum": ["default", "public", "private", "confidential"],
                },
                "workingLocationProperties": {
                    "type": "object",
                    "description": "Working location properties for workingLocation events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["homeOffice", "officeLocation", "customLocation"],
                            "description": "Type of the working location. Required when adding working location properties",
                        },
                        "customLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from a custom location",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "An optional extra label for additional information",
                                }
                            },
                        },
                        "homeOffice": {
                            "type": "object",
                            "description": "If present, specifies that the user is working at home",
                            "properties": {
                                "address_1": {
                                    "type": "string",
                                    "description": "Home Office address 1",
                                },
                                "address_2": {
                                    "type": "string",
                                    "description": "Home Office address 2",
                                },
                                "city": {
                                    "type": "string",
                                    "description": "City located",
                                },
                                "state": {
                                    "type": "string",
                                    "description": "State of the home office",
                                },
                                "postal_code": {
                                    "type": "string",
                                    "description": "Postal code of the home office",
                                },
                            },
                            "additionalProperties": True,
                        },
                        "officeLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from an office",
                            "properties": {
                                "buildingId": {
                                    "type": "string",
                                    "description": "An optional building identifier. This should reference a building ID in the organization's Resources database",
                                },
                                "deskId": {
                                    "type": "string",
                                    "description": "An optional desk identifier",
                                },
                                "floorId": {
                                    "type": "string",
                                    "description": "An optional floor identifier",
                                },
                                "floorSectionId": {
                                    "type": "string",
                                    "description": "An optional floor section identifier",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "The office name that's displayed in Calendar Web and Mobile clients. We recommend you reference a building name in the organization's Resources database",
                                },
                            },
                        },
                    },
                    "required": ["type"],
                },
                "conferenceDataVersion": {
                    "type": "integer",
                    "description": "Conference data version supported (0-1)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "Maximum number of attendees to include in response",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications",
                    "enum": ["all", "externalOnly", "none"],
                },
                "supportsAttachments": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether client supports event attachments",
                },
            },
            "required": ["calendarId", "eventId"],
        },
    },
    {
        "name": "update_event",
        "description": """Fully update an event (complete replacement) following Google Calendar API v3 specification.
        
        Updates an event following Google Calendar API v3 structure. This method does not support patch
        semantics and always updates the entire event resource. To do a partial update, perform a get
        followed by an update using etags to ensure atomicity.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier
          - eventId: Required. Event identifier
          - start: Required. Event start time (dateTime/date object with optional timeZone)
          - end: Required. Event end time (dateTime/date object with optional timeZone)
        
        Optional Properties (Google Calendar API v3 compliant):
          - attachments: File attachments for the event
          - attendees: List of event attendees with email, displayName, responseStatus
          - birthdayProperties: Birthday properties for birthday events
          - colorId: Color ID of the event (1-11 for event colors)
          - conferenceData: Conference/meeting data for video calls
          - description: Event description text
          - extendedProperties: Private and shared extended properties
          - focusTimeProperties: Focus time properties for focusTime events
          - guestsCanInviteOthers: Whether guests can invite others
          - guestsCanModify: Whether guests can modify the event
          - guestsCanSeeOtherGuests: Whether guests can see other guests
          - hangoutLink: Hangout video call link
          - iCalUID: iCalendar UID for external integration
          - location: Geographic location of the event
          - locked: Whether the event is locked against changes
          - outOfOfficeProperties: Out of office properties for outOfOffice events
          - privateCopy: Whether this is a private copy of the event
          - recurrence: List of RRULE, EXRULE, RDATE and EXDATE lines
          - reminders: Reminder settings with useDefault and overrides
          - sequence: iCalendar sequence number
          - source: Source from which the event was created
          - status: Event status (confirmed, tentative, cancelled)
          - summary: Event title/summary
          - transparency: Whether event blocks time (opaque, transparent)
          - visibility: Event visibility (default, public, private, confidential)
          - workingLocationProperties: Working location properties for workingLocation events
        
        Query Parameters:
          - conferenceDataVersion: Conference data version supported (0-1)
          - maxAttendees: Maximum number of attendees to include in response
          - sendUpdates: Guests who should receive notifications (all, externalOnly, none)
          - supportsAttachments: Whether client supports event attachments
        
        Full Update Behavior:
          - All fields are replaced (full replacement operation)
          - Missing optional fields are set to null/defaults
          - start and end are required for PUT operations
          - This is different from PATCH which only updates provided fields
        
        Response Structure:
          - Returns updated event with complete Google Calendar API v3 format
        
        Status Codes:
          - 200: Success - Event updated successfully
          - 400: Bad Request - Invalid update data or missing required fields
          - 404: Not Found - Event or calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventId": {"type": "string", "description": "Event identifier"},
                "start": {
                    "type": "object",
                    "description": "Event start time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "end": {
                    "type": "object",
                    "description": "Event end time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "summary": {"type": "string", "description": "Event title/summary"},
                "description": {"type": "string", "description": "Event description"},
                "location": {"type": "string", "description": "Event location"},
                "attachments": {
                    "type": "array",
                    "description": "File attachments for the event",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fileUrl": {
                                "type": "string",
                                "description": "URL of attached file",
                            },
                        },
                    },
                },
                "attendees": {
                    "type": "array",
                    "description": "List of event attendees",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Attendee email address",
                            },
                            "displayName": {
                                "type": "string",
                                "description": "Attendee display name",
                            },
                            "optional": {
                                "type": "boolean",
                                "description": "Whether attendee is optional",
                            },
                            "resource": {
                                "type": "boolean",
                                "description": "Whether attendee is a resource",
                            },
                            "responseStatus": {
                                "type": "string",
                                "description": "Response status",
                            },
                            "comment": {
                                "type": "string",
                                "description": "Attendee comment",
                            },
                            "additionalGuests": {
                                "type": "integer",
                                "description": "Number of additional guests",
                            },
                        },
                        "required": ["email"],
                    },
                },
                "eventType": {
                    "type": "string",
                    "description": "Event type",
                    "enum": [
                        "default",
                        "birthday",
                        "outOfOffice",
                        "focusTime",
                        "workingLocation",
                    ],
                },
                "birthdayProperties": {
                    "type": "object",
                    "description": "Birthday properties for birthday events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["birthday"],
                            "description": "Type of birthday event, must be 'birthday'. Cannot be changed after event creation.",
                        }
                    },
                    "required": ["type"],
                },
                "colorId": {"type": "string", "description": "Color ID of the event"},
                "conferenceData": {
                    "type": "object",
                    "description": "Conference/meeting data for video calls",
                    "properties": {
                        "conferenceId": {
                            "type": "string",
                            "description": "Conference ID",
                        },
                        "conferenceSolution": {
                            "type": "object",
                            "description": "Conference solution details",
                            "properties": {
                                "iconUri": {
                                    "type": "string",
                                    "description": "Icon URI for the conference solution",
                                },
                                "key": {
                                    "type": "object",
                                    "description": "Conference solution key",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution type",
                                        }
                                    },
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name of the conference solution",
                                },
                            },
                        },
                        "createRequest": {
                            "type": "object",
                            "description": "Conference create request details",
                            "properties": {
                                "conferenceSolutionKey": {
                                    "type": "object",
                                    "description": "Conference solution for the create request",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Conference solution key type",
                                        }
                                    },
                                },
                                "requestId": {
                                    "type": "string",
                                    "description": "Request ID for creating the conference",
                                },
                            },
                        },
                        "entryPoints": {
                            "type": "array",
                            "description": "Conference entry points",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "accessCode": {
                                        "type": "string",
                                        "description": "Access code for the conference",
                                    },
                                    "entryPointType": {
                                        "type": "string",
                                        "description": "Type of entry point",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Label for the entry point",
                                    },
                                    "meetingCode": {
                                        "type": "string",
                                        "description": "Meeting code for the conference",
                                    },
                                    "passcode": {
                                        "type": "string",
                                        "description": "Passcode for the conference",
                                    },
                                    "password": {
                                        "type": "string",
                                        "description": "Password for the conference",
                                    },
                                    "pin": {
                                        "type": "string",
                                        "description": "PIN for the conference",
                                    },
                                    "uri": {
                                        "type": "string",
                                        "description": "URI for the conference entry point",
                                    },
                                },
                            },
                        },
                        "notes": {"type": "string", "description": "Conference notes"},
                        "signature": {
                            "type": "string",
                            "description": "Conference signature",
                        },
                    },
                },
                "extendedProperties": {
                    "type": "object",
                    "description": "Extended properties",
                    "properties": {
                        "private": {
                            "type": "object",
                            "description": "Private extended properties",
                        },
                        "shared": {
                            "type": "object",
                            "description": "Shared extended properties",
                        },
                    },
                },
                "focusTimeProperties": {
                    "type": "object",
                    "description": "Focus time properties for focusTime events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "chatStatus": {
                            "type": "string",
                            "description": "The status to mark the user in Chat and related products",
                            "enum": ["available", "doNotDisturb"],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "guestsCanInviteOthers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can invite others",
                },
                "guestsCanModify": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether guests can modify the event",
                },
                "guestsCanSeeOtherGuests": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether guests can see other guests",
                },
                "hangoutLink": {"type": "string", "description": "Hangout link"},
                "iCalUID": {"type": "string", "description": "iCalendar UID"},
                "locked": {"type": "boolean", "description": "Whether event is locked"},
                "outOfOfficeProperties": {
                    "type": "object",
                    "description": "Out of office properties for outOfOffice events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "privateCopy": {
                    "type": "boolean",
                    "description": "Whether this is a private copy",
                },
                "recurrence": {
                    "type": "array",
                    "description": """List of RRULE, EXRULE, RDATE and EXDATE lines for recurring events following RFC 5545 (iCalendar) standard.
                    
                    Supported Recurrence Types:
                    
                    RRULE (Recurrence Rule) - Defines the pattern for recurring events:
                    • FREQ: Frequency (DAILY, WEEKLY, MONTHLY, YEARLY)
                    • INTERVAL: Interval between occurrences (e.g., INTERVAL=2 for every 2 weeks)
                    • COUNT: Maximum number of occurrences
                    • UNTIL: End date (format: YYYYMMDDTHHMMSSZ)
                    • BYDAY: Days of week (MO, TU, WE, TH, FR, SA, SU)
                    • BYMONTHDAY: Days of month (1-31)
                    • BYMONTH: Months (1-12)
                    • BYSETPOS: Position in set (e.g., 1st, 2nd, -1 for last)
                    
                    EXDATE (Exception Dates) - Exclude specific occurrences:
                    • Format: EXDATE:YYYYMMDDTHHMMSSZ or EXDATE;VALUE=DATE:YYYYMMDD
                    • Use timezone format for timed events, date format for all-day events
                    
                    RDATE (Recurrence Dates) - Add specific occurrences:
                    • Format: RDATE:YYYYMMDDTHHMMSSZ or RDATE;VALUE=DATE:YYYYMMDD
                    
                    Common Examples:
                    • Daily: ["RRULE:FREQ=DAILY"]
                    • Every weekday: ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
                    • Weekly on specific days: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH"]
                    • Monthly on specific date: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=15"]
                    • First Monday of month: ["RRULE:FREQ=MONTHLY;BYDAY=1MO"]
                    • Last Friday of month: ["RRULE:FREQ=MONTHLY;BYDAY=-1FR"]
                    • Yearly: ["RRULE:FREQ=YEARLY"]
                    • Every 2 weeks: ["RRULE:FREQ=WEEKLY;INTERVAL=2"]
                    • 10 times only: ["RRULE:FREQ=WEEKLY;COUNT=10"]
                    • Until specific date: ["RRULE:FREQ=DAILY;UNTIL=20231231T235959Z"]
                    
                    Complex Examples:
                    • Weekly with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH", "EXDATE:20231024T100000Z"]
                    • Monthly with additional dates: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=1", "RDATE:20231215T100000Z"]
                    • Every other month on 2nd Tuesday: ["RRULE:FREQ=MONTHLY;INTERVAL=2;BYDAY=2TU"]
                    
                    All-Day Event Examples:
                    • Daily all-day: ["RRULE:FREQ=DAILY"]
                    • Weekly all-day with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=MO", "EXDATE;VALUE=DATE:20231030"]
                    
                    Note: For all-day events, use VALUE=DATE format for EXDATE/RDATE. For timed events, use full timestamp format.""",
                    "items": {"type": "string"},
                },
                "reminders": {
                    "type": "object",
                    "description": "Reminder settings",
                    "properties": {
                        "useDefault": {
                            "type": "boolean",
                            "description": "Whether to use default reminders",
                        },
                        "overrides": {
                            "type": "array",
                            "description": "Custom reminder overrides",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email", "popup"],
                                        "description": "Reminder method",
                                    },
                                    "minutes": {
                                        "type": "integer",
                                        "description": "Minutes before event",
                                    },
                                },
                                "required": ["method", "minutes"],
                            },
                        },
                    },
                },
                "sequence": {"type": "integer", "description": "Sequence number"},
                "source": {
                    "type": "object",
                    "description": "Source from which the event was created",
                    "properties": {
                        "url": {"type": "string", "description": "Source URL"},
                        "title": {"type": "string", "description": "Source title"},
                    },
                },
                "status": {
                    "type": "string",
                    "description": "Event status",
                    "enum": ["confirmed", "tentative", "cancelled"],
                },
                "transparency": {
                    "type": "string",
                    "description": "Event transparency",
                    "enum": ["opaque", "transparent"],
                },
                "visibility": {
                    "type": "string",
                    "description": "Event visibility",
                    "enum": ["default", "public", "private", "confidential"],
                },
                "workingLocationProperties": {
                    "type": "object",
                    "description": "Working location properties for workingLocation events",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["homeOffice", "officeLocation", "customLocation"],
                            "description": "Type of the working location. Required when adding working location properties",
                        },
                        "customLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from a custom location",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "An optional extra label for additional information",
                                }
                            },
                        },
                        "homeOffice": {
                            "type": "object",
                            "description": "If present, specifies that the user is working at home",
                            "properties": {
                                "address_1": {
                                    "type": "string",
                                    "description": "Home Office address 1",
                                },
                                "address_2": {
                                    "type": "string",
                                    "description": "Home Office address 2",
                                },
                                "city": {
                                    "type": "string",
                                    "description": "City located",
                                },
                                "state": {
                                    "type": "string",
                                    "description": "State of the home office",
                                },
                                "postal_code": {
                                    "type": "string",
                                    "description": "Postal code of the home office",
                                },
                            },
                            "additionalProperties": True,
                        },
                        "officeLocation": {
                            "type": "object",
                            "description": "If present, specifies that the user is working from an office",
                            "properties": {
                                "buildingId": {
                                    "type": "string",
                                    "description": "An optional building identifier. This should reference a building ID in the organization's Resources database",
                                },
                                "deskId": {
                                    "type": "string",
                                    "description": "An optional desk identifier",
                                },
                                "floorId": {
                                    "type": "string",
                                    "description": "An optional floor identifier",
                                },
                                "floorSectionId": {
                                    "type": "string",
                                    "description": "An optional floor section identifier",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "The office name that's displayed in Calendar Web and Mobile clients. We recommend you reference a building name in the organization's Resources database",
                                },
                            },
                        },
                    },
                    "required": ["type"],
                },
                "conferenceDataVersion": {
                    "type": "integer",
                    "description": "Conference data version supported (0-1)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "Maximum number of attendees to include in response",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications: all, externalOnly, none",
                    "enum": ["all", "externalOnly", "none"],
                },
                "supportsAttachments": {
                    "type": "boolean",
                    "description": "Whether client supports event attachments",
                },
            },
            "required": ["calendarId", "eventId", "start", "end"],
        },
    },
    {
        "name": "delete_event",
        "description": """Delete an event from the specified calendar.
        
        Permanently deletes an event following Google Calendar API v3 behavior.
        This action cannot be undone. Use with caution.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier
          - eventId: Required. Event identifier to delete
          - sendUpdates: Optional. Guests who should receive notifications (all, externalOnly, none)
        
        Deletion Behavior:
          - Event is permanently removed from the calendar
          - This operation is irreversible
          - Notifications can be sent to attendees if specified
        
        Response Structure:
          - Returns 204 No Content on successful deletion
          - No response body as per Google Calendar API v3
        
        Status Codes:
          - 204: No Content - Event deleted successfully
          - 404: Not Found - Event or calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventId": {
                    "type": "string",
                    "description": "Event identifier to delete",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications: all, externalOnly, none",
                },
            },
            "required": ["calendarId", "eventId"],
        },
    },
    {
        "name": "move_event",
        "description": """Move an event from one calendar to another.
        
        Moves an event between calendars following Google Calendar API v3 behavior.
        The event retains its properties but changes calendar ownership.
        
        Request Body Requirements:
          - calendarId: Required. Source calendar identifier
          - eventId: Required. Event identifier to move
          - destination: Required. Target calendar identifier where event will be moved
          - sendUpdates: Optional. Guests who should receive notifications (all, externalOnly, none)
        
        Move Operation:
          - Event is transferred from source to destination calendar
          - Event properties (title, time, attendees, etc.) remain unchanged
          - Event ID may change during the move operation
          - Original event is removed from source calendar
        
        Response Structure:
          - Returns moved event with Google Calendar API v3 format
          - Event will have the same content but potentially new ID
        
        Status Codes:
          - 200: Success - Event moved successfully
          - 404: Not Found - Event, source calendar, or destination calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Source calendar identifier",
                },
                "eventId": {
                    "type": "string",
                    "description": "Event identifier to move",
                },
                "destination": {
                    "type": "string",
                    "description": "Target calendar identifier where event will be moved",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications: all, externalOnly, none",
                },
            },
            "required": ["calendarId", "eventId", "destination"],
        },
    },
    {
        "name": "quick_add_event",
        "description": """Create an event using natural language text parsing.
        
        Creates an event from natural language text following Google Calendar API v3 structure.
        Automatically extracts date, time, and other details from the provided text.
        Similar to Quick Add feature in Google Calendar.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier where event will be created
        
        Required Query Parameters:
          - text: Required. Natural language text describing the event (e.g., 'Meeting tomorrow at 2pm')

        Request Body Requirements:
          - sendUpdates: Optional. Guests who should receive notifications (all, externalOnly, none)
        
        Text Parsing:
          - Automatically extracts event title from text
          - Identifies date and time information
          - Recognizes common patterns like 'tomorrow', 'next week', 'at 3pm'
          - May identify location if mentioned in text
          - Creates a properly formatted event from parsed information
        
        Response Structure:
          - Returns created event with Google Calendar API v3 format
          - Event will contain parsed information from the text
        
        Status Codes:
          - 201: Created - Event created successfully from text
          - 400: Bad Request - Unable to parse text or invalid calendar
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Calendar identifier where event will be created",
                },
                "text": {
                    "type": "string",
                    "description": "Natural language text describing the event (e.g., 'Meeting tomorrow at 2pm')",
                },
                "sendUpdates": {
                    "type": "string",
                    "description": "Guests who should receive notifications: all, externalOnly, none",
                },
            },
            "required": ["calendarId", "text"],
        },
    },
    {
        "name": "import_event",
        "description": """Import an event as a private copy to the specified calendar.
        
        This operation is used to add a private copy of an existing event to a calendar, following
        Google Calendar API v3 import specification. Supports comprehensive event data including
        type conversion, iCalUID handling, and attachment processing.
        
        Google Calendar API Import Features:
          - Creates a private copy of the event in the target calendar
          - Handles event type conversion (non-default types may be converted to default)
          - Manages iCalUID uniqueness and conflict resolution
          - Processes attendees, attachments, and conference data based on client capabilities
          - Supports all Google Calendar event types: default, birthday, focusTime, fromGmail, outOfOffice, workingLocation
          
        Request Body Requirements:
          - calendarId: Required. Calendar identifier where event will be imported
          - iCalUID: iCalendar UID for external calendar integration
          - start: Required. Event start time (dateTime or date with optional timeZone)
          - end: Required. Event end time (dateTime or date with optional timeZone)
          
        Advanced Import Properties:
          - attendees: Array of attendee objects with email, displayName, optional, responseStatus
          - conferenceData: Conference/meeting data (processed based on conferenceDataVersion)
          - attachments: File attachments (processed if supportsAttachments=true)
          - extendedProperties: Private and shared extended properties
          - focusTimeProperties: Focus time settings (for focusTime events)
          - outOfOfficeProperties: Out of office details (for outOfOffice events)
          - recurrence: Array of RRULE strings for recurring events
          - reminders: Reminder configuration with useDefault and overrides
          - source: Source information with title and URL
          - transparency: Event transparency (opaque/transparent)
          - visibility: Event visibility (default, public, private, confidential)
          - sequence: iCalendar sequence number
          - sequence: iCalendar sequence number
          - colorId: Color ID of the event (1-11 for event colors)
          - originalStartTime: Original start time for recurring event instances (must match start values)
          
        Query Parameters:
          - conferenceDataVersion: Conference data version support (0 or 1)
          - supportsAttachments: Whether client supports event attachments
          
        Import Processing:
          - Validates event data and user permissions
          - Converts unsupported event types to 'default' with warnings
          - Generates new event ID while preserving iCalUID if provided
          - Creates comprehensive event with all related objects (attendees, attachments, etc.)
          - Handles all-day events vs timed events automatically
          
        Response Structure:
          - Returns EventImportResponse with Google Calendar API v3 format:
            * Complete event data with generated IDs
            * Creator and organizer information
            * Conversion warnings if event type was changed
            * All imported properties and relationships
        
        Status Codes:
          - 201: Created - Event imported successfully as private copy
          - 400: Bad Request - Invalid import data or missing required fields
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Calendar identifier where event will be imported",
                },
                "summary": {
                    "type": "string",
                    "description": "Event title/summary (required)",
                },
                "description": {"type": "string", "description": "Event description"},
                "location": {"type": "string", "description": "Event location"},
                "start": {
                    "type": "object",
                    "description": "Event start time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "end": {
                    "type": "object",
                    "description": "Event end time (required)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "organizer": {
                    "type": "object",
                    "description": "The organizer of the event. If the organizer is also an attendee, this is indicated with a separate entry in attendees with the organizer field set to True. To change the organizer, use the move operation. Read-only, except when importing an event.",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The organizer's email address. Must be a valid email address as per RFC5322.",
                        },
                        "displayName": {
                            "type": "string",
                            "description": "The organizer's name, if available.",
                        },
                    },
                    "required": ["email"],
                },
                "status": {
                    "type": "string",
                    "description": "Event status: confirmed, tentative, cancelled",
                    "enum": ["confirmed", "tentative", "cancelled"],
                },
                "transparency": {
                    "type": "string",
                    "description": "Event transparency: opaque, transparent",
                    "enum": ["opaque", "transparent"],
                },
                "visibility": {
                    "type": "string",
                    "description": "Event visibility: default, public, private, confidential",
                    "enum": ["default", "public", "private", "confidential"],
                },
                "attendees": {
                    "type": "array",
                    "description": "Event attendees",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Attendee email address",
                            },
                            "displayName": {
                                "type": "string",
                                "description": "Attendee display name",
                            },
                            "optional": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether attendee is optional",
                            },
                            "resource": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether attendee is a resource",
                            },
                            "responseStatus": {
                                "type": "string",
                                "default": "needsAction",
                                "description": "Response status",
                            },
                            "comment": {
                                "type": "string",
                                "description": "Attendee comment",
                            },
                            "additionalGuests": {
                                "type": "integer",
                                "default": 0,
                                "description": "Number of additional guests",
                            },
                        },
                        "required": ["email", "responseStatus"],
                    },
                },
                "attendeesOmitted": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether attendees may have been omitted from the event's representation",
                },
                "colorId": {
                    "type": "string",
                    "description": "Color ID of the event (1-11 for event colors)",
                },
                "recurrence": {
                    "type": "array",
                    "description": """List of RRULE, EXRULE, RDATE and EXDATE lines for recurring events following RFC 5545 (iCalendar) standard.
                    
                    Supported Recurrence Types:
                    
                    RRULE (Recurrence Rule) - Defines the pattern for recurring events:
                    • FREQ: Frequency (DAILY, WEEKLY, MONTHLY, YEARLY)
                    • INTERVAL: Interval between occurrences (e.g., INTERVAL=2 for every 2 weeks)
                    • COUNT: Maximum number of occurrences
                    • UNTIL: End date (format: YYYYMMDDTHHMMSSZ)
                    • BYDAY: Days of week (MO, TU, WE, TH, FR, SA, SU)
                    • BYMONTHDAY: Days of month (1-31)
                    • BYMONTH: Months (1-12)
                    • BYSETPOS: Position in set (e.g., 1st, 2nd, -1 for last)
                    
                    EXDATE (Exception Dates) - Exclude specific occurrences:
                    • Format: EXDATE:YYYYMMDDTHHMMSSZ or EXDATE;VALUE=DATE:YYYYMMDD
                    • Use timezone format for timed events, date format for all-day events
                    
                    RDATE (Recurrence Dates) - Add specific occurrences:
                    • Format: RDATE:YYYYMMDDTHHMMSSZ or RDATE;VALUE=DATE:YYYYMMDD
                    
                    Common Examples:
                    • Daily: ["RRULE:FREQ=DAILY"]
                    • Every weekday: ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
                    • Weekly on specific days: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH"]
                    • Monthly on specific date: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=15"]
                    • First Monday of month: ["RRULE:FREQ=MONTHLY;BYDAY=1MO"]
                    • Last Friday of month: ["RRULE:FREQ=MONTHLY;BYDAY=-1FR"]
                    • Yearly: ["RRULE:FREQ=YEARLY"]
                    • Every 2 weeks: ["RRULE:FREQ=WEEKLY;INTERVAL=2"]
                    • 10 times only: ["RRULE:FREQ=WEEKLY;COUNT=10"]
                    • Until specific date: ["RRULE:FREQ=DAILY;UNTIL=20231231T235959Z"]
                    
                    Complex Examples:
                    • Weekly with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=TU,TH", "EXDATE:20231024T100000Z"]
                    • Monthly with additional dates: ["RRULE:FREQ=MONTHLY;BYMONTHDAY=1", "RDATE:20231215T100000Z"]
                    • Every other month on 2nd Tuesday: ["RRULE:FREQ=MONTHLY;INTERVAL=2;BYDAY=2TU"]
                    
                    All-Day Event Examples:
                    • Daily all-day: ["RRULE:FREQ=DAILY"]
                    • Weekly all-day with exceptions: ["RRULE:FREQ=WEEKLY;BYDAY=MO", "EXDATE;VALUE=DATE:20231030"]
                    
                    Note: For all-day events, use VALUE=DATE format for EXDATE/RDATE. For timed events, use full timestamp format.""",
                    "items": {"type": "string"},
                },
                "reminders": {
                    "type": "object",
                    "description": "Event reminders configuration",
                    "properties": {
                        "useDefault": {
                            "type": "boolean",
                            "description": "Whether to use default reminders",
                        },
                        "overrides": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email", "popup"],
                                    },
                                    "minutes": {
                                        "type": "integer",
                                        "description": "Minutes before event",
                                    },
                                },
                            },
                        },
                    },
                },
                "attachments": {
                    "type": "array",
                    "description": "Event attachments (processed if supportsAttachments=true)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fileUrl": {
                                "type": "string",
                                "description": "URL of attached file",
                            }
                        },
                        "required": ["fileUrl"],
                    },
                },
                "conferenceData": {
                    "type": "object",
                    "description": "Conference data (processed based on conferenceDataVersion)",
                    "properties": {
                        "createRequest": {
                            "type": "object",
                            "description": "Request to create conference",
                        },
                        "entryPoints": {
                            "type": "array",
                            "description": "Conference entry points",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entryPointType": {
                                        "type": "string",
                                        "description": "Entry point type (video, phone, sip, more)",
                                    },
                                    "uri": {
                                        "type": "string",
                                        "description": "URI of the entry point",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Label for the entry point",
                                    },
                                    "pin": {
                                        "type": "string",
                                        "description": "PIN to access the conference",
                                    },
                                    "accessCode": {
                                        "type": "string",
                                        "description": "Access code for the conference",
                                    },
                                    "meetingCode": {
                                        "type": "string",
                                        "description": "Meeting code for the conference",
                                    },
                                    "passcode": {
                                        "type": "string",
                                        "description": "Passcode for the conference",
                                    },
                                    "password": {
                                        "type": "string",
                                        "description": "Password for the conference",
                                    },
                                },
                            },
                        },
                        "conferenceSolution": {
                            "type": "object",
                            "description": "Conference solution details",
                        },
                        "conferenceId": {
                            "type": "string",
                            "description": "Conference ID",
                        },
                        "signature": {
                            "type": "string",
                            "description": "Conference signature",
                        },
                    },
                },
                "extendedProperties": {
                    "type": "object",
                    "description": "Extended properties",
                    "properties": {
                        "private": {
                            "type": "object",
                            "description": "Private extended properties",
                        },
                        "shared": {
                            "type": "object",
                            "description": "Shared extended properties",
                        },
                    },
                },
                "focusTimeProperties": {
                    "type": "object",
                    "description": "Focus time properties for focusTime events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "chatStatus": {
                            "type": "string",
                            "description": "The status to mark the user in Chat and related products",
                            "enum": ["available", "doNotDisturb"],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "outOfOfficeProperties": {
                    "type": "object",
                    "description": "Out of office properties for outOfOffice events",
                    "properties": {
                        "autoDeclineMode": {
                            "type": "string",
                            "description": "Whether to decline meeting invitations which overlap Focus Time events",
                            "enum": [
                                "declineNone",
                                "declineAllConflictingInvitations",
                                "declineOnlyNewConflictingInvitations",
                            ],
                        },
                        "declineMessage": {
                            "type": "string",
                            "description": "Response message to set if an existing event or new invitation is automatically declined by Calendar",
                        },
                    },
                },
                "originalStartTime": {
                    "type": "object",
                    "description": "Original start time for event instances (must match start field values)",
                    "properties": {
                        "dateTime": {
                            "type": "string",
                            "description": "RFC3339 timestamp with timezone for timed events",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format for all-day events",
                        },
                        "timeZone": {
                            "type": "string",
                            "description": "IANA timezone identifier",
                        },
                    },
                },
                "source": {
                    "type": "object",
                    "description": "Event source information",
                    "properties": {
                        "url": {"type": "string", "description": "Source URL"},
                        "title": {"type": "string", "description": "Source title"},
                    },
                    "required": ["url", "title"],
                },
                "iCalUID": {
                    "type": "string",
                    "description": "iCalendar UID for external calendar integration",
                },
                "sequence": {
                    "type": "integer",
                    "description": "iCalendar sequence number",
                },
                "guestsCanInviteOthers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether attendees can invite others",
                },
                "guestsCanModify": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether attendees can modify the event",
                },
                "guestsCanSeeOtherGuests": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether attendees can see other guests",
                },
                "conferenceDataVersion": {
                    "type": "integer",
                    "description": "Version number of conference data supported by API client (0 or 1)",
                },
                "supportsAttachments": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether API client supports event attachments",
                },
            },
            "required": ["calendarId", "start", "end", "iCalUID"],
        },
    },
    {
        "name": "get_event_instances",
        "description": """Returns instances of the specified recurring event following Google Calendar API v3 specification.
        
        Returns individual instances of a recurring event with complete Google Calendar API v3 compatibility.
        Expands recurring events into their actual occurrences within the specified time range.
        Supports all official query parameters including filtering, pagination, and response formatting.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier
          - eventId: Required. Recurring event identifier
        
        Optional Parameters (Google Calendar API v3 compliant):
          - maxAttendees: The maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.
          - maxResults: Maximum number of events returned on one result page. By default the value is 250 events. The page size can never be larger than 2500 events. Optional.
          - originalStart: The original start time of the instance in the result. Optional.
          - pageToken: Token specifying which result page to return. Optional.
          - showDeleted: Whether to include deleted events (with status equals 'cancelled') in the result. Cancelled instances of recurring events will still be included if singleEvents is False. Optional.
          - timeMin: Lower bound (inclusive) for an event's end time to filter by. Optional. The default is not to filter by end time. Must be an RFC3339 timestamp with mandatory time zone offset.
          - timeMax: Upper bound (exclusive) for an event's start time to filter by. Optional. The default is not to filter by start time. Must be an RFC3339 timestamp with mandatory time zone offset.
          - timeZone: Time zone used in the response. Optional. The default is the time zone of the calendar.
        
        Instance Expansion Features:
          - Takes a recurring event and expands it into individual occurrences
          - Each instance shows the actual date/time it occurs
          - Useful for displaying recurring events in calendar views
          - Respects time range filters to limit results
          - Supports pagination for large recurring series
          - Handles timezone conversions based on timeZone parameter
          - Includes deleted instances when showDeleted=true
        
        Response Structure:
          - Returns events collection with Google Calendar API v3 format:
            * kind: "calendar#events"
            * etag: ETag of the collection
            * summary: Calendar summary
            * description: Calendar description
            * updated: Last modification time
            * timeZone: Response timezone
            * accessRole: User's access role
            * defaultReminders: Default reminders for the calendar
            * nextPageToken: Token for next page (if applicable)
            * items: Array of individual event instances
        
        Status Codes:
          - 200: Success - Event instances retrieved successfully
          - 400: Bad Request - Invalid parameters or non-recurring event
          - 403: Forbidden - Insufficient permissions
          - 404: Not Found - Event or calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "description": "Calendar identifier"},
                "eventId": {
                    "type": "string",
                    "description": "Recurring event identifier",
                },
                "maxAttendees": {
                    "type": "integer",
                    "description": "The maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
                },
                "maxResults": {
                    "type": "integer",
                    "description": "Maximum number of events returned on one result page. By default the value is 250 events. The page size can never be larger than 2500 events. Optional.",
                    "minimum": 1,
                    "maximum": 2500,
                },
                "originalStart": {
                    "type": "string",
                    "description": "The original start time of the instance in the result. Optional.",
                },
                "pageToken": {
                    "type": "string",
                    "description": "Token specifying which result page to return. Optional.",
                },
                "showDeleted": {
                    "type": "boolean",
                    "description": "Whether to include deleted events (with status equals 'cancelled') in the result. Cancelled instances of recurring events will still be included if singleEvents is False. Optional.",
                },
                "timeMin": {
                    "type": "string",
                    "description": "Lower bound (inclusive) for an event's end time to filter by. Optional. The default is not to filter by end time. Must be an RFC3339 timestamp with mandatory time zone offset.",
                },
                "timeMax": {
                    "type": "string",
                    "description": "Upper bound (exclusive) for an event's start time to filter by. Optional. The default is not to filter by start time. Must be an RFC3339 timestamp with mandatory time zone offset.",
                },
                "timeZone": {
                    "type": "string",
                    "description": "Time zone used in the response. Optional. The default is the time zone of the calendar.",
                },
            },
            "required": ["calendarId", "eventId"],
        },
    },
    {
        "name": "watch_events",
        "description": """Set up a webhook to receive notifications when events change.
        
        Sets up webhook notifications for event changes following Google Calendar API v3 structure.
        Returns a channel for managing the watch. Monitors events in the specified calendar for changes.
        
        Request Body Requirements:
          - calendarId: Required. Calendar identifier to watch for changes

        Optional Query Parameters:
          - eventTypes(string): Acceptable values are:
                    "birthday": Special all-day events with an annual recurrence.
                    "default": Regular events.
                    "focusTime": Focus time events.
                    "fromGmail": Events from Gmail.
                    "outOfOffice": Out of office events.
                    "workingLocation": Working location events. 
        
        Request Body Requirements:
          - id: Required. Unique channel identifier
          - address: Required. Webhook URL to receive notifications
          - type: Optional. Channel type (default: \"web_hook\")
          - token: Optional. Optional token for webhook authentication
          - params: Optional. Additional channel parameters
        
        Webhook Notifications:
          - Server will send POST requests to the specified address
          - Notifications triggered by event changes:
            * Event created, updated, or deleted
            * Event moved between calendars
            * Event status changes
        
        Channel Management:
          - Each watch creates a unique notification channel
          - Channels can expire (set expiration time)
          - Multiple channels can watch the same calendar
          - Use unique channel IDs to avoid conflicts
        
        Response Structure:
          - Returns Channel resource with Google Calendar API v3 format:
            * kind: \"api#channel\"
            * id: Channel identifier
            * resourceId: Resource being watched
            * resourceUri: Resource URI path
        
        Status Codes:
          - 200: Success - Watch channel created successfully
          - 400: Bad Request - Invalid channel configuration
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Calendar identifier to watch for changes",
                },
                "id": {"type": "string", "description": "Unique channel identifier"},
                "type": {"type": "string", "description": "Channel type (web_hook)"},
                "address": {
                    "type": "string",
                    "description": "Webhook URL to receive notifications",
                },
                "token": {
                    "type": "string",
                    "description": "Optional token for webhook authentication",
                },
                "params": {
                    "type": "object",
                    "description": "Additional channel parameters",
                    "properties": {
                        "ttl": {
                            "type": "string",
                            "description": "Time to live in seconds (string).",
                        }
                    },
                },
                "eventTypes": {
                    "type": "string",
                    "description": "Event types of resources to watch. Optional. This parameter can be repeated multiple times to watch resources of different types. If unset, returns all event types. Acceptable values are: 'birthday' - Special all-day events with an annual recurrence, 'default' - Regular events, 'focusTime' - Focus time events, 'fromGmail' - Events from Gmail, 'outOfOffice' - Out of office events, 'workingLocation' - Working location events.",
                },
            },
            "required": ["calendarId", "id", "type", "address"],
        },
    },
]
