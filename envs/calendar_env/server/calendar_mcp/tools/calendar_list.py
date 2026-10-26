"""
CalendarList Tools Module

This module contains tools related to calendar list management.
Covers all 7 Google Calendar API v3 CalendarList endpoints for user calendar list operations.
"""

CALENDAR_LIST_TOOLS = [
    {
        "name": "get_calendar_list",
        "description": """Returns the calendars on the user's calendar list.
        
        Lists all calendars in the user's calendar list with their display settings and access permissions.
        Returns calendars with user-specific customizations like colors, visibility, and notification settings.
        Supports pagination through pageToken and incremental synchronization through syncToken.
        
        Request Body Requirements:
          - maxResults: Optional. Maximum number of entries returned (pagination). If 0, returns no items.
          - minAccessRole: Optional. Minimum access role filter (freeBusyReader, reader, writer, owner)
          - pageToken: Optional. Token specifying which result page to return (for pagination)
          - showDeleted: Optional. Include deleted calendars in results (default: false)
          - showHidden: Optional. Include hidden calendars in results (default: false)
          - syncToken: Optional. Token for incremental synchronization (returns only changed entries)
        
        Pagination Support:
          - Use maxResults to limit the number of calendars returned per page
          - Use pageToken to retrieve subsequent pages of results
          - Check nextPageToken in response to determine if more results are available
          - Pass nextPageToken as pageToken in the next request to get the next page
        
        Incremental Synchronization:
          - Use syncToken to get only entries that have changed since the last request
          - syncToken cannot be used together with minAccessRole parameter
          - When using syncToken, deleted and hidden entries are automatically included
          - If syncToken expires, server returns 410 GONE and client should perform full sync
          - Use nextSyncToken from response for subsequent incremental sync requests
        
        Response Structure:
          - Returns calendar list collection with Google Calendar API v3 format:
            * kind: "calendar#calendarList"
            * etag: ETag of the collection
            * items: Array of CalendarListEntry objects
            * nextPageToken: Token for next page (if more results available)
            * nextSyncToken: Token for incremental sync (always provided when items are returned)
        
        CalendarListEntry Structure:
          - Each item contains calendar metadata plus user-specific settings:
            * id: Calendar identifier
            * summary: Display title (with summaryOverride if set)
            * accessRole: User's permission level
            * primary: Whether it's the user's primary calendar
            * backgroundColor/foregroundColor: Display colors
            * hidden: Whether hidden from calendar list
            * selected: Whether selected in UI
            * defaultReminders: User-specific default reminders
            * notificationSettings: Notification preferences
        
        Status Codes:
          - 200: Success - Calendar list retrieved successfully
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "maxResults": {
                    "type": "integer",
                    "description": "Maximum number of entries returned. If 0, returns no items.",
                    "minimum": 0,
                    "maximum": 250,
                    "default": 100,
                },
                "minAccessRole": {
                    "type": "string",
                    "description": "Minimum access role filter",
                    "enum": ["freeBusyReader", "reader", "writer", "owner"],
                },
                "pageToken": {
                    "type": "string",
                    "description": "Token specifying which result page to return (for pagination)",
                },
                "showDeleted": {
                    "type": "boolean",
                    "description": "Include deleted calendars in results",
                    "default": False,
                },
                "showHidden": {
                    "type": "boolean",
                    "description": "Include hidden calendars in results",
                    "default": False,
                },
                "syncToken": {
                    "type": "string",
                    "description": "Token for incremental synchronization (returns only changed entries since last sync)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_calendar_from_list",
        "description": """Returns a calendar from the user's calendar list.
        
        Retrieves a specific calendar entry from the user's calendar list with all user-specific settings.
        Shows how the calendar appears in the user's list with customizations and access permissions.
        
        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or the keyword 'primary' to refer to the user's primary calendar
        
        Response Structure:
          - Returns CalendarListEntry with Google Calendar API v3 format:
            * kind: "calendar#calendarListEntry"
            * etag: ETag of the resource
            * id: Calendar identifier
            * summary: Display title (with summaryOverride if set)
            * description: Calendar description
            * location: Calendar location
            * timeZone: Calendar timezone
            * summaryOverride: Custom title override for this user
            * colorId: Calendar color ID
            * backgroundColor: Background color (hex)
            * foregroundColor: Foreground color (hex)
            * hidden: Whether hidden from calendar list
            * selected: Whether selected in UI
            * accessRole: User's permission level (freeBusyReader, reader, writer, owner)
            * defaultReminders: User-specific default reminders
            * notificationSettings: Notification preferences
            * primary: Whether it's the user's primary calendar
            * deleted: Whether the calendar is deleted
        
        Status Codes:
          - 200: Success - Calendar entry retrieved successfully
          - 404: Not Found - Calendar not found in user's list
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
        "name": "add_calendar_to_list",
        "description": """Inserts an existing calendar into the user's calendar list.
        
        Adds an existing calendar to the user's calendar list with custom display settings.
        The calendar must already exist - this endpoint only adds it to the user's list with personalization.
        
        Request Body Requirements:
          - id: Required. Calendar ID to add to user's list
          - summaryOverride: Optional. Custom calendar title override
          - colorId: Optional. Calendar color ID
          - backgroundColor: Optional. Background color (hex format like #FF5733)
          - foregroundColor: Optional. Foreground color (hex format like #FFFFFF)
          - hidden: Optional. Whether calendar is hidden from list (default: false)
          - selected: Optional. Whether calendar is selected in UI (default: true)
          - defaultReminders: Optional. Array of default reminder settings
          - notificationSettings: Optional. Notification preferences
        
        Calendar Must Exist:
          - The calendar with the specified ID must already exist in the database
          - This endpoint does not create new calendars, only adds them to user's list
          - Use create_calendar tool first if the calendar doesn't exist
        
        Response Structure:
          - Returns the created CalendarListEntry with Google Calendar API v3 format
          - Includes all user-specific customizations applied
        
        Colors:
          - To set backgroundColor/foregroundColor you must pass query param colorRgbFormat=true
          - When RGB fields are provided, colorId (if present) is ignored
        
        Status Codes:
          - 201: Created - Calendar added to list successfully
          - 404: Not Found - Calendar with specified ID not found
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Calendar ID to add to user's list (UUID)",
                    "minLength": 1,
                },
                "colorRgbFormat": {
                    "type": "boolean",
                    "description": "Query param: if true, allows writing backgroundColor/foregroundColor",
                    "default": False,
                },
                "summaryOverride": {
                    "type": "string",
                    "description": "Custom calendar title override",
                    "maxLength": 255,
                },
                "colorId": {
                    "type": "string",
                    "description": "Calendar color ID",
                    "maxLength": 50,
                },
                "backgroundColor": {
                    "type": "string",
                    "description": "Background color (hex format like #FF5733)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "foregroundColor": {
                    "type": "string",
                    "description": "Foreground color (hex format like #FFFFFF)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "hidden": {
                    "type": "boolean",
                    "description": "Whether calendar is hidden from list",
                    "default": False,
                },
                "selected": {
                    "type": "boolean",
                    "description": "Whether calendar is selected in UI",
                    "default": True,
                },
                "defaultReminders": {
                    "type": "array",
                    "description": "Default reminder settings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["email", "popup"],
                                "description": "Reminder delivery method",
                            },
                            "minutes": {
                                "type": "integer",
                                "description": "Minutes before event to trigger reminder",
                                "minimum": 0,
                            },
                        },
                        "required": ["method", "minutes"],
                    },
                },
                "notificationSettings": {
                    "type": "object",
                    "description": "Notification preferences",
                    "properties": {
                        "notifications": {
                            "type": "array",
                            "description": "List of notification settings",
                            "items": {
                                "type": "object",
                                "description": "Individual notification setting",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email"],
                                        "description": "Notification delivery method (only 'email' supported)",
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "eventCreation",
                                            "eventChange",
                                            "eventCancellation",
                                            "eventResponse",
                                            "agenda",
                                        ],
                                        "description": "Notification type",
                                    },
                                },
                                "required": ["method", "type"],
                            },
                        }
                    },
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "update_calendar_in_list",
        "description": """Updates an entry on the user's calendar list (partial update).
        
        Partially updates calendar list entry settings. Only provided fields will be updated,
        others remain unchanged. This allows fine-grained control over calendar display settings.
        
        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or 'primary'
        
        Request Body (all optional):
          - summaryOverride: Custom calendar title override
          - colorId: Calendar color ID
          - backgroundColor: Background color (hex format like #FF5733)
          - foregroundColor: Foreground color (hex format like #FFFFFF)
          - hidden: Whether calendar is hidden from list
          - selected: Whether calendar is selected in UI
          - defaultReminders: Array of default reminder settings
          - notificationSettings: Notification preferences
        
        Partial Update Behavior:
          - Only fields provided in request body are updated
          - Null values will clear the field (set to null) for string/complex fields
          - 'hidden' and 'selected' cannot be null (booleans are NOT NULL in our DB)
          - Missing fields are left unchanged
          - At least one field must be provided for update

        Response Structure:
          - Returns updated CalendarListEntry with Google Calendar API v3 format
          - Shows all current settings including unchanged fields

        Colors:
          - To set backgroundColor/foregroundColor you must pass query param colorRgbFormat=true
          - When RGB fields are provided, colorId (if present) is ignored

        Behavioral coupling:
          - If hidden=true, the server will force selected=false
          - If hidden=false, the server will force selected=true (matches observed UI behavior)
        
        Status Codes:
          - 200: Success - Calendar list entry updated successfully
          - 400: Bad Request - No fields provided for update
          - 404: Not Found - Calendar not found in user's list
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID)",
                    "minLength": 1,
                },
                "colorRgbFormat": {
                    "type": "boolean",
                    "description": "Query param: if true, allows writing backgroundColor/foregroundColor",
                    "default": False,
                },
                "summaryOverride": {
                    "type": "string",
                    "description": "Custom calendar title override",
                    "maxLength": 255,
                },
                "colorId": {
                    "type": "string",
                    "description": "Calendar color ID",
                    "maxLength": 50,
                },
                "backgroundColor": {
                    "type": "string",
                    "description": "Background color (hex format like #FF5733)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "foregroundColor": {
                    "type": "string",
                    "description": "Foreground color (hex format like #FFFFFF)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "hidden": {
                    "type": "boolean",
                    "description": "Whether calendar is hidden from list",
                    "default": False,
                },
                "selected": {
                    "type": "boolean",
                    "description": "Whether calendar is selected in UI",
                    "default": True,
                },
                "defaultReminders": {
                    "type": "array",
                    "description": "Default reminder settings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["email", "popup"],
                                "description": "Reminder delivery method",
                            },
                            "minutes": {
                                "type": "integer",
                                "description": "Minutes before event to trigger reminder",
                                "minimum": 0,
                            },
                        },
                        "required": ["method", "minutes"],
                    },
                },
                "notificationSettings": {
                    "type": "object",
                    "description": "Notification preferences",
                    "properties": {
                        "notifications": {
                            "type": "array",
                            "description": "List of notification settings",
                            "items": {
                                "type": "object",
                                "description": "Individual notification setting",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["email"],
                                        "description": "Notification delivery method (only 'email' supported)",
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "eventCreation",
                                            "eventChange",
                                            "eventCancellation",
                                            "eventResponse",
                                            "agenda",
                                        ],
                                        "description": "Notification type",
                                    },
                                },
                                "required": ["method", "type"],
                            },
                        }
                    },
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "replace_calendar_in_list",
        "description": """Updates an entry on the user's calendar list (full update).
        
        Fully updates calendar list entry settings. All fields are replaced with provided values.
        Fields not provided will be set to their default values (full replacement).
        
        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID) or 'primary'
        
        Request Body (all optional, but null/missing values will be set to defaults):
          - summaryOverride: Custom calendar title override (null to clear)
          - colorId: Calendar color ID (null to clear)
          - backgroundColor: Background color (hex format like #FF5733, null to clear)
          - foregroundColor: Foreground color (hex format like #FFFFFF, null to clear)
          - hidden: Whether calendar is hidden from list (default: false)
          - selected: Whether calendar is selected in UI (default: true)
          - defaultReminders: Array of default reminder settings (null to clear)
          - notificationSettings: Notification preferences (null to clear)
          - conferenceProperties: Conference properties for this calendar (null to clear)
        
        Full Update Behavior:
          - All fields are replaced (full replacement operation)
          - Missing optional fields are set to null/defaults
          - Required fields (hidden, selected) get default values if not provided
          - This is different from PATCH which only updates provided fields

        Response Structure:
          - Returns updated CalendarListEntry with Google Calendar API v3 format
          - Shows all current settings after full update

        Colors:
          - To set backgroundColor/foregroundColor you must pass query param colorRgbFormat=true
          - When RGB fields are provided, colorId (if present) is ignored

        Behavioral coupling:
          - If hidden=true, the server will force selected=false
          - If hidden=false, the server will force selected=true (matches observed UI behavior)
        
        Status Codes:
          - 200: Success - Calendar list entry updated successfully
          - 404: Not Found - Calendar not found in user's list
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "description": "Unique calendar identifier (UUID)",
                    "minLength": 1,
                },
                "colorRgbFormat": {
                    "type": "boolean",
                    "description": "Query param: if true, allows writing backgroundColor/foregroundColor",
                    "default": False,
                },
                "summaryOverride": {
                    "type": "string",
                    "description": "Custom calendar title override",
                    "maxLength": 255,
                },
                "colorId": {
                    "type": "string",
                    "description": "Calendar color ID",
                    "maxLength": 50,
                },
                "backgroundColor": {
                    "type": "string",
                    "description": "Background color (hex format like #FF5733)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "foregroundColor": {
                    "type": "string",
                    "description": "Foreground color (hex format like #FFFFFF)",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                },
                "hidden": {
                    "type": "boolean",
                    "description": "Whether calendar is hidden from list",
                    "default": False,
                },
                "selected": {
                    "type": "boolean",
                    "description": "Whether calendar is selected in UI",
                    "default": True,
                },
                "defaultReminders": {
                    "type": "array",
                    "description": "Default reminder settings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "description": "Reminder delivery method (email, popup). Empty string allowed to clear",
                            },
                            "minutes": {
                                "type": "integer",
                                "description": "Minutes before event to trigger reminder",
                                "minimum": 0,
                            },
                        },
                    },
                },
                "notificationSettings": {
                    "type": "object",
                    "description": "Notification preferences",
                    "properties": {
                        "notifications": {
                            "type": "array",
                            "description": "List of notification settings",
                            "items": {
                                "type": "object",
                                "description": "Individual notification setting",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "description": "Notification delivery method. Empty string allowed to clear",
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Notification type. Empty string allowed to clear",
                                    },
                                },
                            },
                        }
                    },
                },
                "conferenceProperties": {
                    "type": "object",
                    "description": "Conference properties for this calendar",
                    "properties": {
                        "allowedConferenceSolutionTypes": {
                            "type": "array",
                            "description": "The types of conference solutions that are supported for this calendar",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "eventHangout",
                                    "eventNamedHangout",
                                    "hangoutsMeet",
                                ],
                                "description": "Conference solution type",
                            },
                        }
                    },
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "remove_calendar_from_list",
        "description": """Removes a calendar from the user's calendar list.
        
        Removes a calendar from the user's calendar list (soft delete).
        The calendar itself remains in the database but is no longer visible in the user's list.
        Primary calendars cannot be removed from the calendar list.
        
        Request Body Requirements:
          - calendarId: Required. Unique calendar identifier (UUID)
        
        Primary Calendar Protection:
          - Primary calendars cannot be removed from the calendar list
          - Attempting to remove primary calendar returns 400 Bad Request
          - Primary calendars are always part of the user's list
        
        Operation Details:
          - Calendar is soft-deleted (marked as deleted, not physically removed)
          - Calendar data remains in database but is hidden from list
          - Calendar can potentially be re-added to list later
          - Does not affect the underlying calendar data or events
        
        Response Structure:
          - Returns 204 No Content on successful removal
          - No response body as per Google Calendar API v3
        
        Status Codes:
          - 204: No Content - Calendar removed from list successfully
          - 400: Bad Request - Attempted to remove primary calendar from list
          - 404: Not Found - Calendar not found in user's list
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
        "name": "watch_calendar_list",
        "description": """Watch for changes to CalendarList resources.
        
        Sets up webhook notifications (Channel) for changes to the user's calendar list.
        Monitors for additions, removals, and updates to calendar list entries.
        
        Request Body (Channel):
          - id: Required. Unique channel identifier for this watch
          - type: Channel type (only "web_hook" (or "webhook") supported)
          - address: Required. HTTPS URL where notifications will be sent
          - token: Optional. Verification token for webhook security
        
        Webhook Notifications (simplified):
          - Server will send POST requests to the specified address
          - Notifications triggered by calendar list changes:
            * Calendar added to or removed from list
            * Calendar list entry settings updated
            * Calendar permissions changed
        
        Channel Response:
          - Returns Channel object with fields: kind, id, resourceId, resourceUri, token, expiration, type, address
          - resourceUri is the collection path: /users/me/calendarList
          - resourceId is generated by the server

        Channel Management:
          - Each watch creates a unique notification channel
          - Channels can expire (set expiration time)
          - Multiple channels can watch the same resource
          - Use unique channel IDs to avoid conflicts
        
        Response Structure:
          - Returns Channel resource with Google Calendar API v3 format:
            * kind: "api#channel"
            * id: Channel identifier
            * resourceId: Resource being watched
            * resourceUri: Resource URI path
            * token: Verification token (if provided)
            * expiration: Channel expiration time (if set)
            * type: Channel type "web_hook" (or "webhook").
            * address: Notification delivery address
        
        Status Codes:
          - 200: Success - Watch channel created successfully
          - 400: Bad Request - Invalid channel configuration or query parameters
          - 500: Internal Server Error
          
        Note: This is a simplified implementation. In production, you would need:
          - Webhook endpoint verification
          - Channel management and cleanup
          - Actual change detection and notification dispatch""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique channel identifier for this watch",
                    "minLength": 1,
                },
                "type": {
                    "type": "string",
                    "description": "Channel type (only web_hook supported; 'webhook' accepted as alias)",
                    "enum": ["web_hook", "webhook"],
                    "default": "web_hook",
                },
                "address": {
                    "type": "string",
                    "description": "HTTPS URL where notifications will be sent",
                    "format": "uri",
                    "minLength": 1,
                },
                "token": {
                    "type": "string",
                    "description": "Verification token for webhook security",
                    "maxLength": 256,
                },
                "params": {
                    "type": "object",
                    "description": "Optional parameters (Google spec supports 'ttl' in seconds as string)",
                    "properties": {
                        "ttl": {
                            "type": "string",
                            "description": "Time to live in seconds (string). Server computes expiration = now + ttl",
                        }
                    },
                },
            },
            "required": ["id", "type", "address"],
        },
    },
]
