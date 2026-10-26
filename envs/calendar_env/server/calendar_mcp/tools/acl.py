"""
ACL Tools Module

This module contains tools related to ACL.
Follows Google Calendar API v3 structure for ACL operations.
"""

ACL_TOOLS = [
    {
        "name": "get_acl_rule",
        "description": """Retrieve an access control rule by ID.

        Fetches the ACL rule for a calendar by rule ID.
        Follows the structure of Google Calendar API v3 `/calendars/{calendarId}/acl/{ruleId}`.

        Required Parameters:
          - calendarId: Calendar identifier
          - ruleId: ACL rule identifier

        Response Structure:
          - kind: "calendar#aclRule"
          - etag: ETag
          - id: ACL rule ID
          - scope: { type, value }
          - role: ACL role (e.g., "reader", "writer")

        Status Codes:
          - 200: Success
          - 404: Not Found
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "ruleId": {"type": "string", "minLength": 1},
            },
            "required": ["calendarId", "ruleId"],
        },
    },
    {
        "name": "list_acl_rules",
        "description": """List ACL rules for a calendar with pagination and filtering support.

        Returns access control rules for the given calendar with support for pagination and incremental synchronization.
        Follows the structure of Google Calendar API v3 `/calendars/{calendarId}/acl`.

        Required Parameters:
          - calendarId: Calendar identifier

        Optional Parameters:
          - maxResults: Maximum number of entries returned on one result page (1-250, default 100)
          - pageToken: Token specifying which result page to return
          - showDeleted: Whether to include deleted ACLs (role="none") in the result (default False)
          - syncToken: Token for incremental synchronization, returns only entries changed since token

        Response Structure:
          - kind: "calendar#acl"
          - etag: ETag of the ACL collection
          - items: Array of ACL rules
          - nextPageToken: Token for next page (if more results available)
          - nextSyncToken: Token for next sync operation

        Synchronization:
          - When syncToken is provided, showDeleted is automatically set to True
          - Deleted ACLs are always included in sync responses
          - If syncToken expires, server responds with 410 GONE

        Status Codes:
          - 200: Success
          - 400: Bad Request (invalid parameters)
          - 410: Gone (sync token expired)
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "maxResults": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 250,
                    "default": 100,
                    "description": "Maximum number of entries returned on one result page",
                },
                "pageToken": {
                    "type": "string",
                    "description": "Token specifying which result page to return",
                },
                "showDeleted": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include deleted ACLs in the result",
                },
                "syncToken": {
                    "type": "string",
                    "description": "Token for incremental synchronization",
                },
            },
            "required": ["calendarId"],
        },
    },
    {
        "name": "insert_acl_rule",
        "description": """Insert a new access control rule.

        Adds a new ACL rule to the specified calendar.
        Equivalent to: POST /calendars/{calendarId}/acl

        Required Parameters:
          - calendarId: Calendar identifier
          - rule: ACL rule input (scope, role)

        Optional Parameters:
          - sendNotifications: Whether to send notifications about the calendar sharing change (default: True)

        Response Structure:
          - kind: "calendar#aclRule"
          - etag: ETag
          - id: ACL rule ID
          - scope: { type, value }
          - role: ACL role

        Status Codes:
          - 201: Created
          - 400: Bad Request
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "scope": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "minLength": 1},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["type"],
                },
                "role": {"type": "string", "minLength": 1},
                "sendNotifications": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to send notifications about the calendar sharing change",
                },
            },
            "required": ["calendarId", "scope", "role"],
        },
    },
    {
        "name": "update_acl_rule",
        "description": """Fully update an existing ACL rule.

        Replaces an ACL rule with a new one.
        Equivalent to: PUT /calendars/{calendarId}/acl/{ruleId}

        Required Parameters:
          - calendarId: Calendar identifier
          - ruleId: ACL rule ID
          - rule: Complete rule replacement (scope, role)

        Optional Parameters:
          - sendNotifications: Whether to send notifications about the calendar sharing change (default: True)

        Response:
          - Updated ACL rule object

        Status Codes:
          - 200: Success
          - 404: Not Found
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "ruleId": {"type": "string", "minLength": 1},
                "scope": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "minLength": 1},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["type"],
                },
                "role": {"type": "string", "minLength": 1},
                "sendNotifications": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to send notifications about the calendar sharing change",
                },
            },
            "required": ["calendarId", "ruleId", "scope"],
        },
    },
    {
        "name": "patch_acl_rule",
        "description": """Partially update an ACL rule.

        Allows modifying select fields of an ACL rule.
        Equivalent to: PATCH /calendars/{calendarId}/acl/{ruleId}

        Required Parameters:
          - calendarId: Calendar identifier
          - ruleId: ACL rule ID
          - rule: Partial updates (any of: scope, role)

        Optional Parameters:
          - sendNotifications: Whether to send notifications about the calendar sharing change (default: True)

        Response:
          - Updated ACL rule object

        Status Codes:
          - 200: Success
          - 404: Not Found
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "ruleId": {"type": "string", "minLength": 1},
                "scope": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
                "role": {"type": "string"},
                "sendNotifications": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to send notifications about the calendar sharing change",
                },
            },
            "required": ["calendarId", "ruleId"],
        },
    },
    {
        "name": "delete_acl_rule",
        "description": """Delete an ACL rule.

        Deletes a specific ACL rule from the calendar.
        Equivalent to: DELETE /calendars/{calendarId}/acl/{ruleId}

        Required Parameters:
          - calendarId: Calendar identifier
          - ruleId: ACL rule ID

        Response:
          - No content (204) on success

        Status Codes:
          - 204: No Content
          - 404: Not Found
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {"type": "string", "minLength": 1},
                "ruleId": {"type": "string", "minLength": 1},
            },
            "required": ["calendarId", "ruleId"],
        },
    },
    {
        "name": "watch_acl",
        "description": """Set up a webhook to receive notifications when ACL rules change.

        Sets up webhook notifications for ACL rule changes following Google Calendar API v3 structure.
        Returns a channel for managing the watch. Monitors ACL rules in the specified calendar for changes.

        Equivalent to: POST /calendars/{calendarId}/acl/watch

        Required Parameters:
          - calendarId: Calendar identifier to watch for ACL changes
          - id: Unique channel identifier
          - address: Webhook URL to receive notifications
          - type: Channel type (default: "web_hook")
          

        Optional Parameters:
          - token: Optional token for webhook authentication
          - params: Additional channel parameters

        Response Structure:
          - Returns Channel resource with Google Calendar API v3 format:
            * kind: "api#channel"
            * id: Channel identifier
            * resourceId: Resource being watched
            * resourceUri: Resource URI path
            * token: Authentication token (if provided)
            * expiration: Channel expiration time (if set)

        Channel Management:
          - Each watch creates a unique notification channel
          - Channels can expire (set expiration time)
          - Multiple channels can watch the same calendar ACL
          - Use unique channel IDs to avoid conflicts

        Status Codes:
          - 200: Success - Watch channel created successfully
          - 400: Bad Request - Invalid channel configuration
          - 404: Not Found - Calendar not found
          - 500: Internal Server Error
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Calendar identifier to watch for ACL changes",
                },
                "id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Unique channel identifier",
                },
                "type": {
                    "type": "string",
                    "default": "web_hook",
                    "description": "Channel type (only 'web_hook' supported)",
                },
                "address": {
                    "type": "string",
                    "minLength": 1,
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
            },
            "required": ["calendarId", "id", "type", "address"],
        },
    },
]
