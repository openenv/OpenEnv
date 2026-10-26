"""
Settings Tools Module

This module contains tools related to user settings management.
Follows Google Calendar API v3 structure for settings operations.
Covers listing and retrieving settings only.
"""

SETTINGS_TOOLS = [
    {
        "name": "get_settings",
        "description": """Retrieve a specific user setting by ID.

        Returns the value of a setting resource by its ID.
        Follows Google Calendar API v3 `/settings/{settingId}` structure.

        Request Body Requirements:
          - settingId: Required. Unique identifier of the setting (e.g., "timezone")

        Response Structure:
          - kind: "calendar#setting"
          - etag: ETag of the setting
          - id: Unique setting identifier
          - value: Current value of the setting
          - user_id: ID of the user who owns the setting

        Status Codes:
          - 200: Success - Setting returned successfully
          - 404: Not Found - No setting exists with the given ID
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "settingId": {
                    "type": "string",
                    "description": "Unique identifier of the setting (e.g., 'timezone')",
                    "minLength": 1,
                }
            },
            "required": ["settingId"],
        },
    },
    {
        "name": "list_settings",
        "description": """List all visible user settings.

        Retrieves all settings that are visible to the authenticated user.
        Follows the structure of Google Calendar API v3 `/settings` endpoint.

        No request body required.

        Response Structure:
          - kind: "calendar#settings"
          - etag: ETag for the entire settings collection
          - items: Array of setting resources, each containing:
            * kind: "calendar#setting"
            * etag: ETag of the setting
            * id: Unique identifier of the setting (e.g., "timezone")
            * value: Current value of the setting

        Status Codes:
          - 200: Success - List of settings returned
          - 500: Internal Server Error""",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "watch_settings",
        "description": """Watch for changes to user settings.

        Sets up a notification channel to receive updates when settings change.
        Follows Google Calendar API v3 `/settings/watch` structure.

        Creates a watch channel that will send webhook notifications to the specified
        address whenever settings are modified. The channel will automatically expire
        after a maximum of 24 hours or at the specified expiration time.

        Request Body Requirements:
          - id: Required. Unique identifier for the channel
          - type: Optional. Channel type (defaults to "web_hook")
          - address: Required. URL where notifications will be sent
          - token: Optional. Verification token for webhook security
          - params: Optional. Additional parameters as key-value pairs

        Optional Parameters:
          - token: Optional token for webhook authentication
          - params: Additional channel parameters

        Response Structure:
          - kind: "api#channel"
          - id: Channel identifier
          - resourceId: Resource being watched
          - resourceUri: URI of the resource ("/settings")
          - token: Verification token (if provided)
          - expiration: Channel expiration time

        Webhook Notification Format:
        Your webhook will receive POST requests with this payload:
        {
          "kind": "api#channel",
          "id": "channel-id",
          "resourceId": "settings-user-id",
          "resourceUri": "/settings",
          "eventType": "update|insert|delete",
          "resourceState": "sync",
          "timestamp": "2024-10-01T18:30:00Z",
          "data": {
            "kind": "calendar#setting",
            "id": "setting-id",
            "value": "new-value",
            "oldValue": "previous-value",
            "user_id": "user-id"
          }
        }

        Status Codes:
          - 200: Success - Watch channel created
          - 400: Bad Request - Invalid request parameters
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the watch channel",
                    "minLength": 1,
                },
                "type": {
                    "type": "string",
                    "description": "Type of notification channel",
                    "default": "web_hook",
                    "enum": ["web_hook"],
                },
                "address": {
                    "type": "string",
                    "description": "URL where webhook notifications will be sent",
                    "format": "uri",
                    "minLength": 1,
                },
                "token": {
                    "type": "string",
                    "description": "Optional verification token for webhook security",
                },
                "params": {
                    "type": "object",
                    "description": "Additional parameters as key-value pairs",
                    "properties": {
                        "ttl": {
                            "type": "string",
                            "description": "Time to live in seconds (string).",
                        }
                    },
                },
            },
            "required": ["id", "type", "address"],
        },
    },
]
