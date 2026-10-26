"""
MCP Tools for User Management
"""

USERS_TOOLS = [
    {
        "name": "get_user_by_email",
        "description": """Get user details by email address

This tool retrieves complete user information using their email address, including user ID, name, and other profile details.

**API Endpoint:** GET /users/email/{email}

**Parameters:**
- email (required): The email address to lookup

**Returns:**
Complete user information including id, name, given_name, family_name, picture, locale, timezone, is_active, is_verified, etc.

**Example:**
```json
{
  "email": "user@example.com"
}
```""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email address to lookup"}
            },
            "required": ["email"],
        },
    }
]
