"""
Colors Tools Module

This module contains tools related to Google Calendar color definitions.
Provides static color palettes for calendars and events.
"""

COLORS_TOOLS = [
    {
        "name": "get_colors",
        "description": """Retrieve the color definitions for calendars and events.
        
        Returns the global palette of color definitions used by Google Calendar.
        This endpoint provides static color data and does not require authentication 
        or user context as it returns predefined color schemes.
        
        Color Structure:
          - calendar: Object mapping color IDs to calendar color definitions
          - event: Object mapping color IDs to event color definitions
          
        Each color definition contains:
          - background: The background color (hex format)
          - foreground: The foreground color for text (hex format)
          
        Available Calendar Colors: 24 predefined colors (IDs 1-24)
        Available Event Colors: 11 predefined colors (IDs 1-11)
        
        Usage Examples:
          - Get all available colors for UI color pickers
          - Validate color IDs before setting calendar/event colors
          - Display color options to users
          
        Response Structure:
          - Returns colors with Google Calendar API v3 format:
            * kind: "calendar#colors"
            * updated: Last modification timestamp
            * calendar: Object with calendar color definitions
            * event: Object with event color definitions
            
        Color Examples:
          - Calendar Color 1: Background "#ac725e" (brown), Foreground "#1d1d1d" (dark)
          - Event Color 1: Background "#a4bdfc" (light blue), Foreground "#1d1d1d" (dark)
          - Event Color 11: Background "#dc2127" (red), Foreground "#1d1d1d" (dark)
        
        API Endpoint: GET /colors
        
        Status Codes:
          - 200: Success - Color definitions retrieved
          - 500: Internal Server Error""",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
]
