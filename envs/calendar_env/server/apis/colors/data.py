"""
Google Calendar API v3 Colors Data
Static color definitions for calendars and events
"""

from datetime import datetime, timezone
from typing import Any, Dict

# Google Calendar color definitions based on official API
CALENDAR_COLORS = {
    "1": {"background": "#ac725e", "foreground": "#1d1d1d"},
    "2": {"background": "#d06b64", "foreground": "#1d1d1d"},
    "3": {"background": "#f83a22", "foreground": "#1d1d1d"},
    "4": {"background": "#fa57c4", "foreground": "#1d1d1d"},
    "5": {"background": "#9fc6e7", "foreground": "#1d1d1d"},
    "6": {"background": "#9a9cff", "foreground": "#1d1d1d"},
    "7": {"background": "#4986e7", "foreground": "#1d1d1d"},
    "8": {"background": "#9aa116", "foreground": "#1d1d1d"},
    "9": {"background": "#ef6c00", "foreground": "#1d1d1d"},
    "10": {"background": "#ff7537", "foreground": "#1d1d1d"},
    "11": {"background": "#42d692", "foreground": "#1d1d1d"},
    "12": {"background": "#16a765", "foreground": "#1d1d1d"},
    "13": {"background": "#7bd148", "foreground": "#1d1d1d"},
    "14": {"background": "#b3dc6c", "foreground": "#1d1d1d"},
    "15": {"background": "#fbe983", "foreground": "#1d1d1d"},
    "16": {"background": "#fad165", "foreground": "#1d1d1d"},
    "17": {"background": "#92e1c0", "foreground": "#1d1d1d"},
    "18": {"background": "#9fe1e7", "foreground": "#1d1d1d"},
    "19": {"background": "#9fc6e7", "foreground": "#1d1d1d"},
    "20": {"background": "#4986e7", "foreground": "#1d1d1d"},
    "21": {"background": "#9aa116", "foreground": "#1d1d1d"},
    "22": {"background": "#16a765", "foreground": "#1d1d1d"},
    "23": {"background": "#ff7537", "foreground": "#1d1d1d"},
    "24": {"background": "#ffad46", "foreground": "#1d1d1d"},
}

EVENT_COLORS = {
    "1": {"background": "#a4bdfc", "foreground": "#1d1d1d"},
    "2": {"background": "#7ae7bf", "foreground": "#1d1d1d"},
    "3": {"background": "#dbadff", "foreground": "#1d1d1d"},
    "4": {"background": "#ff887c", "foreground": "#1d1d1d"},
    "5": {"background": "#fbd75b", "foreground": "#1d1d1d"},
    "6": {"background": "#ffb878", "foreground": "#1d1d1d"},
    "7": {"background": "#46d6db", "foreground": "#1d1d1d"},
    "8": {"background": "#e1e1e1", "foreground": "#1d1d1d"},
    "9": {"background": "#5484ed", "foreground": "#1d1d1d"},
    "10": {"background": "#51b749", "foreground": "#1d1d1d"},
    "11": {"background": "#dc2127", "foreground": "#1d1d1d"},
}


def get_colors_response() -> Dict[str, Any]:
    """
    Get the complete colors response in Google Calendar API v3 format

    Returns:
        Dict containing calendar and event colors with metadata
    """
    return {
        "kind": "calendar#colors",
        "updated": "2023-01-01T00:00:00.000Z",
        "calendar": CALENDAR_COLORS,
        "event": EVENT_COLORS,
    }


def get_color_by_id(color_type: str, color_id: str) -> Dict[str, str]:
    """
    Get a specific color by type and ID

    Args:
        color_type: Either 'calendar' or 'event'
        color_id: The color ID (e.g., '1', '2', etc.)

    Returns:
        Dict containing background and foreground colors

    Raises:
        ValueError: If color_type or color_id is invalid
    """
    if color_type == "calendar":
        colors = CALENDAR_COLORS
    elif color_type == "event":
        colors = EVENT_COLORS
    else:
        raise ValueError(
            f"Invalid color type: {color_type}. Must be 'calendar' or 'event'"
        )

    if color_id not in colors:
        raise ValueError(f"Invalid color ID: {color_id} for type {color_type}")

    return colors[color_id]


def get_all_calendar_colors() -> Dict[str, Dict[str, str]]:
    """Get all calendar colors"""
    return CALENDAR_COLORS.copy()


def get_all_event_colors() -> Dict[str, Dict[str, str]]:
    """Get all event colors"""
    return EVENT_COLORS.copy()


def validate_color_id(color_type: str, color_id: str) -> bool:
    """
    Validate if a color ID exists for the given type

    Args:
        color_type: Either 'calendar' or 'event'
        color_id: The color ID to validate

    Returns:
        True if color ID exists, False otherwise
    """
    try:
        get_color_by_id(color_type, color_id)
        return True
    except ValueError:
        return False
