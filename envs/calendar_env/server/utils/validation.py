"""
Validation utilities for Calendar API
Provides common validation functions used across different API endpoints
"""

from typing import Optional, Tuple

from apis.colors.data import CALENDAR_COLORS, EVENT_COLORS
from database.managers.color_manager import ColorManager


def validate_calendar_color_id(
    color_id: Optional[str], database_id: str
) -> Optional[str]:
    """
    Validate calendar colorId against database

    Args:
        color_id: The color ID to validate (can be None)
        database_id: Database ID for color manager

    Returns:
        None if validation passes, error message if invalid
    """
    if color_id is None:
        return None  # Optional field, None is valid

    if not isinstance(color_id, str):
        return "colorId must be a string"

    try:
        color_manager = ColorManager(database_id)
        if not color_manager.validate_color_id("calendar", color_id):
            return f"Invalid calendar colorId: '{color_id}'. Check available colorId with GET /colors"
    except Exception:
        # Fallback validation if database fails
        return (
            f"Could not validate colorId: '{color_id}'. Database may not be initialized"
        )

    return None


def validate_event_color_id(color_id: Optional[str], database_id: str) -> Optional[str]:
    """
    Validate event colorId against database

    Args:
        color_id: The color ID to validate (can be None)
        database_id: Database ID for color manager

    Returns:
        None if validation passes, error message if invalid
    """
    if color_id is None:
        return None  # Optional field, None is valid

    if not isinstance(color_id, str):
        return "colorId must be a string"

    try:
        color_manager = ColorManager(database_id)
        if not color_manager.validate_color_id("event", color_id):
            return f"Invalid event colorId: '{color_id}'. Check available colors with GET /colors"
    except Exception:
        # Fallback validation if database fails
        return (
            f"Could not validate colorId: '{color_id}'. Database may not be initialized"
        )

    return None


def validate_request_colors(
    data: dict, color_type: str, database_id: str
) -> Optional[str]:
    """
    Validate colorId in request data against database

    Args:
        data: Request data dictionary
        color_type: Either 'calendar' or 'event'
        database_id: Database ID for color manager

    Returns:
        None if validation passes, error message if invalid
    """
    if "colorId" not in data:
        return None  # No colorId in request, that's fine

    color_id = data.get("colorId")

    if color_type == "calendar":
        return validate_calendar_color_id(color_id, database_id)
    elif color_type == "event":
        return validate_event_color_id(color_id, database_id)
    else:
        return f"Unknown color type: {color_type}"


def validate_color_combination(
    background_color: str, foreground_color: str, color_type: str
) -> Optional[str]:
    """
    Validate that backgroundColor and foregroundColor combination exists in color data

    Args:
        background_color: Background color in hex format (e.g., "#ac725e")
        foreground_color: Foreground color in hex format (e.g., "#1d1d1d")
        color_type: Either 'calendar' or 'event'

    Returns:
        None if combination is valid, error message if invalid
    """
    if color_type == "calendar":
        colors = CALENDAR_COLORS
    elif color_type == "event":
        colors = EVENT_COLORS
    else:
        return f"Unknown color type: {color_type}"

    # Check if the combination exists in the color data
    for color_id, color_data in colors.items():
        if (
            color_data["background"].lower() == background_color.lower()
            and color_data["foreground"].lower() == foreground_color.lower()
        ):
            return None  # Valid combination found

    return f"Invalid color combination: backgroundColor='{background_color}' and foregroundColor='{foreground_color}' is not a valid {color_type} color combination"


def find_color_id_by_combination(
    background_color: str, foreground_color: str, color_type: str
) -> Optional[str]:
    """
    Find the colorId that matches the given backgroundColor and foregroundColor combination

    Args:
        background_color: Background color in hex format (e.g., "#ac725e")
        foreground_color: Foreground color in hex format (e.g., "#1d1d1d")
        color_type: Either 'calendar' or 'event'

    Returns:
        colorId if combination is found, None if not found
    """
    if color_type == "calendar":
        colors = CALENDAR_COLORS
    elif color_type == "event":
        colors = EVENT_COLORS
    else:
        return None

    # Find the colorId that matches the combination
    for color_id, color_data in colors.items():
        if (
            color_data["background"].lower() == background_color.lower()
            and color_data["foreground"].lower() == foreground_color.lower()
        ):
            return color_id

    return None


def set_colors_from_color_id(data: dict, color_type: str) -> Optional[str]:
    """
    Set backgroundColor and foregroundColor from colorId if they are not provided

    Args:
        data: Request data dictionary (will be modified to add colors if colorId is valid)
        color_type: Either 'calendar' or 'event'

    Returns:
        None if successful, error message if colorId is invalid
    """
    color_id = data.get("colorId")

    # Only process if colorId is provided and RGB colors are not provided
    if not color_id:
        return None

    # Skip if RGB colors are already provided
    if data.get("backgroundColor") or data.get("foregroundColor"):
        return None

    if color_type == "calendar":
        colors = CALENDAR_COLORS
    elif color_type == "event":
        colors = EVENT_COLORS
    else:
        return f"Unknown color type: {color_type}"

    # Validate colorId and get colors
    if color_id not in colors:
        return f"Invalid {color_type} colorId: '{color_id}'. Check available colorId with GET /colors"

    # Set the colors from the colorId
    color_data = colors[color_id]
    data["backgroundColor"] = color_data["background"]
    data["foregroundColor"] = color_data["foreground"]

    return None


def validate_and_set_color_id(data: dict, color_type: str) -> Optional[str]:
    """
    Validate RGB color combination and set appropriate colorId if valid combination exists

    Args:
        data: Request data dictionary (will be modified to add colorId if found)
        color_type: Either 'calendar' or 'event'

    Returns:
        None if validation passes, error message if invalid combination
    """
    background_color = data.get("backgroundColor")
    foreground_color = data.get("foregroundColor")

    # Only validate if both colors are provided
    if not (background_color and foreground_color):
        return None

    # Validate the combination exists
    validation_error = validate_color_combination(
        background_color, foreground_color, color_type
    )
    if validation_error:
        return validation_error

    # Find and set the matching colorId
    color_id = find_color_id_by_combination(
        background_color, foreground_color, color_type
    )
    if color_id:
        data["colorId"] = color_id

    return None
