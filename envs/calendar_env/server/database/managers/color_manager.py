"""
Color database manager with Google Calendar API v3 compatible operations
Handles color definitions for calendars and events
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.models.color import Color, ColorType
from database.session_utils import get_session, init_database
from sqlalchemy import and_, func

logger = logging.getLogger(__name__)


class ColorManager:
    """Color manager for database operations"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        # Initialize database on first use
        init_database(database_id)

    def get_colors_response(self) -> Dict[str, Any]:
        """
        Get the complete colors response in Google Calendar API v3 format

        Returns:
            Dict containing calendar and event colors with metadata
        """
        session = get_session(self.database_id)
        try:
            # Get all colors from database
            colors = session.query(Color).all()

            # Get last updated timestamp
            last_updated = session.query(func.max(Color.updated_at)).scalar()
            if not last_updated:
                last_updated = datetime.now(timezone.utc)

            # Organize colors by type
            calendar_colors = {}
            event_colors = {}

            for color in colors:
                color_dict = color.to_dict()
                if color.color_type == ColorType.CALENDAR:
                    calendar_colors[color.color_id] = color_dict
                elif color.color_type == ColorType.EVENT:
                    event_colors[color.color_id] = color_dict

            return {
                "kind": "calendar#colors",
                "updated": last_updated.isoformat().replace("+00:00", ".000Z"),
                "calendar": calendar_colors,
                "event": event_colors,
            }

        except Exception as e:
            logger.error(f"Error getting colors response: {e}")
            raise
        finally:
            session.close()

    def get_color_by_id(
        self, color_type: str, color_id: str
    ) -> Optional[Dict[str, str]]:
        """
        Get a specific color by type and ID

        Args:
            color_type: Either 'calendar' or 'event'
            color_id: The color ID (e.g., '1', '2', etc.)

        Returns:
            Dict containing background and foreground colors, or None if not found
        """
        session = get_session(self.database_id)
        try:
            # Convert string type to enum
            if color_type == "calendar":
                type_enum = ColorType.CALENDAR
            elif color_type == "event":
                type_enum = ColorType.EVENT
            else:
                return None

            color = (
                session.query(Color)
                .filter(and_(Color.color_type == type_enum, Color.color_id == color_id))
                .first()
            )

            if color:
                return color.to_dict()
            return None

        except Exception as e:
            logger.error(f"Error getting color {color_type}:{color_id}: {e}")
            return None
        finally:
            session.close()

    def validate_color_id(self, color_type: str, color_id: str) -> bool:
        """
        Validate if a color ID exists for the given type

        Args:
            color_type: Either 'calendar' or 'event'
            color_id: The color ID to validate

        Returns:
            True if color ID exists, False otherwise
        """
        return self.get_color_by_id(color_type, color_id) is not None

    def get_all_colors_by_type(self, color_type: str) -> Dict[str, Dict[str, str]]:
        """
        Get all colors of a specific type

        Args:
            color_type: Either 'calendar' or 'event'

        Returns:
            Dict mapping color IDs to their color definitions
        """
        session = get_session(self.database_id)
        try:
            # Convert string type to enum
            if color_type == "calendar":
                type_enum = ColorType.CALENDAR
            elif color_type == "event":
                type_enum = ColorType.EVENT
            else:
                return {}

            colors = session.query(Color).filter(Color.color_type == type_enum).all()

            result = {}
            for color in colors:
                result[color.color_id] = color.to_dict()

            return result

        except Exception as e:
            logger.error(f"Error getting all colors for type {color_type}: {e}")
            return {}
        finally:
            session.close()

    def load_sample_colors(self, color_data: List[Dict[str, Any]]) -> int:
        """
        Load sample color data into the database

        Args:
            color_data: List of color dictionaries with color_id, color_type, background, foreground

        Returns:
            Number of colors loaded
        """
        session = get_session(self.database_id)
        try:
            loaded_count = 0

            for color_info in color_data:
                # Check if color already exists
                existing = (
                    session.query(Color)
                    .filter(
                        and_(
                            Color.color_id == color_info["color_id"],
                            Color.color_type == ColorType(color_info["color_type"]),
                        )
                    )
                    .first()
                )

                if not existing:
                    # Create new color
                    color = Color(
                        color_id=color_info["color_id"],
                        color_type=ColorType(color_info["color_type"]),
                        background=color_info["background"],
                        foreground=color_info["foreground"],
                    )
                    session.add(color)
                    loaded_count += 1
                else:
                    # Update existing color
                    existing.background = color_info["background"]
                    existing.foreground = color_info["foreground"]
                    existing.updated_at = datetime.now(timezone.utc)
                    loaded_count += 1

            session.commit()
            logger.info(f"Loaded {loaded_count} colors into database")
            return loaded_count

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading sample colors: {e}")
            raise
        finally:
            session.close()

    def clear_all_colors(self) -> int:
        """
        Clear all colors from the database

        Returns:
            Number of colors deleted
        """
        session = get_session(self.database_id)
        try:
            count = session.query(Color).count()
            session.query(Color).delete()
            session.commit()

            logger.info(f"Cleared {count} colors from database")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing colors: {e}")
            raise
        finally:
            session.close()
