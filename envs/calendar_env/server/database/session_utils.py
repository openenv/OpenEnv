"""
Database session utilities - Simplified session management
"""

from database.session_manager import CalendarSessionManager
from sqlalchemy.orm import Session

# Global session manager instance
_session_manager = CalendarSessionManager()


def get_session(database_id: str) -> Session:
    """Get database session for the specified database ID"""
    return _session_manager.get_session(database_id)


def init_database(database_id: str):
    """Initialize database for the specified database ID"""
    return _session_manager.init_database(database_id)
