"""
Calendar database model
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .base import Base


class Calendar(Base):
    """Calendar database model"""

    __tablename__ = "calendars"

    calendar_id = Column(String(255), primary_key=True, nullable=False)
    user_id = Column(
        String(255), ForeignKey("users.user_id"), nullable=False, index=True
    )
    summary = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(500), nullable=True)
    time_zone = Column(String(100), nullable=False, default="UTC")
    conference_properties = Column(Text, nullable=True)  # JSON string
    is_primary = Column(
        Boolean, default=False, nullable=False
    )  # Primary vs secondary calendar

    # Calendar List specific fields
    summary_override = Column(
        String(255), nullable=True
    )  # Custom summary override for list display
    color_id = Column(String(50), nullable=True)  # Calendar color ID
    background_color = Column(String(7), nullable=True)  # Hex color for background
    foreground_color = Column(String(7), nullable=True)  # Hex color for foreground
    hidden = Column(Boolean, default=False, nullable=False)  # Hidden from calendar list
    selected = Column(Boolean, default=True, nullable=False)  # Selected in UI
    default_reminders = Column(Text, nullable=True)  # JSON string of default reminders
    notification_settings = Column(
        Text, nullable=True
    )  # JSON string of notification settings
    deleted = Column(Boolean, default=False, nullable=False)  # Soft delete flag

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user = relationship("User", back_populates="calendars")
    events = relationship(
        "Event", back_populates="calendar", cascade="all, delete-orphan"
    )
    acls = relationship("ACLs", back_populates="calendar", cascade="all, delete-orphan")

    # Table constraints
    __table_args__ = (
        # Ensure each user has exactly one primary calendar (only when is_primary=True)
        Index(
            "idx_unique_primary_per_user",
            "user_id",
            unique=True,
            sqlite_where=Column("is_primary") == True,
        ),
    )

    def __repr__(self):
        return f"<Calendar(calendar_id='{self.calendar_id}', user_id='{self.user_id}', summary='{self.summary}', is_primary={self.is_primary})>"
