"""
User database model
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base


class User(Base):
    """User database model"""

    __tablename__ = "users"

    user_id = Column(String(255), primary_key=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    given_name = Column(String(255), nullable=True)
    family_name = Column(String(255), nullable=True)
    picture = Column(String(500), nullable=True)  # Profile picture URL
    locale = Column(String(10), nullable=True, default="en")
    timezone = Column(String(100), nullable=False, default="UTC")

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # OAuth/Authentication data
    provider = Column(String(50), nullable=True)  # google, microsoft, etc.
    provider_id = Column(String(255), nullable=True)  # OAuth provider user ID
    # Static API token to authenticate MCP calls (mapped 1:1 to a user)
    static_token = Column(String(255), unique=True, nullable=False, index=True)
    access_token_hash = Column(Text, nullable=True)  # Hashed for security
    refresh_token_hash = Column(Text, nullable=True)  # Hashed for security

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    calendars = relationship(
        "Calendar", back_populates="user", cascade="all, delete-orphan"
    )
    attendees = relationship(
        "Attendees", back_populates="user", cascade="all, delete-orphan"
    )
    acls = relationship("ACLs", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}', name='{self.name}')>"

    def to_dict(self):
        """Convert to dictionary format for API responses"""
        return {
            "id": self.user_id,
            "email": self.email,
            "name": self.name,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "picture": self.picture,
            "locale": self.locale,
            "timezone": self.timezone,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat()
            if self.last_login_at
            else None,
        }
