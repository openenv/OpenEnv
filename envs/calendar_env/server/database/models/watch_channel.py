import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from .base import Base


class WatchChannel(Base):
    """Database model for storing watch channel subscriptions"""

    __tablename__ = "watch_channels"

    id = Column(String, primary_key=True, index=True)  # Channel ID
    resource_id = Column(
        String, nullable=False, index=True
    )  # e.g., "acl-{calendar_id}"
    resource_uri = Column(
        String, nullable=False
    )  # e.g., "/calendars/{calendar_id}/acl"
    resource_type = Column(
        String, nullable=False, default="acl"
    )  # Type of resource being watched

    calendar_id = Column(String, nullable=False, index=True)  # Calendar being watched
    user_id = Column(String, nullable=False, index=True)  # User who created the watch

    # Webhook details
    webhook_address = Column(String, nullable=False)  # URL to send notifications to
    webhook_token = Column(String, nullable=True)  # Optional verification token
    webhook_type = Column(String, nullable=False, default="web_hook")

    # Channel metadata
    params = Column(Text, nullable=True)  # JSON string of additional parameters

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # When this channel expires
    last_notification_at = Column(
        DateTime, nullable=True
    )  # Last time a notification was sent

    # Status tracking
    is_active = Column(
        String, default="true", nullable=False
    )  # "true" or "false" as string
    notification_count = Column(
        Integer, default=0, nullable=False
    )  # Number of notifications sent

    def is_expired(self) -> bool:
        """Check if the channel has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_channel_dict(self):
        """Convert to Channel schema format"""
        return {
            "kind": "api#channel",
            "id": self.id,
            "resourceId": self.resource_id,
            "resourceUri": self.resource_uri,
            "token": self.webhook_token,
            "expiration": self.expires_at.isoformat() if self.expires_at else None,
            "type": self.webhook_type,
            "address": self.webhook_address,
            "params": json.loads(self.params) if self.params else None,
        }
