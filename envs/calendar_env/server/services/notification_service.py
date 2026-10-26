"""
Webhook notification service for calendar watch channels
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from database.models.watch_channel import WatchChannel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending webhook notifications to watch channels"""

    def __init__(self, db: Session):
        self.db = db

    def send_notification(
        self,
        channel_id: str,
        event_type: str,
        resource_data: Dict[str, Any],
        resource_state: str = "sync",
    ) -> bool:
        """
        Send a notification to a specific watch channel

        Args:
            channel_id: The watch channel ID
            event_type: Type of event (e.g., "sync", "exists", "not_exists")
            resource_data: The actual resource data that changed
            resource_state: State of the resource

        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            # Get the watch channel from database
            channel = (
                self.db.query(WatchChannel)
                .filter(WatchChannel.id == channel_id, WatchChannel.is_active == "true")
                .first()
            )

            if not channel:
                logger.warning(f"Watch channel {channel_id} not found or inactive")
                return False

            # Check if channel has expired
            if channel.is_expired():
                logger.info(f"Watch channel {channel_id} has expired")
                self._deactivate_channel(channel)
                return False

            # Prepare notification payload
            notification_payload = {
                "kind": "api#channel",
                "id": channel.id,
                "resourceId": channel.resource_id,
                "resourceUri": channel.resource_uri,
                "eventType": event_type,
                "resourceState": resource_state,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": resource_data,
            }

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Calendar-Notifications/1.0",
            }

            # Add token to headers if present
            if channel.webhook_token:
                headers["X-Goog-Channel-Token"] = channel.webhook_token

            # Send the webhook notification
            response = requests.post(
                channel.webhook_address,
                json=notification_payload,
                headers=headers,
                timeout=30,
            )

            # Check if notification was successful
            if response.status_code in [200, 201, 202, 204]:
                # Update channel statistics
                self._update_channel_stats(channel)
                logger.info(
                    f"Notification sent successfully to {channel.webhook_address}"
                )
                return True
            else:
                logger.error(
                    f"Webhook failed with status {response.status_code}: {response.text}"
                )
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Webhook timeout for channel {channel_id}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Webhook request failed for channel {channel_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return False

    def notify_acl_change(
        self, calendar_id: str, change_type: str, acl_rule_data: Dict[str, Any]
    ) -> int:
        """
        Send ACL change notifications to all active watch channels for a calendar

        Args:
            calendar_id: The calendar ID
            change_type: Type of change ("insert", "update", "delete")
            acl_rule_data: The ACL rule data

        Returns:
            Number of successful notifications sent
        """
        try:
            # Find all active watch channels for this calendar's ACL
            channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.calendar_id == calendar_id,
                    WatchChannel.resource_type == "acl",
                    WatchChannel.is_active == "true",
                )
                .all()
            )
            if not channels:
                logger.debug(
                    f"No active watch channels found for calendar {calendar_id}"
                )
                return 0

            successful_notifications = 0

            for channel in channels:
                # Check if channel has expired
                if channel.is_expired():
                    self._deactivate_channel(channel)
                    continue

                # Send notification
                success = self.send_notification(
                    channel.id, change_type, acl_rule_data, "sync"
                )

                if success:
                    successful_notifications += 1

            logger.info(
                f"Sent {successful_notifications} ACL change notifications for calendar {calendar_id}"
            )
            return successful_notifications

        except Exception as e:
            logger.error(f"Error sending ACL change notifications: {e}")
            return 0

    def notify_settings_change(
        self, user_id: str, change_type: str, setting_data: Dict[str, Any]
    ) -> int:
        """
        Send settings change notifications to all active watch channels for a user

        Args:
            user_id: The user ID whose settings changed
            change_type: Type of change ("insert", "update", "delete")
            setting_data: The setting data

        Returns:
            Number of successful notifications sent
        """
        try:
            # Find all active watch channels for this user's settings
            channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.user_id == user_id,
                    WatchChannel.resource_type == "settings",
                    WatchChannel.is_active == "true",
                )
                .all()
            )

            if not channels:
                logger.debug(
                    f"No active settings watch channels found for user {user_id}"
                )
                return 0

            successful_notifications = 0

            for channel in channels:
                # Check if channel has expired
                if channel.is_expired():
                    self._deactivate_channel(channel)
                    continue

                # Send notification
                success = self.send_notification(
                    channel.id, change_type, setting_data, "sync"
                )

                if success:
                    successful_notifications += 1

            logger.info(
                f"Sent {successful_notifications} settings change notifications for user {user_id}"
            )
            return successful_notifications

        except Exception as e:
            logger.error(f"Error sending settings change notifications: {e}")
            return 0

    def cleanup_expired_channels(self) -> int:
        """
        Clean up expired watch channels

        Returns:
            Number of channels cleaned up
        """
        try:
            current_time = datetime.utcnow()

            # Find expired channels
            expired_channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.expires_at < current_time,
                    WatchChannel.is_active == "true",
                )
                .all()
            )

            cleanup_count = 0
            for channel in expired_channels:
                self._deactivate_channel(channel)
                cleanup_count += 1

            if cleanup_count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {cleanup_count} expired watch channels")

            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up expired channels: {e}")
            self.db.rollback()
            return 0

    def _update_channel_stats(self, channel: WatchChannel):
        """Update channel statistics after successful notification"""
        try:
            channel.last_notification_at = datetime.utcnow()
            channel.notification_count += 1
            self.db.commit()
        except Exception as e:
            logger.error(f"Error updating channel stats: {e}")
            self.db.rollback()

    def _deactivate_channel(self, channel: WatchChannel):
        """Deactivate a watch channel"""
        try:
            channel.is_active = "false"
            self.db.commit()
            logger.info(f"Deactivated watch channel {channel.id}")
        except Exception as e:
            logger.error(f"Error deactivating channel {channel.id}: {e}")
            self.db.rollback()


def get_notification_service(db: Session) -> NotificationService:
    """Factory function to get notification service instance"""
    return NotificationService(db)
