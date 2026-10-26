"""
Setting Manager - Database operations for settings management using SQLAlchemy
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from database.models.settings import Settings
from database.models.watch_channel import WatchChannel
from database.session_utils import get_session, init_database
from schemas.settings import Channel, SettingItem, SettingsWatchRequest

logger = logging.getLogger(__name__)


class SettingManager:
    """Manager for settings database operations using SQLAlchemy"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        init_database(database_id)

    def list_settings(self, user_id: str) -> list[SettingItem]:
        """List all settings"""
        session = get_session(self.database_id)
        try:
            settings = session.query(Settings).filter(Settings.user_id == user_id).all()
            return [SettingItem.model_validate(s) for s in settings]
        except Exception as e:
            logger.error(f"Error listing settings for user '{user_id}': {e}")
            raise
        finally:
            session.close()

    def get_setting_by_id(self, setting_id: str, user_id: str) -> Optional[Dict]:
        """Get a setting by its ID"""
        session = get_session(self.database_id)
        try:
            setting = (
                session.query(Settings)
                .filter(Settings.id == setting_id, Settings.user_id == user_id)
                .first()
            )
            if setting:
                return self._format_setting(setting)
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving setting '{setting_id}' for user '{user_id}': {e}"
            )
            raise
        finally:
            session.close()

    def _format_setting(self, setting: Settings) -> Dict:
        """Format a setting for API response"""
        return {
            "kind": "calendar#setting",
            "etag": setting.etag,
            "id": setting.id,
            "value": setting.value,
            "user_id": setting.user_id,
        }

    def watch_settings(
        self, watch_request: SettingsWatchRequest, user_id: str
    ) -> Channel:
        """
        Set up a watch channel for settings changes

        Args:
            watch_request: The watch request parameters
            user_id: The user setting up the watch

        Returns:
            Channel: The created watch channel
        """
        session = get_session(self.database_id)
        try:
            # Generate unique resource ID for settings watch
            resource_id = f"settings-{user_id}-{uuid.uuid4().hex[:8]}"
            resource_uri = f"/calendars/{user_id}/settings"

            # Calculate expiration time (max 24 hours from now if not specified)
            now = datetime.utcnow()
            expires_at = now + timedelta(hours=24)

            if (
                session.query(WatchChannel)
                .filter(WatchChannel.id == watch_request.id)
                .first()
            ):
                raise ValueError(f"Channel with Id {watch_request.id} already exists")

            # Create watch channel record
            watch_channel = WatchChannel(
                id=watch_request.id,
                resource_id=resource_id,
                resource_uri=resource_uri,
                resource_type="settings",
                calendar_id="",
                user_id=user_id,
                webhook_address=watch_request.address,
                webhook_token=watch_request.token,
                webhook_type=watch_request.type,
                params=json.dumps(watch_request.params.model_dump())
                if watch_request.params
                else None,
                created_at=now,
                expires_at=expires_at,
                is_active="true",
                notification_count=0,
            )

            # Save to database
            session.add(watch_channel)
            session.commit()

            logger.info(
                f"Created settings watch channel {watch_request.id} for user {user_id}"
            )

            # Return channel response
            return Channel(
                kind="api#channel",
                id=watch_channel.id,
                resourceId=resource_id,
                resourceUri=resource_uri,
                token=watch_channel.webhook_token,
                expiration=expires_at.isoformat() + "Z" if expires_at else None,
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Error creating settings watch channel: {e}")
            raise
        finally:
            session.close()
