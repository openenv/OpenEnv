"""
Calendar List Manager - Database operations for calendar list management using SQLAlchemy
Manages user-specific calendar settings and access in a database-per-user architecture
"""

import base64
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from database.models import Calendar, Event, User
from database.models.acl import ACLs, Scope
from database.models.watch_channel import WatchChannel
from database.session_utils import get_session, init_database
from fastapi import HTTPException, status
from schemas.calendar_list import WatchRequest
from schemas.settings import Channel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Allowed values aligning with Google Calendar API v3
ALLOWED_REMINDER_METHODS = {"email", "popup"}
ALLOWED_NOTIFICATION_METHODS = {"email"}
ALLOWED_NOTIFICATION_TYPES = {
    "eventCreation",
    "eventChange",
    "eventCancellation",
    "eventResponse",
    "agenda",
}


class CalendarListManager:
    """Manager for calendar list database operations using SQLAlchemy"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        # Initialize database on first use
        init_database(database_id)

    def list_calendar_entries(
        self,
        user_id: str,
        max_results: Optional[int] = None,
        min_access_role: Optional[str] = None,
        show_deleted: Optional[bool] = False,
        show_hidden: Optional[bool] = False,
        page_token: Optional[str] = None,
        sync_token: Optional[str] = None,
    ) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """List all calendar entries in user's calendar list

        Notes:
        - If max_results is None, return all (subject to other filters)
        - If max_results is 0, return an empty list (explicit zero results)
        - If max_results > 0, limit the results accordingly
        - min_access_role filters calendars based on user's minimum access role
        - page_token specifies which result page to return
        - sync_token enables incremental synchronization

        Returns:
        - Tuple of (calendar_entries, next_page_token, next_sync_token)
        """
        session = get_session(self.database_id)
        try:
            # Handle sync token for incremental synchronization
            sync_timestamp = None
            if sync_token:
                try:
                    sync_timestamp = self._decode_sync_token(sync_token)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid sync token: {sync_token}, error: {e}")
                    raise ValueError(
                        f"Invalid sync token. Please perform full synchronization."
                    )

            # Parse page token to get offset
            offset = 0
            if page_token:
                try:
                    offset = self._decode_page_token(page_token)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid page token: {page_token}, error: {e}")
                    offset = 0

            query = session.query(Calendar).filter(Calendar.user_id == user_id)

            # Apply sync token filtering for incremental sync
            if sync_token and sync_timestamp:
                # For incremental sync, return entries modified since sync timestamp
                query = query.filter(Calendar.updated_at > sync_timestamp)
                # Include deleted and hidden entries when using sync token
                show_deleted = True
                show_hidden = True
            else:
                # Filter deleted calendars unless specifically requested
                if not show_deleted:
                    query = query.filter(Calendar.deleted.is_(False))

            # Order consistently for pagination (by calendar_id for deterministic results)
            query = query.order_by(Calendar.calendar_id)

            # Apply offset for pagination
            if offset > 0:
                query = query.offset(offset)

            # Apply limit if specified (support 0 to return empty set)
            # For pagination, we fetch max_results + 1 to determine if there are more pages
            fetch_limit = None
            if max_results is not None:
                # Negative values are treated as 0 (empty set)
                limit_value = max(0, int(max_results))
                if limit_value == 0:
                    return [], None, None
                fetch_limit = limit_value + 1  # Fetch one extra to check for next page
                query = query.limit(fetch_limit)

            calendars = query.all()

            # Define access role hierarchy for filtering
            role_hierarchy = {"freeBusyReader": 1, "reader": 2, "writer": 3, "owner": 4}

            min_role_level = (
                role_hierarchy.get(min_access_role, 0) if min_access_role else 0
            )

            result = []
            for calendar in calendars:
                # Check if user has sufficient access role if min_access_role is specified
                if min_access_role:
                    user_access_role = self._get_user_access_role(user_id, calendar)
                    user_role_level = role_hierarchy.get(user_access_role, 0)

                    # Skip calendars where user doesn't meet minimum access role
                    if user_role_level < min_role_level:
                        continue

                entry = self._format_calendar_list_entry(calendar, show_hidden, user_id)
                if entry:
                    result.append(entry)

            # Determine if there are more pages
            next_page_token = None
            if max_results is not None and len(result) > max_results:
                # Remove the extra item and generate next page token
                result = result[:max_results]
                next_offset = offset + max_results
                next_page_token = self._encode_page_token(next_offset)

            # Generate next sync token for incremental sync
            next_sync_token = None
            if len(result) > 0:
                # Generate sync token based on current timestamp
                next_sync_token = self._encode_sync_token(datetime.utcnow())

            return result, next_page_token, next_sync_token

        except Exception as e:
            logger.error(f"Error listing calendar entries: {e}")
            raise
        finally:
            session.close()

    def get_calendar_entry(self, user_id: str, calendar_id: str) -> Optional[Dict]:
        """Get a specific calendar entry from user's calendar list"""
        session = get_session(self.database_id)
        try:
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )

            if not calendar or calendar.deleted:
                return None

            return self._format_calendar_list_entry(
                calendar, show_hidden=True, user_id=user_id
            )

        except Exception as e:
            logger.error(f"Error getting calendar entry {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def insert_calendar_entry(
        self, user_id: str, calendar_id: str, entry_data: Dict
    ) -> Optional[Dict]:
        """Insert an existing calendar into user's calendar list"""
        session = get_session(self.database_id)
        try:
            # Check if calendar exists
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )

            if not calendar:
                # If calendar doesn't exist, we could create it or return None
                # For this implementation, we'll return None (calendar must exist first)
                return None

            # If calendar is soft-deleted from list, restore it on insert (Google API semantics)
            if calendar.deleted:
                calendar.deleted = False

            # Update calendar with list-specific settings
            if "summaryOverride" in entry_data:
                calendar.summary_override = entry_data["summaryOverride"]
            if "colorId" in entry_data:
                calendar.color_id = entry_data["colorId"]
            if "backgroundColor" in entry_data:
                calendar.background_color = entry_data["backgroundColor"]
            if "foregroundColor" in entry_data:
                calendar.foreground_color = entry_data["foregroundColor"]
            if "hidden" in entry_data:
                calendar.hidden = entry_data["hidden"]
            if "selected" in entry_data:
                calendar.selected = entry_data["selected"]
            if "defaultReminders" in entry_data:
                calendar.default_reminders = (
                    json.dumps(entry_data["defaultReminders"])
                    if entry_data["defaultReminders"]
                    else None
                )
            if "notificationSettings" in entry_data:
                calendar.notification_settings = (
                    json.dumps(entry_data["notificationSettings"])
                    if entry_data["notificationSettings"]
                    else None
                )

            session.commit()

            return self._format_calendar_list_entry(
                calendar, show_hidden=True, user_id=user_id
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting calendar entry {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def update_calendar_entry(
        self, user_id: str, calendar_id: str, entry_data: Dict, is_patch: bool = True
    ) -> Optional[Dict]:
        """Update a calendar entry in user's calendar list"""
        session = get_session(self.database_id)
        try:
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )

            if not calendar or calendar.deleted:
                return None

            # Update fields based on entry data
            if "summaryOverride" in entry_data:
                calendar.summary_override = entry_data["summaryOverride"]
            elif not is_patch:
                # For PUT requests, clear the field if not provided
                calendar.summary_override = None

            if "colorId" in entry_data:
                calendar.color_id = entry_data["colorId"]
            elif not is_patch:
                calendar.color_id = None

            if "backgroundColor" in entry_data:
                calendar.background_color = entry_data["backgroundColor"]
            elif not is_patch:
                calendar.background_color = None

            if "foregroundColor" in entry_data:
                calendar.foreground_color = entry_data["foregroundColor"]
            elif not is_patch:
                calendar.foreground_color = None

            if "hidden" in entry_data:
                calendar.hidden = entry_data["hidden"]
            elif not is_patch:
                # For PUT (full update), set default for NOT NULL fields when not provided
                calendar.hidden = False

            if "selected" in entry_data:
                calendar.selected = entry_data["selected"]
            elif not is_patch:
                # For PUT (full update), set default for NOT NULL fields when not provided
                calendar.selected = True

            if "defaultReminders" in entry_data:
                calendar.default_reminders = (
                    json.dumps(entry_data["defaultReminders"])
                    if entry_data["defaultReminders"]
                    else None
                )
            elif not is_patch:
                calendar.default_reminders = None

            if "notificationSettings" in entry_data:
                calendar.notification_settings = (
                    json.dumps(entry_data["notificationSettings"])
                    if entry_data["notificationSettings"]
                    else None
                )
            elif not is_patch:
                calendar.notification_settings = None

            if "conferenceProperties" in entry_data:
                calendar.conference_properties = (
                    json.dumps(entry_data["conferenceProperties"])
                    if entry_data["conferenceProperties"]
                    else None
                )

            session.commit()

            return self._format_calendar_list_entry(
                calendar, show_hidden=True, user_id=user_id
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating calendar entry {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def delete_calendar_entry(self, user_id: str, calendar_id: str) -> bool:
        """Remove a calendar from user's calendar list"""
        session = get_session(self.database_id)
        try:
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )

            if not calendar or calendar.deleted:
                return False

            # For primary calendar, we can't remove it from the list
            if calendar.is_primary:
                raise ValueError("Cannot remove primary calendar from calendar list")

            # Mark as deleted (soft delete for calendar list)
            calendar.deleted = True
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting calendar entry {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def _has_acl_role(
        self, user_id: str, calendar: Calendar, allowed_roles: List[str]
    ) -> bool:
        """Check if user has required ACL permissions for calendar operations"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                logger.warning(f"No user found with ID: {user_id}")
                return False

            acls = (
                session.query(ACLs)
                .join(Scope, ACLs.scope_id == Scope.id)
                .filter(
                    ACLs.calendar_id == calendar.calendar_id,
                    Scope.type == "user",
                    Scope.value == user.email,
                )
                .all()
            )

            if not acls:
                logger.warning(
                    f"No ACL found for user {user.email} on calendar {calendar.calendar_id}"
                )
                return False

            for acl in acls:
                if acl.role.value in allowed_roles:
                    return True

            logger.warning(
                f"User {user.email} has ACLs but lacks required roles: {allowed_roles}"
            )
            return False
        finally:
            session.close()

    def check_calendar_acl_permissions(
        self, user_id: str, calendar_id: str, allowed_roles: List[str]
    ) -> Calendar:
        """Check if user has required ACL permissions for calendar operations and return calendar"""
        session = get_session(self.database_id)
        try:
            calendar = (
                session.query(Calendar)
                .filter(Calendar.calendar_id == calendar_id)
                .first()
            )
            if not calendar:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Calendar with id '{calendar_id}' does not exist",
                )

            if not self._has_acl_role(user_id, calendar, allowed_roles):
                raise PermissionError(
                    f"User '{user_id}' lacks required roles: {allowed_roles}"
                )

            return calendar
        finally:
            session.close()

    def watch_calendar_list(self, watch_request: WatchRequest, user_id: str) -> Dict:
        """Set up watch notifications for calendar list changes"""

        session = get_session(self.database_id)
        try:
            # Generate unique resource ID for settings watch
            resource_id = f"calendarList-{user_id}-{uuid.uuid4().hex[:8]}"
            resource_uri = "/calendars/me/calendarList"

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
                resource_type="calendar_list",
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
            return {
                "kind": "api#channel",
                "id": watch_channel.id,
                "resourceId": resource_id,
                "resourceUri": resource_uri,
                "token": watch_channel.webhook_token,
                "expiration": expires_at.isoformat() + "Z" if expires_at else None,
            }

        except Exception as e:
            session.rollback()
            logger.error(f"Error creating settings watch channel: {e}")
            raise
        finally:
            session.close()

    def _get_user_access_role(self, user_id: str, calendar: Calendar) -> str:
        """Get the user's actual access role for a calendar based on ACL entries"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                return "none"

            acls = (
                session.query(ACLs)
                .join(Scope, ACLs.scope_id == Scope.id)
                .filter(
                    ACLs.calendar_id == calendar.calendar_id,
                    Scope.type == "user",
                    Scope.value == user.email,
                )
                .all()
            )

            if not acls:
                return "none"

            # Return the highest permission level found
            role_hierarchy = {
                "none": 0,
                "freeBusyReader": 1,
                "reader": 2,
                "writer": 3,
                "owner": 4,
            }
            highest_role = "none"
            highest_weight = 0

            for acl in acls:
                role_weight = role_hierarchy.get(acl.role.value, 0)
                if role_weight > highest_weight:
                    highest_weight = role_weight
                    highest_role = acl.role.value

            return highest_role
        finally:
            session.close()

    def _format_calendar_list_entry(
        self, calendar: Calendar, show_hidden: bool = False, user_id: str = None
    ) -> Optional[Dict]:
        """Format calendar model for calendar list API response"""
        # Skip hidden calendars unless explicitly requested
        if calendar.hidden and not show_hidden:
            return None

        # Determine access role from actual ACL data
        access_role = "owner" if calendar.is_primary else "writer"  # Default fallback
        if user_id:
            access_role = self._get_user_access_role(user_id, calendar)

        formatted = {
            "kind": "calendar#calendarListEntry",
            "etag": f"etag-list-{calendar.calendar_id}-{calendar.updated_at.isoformat() if calendar.updated_at else ''}",
            "id": calendar.calendar_id,
            "summary": calendar.summary_override or calendar.summary,
            "description": calendar.description,
            "location": calendar.location,
            "timeZone": calendar.time_zone,
            "accessRole": access_role,
            "primary": calendar.is_primary,
            "hidden": calendar.hidden or False,
            "selected": calendar.selected if calendar.selected is not None else True,
            "deleted": calendar.deleted or False,
        }

        # Add optional fields if present
        if calendar.summary_override:
            formatted["summaryOverride"] = calendar.summary_override

        if calendar.color_id:
            formatted["colorId"] = calendar.color_id

        if calendar.background_color:
            formatted["backgroundColor"] = calendar.background_color

        if calendar.foreground_color:
            formatted["foregroundColor"] = calendar.foreground_color

        if calendar.default_reminders:
            try:
                raw_items = json.loads(calendar.default_reminders)
                sanitized: List[Dict] = []
                if isinstance(raw_items, list):
                    for item in raw_items:
                        if not isinstance(item, dict):
                            continue
                        method = item.get("method")
                        minutes = item.get("minutes")
                        if (
                            method in ALLOWED_REMINDER_METHODS
                            and isinstance(minutes, int)
                            and minutes >= 0
                        ):
                            sanitized.append({"method": method, "minutes": minutes})
                if sanitized:
                    formatted["defaultReminders"] = sanitized
            except Exception:
                # Ignore malformed stored data
                pass

        if calendar.notification_settings:
            try:
                raw = json.loads(calendar.notification_settings)
                if isinstance(raw, dict):
                    notifs = raw.get("notifications")
                    sanitized_notifs: List[Dict] = []
                    if isinstance(notifs, list):
                        for n in notifs:
                            if not isinstance(n, dict):
                                continue
                            m = n.get("method")
                            t = n.get("type")
                            if (
                                m in ALLOWED_NOTIFICATION_METHODS
                                and t in ALLOWED_NOTIFICATION_TYPES
                            ):
                                sanitized_notifs.append({"method": m, "type": t})
                    if sanitized_notifs:
                        formatted["notificationSettings"] = {
                            "notifications": sanitized_notifs
                        }
            except Exception:
                # Ignore malformed stored data
                pass

        # Add conference properties
        if calendar.conference_properties:
            try:
                # conference_properties is stored as JSON string, parse it
                conf_props = json.loads(calendar.conference_properties)
                formatted["conferenceProperties"] = conf_props
            except Exception:
                # If parsing fails, provide default
                formatted["conferenceProperties"] = {
                    "allowedConferenceSolutionTypes": []
                }
        else:
            formatted["conferenceProperties"] = {"allowedConferenceSolutionTypes": []}

        return formatted

    def _encode_page_token(self, offset: int) -> str:
        """Encode offset as a page token"""
        try:
            token_data = str(offset)
            return base64.b64encode(token_data.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding page token: {e}")
            return ""

    def _decode_page_token(self, token: str) -> int:
        """Decode page token to get offset"""
        try:
            # Handle legacy case where raw numbers might be passed
            if token.isdigit():
                logger.warning(
                    f"Raw numeric page token received: {token}. This should be a base64-encoded token."
                )
                return int(token)

            # Add padding if needed for base64 decoding
            missing_padding = len(token) % 4
            if missing_padding:
                token += "=" * (4 - missing_padding)

            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            return int(decoded)
        except Exception as e:
            logger.error(f"Error decoding page token: {e}")
            raise ValueError(
                f"Invalid page token: {token}. Page tokens should only be generated by the API."
            )

    def _encode_sync_token(self, timestamp: datetime) -> str:
        """Encode timestamp as a sync token"""
        try:
            # Use ISO format timestamp for sync token
            token_data = timestamp.isoformat()
            return base64.b64encode(token_data.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding sync token: {e}")
            return ""

    def _decode_sync_token(self, token: str) -> datetime:
        """Decode sync token to get timestamp"""
        try:
            # Add padding if needed for base64 decoding
            missing_padding = len(token) % 4
            if missing_padding:
                token += "=" * (4 - missing_padding)

            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            return datetime.fromisoformat(decoded)
        except Exception as e:
            logger.error(f"Error decoding sync token: {e}")
            raise ValueError(
                f"Invalid sync token. Token may have expired or is malformed."
            )
