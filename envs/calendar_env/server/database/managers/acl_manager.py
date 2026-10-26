import base64
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4

from database.models.acl import AclRole, ACLs, Scope
from database.models.calendar import Calendar
from database.models.user import User
from database.models.watch_channel import WatchChannel
from schemas.acl import (
    ACLListResponse,
    ACLRule,
    ACLRuleInput,
    Channel,
    PatchACLRuleInput,
    ScopeInput,
    ScopeOutput,
)
from services.notification_service import get_notification_service
from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ACLManager:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    def validate_calendar_id(self, calendar_id, user_id):
        calendar = (
            self.db.query(Calendar)
            .filter(
                Calendar.calendar_id == calendar_id,
                Calendar.user_id == user_id,
                Calendar.deleted == False,
            )
            .first()
        )
        if not calendar:
            return False
        return True

    def list_rules(
        self,
        calendar_id: str,
        max_results: int = 100,
        page_token: Optional[str] = None,
        show_deleted: bool = False,
        sync_token: Optional[str] = None,
    ) -> ACLListResponse:
        """
        List ACL rules for a calendar with pagination and filtering support.

        Args:
            calendar_id: The calendar ID
            max_results: Maximum number of entries per page (1-250, default 100)
            page_token: Token for pagination continuation
            show_deleted: Whether to include deleted ACLs (role == "none")
            sync_token: Token for incremental synchronization

        Returns:
            ACLListResponse with items and pagination tokens
        """
        try:
            # Handle sync token for incremental synchronization
            if sync_token:
                return self._handle_sync_request(
                    calendar_id, sync_token, max_results, show_deleted
                )

            # Base query
            query = self.db.query(ACLs).filter(
                ACLs.calendar_id == calendar_id, ACLs.user_id == self.user_id
            )

            # Filter deleted ACLs if not requested
            if not show_deleted:
                query = query.filter(ACLs.role != AclRole.none)

            # Order by created_at for consistent pagination
            query = query.order_by(ACLs.created_at, ACLs.id)

            # Handle pagination
            offset = 0
            if page_token:
                try:
                    offset = int(base64.b64decode(page_token).decode("utf-8"))
                except (ValueError, TypeError):
                    raise ValueError("Invalid pageToken")

            # Get one extra item to determine if there's a next page
            items = query.offset(offset).limit(max_results + 1).all()

            # Determine if there's a next page
            has_next_page = len(items) > max_results
            if has_next_page:
                items = items[:max_results]  # Remove the extra item
                next_page_token = base64.b64encode(
                    str(offset + max_results).encode("utf-8")
                ).decode("utf-8")
            else:
                next_page_token = None

            # Generate next sync token
            latest_updated = (
                self.db.query(ACLs.updated_at)
                .filter(ACLs.calendar_id == calendar_id, ACLs.user_id == self.user_id)
                .order_by(desc(ACLs.updated_at))
                .first()
            )

            next_sync_token = None
            if latest_updated and latest_updated[0]:
                sync_data = {
                    "calendar_id": calendar_id,
                    "timestamp": latest_updated[0].isoformat(),
                }
                next_sync_token = base64.b64encode(
                    json.dumps(sync_data).encode("utf-8")
                ).decode("utf-8")

            # Generate collection etag
            etag = f'"{uuid4()}"'

            # Convert database models to schema models
            acl_rules = []
            for item in items:
                if item.scope.type == "default":
                    scope = ScopeOutput(type=item.scope.type)
                else:
                    scope = ScopeOutput(type=item.scope.type, value=item.scope.value)
                acl_rule = ACLRule(
                    id=item.id,
                    calendar_id=item.calendar_id,
                    user_id=item.user_id,
                    role=item.role,
                    etag=item.etag,
                    scope=scope,
                )
                acl_rules.append(acl_rule)

            return ACLListResponse(
                etag=etag,
                items=acl_rules,
                nextPageToken=next_page_token,
                nextSyncToken=next_sync_token,
            )

        except Exception as e:
            logger.error(f"Error listing ACL rules: {e}")
            raise

    def _handle_sync_request(
        self, calendar_id: str, sync_token: str, max_results: int, show_deleted: bool
    ) -> ACLListResponse:
        """
        Handle incremental synchronization request.

        Args:
            calendar_id: The calendar ID
            sync_token: The sync token from previous request
            max_results: Maximum number of entries per page
            show_deleted: Whether to include deleted ACLs

        Returns:
            ACLListResponse with changes since the sync token
        """
        try:
            # Decode sync token
            sync_data = json.loads(base64.b64decode(sync_token).decode("utf-8"))
            last_sync_time = datetime.fromisoformat(sync_data["timestamp"])

            # Verify calendar ID matches
            if sync_data.get("calendar_id") != calendar_id:
                raise ValueError("Sync token calendar ID mismatch")

            # Check if sync token is too old (expired)
            if (datetime.utcnow() - last_sync_time).days > 7:  # 7 days expiration
                from fastapi import HTTPException

                raise HTTPException(status_code=410, detail="Sync token expired")

            # Query for changes since last sync
            query = self.db.query(ACLs).filter(
                ACLs.calendar_id == calendar_id,
                ACLs.user_id == self.user_id,
                ACLs.updated_at > last_sync_time,
            )

            # Always include deleted ACLs in sync requests (Google API behavior)
            # Order by updated_at for consistent results
            query = query.order_by(ACLs.updated_at, ACLs.id)

            # Apply limit
            items = query.limit(max_results + 1).all()

            # Determine if there's more data
            has_more = len(items) > max_results
            if has_more:
                items = items[:max_results]
                # For sync requests, we don't use page tokens, just return what we have
                # Client should make another sync request with updated sync token

            # Generate new sync token based on latest item
            next_sync_token = None
            if items:
                latest_time = max(item.updated_at for item in items)
                sync_data = {
                    "calendar_id": calendar_id,
                    "timestamp": latest_time.isoformat(),
                }
                next_sync_token = base64.b64encode(
                    json.dumps(sync_data).encode("utf-8")
                ).decode("utf-8")
            else:
                # No changes, return same sync token
                next_sync_token = sync_token

            # Generate collection etag
            etag = f'"{uuid4()}"'

            # Convert database models to schema models
            acl_rules = []
            for item in items:
                scope = ScopeInput(type=item.scope.type, value=item.scope.value)
                acl_rule = ACLRule(
                    id=item.id,
                    calendar_id=item.calendar_id,
                    user_id=item.user_id,
                    role=item.role,
                    etag=item.etag,
                    scope=scope,
                )
                acl_rules.append(acl_rule)

            return ACLListResponse(
                etag=etag,
                items=acl_rules,
                nextPageToken=None,  # No page tokens in sync mode
                nextSyncToken=next_sync_token,
            )

        except json.JSONDecodeError:
            raise ValueError("Invalid sync token format")
        except Exception as e:
            logger.error(f"Error handling sync request: {e}")
            raise

    def get_rule(self, calendar_id: str, rule_id: str):
        """
        Retrieve a specific ACL rule by calendar ID and rule ID (must be owned).
        """
        return (
            self.db.query(ACLs)
            .filter_by(id=rule_id, calendar_id=calendar_id, user_id=self.user_id)
            .first()
        )

    def insert_rule(
        self, calendar_id: str, rule: ACLRuleInput, send_notifications: bool = True
    ):
        """
        Insert a new ACL rule after validating scope existence.

        Args:
            calendar_id: The calendar ID
            rule: The ACL rule input data
            send_notifications: Whether to send notifications about the calendar sharing change (default: True)

        Returns the inserted ACL rule.
        """
        # Look up scope (must exist)
        if rule.scope.type == "default":
            scope = self.db.query(Scope).filter(Scope.type == rule.scope.type).first()
        else:
            if rule.scope.value is None:
                scope = (
                    self.db.query(Scope).filter(Scope.type == rule.scope.type).first()
                )
            else:
                scope = (
                    self.db.query(Scope)
                    .filter(
                        Scope.type == rule.scope.type, Scope.value == rule.scope.value
                    )
                    .first()
                )

        if not scope:
            raise ValueError(f"Scope ({rule.scope.type}, {rule.scope.value}) not found")

        # Create ACL rule
        rule_id = f"{uuid4()}"
        db_rule = ACLs(
            id=rule_id,
            calendar_id=calendar_id,
            user_id=self.user_id,
            role=rule.role,
            scope_id=scope.id,
            etag=f'"{uuid4()}"',
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.db.add(db_rule)
        self.db.commit()
        self.db.refresh(db_rule)

        # Send notification for ACL rule insertion if notifications are enabled
        if send_notifications:
            self._send_acl_notification(
                calendar_id,
                "insert",
                {
                    "id": db_rule.id,
                    "calendar_id": db_rule.calendar_id,
                    "user_id": db_rule.user_id,
                    "role": db_rule.role.value,
                    "scope": scope.as_dict(),
                    "etag": db_rule.etag,
                },
            )

        response = {
            "kind": "calendar#aclRule",
            "etag": db_rule.etag,
            "id": db_rule.id,
            "scope": {},
            "role": db_rule.role.value,
        }
        scope_dict = scope.as_dict()
        response["scope"]["type"] = scope_dict.get("type")
        if scope_dict.get("value") != "public":
            response["scope"]["value"] = scope_dict.get("value")
        return response

    def update_rule(
        self,
        calendar_id: str,
        rule_id: str,
        rule: ACLRuleInput,
        send_notifications: bool = True,
    ):
        """
        Fully replace an existing ACL rule's role and scope.

        Returns the updated rule or None if not found.
        """
        db_rule = self.get_rule(calendar_id, rule_id)

        if not db_rule:
            return None

        if rule.role is not None:
            db_rule.role = rule.role
        # Update scope
        if rule.scope is not None:
            if db_rule.scope:
                db_rule.scope.type = rule.scope.type
                if rule.scope.value is not None and rule.scope.type != "default":
                    if rule.scope.type in ["user", "group"]:
                        # Validate whether value contains valid email address
                        user = (
                            self.db.query(User)
                            .filter(
                                User.email == rule.scope.value, User.is_active == True
                            )
                            .first()
                        )
                        if user is None:
                            raise ValueError(
                                "Invalid data in 'value field'. Please enter an existing email id in 'value' field"
                            )
                    db_rule.scope.value = rule.scope.value
            else:
                raise ValueError("ACL rule has no associated scope object to update.")
        db_rule.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_rule)

        # Send notification for ACL rule update if notifications are enabled
        if send_notifications:
            self._send_acl_notification(
                calendar_id,
                "update",
                {
                    "id": db_rule.id,
                    "calendar_id": db_rule.calendar_id,
                    "user_id": db_rule.user_id,
                    "role": db_rule.role.value,
                    "scope": db_rule.scope.as_dict(),
                    "etag": db_rule.etag,
                },
            )

        response = {
            "kind": "calendar#aclRule",
            "etag": db_rule.etag,
            "id": db_rule.id,
            "scope": {},
            "role": db_rule.role.value,
        }
        scope_dict = db_rule.scope.as_dict()
        response["scope"]["type"] = scope_dict.get("type")
        if scope_dict.get("value") != "public":
            response["scope"]["value"] = scope_dict.get("value")
        return response

    def patch_rule(
        self,
        calendar_id: str,
        rule_id: str,
        rule: PatchACLRuleInput,
        send_notifications: bool = True,
    ):
        """
        Partially update an ACL rule's role or scope if provided.

        Returns the updated rule or None if not found.
        """
        db_rule = self.get_rule(calendar_id, rule_id)
        if not db_rule:
            return None

        if rule.role is not None:
            db_rule.role = rule.role

        if rule.scope is not None:
            # Patch the related Scope object, not as a dict
            if db_rule.scope:
                db_rule.scope.type = rule.scope.type
                if rule.scope.value is not None and rule.scope.type != "default":
                    if rule.scope.type in ["user", "group"]:
                        # Validate whether value contains valid email address
                        user = (
                            self.db.query(User)
                            .filter(
                                User.email == rule.scope.value, User.is_active == True
                            )
                            .first()
                        )
                        if user is None:
                            raise ValueError(
                                "Invalid data in 'value field'. Please enter an existing email id in 'value' field"
                            )
                    db_rule.scope.value = rule.scope.value
            else:
                raise ValueError("ACL rule has no associated scope object to patch.")

        db_rule.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_rule)

        # Send notification for ACL rule patch if notifications are enabled
        if send_notifications:
            self._send_acl_notification(
                calendar_id,
                "update",
                {
                    "id": db_rule.id,
                    "calendar_id": db_rule.calendar_id,
                    "user_id": db_rule.user_id,
                    "role": db_rule.role.value,
                    "scope": db_rule.scope.as_dict(),
                    "etag": db_rule.etag,
                },
            )

        response = {
            "kind": "calendar#aclRule",
            "etag": db_rule.etag,
            "id": db_rule.id,
            "scope": {},
            "role": db_rule.role.value,
        }
        scope_dict = db_rule.scope.as_dict()
        response["scope"]["type"] = scope_dict.get("type")
        if scope_dict.get("value") != "public":
            response["scope"]["value"] = scope_dict.get("value")
        return response

    def delete_rule(self, calendar_id: str, rule_id: str) -> bool:
        """
        Delete an ACL rule by ID and calendar ID. Only the calendar owner can delete ACLs.

        Returns:
            True if deleted, False if rule not found.
        Raises:
            Exception if DB operation fails.
        """
        session = self.db
        try:
            from database.models import Calendar  # to resolve join dependency

            rule = (
                session.query(ACLs)
                .join(Calendar)
                .filter(
                    ACLs.id == rule_id,
                    ACLs.calendar_id == calendar_id,
                    Calendar.user_id == self.user_id,
                )
                .first()
            )

            if not rule:
                return False

            # Capture rule data before deletion for notification
            rule_data = {
                "id": rule.id,
                "calendar_id": rule.calendar_id,
                "user_id": rule.user_id,
                "role": rule.role.value,
                "scope": rule.scope.as_dict() if rule.scope else {},
                "etag": rule.etag,
            }

            session.delete(rule)
            session.commit()

            # Send notification for ACL rule deletion
            self._send_acl_notification(calendar_id, "delete", rule_data)

            return True

        except Exception as e:
            session.rollback()
            raise

    def watch_acl(
        self, user_id: str, calendar_id: str, watch_request: Dict[str, Any]
    ) -> Channel:
        """
        Set up watch notifications for ACL changes.

        POST /calendars/{calendarId}/acl/watch
        """
        try:
            session = self.db

            # Generate unique resource ID for events watch
            resource_id = f"acl-{calendar_id}-{uuid.uuid4().hex[:8]}"
            resource_uri = f"/calendars/{calendar_id}/acl"
            # Validate that the user has access to the calendar

            # Default expiration: 24 hours from now
            expires_at = datetime.utcnow() + timedelta(hours=24)

            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

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
                resource_type="acl",
                calendar_id=calendar_id,
                user_id=user_id,
                webhook_address=watch_request.address,
                webhook_token=watch_request.token,
                webhook_type=watch_request.type,
                params=json.dumps(watch_request.params.model_dump())
                if watch_request.params
                else None,
                created_at=datetime.utcnow(),
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

            # Create response channel
            channel = Channel(
                id=watch_channel.id,
                resourceId=resource_id,
                resourceUri=watch_channel.resource_uri,
                token=watch_channel.webhook_token,
                expiration=watch_channel.expires_at.isoformat() + "Z"
                if watch_channel.expires_at
                else None,
            )

            logger.info(
                f"Set up watch channel {watch_channel.id} for ACL changes in calendar {calendar_id}"
            )
            return channel

        except Exception as e:
            logger.error(f"Error setting up ACL watch for calendar {calendar_id}: {e}")
            self.db.rollback()
            raise

    def cleanup_expired_channels(self) -> int:
        """
        Clean up expired watch channels for this user

        Returns:
            Number of channels cleaned up
        """
        try:
            current_time = datetime.utcnow()

            # Find expired channels for this user
            expired_channels = (
                self.db.query(WatchChannel)
                .filter(
                    WatchChannel.user_id == self.user_id,
                    WatchChannel.expires_at < current_time,
                    WatchChannel.is_active == "true",
                )
                .all()
            )

            cleanup_count = 0
            for channel in expired_channels:
                channel.is_active = "false"
                cleanup_count += 1

            if cleanup_count > 0:
                self.db.commit()
                logger.info(
                    f"Cleaned up {cleanup_count} expired watch channels for user {self.user_id}"
                )

            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up expired channels: {e}")
            self.db.rollback()
            return 0

    def _send_acl_notification(
        self, calendar_id: str, change_type: str, acl_data: Dict[str, Any]
    ):
        """
        Send ACL change notification to all active watch channels for the calendar

        Args:
            calendar_id: The calendar ID
            change_type: Type of change ("insert", "update", "delete")
            acl_data: The ACL rule data
        """
        try:
            notification_service = get_notification_service(self.db)
            notifications_sent = notification_service.notify_acl_change(
                calendar_id, change_type, acl_data
            )

            if notifications_sent > 0:
                logger.debug(
                    f"Sent {notifications_sent} notifications for ACL {change_type} in calendar {calendar_id}"
                )

        except Exception as e:
            logger.error(f"Error sending ACL notification: {e}")
            # Don't raise the exception as notification failure shouldn't break the main operation


def get_acl_manager(db: Session, user_id: str) -> ACLManager:
    return ACLManager(db, user_id)
