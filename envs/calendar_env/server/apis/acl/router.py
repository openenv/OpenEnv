"""
ACL (Access Control List) API endpoints following Google Calendar API v3 structure.
Handles CRUD operations for calendar access rules.
"""

import logging
from typing import List, Optional

from database.managers.acl_manager import ACLManager, get_acl_manager
from database.session_manager import CalendarSessionManager
from fastapi import APIRouter, Depends, HTTPException, Query, status
from middleware.auth import get_user_context
from pydantic import ValidationError
from schemas.acl import (
    ACLListResponse,
    ACLRule,
    ACLRuleInput,
    ACLWatchRequest,
    Channel,
    InsertACLRule,
    PatchACLRuleInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calendars", tags=["acl"])
session_manager = CalendarSessionManager()


def get_acl_manager_instance(database_id: str, user_id: str) -> ACLManager:
    session = session_manager.get_session(database_id)
    return get_acl_manager(session, user_id)


@router.get(
    "/{calendarId}/acl", response_model=ACLListResponse, operation_id="list_acl_rules"
)
def list_acl_rules(
    calendarId: str,
    maxResults: Optional[int] = Query(
        100,
        ge=1,
        le=250,
        description="Maximum number of entries returned on one result page",
    ),
    pageToken: Optional[str] = Query(
        None, description="Token specifying which result page to return"
    ),
    showDeleted: bool = Query(
        False, description="Whether to include deleted ACLs in the result"
    ),
    syncToken: Optional[str] = Query(
        None, description="Token for incremental synchronization"
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)

        if not manager.validate_calendar_id(calendarId, user_id):
            raise ValueError(f"Calendar {calendarId} not found for user {user_id}")

        # If syncToken is provided, showDeleted must be True (Google Calendar API behavior)
        if syncToken and not showDeleted:
            showDeleted = True

        result = manager.list_rules(
            calendarId,
            max_results=maxResults,
            page_token=pageToken,
            show_deleted=showDeleted,
            sync_token=syncToken,
        )
        return result

    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{str(verr)}"
        )
    except HTTPException as he:
        if he.status_code == 410:  # Handle sync token expiration
            raise he
        raise
    except Exception as e:
        logger.error(f"Error listing ACL rules: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{calendarId}/acl/{ruleId}", response_model=ACLRule, operation_id="get_acl_rule"
)
def get_acl_rule(
    calendarId: str,
    ruleId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)
        rule = manager.get_rule(calendarId, ruleId)
        if not rule:
            raise HTTPException(status_code=404, detail="ACL rule not found")
        return rule
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving ACL rule {ruleId}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{calendarId}/acl", status_code=201, operation_id="insert_acl_rule")
def insert_acl_rule(
    calendarId: str,
    rule: ACLRuleInput,
    sendNotifications: bool = Query(
        True,
        description="Whether to send notifications about the calendar sharing change",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)

        if not manager.validate_calendar_id(calendarId, user_id):
            raise ValueError(f"Calendar {calendarId} not found for user {user_id}")

        return manager.insert_rule(
            calendarId, rule, send_notifications=sendNotifications
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error inserting ACL rule: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{calendarId}/acl/{ruleId}", operation_id="update_acl_rule")
def update_acl_rule(
    calendarId: str,
    ruleId: str,
    rule: ACLRuleInput,
    sendNotifications: bool = Query(
        True,
        description="Whether to send notifications about the calendar sharing change",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)

        if not manager.validate_calendar_id(calendarId, user_id):
            raise ValueError(f"Calendar {calendarId} not found for user {user_id}")

        updated = manager.update_rule(
            calendarId, ruleId, rule, send_notifications=sendNotifications
        )
        if not updated:
            raise ValueError(f"ACL rule not found")
        return updated
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating ACL rule {ruleId}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred {e}")


@router.patch("/{calendarId}/acl/{ruleId}", operation_id="patch_acl_rule")
def patch_acl_rule(
    calendarId: str,
    ruleId: str,
    rule: PatchACLRuleInput,
    sendNotifications: bool = Query(
        True,
        description="Whether to send notifications about the calendar sharing change",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)
        updated = manager.patch_rule(calendarId, ruleId, rule, sendNotifications)
        if not updated:
            raise ValueError(f"ACL rule not found")
        return updated
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error patching ACL rule {ruleId}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{calendarId}/acl/{ruleId}", status_code=status.HTTP_204_NO_CONTENT)
def delete_acl_rule(
    calendarId: str,
    ruleId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Deletes an ACL rule from the specified calendar.

    DELETE /calendars/{calendarId}/acl/{ruleId}
    """
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)

        success = manager.delete_rule(calendarId, ruleId)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ACL rule '{ruleId}' not found for calendar '{calendarId}'",
            )

        return None  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting ACL rule {ruleId} from calendar {calendarId}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{calendarId}/acl/watch", response_model=Channel, operation_id="watch_acl"
)
def watch_acl(
    calendarId: str,
    watch_request: ACLWatchRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Set up watch notifications for ACL changes on the specified calendar.

    POST /calendars/{calendarId}/acl/watch
    """
    try:
        database_id, user_id = user_context
        manager = get_acl_manager_instance(database_id, user_id)

        # Log the received request for debugging
        logger.info(f"Received watch request for calendar {calendarId}")
        logger.debug(f"Watch request data: {watch_request}")

        from database.models.user import User

        # Validate user exists in this database (ensures ownership context)
        from database.session_utils import get_session

        session = get_session(database_id)
        try:
            user_row = session.query(User).filter(User.user_id == user_id).first()
            if not user_row:
                raise HTTPException(
                    status_code=404, detail=f"User not found: {user_id}"
                )
        finally:
            session.close()

        # Set up watch channel
        channel = manager.watch_acl(user_id, calendarId, watch_request)

        return channel

    except ValidationError as e:
        logger.error(f"Validation error for calendar {calendarId}: {e.errors()}")
        validation_errors = []
        for error in e.errors():
            field = error.get("loc", ["unknown"])[0] if error.get("loc") else "unknown"
            message = error.get("msg", "validation failed")
            validation_errors.append(f"{field}: {message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation errors: - {' - '.join(validation_errors)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up ACL watch for calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
